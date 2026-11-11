"""
Main Multi-GPU Training Script for Hybrid CNN-BiLSTM-CTC ASR Pipeline.
Supports PyTorch Distributed Data Parallel (DDP), Mixed Precision (bfloat16/float16),
Two-Stage Curriculum Learning, and Model Checkpointing.
"""

import os
import sys
import tempfile

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    mpl_dir = os.path.join(tempfile.gettempdir(), ".matplotlib")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
except Exception:
    pass

# Guard against broken/incompatible torchvision in environment (e.g. RuntimeError: operator torchvision::nms does not exist)
try:
    import torchvision
except Exception:
    sys.modules["torchvision"] = None
    sys.modules["torchvision.transforms"] = None
    sys.modules["torchvision.ops"] = None
    try:
        import transformers.utils.import_utils as _iu
        _iu._torchvision_available = False
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

import math
import yaml
import json
import argparse
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import LibriSpeechDataset
from src.data.collate import DynamicBucketSampler, ASRCollateFn
from src.model.hybrid_asr import HybridASRModel
from src.decoding.beam_search import CTCDecoder
from src.decoding.evaluate_wer import compute_metrics
from src.visualization import TrainingHistory, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN-Enhanced ASR Model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()


def setup_distributed() -> Tuple[bool, int, int, torch.device]:
    """Initializes distributed environment if launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_distributed = False
    return is_distributed, rank, world_size, device


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def build_optimizer_and_scheduler(
    model: HybridASRModel,
    config: Dict[str, Any],
    epoch: int,
    total_steps_per_epoch: int
):
    """
    Builds optimizer & scheduler according to the two-stage curriculum:
    Stage 1 (Epochs 1-10): Backbone frozen, Custom head lr = 3e-4.
    Stage 2 (Epochs 11-30): Backbone unfrozen, discriminative lrs.
    """
    freeze_epochs = config["model"]["backbone"]["freeze_epochs"]
    is_stage1 = epoch <= freeze_epochs

    if is_stage1:
        model.freeze_backbone()
        lr_head = float(config["training"]["stage1"]["lr"])
        weight_decay = float(config["training"]["stage1"]["weight_decay"])
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=weight_decay)
        
        # Warmup (1 epoch) + Cosine Annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(freeze_epochs * total_steps_per_epoch), eta_min=1e-5
        )
    else:
        model.unfreeze_backbone()
        backbone_lr = float(config["training"]["stage2"]["backbone_lr"])
        head_lr = float(config["training"]["stage2"]["head_lr"])
        weight_decay = float(config["training"]["stage2"]["weight_decay"])

        param_groups = model.get_parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        remaining_epochs = max(config["training"]["total_epochs"] - freeze_epochs, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(remaining_epochs * total_steps_per_epoch, 1), eta_min=1e-6
        )

    return optimizer, scheduler, is_stage1


def train_one_epoch(
    model: nn.Module,
    raw_model: HybridASRModel,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    rank: int,
    amp_dtype: torch.dtype,
    max_grad_norm: float,
    grad_accum_steps: int = 1,
    is_distributed: bool = False
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Train", disable=(rank != 0))
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        audio_padded = batch["audio_padded"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        targets_padded = batch["targets_padded"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            # Forward pass: log_probs (Batch, Time, Vocab)
            log_probs = model(audio_padded)

            # Compute output frame lengths from wav2vec2 feature extractor
            feat_lens = raw_model.wav2vec2._get_feat_extract_output_lengths(input_lengths)
            # Ensure output lengths don't exceed model temporal dimension
            feat_lens = torch.clamp(feat_lens, max=log_probs.size(1))

            # CTCLoss expects (Time, Batch, Vocab)
            log_probs_transposed = log_probs.permute(1, 0, 2).float()

            raw_loss = criterion(log_probs_transposed, targets_padded, feat_lens, target_lengths)
            loss = raw_loss / grad_accum_steps

        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            if rank == 0:
                print(f"[Warning] NaN/Inf loss encountered at step {step}, skipping batch.")
            continue

        is_accumulating = (step + 1) % grad_accum_steps != 0 and (step + 1) < len(dataloader)
        if is_distributed and is_accumulating and hasattr(model, "no_sync"):
            with model.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        if not is_accumulating:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        loss_val = raw_loss.item()
        total_loss += loss_val
        num_batches += 1

        if rank == 0:
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    raw_model: HybridASRModel,
    dataloader: DataLoader,
    decoder: CTCDecoder,
    device: torch.device,
    rank: int,
    amp_dtype: torch.dtype
) -> Dict[str, float]:
    model.eval()
    all_predictions = []
    all_references = []

    pbar = tqdm(dataloader, desc="Validation", disable=(rank != 0))

    for batch in pbar:
        audio_padded = batch["audio_padded"].to(device, non_blocking=True)
        refs = batch["transcripts"]

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            log_probs = model(audio_padded)

        # Fast greedy decoding during validation epochs
        preds = decoder.decode_greedy(log_probs)

        all_predictions.extend(preds)
        all_references.extend(refs)

    metrics = compute_metrics(all_predictions, all_references)
    return metrics


def save_checkpoint(
    checkpoint_dir: str,
    raw_model: HybridASRModel,
    tokenizer: CharacterTokenizer,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    epoch: int
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 1. Model weights
    weights_path = os.path.join(checkpoint_dir, "model_weights.pt")
    torch.save(raw_model.state_dict(), weights_path)

    # 2. Vocabulary
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    tokenizer.save_vocab(vocab_path)

    # 3. Model config
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config["model"], f, indent=2)

    # 4. Metrics
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, **metrics}, f, indent=2)

    print(f"✓ Saved best model checkpoint to {checkpoint_dir} (Epoch {epoch}, WER: {metrics['wer_percent']:.2f}%)")


def main():
    args = parse_args()
    config_path = args.config
    if not os.path.exists(config_path):
        if os.path.exists(os.path.join("configs", config_path)):
            config_path = os.path.join("configs", config_path)
        elif config_path in ("config.yaml", "train_config.yaml") and os.path.exists("configs/train_config.yaml"):
            config_path = "configs/train_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    is_distributed, rank, world_size, device = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("Starting CNN-Enhanced ASR Pipeline Training")
        print(f"Distributed: {is_distributed} | World Size: {world_size} | Device: {device}")
        print("=" * 60)

    # Tokenizer
    tokenizer = CharacterTokenizer()

    # Datasets
    data_cfg = config["data"]
    train_dataset = LibriSpeechDataset(
        dataset_root=data_cfg["dataset_root"],
        split=data_cfg["train_split"],
        tokenizer=tokenizer,
        target_sample_rate=data_cfg["sample_rate"],
        is_train=True,
        noise_prob=data_cfg["augmentation"]["noise_prob"],
        noise_snr_min=data_cfg["augmentation"]["noise_snr_min"],
        noise_snr_max=data_cfg["augmentation"]["noise_snr_max"]
    )

    val_dataset = LibriSpeechDataset(
        dataset_root=data_cfg["dataset_root"],
        split=data_cfg["val_split"],
        tokenizer=tokenizer,
        target_sample_rate=data_cfg["sample_rate"],
        is_train=False
    )

    batch_size = args.batch_size or config["training"]["batch_size"]
    collate_fn = ASRCollateFn(target_pad_value=tokenizer.blank_id)

    # DDP Sampler or Dynamic Bucket Sampler
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=config["system"]["num_workers"],
            pin_memory=config["system"]["pin_memory"]
        )
    else:
        train_sampler = None
        lengths = [s["audio_len"] for s in train_dataset.samples]
        bucket_sampler = DynamicBucketSampler(lengths=lengths, batch_size=batch_size, num_buckets=10, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=bucket_sampler,
            collate_fn=collate_fn,
            num_workers=config["system"]["num_workers"],
            pin_memory=config["system"]["pin_memory"]
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["system"]["num_workers"],
        pin_memory=config["system"]["pin_memory"]
    )

    # Instantiate Model
    model_cfg = config["model"]
    raw_model = HybridASRModel(
        backbone_name=model_cfg["backbone"]["name"],
        freeze_backbone=True,
        feature_dim=model_cfg["backbone"]["feature_dim"],
        cnn_channels=model_cfg["cnn"]["channels"],
        cnn_projection_dim=model_cfg["cnn"]["projection_dim"],
        lstm_hidden_size=model_cfg["rnn"]["hidden_size"],
        lstm_num_layers=model_cfg["rnn"]["num_layers"],
        lstm_dropout=model_cfg["rnn"]["dropout"],
        dense_hidden_dim=model_cfg["dense"]["hidden_dim"],
        dense_dropout=model_cfg["dense"]["dropout"],
        vocab_size=model_cfg["vocab_size"]
    ).to(device)

    if is_distributed:
        device_ids = [device.index] if device.type == "cuda" else None
        model = DDP(raw_model, device_ids=device_ids, find_unused_parameters=False)
    else:
        model = raw_model

    criterion = nn.CTCLoss(blank=config["training"]["ctc_blank_idx"], zero_infinity=True)

    # Precision
    precision_str = config["system"]["mixed_precision"]
    amp_dtype = torch.bfloat16 if precision_str == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

    decoder = CTCDecoder(tokenizer=tokenizer)

    total_epochs = args.epochs or config["training"]["total_epochs"]
    freeze_epochs = model_cfg["backbone"]["freeze_epochs"]
    checkpoint_dir = config["system"]["checkpoint_dir"]
    grad_accum_steps = config["training"].get("grad_accum_steps", 1)

    best_wer = float("inf")
    current_stage = None
    optimizer, scheduler = None, None
    history = TrainingHistory(log_path=os.path.join(checkpoint_dir, "training_history.json"))

    for epoch in range(1, total_epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Check curriculum transition
        is_stage1 = (epoch <= freeze_epochs)
        stage_name = "Stage 1 (Head Warmup)" if is_stage1 else "Stage 2 (Joint Fine-Tuning)"

        if stage_name != current_stage:
            current_stage = stage_name
            if rank == 0:
                print(f"\n>>> Transitioning to {current_stage} at Epoch {epoch}")
            steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
            optimizer, scheduler, _ = build_optimizer_and_scheduler(
                raw_model, config, epoch, steps_per_epoch
            )

        avg_loss = train_one_epoch(
            model=model,
            raw_model=raw_model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            rank=rank,
            amp_dtype=amp_dtype,
            max_grad_norm=config["training"]["max_grad_norm"],
            grad_accum_steps=grad_accum_steps,
            is_distributed=is_distributed
        )

        if rank == 0:
            print(f"Epoch {epoch}/{total_epochs} [{current_stage}] Complete. Avg Loss: {avg_loss:.4f}")

            # Validation
            metrics = evaluate(
                model=model,
                raw_model=raw_model,
                dataloader=val_loader,
                decoder=decoder,
                device=device,
                rank=rank,
                amp_dtype=amp_dtype
            )
            print(f"Epoch {epoch} Validation - WER: {metrics['wer_percent']:.2f}% | CER: {metrics['cer_percent']:.2f}%")

            # Extract current learning rates
            lr_h = optimizer.param_groups[-1]["lr"]
            lr_bb = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0

            # Record history
            history.record(
                epoch=epoch,
                train_loss=avg_loss,
                val_wer_percent=metrics["wer_percent"],
                val_cer_percent=metrics["cer_percent"],
                stage=stage_name,
                lr_head=lr_h,
                lr_backbone=lr_bb
            )

            # Auto-generate training progress curves
            plot_training_curves(
                history=history,
                output_dir=checkpoint_dir,
                filename_prefix="training_curves",
                stage1_epochs=freeze_epochs
            )

            if metrics["wer"] < best_wer:
                best_wer = metrics["wer"]
                save_checkpoint(checkpoint_dir, raw_model, tokenizer, config, metrics, epoch)

    cleanup_distributed(is_distributed)
    if rank == 0:
        print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()

