"""
Standalone Benchmark Evaluation Script.
Evaluates trained CNN-Enhanced ASR Model on LibriSpeech test-clean benchmark,
computes WER/CER with Beam Search & Language Model, and runs
Educational Reading Fluency & Articulation Assessment.
"""

import os
import sys
import tempfile

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

import json
import argparse
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import LibriSpeechDataset
from src.data.collate import ASRCollateFn
from src.model.hybrid_asr import HybridASRModel
from src.decoding.beam_search import CTCDecoder
from src.decoding.evaluate_wer import compute_metrics
from src.assessment.metrics import evaluate_reading_fluency
from src.visualization import plot_evaluation_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CNN-Enhanced ASR Model")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/best_model", help="Path to checkpoint directory")
    parser.add_argument("--dataset_root", type=str, default="dataset/LibriSpeech", help="LibriSpeech root directory")
    parser.add_argument("--split", type=str, default="test-clean", help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--use_beam_search", action="store_true", help="Enable CTC Beam Search decoding")
    parser.add_argument("--lm_path", type=str, default="", help="Path to KenLM ARPA model")
    parser.add_argument("--output_json", type=str, default="evaluation_results.json", help="Output metrics path")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("Evaluating CNN-Enhanced ASR on Benchmark:", args.split)
    print(f"Device: {device} | Checkpoint: {args.checkpoint_dir}")
    print("=" * 65)

    # 1. Load Tokenizer
    vocab_path = os.path.join(args.checkpoint_dir, "vocab.json")
    if os.path.exists(vocab_path):
        tokenizer = CharacterTokenizer.load_vocab(vocab_path)
    else:
        tokenizer = CharacterTokenizer()

    # 2. Load Model Config & Weights
    config_path = os.path.join(args.checkpoint_dir, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
    else:
        # Default config
        model_cfg = {
            "backbone": {"name": "facebook/wav2vec2-base-960h", "feature_dim": 768},
            "cnn": {"channels": [32, 64, 128], "projection_dim": 512},
            "rnn": {"hidden_size": 256, "num_layers": 2, "dropout": 0.2},
            "dense": {"hidden_dim": 256, "dropout": 0.3},
            "vocab_size": 29
        }

    model = HybridASRModel(
        backbone_name=model_cfg["backbone"]["name"],
        freeze_backbone=False,
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

    weights_path = os.path.join(args.checkpoint_dir, "model_weights.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Successfully loaded weights from {weights_path}")
    else:
        print(f"Notice: No checkpoint found at {weights_path}, evaluating un-finetuned model initialized state.")

    model.eval()

    # 3. Setup Dataset
    dataset = LibriSpeechDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        tokenizer=tokenizer,
        is_train=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ASRCollateFn(target_pad_value=tokenizer.blank_id),
        num_workers=4
    )

    # 4. Setup Decoder
    decoder = CTCDecoder(
        tokenizer=tokenizer,
        lm_path=args.lm_path if args.lm_path else None
    )

    all_predictions = []
    all_references = []
    all_durations = []
    sample_assessments = []

    print(f"\nDecoding {len(dataset)} utterances...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            audio_padded = batch["audio_padded"].to(device)
            refs = batch["transcripts"]
            durations = batch["durations"]

            log_probs = model(audio_padded)

            if args.use_beam_search:
                preds = decoder.decode_beam(log_probs)
            else:
                preds = decoder.decode_greedy(log_probs)

            all_predictions.extend(preds)
            all_references.extend(refs)
            all_durations.extend(durations)

            # Collect sample educational assessments
            if len(sample_assessments) < 5:
                for r, p, d in zip(refs, preds, durations):
                    if len(sample_assessments) < 5:
                        report = evaluate_reading_fluency(reference_text=r, hypothesis_text=p, duration_seconds=d)
                        sample_assessments.append(report)

    # 5. Compute Benchmark Metrics & Full Empirical Distributions
    import jiwer
    from src.assessment.alignment import clean_token

    metrics = compute_metrics(all_predictions, all_references)
    word_stats = jiwer.process_words(all_references, all_predictions)
    global_alignment = {
        "correct": int(word_stats.hits),
        "substitutions": int(word_stats.substitutions),
        "omissions": int(word_stats.deletions),
        "insertions": int(word_stats.insertions),
    }

    utterance_wers = []
    utterance_durations = []
    utterance_cwpms = []
    utterance_accuracies = []

    for r, p, d in zip(all_references, all_predictions, all_durations):
        r_tokens = [w for w in r.split() if clean_token(w)]
        n_ref = len(r_tokens)
        dur_min = max(float(d) / 60.0, 1e-4)

        if n_ref > 0:
            u_stat = jiwer.process_words(r, p)
            u_wer = min(u_stat.wer * 100.0, 100.0)
            u_hits = u_stat.hits
            u_acc = (u_hits / n_ref) * 100.0
            u_cwpm = u_hits / dur_min
        else:
            u_wer = 0.0
            u_acc = 100.0
            u_cwpm = 0.0

        utterance_wers.append(round(float(u_wer), 2))
        utterance_durations.append(round(float(d), 2))
        utterance_cwpms.append(round(float(u_cwpm), 1))
        utterance_accuracies.append(round(float(u_acc), 2))

    utterance_stats = {
        "wers": utterance_wers,
        "durations": utterance_durations,
        "cwpms": utterance_cwpms,
        "accuracies": utterance_accuracies,
    }

    print("\n" + "=" * 65)
    print("BENCHMARK EVALUATION RESULTS ON:", args.split)
    print(f"Word Error Rate (WER):      {metrics['wer_percent']:.2f}%")
    print(f"Character Error Rate (CER): {metrics['cer_percent']:.2f}%")
    print(f"Alignment Breakdown: Hits={global_alignment['correct']} | Sub={global_alignment['substitutions']} | Del={global_alignment['omissions']} | Ins={global_alignment['insertions']}")
    print("=" * 65)

    if sample_assessments:
        print("\n--- SAMPLE EDUCATIONAL READING FLUENCY ASSESSMENT ---")
        print(sample_assessments[0].summary())

    # Save output JSON
    results_payload = {
        "split": args.split,
        "metrics": metrics,
        "total_samples": len(dataset),
        "use_beam_search": args.use_beam_search,
        "global_alignment": global_alignment,
        "utterance_stats": utterance_stats,
        "sample_assessments": [a.__dict__ for a in sample_assessments],
        "predictions": all_predictions,
        "references": all_references
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n✓ Saved evaluation results to {args.output_json}")

    # Automatically create publication-grade evaluation diagnostic figures
    out_dir = os.path.dirname(args.output_json) or "."
    plot_evaluation_diagnostics(results_payload, output_dir=out_dir, filename_prefix="evaluation_diagnostics")
    plot_evaluation_diagnostics(results_payload, output_dir="docs/visuals", filename_prefix="fig_evaluation_diagnostics")


if __name__ == "__main__":
    main()

