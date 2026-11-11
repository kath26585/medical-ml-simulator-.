# Project Command Reference Guide

This document lists all execution commands for training, evaluating, visualizing, testing, and running the web application for the **CNN-Enhanced ASR Reading Fluency Assessment** project.

---

## 1. Environment & Setup

Always use the project Python virtual environment located in `.venv`:

```bash
# Option A: Activate venv in your active shell
source .venv/bin/activate

# Option B: Run directly via venv binary without activation
./.venv/bin/python <script.py>
```

---

## 2. Interactive Web Application (Streamlit)

Launch the Streamlit Oral Reading Fluency & Articulation Assessment interface:

```bash
# Primary command:
./.venv/bin/python -m streamlit run streamlit/app.py --server.port 8501

# Or use the convenience runner script:
./run_streamlit.sh

# Specify a custom port with the runner:
./run_streamlit.sh 8502

# Run headlessly for remote server access:
./.venv/bin/python -m streamlit run streamlit/app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
```

---

## 3. Automated Test Suite

Run unit and integration tests across data, models, decoding, audio processing, and reading assessment:

```bash
# Run all tests via unittest discovery:
./.venv/bin/python -m unittest discover tests

# Run all tests on CPU (recommended when GPU VRAM is occupied):
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m unittest discover tests

# Run core ASR pipeline tests only:
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m unittest tests/test_pipeline.py

# Run Streamlit audio & passage pipeline tests only:
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m unittest tests/test_streamlit_pipeline.py

# Run collected dataset processor unit tests only:
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m unittest tests/test_collected_data_processor.py

# Run using pytest:
./.venv/bin/python -m pytest tests/
```

---

## 4. Acoustic Model Training

Train the 10-stage Hybrid CNN-BiLSTM-CTC model using curriculum learning (Stage 1 Warmup + Stage 2 Fine-Tuning):

```bash
# Full 4-GPU Distributed Data Parallel (DDP) training:
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml

# Run DDP training in the background with logging:
nohup torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml > train.log 2>&1 &
tail -f train.log

# Override epochs and batch size (e.g., 30 epochs, batch size 16 per GPU):
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml --epochs 30 --batch_size 16

# Single-GPU debug / smoke test (GPU 0):
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python train.py --config configs/train_config.yaml --epochs 1
```

---

## 5. Benchmark Evaluation

Evaluate the trained model on LibriSpeech benchmarks (`test-clean` or `dev-clean`) and generate fluency diagnostics:

```bash
# Fast greedy decoding evaluation on test-clean:
./.venv/bin/python evaluate.py --checkpoint_dir checkpoints/best_model --split test-clean

# Full KenLM 4-gram Beam Search decoding on test-clean:
./.venv/bin/python evaluate.py \
  --checkpoint_dir checkpoints/best_model \
  --split test-clean \
  --use_beam_search \
  --lm_path lm/4-gram.binary

# Custom evaluation output file:
./.venv/bin/python evaluate.py \
  --checkpoint_dir checkpoints/best_model \
  --split test-clean \
  --use_beam_search \
  --lm_path lm/4-gram.binary \
  --output_json evaluation_results.json
```

---

## 6. Publication Figures & Visualizations

Generate publication-grade diagnostic plots, architectural diagrams, and training curves:

```bash
# Render all 5 architectural & methodology thesis figures:
./.venv/bin/python scripts/generate_thesis_visuals.py

# Plot training curves from an existing training history file:
./.venv/bin/python scripts/plot_results.py --history checkpoints/best_model/training_history.json

# Plot evaluation diagnostics from benchmark results:
./.venv/bin/python scripts/plot_results.py --results evaluation_results.json

# Generate complete demonstration figure suite:
./.venv/bin/python scripts/plot_results.py --demo
```

---

## 7. KenLM Language Model Compilation

Convert text ARPA language models into memory-efficient binary format for instant loading:

```bash
# Compile ARPA language model to fast binary format:
./kenlm/build/bin/build_binary lm/4-gram.arpa lm/4-gram.binary
```

---

## 8. GPU Hardware Monitoring

Monitor CUDA device utilization and VRAM across the 4x RTX A5000 GPUs:

```bash
# One-time GPU status check:
nvidia-smi

# Live monitoring updated every second:
watch -n 1 nvidia-smi
```

---

## 9. Repository Archiving & Lightweight Export

Package the workspace into a minimal compressed archive (excluding 7 GB datasets, 5.6 GB LM files, virtual environments, and caches):

```bash
# Generate ultra-lightweight source archive (~4.6 MB using XZ):
./scripts/archive_project.sh -f tar.xz

# Generate standard tar.gz (~5.5 MB):
./scripts/archive_project.sh -f tar.gz

# Generate zip archive (~5.5 MB):
./scripts/archive_project.sh -f zip

# Include trained PyTorch weights bundle (~425 MB):
./scripts/archive_project.sh -w
```

---

## 10. Collected Student Audio Processing & MP3 Transcoding

Unzip student recordings archive, transcode diverse formats (AAC, WAV) to standardized MP3 (192 kbps), and organize into grade-level, passage-level, and learner-indexed structures:

```bash
# Run default batch extraction, transcoding, and organization (auto-detects zip archive):
./.venv/bin/python scripts/process_collected_data.py

# Run with custom parallel workers (e.g., 16 workers) and custom bitrate:
./.venv/bin/python scripts/process_collected_data.py --workers 16 --bitrate 192

# Force re-transcoding and re-extraction:
./.venv/bin/python scripts/process_collected_data.py --force
```

