# AGENTS.md — CNN-Enhanced ASR for Reading Fluency Assessment

> **Repository Master Guide for Autonomous AI Agents and Core Developers**  
> *Last Verified:* October 2026 | *Framework:* PyTorch 2.7.1 + CUDA 12.6 + Hugging Face + KenLM

---

> [!IMPORTANT]
> ### 🚨 MANDATORY DIRECTIVE: READ BEFORE EXPLORING & KEEP CONTINUOUSLY UPDATED
> 1. **Prerequisite Reading Before Workspace Actions:**  
>    **Every autonomous agent and developer MUST thoroughly read this `AGENTS.md` file before diving into or exploring the workspace.** This document establishes the exact architectural context, tensor flows, hardware constraints, environment traps, and directory layout. Reading this first ensures you have complete context and know precisely where components live without wastefully inspecting every file in the repository.
> 2. **Mandatory Documentation Synchronization Rule:**  
>    **For EVERY change made to the workspace**, `AGENTS.md` **MUST be updated immediately.** Whenever you add or refactor code, adjust hyperparameters, introduce new unit tests, upgrade signal processing algorithms, update benchmark results, or add new scripts/features, you **MUST update the corresponding sections of `AGENTS.md`** as part of that change. This file is the authoritative living single source of truth across all agent sessions.

---

## 1. Project Identity & Research Overview

### 1.1 Thesis Title & Objective
- **Thesis Title:** *CNN-Enhanced Automatic Speech Recognition Model for Automated Articulation and Reading Fluency Assessment in Primary Education*
- **Primary Domain:** Automatic Speech Recognition (ASR), Acoustic-Phonetic Representation Learning, Computer-Assisted Language Learning (CALL), and Primary Oral Reading Fluency (ORF) Assessment.
- **Target Performance Metric:** **3%–5% Word Error Rate (WER)** on the LibriSpeech `test-clean` benchmark.
- **Achieved Milestone:**
  - **Validation WER (Epoch 9):** `3.38%` (CER: `1.01%`)
  - **Benchmark Evaluation (`test-clean`, 2,620 Utterances):** `3.90%–4.06%` WER, `1.12%–2.55%` CER
  - **Global Alignment Benchmark (`test-clean`):** `50,889` correct word tokens ($96.79\%$), `1,558` substitutions ($2.96\%$), `129` omissions ($0.25\%$), and `446` insertions ($0.85\%$)

### 1.2 Core Research Contributions & Architecture
Raw acoustic models trained from scratch on 100 hours of audio typically plateau at 15%–20% WER. This project achieves production/clinical accuracy (3%–5% WER) while preserving a lightweight custom architecture by pairing **self-supervised transfer learning** with a novel **10-stage Hybrid CNN-BiLSTM-CTC** model and **KenLM 4-gram beam search decoding**:
1. **Acoustic Backbone:** Self-supervised Wav2Vec 2.0 Base (`facebook/wav2vec2-base-960h`) providing rich 768-dimensional temporal representations.
2. **Hierarchical CNN Acoustic Refiner:** 3-stage 2D CNN blocks with **time-preserving max pooling** `(1, 2)`. Features are compressed along frequency/channel dimensions while keeping time resolution intact to preserve CTC character-level alignment.
3. **Temporal Sequence Modeling:** 2-layer Bidirectional LSTM (512 total hidden units).
4. **Decoding & Language Modeling:** Fast CTC Beam Search via `pyctcdecode` backed by a 4-gram KenLM binary language model (`lm/4-gram.binary`) with **Passage Contextual Biasing (Hotword Boosting)**.
5. **Educational Assessment Engine:** Word-level dynamic Levenshtein alignment with **punctuation stripping** and **dialect/spelling variant equivalence** (e.g., US vs. UK orthography: *travelers/travellers*, *theater/theatre*, *color/colour*), classifying learner errors into `CORRECT`, `SUBSTITUTION`, `OMISSION`, and `INSERTION`, computing Articulation Accuracy (%), Words Per Minute (WPM), Correct Words Per Minute (CWPM), and standardized primary education proficiency tiers (`Independent`, `Instructional`, `Frustration`).

### 1.3 Implementation Roadmap & Project Scope
- **Phase 1 (Completed & Verified):**
  - Full modular ASR codebase (`src/model/`, `src/data/`, `src/decoding/`, `src/assessment/`, `src/visualization/`).
  - Multi-GPU PyTorch Distributed Data Parallel (DDP) training script with 2-stage curriculum (`train.py`).
  - Standalone evaluation script with automated diagnostics generation (`evaluate.py`).
  - Automated publication-grade visual rendering engine (`scripts/generate_thesis_visuals.py`, `scripts/plot_results.py`).
  - Core pipeline unit test suite (`tests/test_pipeline.py`).
- **Phase 2 (Completed & Hardened):**
  - Interactive Streamlit web application for teachers and primary school students (`streamlit/app.py`).
  - In-browser microphone recording (`st.audio_input`) and multi-format audio file upload (`st.file_uploader`).
  - Signal conditioning pipeline (`streamlit/audio_processor.py`): 16 kHz mono downmixing, 80 Hz – 7.5 kHz Butterworth bandpass filter, STFT Wiener stationary noise suppression, RMS loudness normalization, and WAV byte export.
  - Side-by-side original vs. denoised audio preview players.
  - Calibrated Grade 1–3 reading passage presets with editable custom passage override and punctuation-aware word counting (`streamlit/passages.py`).
  - Real-time CNN-BiLSTM-CTC inference + KenLM 4-gram beam search decoding with **Passage Contextual Biasing** (hotword weight slider, default 8.0).
  - Robust Levenshtein alignment engine with punctuation normalization and UK/US spelling variant equivalence mapping (`src/assessment/alignment.py`).
  - Educational ORF scorecard: Articulation Accuracy (%), WPM, CWPM, error breakdown (Substitutions, Omissions, Insertions), standard primary proficiency tier rating (*Independent*, *Instructional*, *Frustration*), and color-coded word-by-word alignment tokens.
  - Cascaded Neural & DSP Signal Conditioning Pipeline (`streamlit/audio_processor.py`): Multi-harmonic zero-phase notch filter (50/60 Hz line hum and upper harmonics), autonomous Voice Activity Detection (VAD) with primary student isolation and 250ms hangover protection, temporally-smoothed spectral denoising (eliminating 'musical noise' chirping and protecting unvoiced consonants), and ITU-R BS.1770-4 / EBU R128 integrated perceptual loudness normalization (LUFS).
  - Configurable Classroom Noise Presets in Streamlit UI: `🟢 Quiet Room` (soft whisper sensitivity), `🟡 Typical Classroom` (balanced default), and `🔴 High Chatter / Active Classroom` (strict speech isolation with adaptive hotword biasing).
  - Empirical distribution logging (`global_alignment`, `utterance_stats`) across all test benchmark samples in `evaluate.py` and diagnostic plotting in `src/visualization/plotter.py`.
  - Dedicated unit & integration test suite (`tests/test_pipeline.py`, `tests/test_streamlit_pipeline.py`, `tests/test_audio_processor.py`, `tests/test_linguistic_and_field_evaluation.py`, `tests/test_collected_data_processor.py`). **All 44 project unit tests passing.**
  - **Modernized Distraction-Free UI/UX Redesign (`streamlit/app.py`, `streamlit/passages.py`):**
    - Single-page linear state machine ($\mathbf{READING} \to \mathbf{PROCESSING} \to \mathbf{RESULTS}$) eliminating student distraction.
    - Replaced persistent sidebar with an on-demand `@st.dialog("Settings & Audio Customization")` modal.
    - **Child-Proof Spatial Separation & Low-Saliency Settings Footer:**
      - Removed the high-contrast `Settings` button from the top student header, rebalancing the header into an uncluttered 2-column layout (`Brand Title` + `Grade Selector`) giving comfortable breathing room to the grade selector dropdown.
      - Relocated settings trigger to a discreet facilitator footer (`Room & Audio Settings`) at the very bottom of the page, styled as a subtle, low-saliency ghost button (11px typography, 55% opacity, borderless transparent background, zero emojis) outside the primary pupil's visual reading path while preserving 1-click access for teachers across laptops and touch tablets.
    - Hero Reading Card featuring enlarged **Lexend** typography ($28–32\text{px}$ desktop, $22–24\text{px}$ mobile, line-height 1.75) engineered for non-scrollable, continuous reading without screen movement.
    - Grade-calibrated word count guidance ($20–60$ for Grade 1, $35–70$ for Grade 2, $40–105$ for Grade 3) with live guidance feedback chip, inline passage editor, and `[Save Story]` commit button.
    - **Custom Story & Reading Passage Support:**
      - Editable story title (`st.text_input("Story Title")`) alongside passage text in inline editor, enabling evaluation of custom classroom stories.
      - Dynamic Reading Hero Card rendering with custom story indicator chip (`Custom`) and grade-calibrated target range subtitle.
      - Dynamic Results scorecard integration displaying assessed story title.
      - Dedicated benchmark reset (`Reset to Benchmark`) reverting both story title and passage text back to default grade presets.
    - **Automatic Real-Time Word Counter & Word Limit Enforcer:**
      - Live keystroke-level word counting and typing prevention via client-side DOM event interception (`keydown`, `input`, `paste`).
      - Physical keystroke blocking (`e.preventDefault()`) when the grade-level word limit is reached ($65$ for Grade 1, $75$ for Grade 2, $115$ for Grade 3), preserving word edits while blocking excess token creation.
      - Clear SVG warning badge and card border alert highlight when the limit is reached (zero emojis).
      - Server-side text clamping guarantee (`passages.clamp_words`) on commit.
    - Segmented button controls for `Record Voice` and `Upload Audio File` with hero microphone capsule, live dynamic WaveSurfer waveform (reacts to real voice, flat when quiet), and automatic progression to assessment upon stopping recording.
    - Academic Deep Royal Blue (`#1b509b`) primary brand theming with atmospheric canvas gradients and robust Light/Dark mode contrast calibration.
    - Mobile-first container shell ($620\text{px}$ centered on desktop, fluid 100% on mobile) with native mobile app ergonomics and pill touch targets.
    - Full-focus CSS pulsing soundwave equalizer animation during ASR inference.
    - Complete removal of emojis in favor of clean, professional SVG icons and accessible color badges.
  - **Linguistic & Word-Type Error Breakdown Engine (`scripts/analyze_linguistic_errors.py`):**
    - Dynamic Levenshtein alignment breakdown across syllable complexity (Monosyllabic, Disyllabic, Polysyllabic), grammatical role (Function vs. Content Words), and token length categories.
    - Exports structured JSON (`docs/tables/linguistic_error_breakdown.json`), Markdown (`docs/tables/linguistic_error_breakdown.md`), and publication LaTeX table (`docs/tables/table_linguistic_breakdown.tex`).
  - **Primary Grade 1–3 Oral Reading Fluency Curriculum Standards (`streamlit/passages.py`):**
    - Calibrated benchmark passage presets strictly focused on primary education: Grade 1 (Foundational: "The Cat and the Ball", 23 words, target 20–60 words, limit 65), Grade 2 (Transitional: "A Day at the Park", 39 words, target 35–70 words, limit 75), and Grade 3 (Fluent: "Watching the Night Sky", 43 words, target 40–105 words, limit 115). Grade 6 advanced option removed to strictly align with primary education scope.
  - **Statistical Confidence Interval (95% CI) Reporting Engine (`scripts/generate_statistical_tables.py`):**
    - Parametric (Mean, SEM, 95% CI) and non-parametric (Median, IQR, 95% percentile bounds) reporting over all 2,620 test utterances.
    - Exports `docs/tables/table_benchmark_statistical_ci.tex` (booktabs LaTeX table) and `docs/tables/benchmark_statistical_summary.md`.
  - **Field Pilot & Human Teacher Correlation Harness (`scripts/evaluate_field_pilot.py`):**
    - Evaluates student reading trials against human teacher ratings, computing Pearson correlation ($r$), Spearman rank correlation ($\rho$), MAE, RMSE, and pronunciation error detection Precision, Recall, and $F_1$ score across primary school learners (Grades 1–3).
    - Provides reproducible synthetic validation fixture (`tests/fixtures/mock_student_evaluations.json`) and exports LaTeX table (`docs/tables/table_field_pilot_validation.tex`).
  - **Expanded Test Suite (`tests/test_linguistic_and_field_evaluation.py`):**
    - 7 comprehensive unit tests covering syllabification, grammatical classification, Grade 1–3 passage bounds and Grade 6 removal verification, statistical CIs, and correlation matrices.
  - **Collected Student Audio Processing & MP3 Transcoding Engine (`src/data/collector_organizer.py`, `scripts/process_collected_data.py`):**
    - High-throughput unzipping and transcoding engine for collected primary school reading recordings (`dataset/collected-data/Data Collection -20261010T032236Z-1-001.zip`).
    - Robust metadata parsing normalizing heterogeneous filenames across all 48 learners (Learners 01–48), resolving grade levels (Grades 1–3) with automated typo resolution (e.g., `GRADE E` $\to$ Grade 3), and linking standardized passage catalog presets calibrated to ground truth scripts (`dataset/collected-data/script_grade1-3.txt`: Passage 1 "My New Bag", Passage 2 "Benny the Busy Bee", Passage 3 "Oliver and the Wise Owl").
    - Parallel multi-process audio transcoding to 192 kbps MP3 via PyAV (`libmp3lame`) with frame alignment and layout normalization (mono/stereo).
    - Multi-hierarchical organization: `by_grade/Grade_1..3`, `by_passage/Passage_1..3`, and `all_mp3/` (48 standardized MP3 recordings sorted by learner ID).
    - Exports tabular manifest (`dataset/collected-data/organized/metadata.csv`), machine-readable JSON manifest (`metadata.json`), and comprehensive dataset documentation (`README.md`) embedding full reference transcripts and word counts.
    - Dedicated test suite (`tests/test_collected_data_processor.py`). **All 44 project unit tests passing.**
- **Technical Investigation & Training Reports (`TRAINING_REPORT.md`, `REPORTS.md`):**
  - **Comprehensive Model Training & Evaluation Report (`TRAINING_REPORT.md` / `REP-2026-ASR-TRAIN-001`):** Complete machine learning architecture, dependencies, data loading, 2-stage curriculum, empirical convergence history, 4.06% benchmark WER on test-clean, Levenshtein alignment breakdown (50,889 hits), 95% confidence intervals, educational reading fluency metrics, and findings.
  - Rigorous engineering post-mortem (`REPORTS.md` / `REP-2026-ASR-ORF-001`) investigating the **Benchmark-to-Prototype Performance Disparity** ("Laboratory-to-Field Gap"): hardware microphone hum ($50/60\text{ Hz}$ harmonics), WebRTC MediaRecorder packet jitter, Wiener spectral subtraction "musical noise" & formant erosion, naive peak normalization traps, and pediatric formant distribution shift ($F_0 \approx 250 - 400\text{ Hz}$).
  - Actionable **Three-Tier Hardening Roadmap**: Tier 1 (Hardware standards & mic placement), Tier 2 (DSP overhaul: multi-harmonic notch filter, VAD, EBU R128 LUFS loudness normalization, neural denoising), Tier 3 (Pediatric/RIR fine-tuning and contextual biasing).
  - **Tier 2 DSP Overhaul Completed & Verified**: Implemented in `streamlit/audio_processor.py` and `streamlit/app.py`.
- **Phase 3 (Next Milestone):**
  - Teacher session history tracking and downloadable PDF assessment report generation.
  - Pediatric vocal tract length perturbation fine-tuning and room impulse response (RIR) classroom noise simulation.

---

## 2. System Architecture & Tensor Flow Specification

The acoustic model executes a 10-stage processing pipeline defined in `src/model/hybrid_asr.py`:

```text
[Raw Audio Waveform (16 kHz Mono)]
                 │
                 ▼
[Stage 1: Pre-trained Feature Representation]
  - Wav2Vec 2.0 Base Feature Extractor (768-dim temporal acoustic frames)
  - Tensor: (Batch, Time_samples) ──► (Batch, T', 768)
                 │
                 ▼
[Stage 2: Reshape to 2D Feature Map]
  - Tensor: (Batch, 1, T', 768)
                 │
                 ▼
[Stages 3–6: Hierarchical CNN Enhancement Blocks (Time-Preserving)]
  ├─ Conv Block 1: Conv2D(1 -> 32, 3×3, pad=1) + BatchNorm2d + ReLU
  ├─ Pooling 1:    MaxPool2d(kernel=(1, 2), stride=(1, 2)) ──► (Batch, 32, T', 384)
  ├─ Conv Block 2: Conv2D(32 -> 64, 3×3, pad=1) + BatchNorm2d + ReLU
  ├─ Pooling 2:    MaxPool2d(kernel=(1, 2), stride=(1, 2)) ──► (Batch, 64, T', 192)
  └─ Conv Block 3: Conv2D(64 -> 128, 3×3, pad=1) + BatchNorm2d + ReLU ──► (Batch, 128, T', 192)
                 │
                 ▼
[Stage 7: Reshape & Linear Projection]
  - Flatten channel & feature dims: 128 × 192 = 24,576
  - Linear(24576 -> 512) + LayerNorm(512) + ReLU ──► (Batch, T', 512)
                 │
                 ▼
[Stage 8: Bidirectional LSTM Sequence Modeling]
  - 2-Layer BiLSTM (hidden_size=256 per direction, dropout=0.2) ──► (Batch, T', 512)
                 │
                 ▼
[Stage 9: Dense Layer & Regularization]
  - Linear(512 -> 256) + LayerNorm(256) + ReLU + Dropout(p=0.3) ──► (Batch, T', 256)
                 │
                 ▼
[Stage 10: CTC Output Classifier]
  - Linear(256 -> 29) + LogSoftmax(dim=-1) ──► (Batch, T', 29)
```

### 2.1 Critical Architectural Invariants
1. **Time-Preserving Pooling Rule:**  
   The max-pooling layers in `src/model/cnn_blocks.py` **MUST** use kernel `(1, 2)` and stride `(1, 2)`. Pooling across time `(2, 2)` is strictly forbidden because subsampling temporal frames violates the CTC timing condition ($T' \ge 2 \times U$, where $U$ is transcript length) and degrades alignment precision.
2. **Vocabulary Schema (29 Tokens):**  
   Implemented in `src/data/tokenizer.py`:
   - Index `0`: `<blank>` (CTC blank token)
   - Index `1`: `' '` (Word boundary whitespace)
   - Indices `2..27`: Uppercase ASCII letters `'A'` through `'Z'`
   - Index `28`: `'\''` (Apostrophe for English contractions e.g., *can't*, *it's*)

---

## 3. Repository Directory Map

```
/home/kathlenelaresma/tsis/
├── configs/
│   └── train_config.yaml          # Master configuration (hardware, hyperparameters, data paths)
├── config.yaml                    # Symlink pointing to configs/train_config.yaml
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── cnn_blocks.py          # 3-Stage Conv2D + BatchNorm + Time-Preserving MaxPool2d
│   │   ├── rnn_encoder.py         # 2-Layer Bidirectional LSTM sequence modeling
│   │   └── hybrid_asr.py          # Complete Wav2Vec2 + CNN + BiLSTM + CTC model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py           # 29-character tokenizer with CTC blank collapse & normalization
│   │   ├── dataset.py             # LibriSpeech loader with Gaussian SNR noise augmentation
│   │   ├── collate.py             # Dynamic duration bucketing & padded batch collator
│   │   └── collector_organizer.py # Archive extractor, metadata parser, and PyAV MP3 transcoding engine
│   ├── decoding/
│   │   ├── __init__.py
│   │   ├── beam_search.py         # pyctcdecode + KenLM 4-gram binary beam search & greedy decoder (with hotwords)
│   │   └── evaluate_wer.py        # JiWER benchmark evaluation (WER % & CER %)
│   ├── assessment/
│   │   ├── __init__.py
│   │   ├── alignment.py           # Dynamic programming Levenshtein alignment (clean_token & spelling variants)
│   │   └── metrics.py             # Articulation Accuracy, WPM, CWPM, and ReadingTier rating
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py             # Automated training curve & empirical diagnostic visualizer
├── .streamlit/
│   └── config.toml                # Streamlit server config (disables torch class file watcher)
├── streamlit/                     # Phase 2 interactive web application
│   ├── app.py                     # Main Streamlit assessment dashboard (with classroom presets & contextual biasing)
│   ├── audio_processor.py         # Cascaded DSP/VAD engine (mains notch, VAD gating, smoothed denoise, LUFS norm)
│   └── passages.py                # Grade 1-3 reading passage presets & punctuation-aware word counter
├── scripts/
│   ├── generate_thesis_visuals.py # Publication figure rendering (Architecture, Curricula, Tiers)
│   ├── plot_results.py            # CLI visual generation from training logs or evaluation results
│   ├── analyze_linguistic_errors.py # Linguistic & word-type error analyzer (syllables, function words, length)
│   ├── generate_statistical_tables.py # Parametric & non-parametric 95% CI statistical table generator
│   ├── evaluate_field_pilot.py    # Human teacher correlation & error detection (P/R/F1) evaluation harness
│   └── process_collected_data.py  # High-throughput collected dataset unzipper, MP3 transcoder & organizer CLI
├── tests/
│   ├── test_pipeline.py           # Core ASR pipeline unit tests (9 tests passing)
│   ├── test_streamlit_pipeline.py # Streamlit audio processor, UI architecture & end-to-end tests (11 tests passing)
│   ├── test_audio_processor.py    # Cascaded DSP/VAD unit & integration tests (9 tests passing)
│   ├── test_linguistic_and_field_evaluation.py # Linguistic, Grade 1-3 bounds & Grade 6 removal, CI, and correlation tests (7 tests passing)
│   ├── test_collected_data_processor.py # Collected dataset parsing & MP3 transcoding unit tests (7 tests passing)
│   └── fixtures/
│       └── mock_student_evaluations.json # 15-pupil synthetic validation benchmark fixture (Grades 1-3)
├── checkpoints/
│   └── best_model/                # Phase 1 deployment artifact bundle
│       ├── model_weights.pt       # PyTorch state dictionary (422 MB)
│       ├── model_config.json      # Full architecture hyperparameters
│       ├── vocab.json             # Character token-to-ID mapping
│       ├── metrics.json           # Best validation metrics (WER: 3.38%, CER: 1.01%)
│       ├── training_history.json  # Epoch metrics log across all training stages
│       ├── training_curves.png    # 300 DPI multi-panel training progression curve
│       ├── training_curves.pdf    # Vector PDF format for LaTeX inclusion
│       └── training_curves.svg    # Vector SVG format for web/presentations
├── dataset/
│   ├── LibriSpeech/               # Local benchmark dataset
│   │   ├── train-clean-100/       # 28,539 audio samples (~100 hours)
│   │   ├── dev-clean/             # 2,703 audio samples (validation split)
│   │   ├── test-clean/            # 2,620 audio samples (test benchmark split)
│   │   └── *.TXT                  # LibriSpeech metadata manifests
│   └── collected-data/            # Primary school reading fluency field recording collection
│       ├── script_grade1-3.txt    # Ground truth reading scripts for Grades 1–3
│       ├── raw_extracted/         # 48 uncompressed source recordings (AAC, WAV)
│       └── organized/             # Standardized MP3 recordings & multi-hierarchical catalog
│           ├── all_mp3/           # 48 standardized recordings sorted by Learner ID
│           ├── by_grade/          # Grouped by Grade (Grade_1: 9, Grade_2: 15, Grade_3: 24)
│           ├── by_passage/        # Grouped by Passage (Passage_1: 9, Passage_2: 16, Passage_3: 23)
│           ├── metadata.csv       # Tabular manifest with durations, sample rates, and file metadata
│           ├── metadata.json      # Structured machine-readable JSON manifest
│           └── README.md          # Dataset documentation & curriculum standards
├── lm/
│   ├── 4-gram.binary              # Pre-compiled binary KenLM model (1.5 GB, fast load)
│   ├── 4-gram.arpa                # Text ARPA language model (4.1 GB)
│   └── unigrams.txt               # Unigram vocabulary list for pyctcdecode
├── docs/
│   ├── tables/                    # Publication LaTeX & Markdown benchmark tables
│   │   ├── benchmark_statistical_summary.md # Parametric/non-parametric CI report
│   │   ├── statistical_ci_summary.json     # Machine-readable 95% CI distributions
│   │   ├── table_benchmark_statistical_ci.tex # Publication LaTeX CI table
│   │   ├── linguistic_error_breakdown.md   # Syllable, grammar, length error analysis
│   │   ├── linguistic_error_breakdown.json # Categorical linguistic error distributions
│   │   ├── table_linguistic_breakdown.tex  # Publication LaTeX linguistic table
│   │   ├── field_pilot_correlation.md      # Human teacher correlation & error matrix
│   │   ├── field_pilot_correlation_results.json # 20-pupil validation results
│   │   └── table_field_pilot_validation.tex # Publication LaTeX teacher validation table
│   ├── visuals/                   # 7 publication figures (PDF, PNG, SVG, TikZ) & guide
│   └── superpowers/
│       ├── specs/                 # Architecture & design specifications
│       └── plans/                 # Step-by-step implementation plans
├── train.py                       # Multi-GPU PyTorch DDP training script with 2-stage curriculum
├── evaluate.py                    # Standalone benchmark evaluation on test-clean with empirical distributions
├── evaluation_results.json        # Output evaluation metrics, global alignment breakdown, and utterance distributions
├── evaluation_diagnostics.png     # 4-panel assessment diagnostic dashboard
├── run_streamlit.sh               # Shell script launcher for Streamlit web app
├── COMMANDS.md                    # Master CLI command execution reference guide
├── REPORTS.md                     # Technical investigation report: benchmark vs. prototype disparity & DSP hardening
├── TRAINING_REPORT.md             # Comprehensive model training & evaluation report (REP-2026-ASR-TRAIN-001)
├── .venv/                         # Project Python virtual environment (Python 3.10.12)
├── AGENTS.md                      # This agent master reference document
└── skills-lock.json               # Locked agent capabilities
```

---

## 4. Hardware & Environment Quirks (Must Read!)

Agents working in this repository **must observe the following runtime environment guidelines**:

### 4.1 Python Interpreter & Virtual Environment
- **Path:** Always invoke the project virtual environment located at:
  ```bash
  /home/kathlenelaresma/tsis/.venv/bin/python
  # or activate within shell:
  source .venv/bin/activate
  ```
- Do **not** use system `/usr/bin/python3` directly, as scientific dependencies (`soundfile`, `transformers`, `torch`, `pyctcdecode`, `kenlm`) reside in `.venv`.

### 4.2 Known Environment Traps & Pre-Configured Guards
1. **Torchvision Binary Symbol Conflict:**  
   The underlying system Python has a broken `torchvision` installation that raises `RuntimeError: operator torchvision::nms does not exist` when imported.  
   *Guard:* Every executable script (`train.py`, `evaluate.py`, `src/model/hybrid_asr.py`) contains a guard that sets `sys.modules["torchvision"] = None` and `transformers.utils.import_utils._torchvision_available = False`. If creating new entrypoint scripts, always copy this guard at the very top.
2. **Matplotlib Cache Directory:**  
   In sandboxed or headless execution, matplotlib may fail if `$HOME/.cache` is restricted.  
   *Guard:* All visual scripts set `MPLCONFIGDIR=/tmp/matplotlib` or use `tempfile.gettempdir()`, and set backend to `matplotlib.use("Agg")`.
3. **Torchaudio Resampling Fallback:**  
   If `torchaudio` audio transforms fail to find SOX/FFmpeg bindings, the data loader implements a PyTorch linear interpolation fallback.
4. **KenLM Binary vs. ARPA:**  
   Always load `lm/4-gram.binary` rather than `lm/4-gram.arpa`. The binary trie loads in $<1$ second with minimal memory footprint, whereas parsing the 4.1 GB text ARPA file takes $>45$ seconds and consumes substantial RAM.
5. **Streamlit PyTorch C++ Class Inspection Trap:**  
   When Streamlit starts, its file watcher (`local_sources_watcher`) scans all modules in `sys.modules`. Accessing attributes on `torch.classes` triggers PyTorch's dynamic C++ class loader and raises `RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!`.  
   *Guard:*  
   - In `streamlit/app.py`: Monkey-patch `torch.classes.__path__ = []` and set `os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"`.  
   - In `.streamlit/config.toml`: Set `[server] fileWatcherType = "none"`.  
6. **HuggingFace Transformers Weight Initialization Warnings:**  
   Loading the Wav2Vec 2.0 backbone emits verbose warnings regarding uninitialized task-specific heads.  
   *Guard:* All entrypoints configure `transformers.logging.set_verbosity_error()` to keep stdout clean.
7. **Test Runner Framework:**  
   The project virtual environment uses the built-in `unittest` runner (`python -m unittest discover tests/`). `pytest` is not installed by default in the virtualenv.
8. **Streamlit Mobile Audio Upload & LAN Connectivity Trap:**  
   When accessing Streamlit over LAN or mobile devices (`http://<LAN_IP>:8501`), Streamlit's default `enableCORS = true` and `enableXsrfProtection = true` reject `st.file_uploader` requests because `browser.serverAddress` defaults to `localhost`, mismatched with the client IP origin, causing preflight CORS and XSRF cookie failures on mobile WebKit/Chromium. Furthermore, mobile voice recordings (iOS Voice Memos, Android Recorder) default to MPEG-4 `.m4a` / AAC, which `soundfile` (libsndfile) cannot decode alone.  
   *Guard:*  
   - In `.streamlit/config.toml`: Set `enableCORS = false`, `enableXsrfProtection = false`, and `address = "0.0.0.0"`.  
   - In `streamlit/audio_processor.py`: Implemented robust fallback decoding with PyAV (`av`) and clear diagnostic error guidance when unsupported containers are uploaded.

### 4.3 Compute Infrastructure
- **Hardware:** 4× NVIDIA RTX A5000 GPUs (24GB VRAM per GPU, 96GB aggregate VRAM).
- **CUDA Version:** CUDA 12.6, PyTorch 2.7.1, NCCL multi-GPU backend.
- **Precision:** `bfloat16` Native Mixed Precision via `torch.amp.autocast`.

---

## 5. Training Curriculum & Optimization Strategy

The system uses a **Two-Stage Curriculum Learning Strategy** to maintain numerical stability and prevent catastrophic forgetting of pre-trained acoustic representations:

### Stage 1: Custom Head Warmup (Epochs 1–10)
- **Backbone State:** Frozen (`requires_grad=False`, `model.wav2vec2.eval()`, `torch.no_grad()`).
- **Trained Modules:** CNN Enhancement Blocks, BiLSTM, Dense Projection, CTC Classifier.
- **Optimizer:** `AdamW(lr=3.0e-4, weight_decay=0.01)`.
- **LR Schedule:** 1-epoch linear warmup followed by Cosine Annealing to `1e-5`.
- **Performance:** Reaches ~3.4% validation WER by Epoch 9.

### Stage 2: End-to-End Joint Fine-Tuning (Epochs 11–15 / up to 30)
- **Configuration:** Master setting `total_epochs: 15` in `configs/train_config.yaml` (Stage 1 Warmup: Epochs 1–10, Stage 2 Joint Fine-Tuning: Epochs 11–15; extensible to 30 epochs).
- **Backbone State:** Unfrozen with PyTorch gradient checkpointing enabled (`gradient_checkpointing_enable()`).
- **Discriminative Learning Rates:**
  - Wav2Vec 2.0 Backbone: `lr = 1.0e-5`
  - Custom Heads (CNN + BiLSTM + CTC): `lr = 1.0e-4`
- **Optimizer:** `AdamW(weight_decay=0.01)`.
- **LR Schedule:** Cosine Annealing over remaining epochs down to `1e-6`.

### Memory & Distributed Scaling Tactics
- **Dynamic Duration Bucketing:** 10 buckets via `DynamicBucketSampler` eliminate batch padding overhead.
- **Gradient Accumulation:** Batch size 4 per GPU × 4 GPUs × 4 accumulation steps = **64 effective global batch size**.
- **DDP Sync Prevention:** Gradient accumulation uses `model.no_sync()` on accumulation steps to eliminate inter-GPU communication overhead.
- **CTC Stability:** Output frame lengths are clamped to `log_probs.size(1)`, and log probabilities are cast to `float32` before `nn.CTCLoss(blank=0, zero_infinity=True)` to prevent NaN/Inf gradients.

---

## 6. Decoding & Educational Assessment Engine

### 6.1 CTC Decoding Pipeline
- **Greedy Decoding:** Fast argmax decoding across temporal frames. Used during validation epochs for real-time progress monitoring.
- **Beam Search Decoding:** Powered by `pyctcdecode` and KenLM 4-gram binary model.
  - Baseline Hyperparameters: `beam_width=100`, `alpha=0.5` (LM weight), `beta=1.5` (Word insertion bonus).
- **Passage Contextual Biasing (Hotword Boosting):**
  - Implemented in `src/decoding/beam_search.py` via `decoder.decode(..., hotwords=..., hotword_weight=...)`.
  - In reading assessment scenarios where the target passage is known in advance, extracting unique passage vocabulary via `clean_token` and biasing beam search (weight `8.0`–`12.0`) prevents beam search from pruning valid passage vocabulary, significantly reducing out-of-passage substitution hallucinations.

### 6.2 Educational Assessment Metrics
Implemented in `src/assessment/`:
1. **Token Sanitization & Dialect Equivalence (`clean_token`):**
   - Surrounding punctuation (commas, periods, quotation marks, exclamation marks) is stripped while valid internal characters and apostrophes (e.g., *can't*, *it's*) are preserved.
   - Built-in `SPELLING_VARIANTS` equivalence dictionary maps British and American spelling variations (e.g., *TRAVELLERS* $\to$ *TRAVELERS*, *THEATRE* $\to$ *THEATER*, *COLOUR* $\to$ *COLOR*, *CENTRE* $\to$ *CENTER*, *PRACTISE* $\to$ *PRACTICE*) to eliminate false mispronunciation/substitution penalties in international educational settings.
   - Punctuation-isolated tokens are filtered out prior to dynamic programming alignment and in `streamlit/passages.py:count_words()`.
2. **Dynamic Levenshtein DP Alignment:** Aligns predicted hypothesis against reference reading passage, tagging tokens as:
   - `CORRECT`: Matched word
   - `SUBSTITUTION`: Mispronounced/substituted word
   - `OMISSION`: Word skipped by reader
   - `INSERTION`: Extraneous word spoken
3. **Metric Formulas:**
   $$\text{Articulation Accuracy (\%)} = \left(\frac{\text{Correct Words}}{\text{Total Reference Words}}\right) \times 100$$
   $$\text{Words Per Minute (WPM)} = \frac{\text{Total Spoken Words}}{\text{Duration in Minutes}}$$
   $$\text{Correct Words Per Minute (CWPM)} = \frac{\text{Correct Words}}{\text{Duration in Minutes}}$$
4. **Primary Education Proficiency Standards:**
   - **Independent Reading Level:** Accuracy $\ge 97.0\%$
   - **Instructional Reading Level:** $90.0\% \le \text{Accuracy} < 97.0\%$
   - **Frustration Reading Level:** $\text{Accuracy} < 90.0\%$
5. **Empirical Distribution Logging:**
   - Standalone evaluation in `evaluate.py` extracts utterance-level `wers`, `durations`, `cwpms`, and `accuracies` across the entire evaluation set alongside corpus-wide `global_alignment` counts (`hits`, `substitutions`, `deletions`, `insertions`).
   - Diagnostic visualizer `src/visualization/plotter.py` plots true empirical distributions when available in `evaluation_results.json`.

---

## 7. Developer & Agent Standard Operating Procedures

### 7.1 Running the Automated Test Suite
Always run the automated test suite before committing any changes or modifying model, decoding, or assessment logic:
```bash
# Run all 23 unit tests using unittest (recommended):
./.venv/bin/python -m unittest discover tests/

# Run tests on CPU (recommended when GPUs are under heavy load):
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -m unittest discover tests/

# Run core ASR pipeline tests only (9 tests):
./.venv/bin/python -m unittest tests/test_pipeline.py

# Run Streamlit audio & passage pipeline tests only (11 tests):
./.venv/bin/python -m unittest tests/test_streamlit_pipeline.py

# Run cascaded DSP/VAD audio processor tests only (9 tests):
./.venv/bin/python -m unittest tests/test_audio_processor.py

# Run linguistic breakdown and field validation tests only (7 tests):
./.venv/bin/python -m unittest tests/test_linguistic_and_field_evaluation.py

# Run collected data organizer and MP3 transcoding tests only (7 tests):
./.venv/bin/python -m unittest tests/test_collected_data_processor.py
```
*Current Status:* **All 44 project unit tests passing** across `tests/test_pipeline.py` (ASR core: Tokenizer, CNN time preservation, BiLSTM, Forward/Backward, CTCLoss, Fluency Assessment, Punctuation Stripping, Spelling Equivalence, Hotwords Interface), `tests/test_streamlit_pipeline.py` (Audio preprocessing, presets, 16kHz resampling, bandpass filter, denoising, WAV serialization, passage presets, custom story title editing/reset, calibrated word limits, and clean 2-column header / discreet settings footer architecture), `tests/test_audio_processor.py` (Mains hum notch filtering, VAD segmentation, temporal spectral smoothing, LUFS loudness normalization, classroom presets, format validation, and native AAC/M4A PyAV decoding), `tests/test_linguistic_and_field_evaluation.py` (Syllabification, grammatical categories, Grade 1-3 bounds and Grade 6 removal verification, statistical CIs, and human teacher correlation matrix), and `tests/test_collected_data_processor.py` (Zip extraction, filename metadata parsing, typo resolution, PyAV MP3 transcoding, and catalog consistency).

### 7.2 Launching Distributed Training (Multi-GPU DDP)
```bash
# Full 4-GPU DDP training:
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml

# Override epochs or batch size for testing:
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml --epochs 15 --batch_size 4
```
*Outputs Created:* Checkpoints and live training curve plots (`training_curves.png`, `.pdf`, `.svg`) are continuously updated in `checkpoints/best_model/`.

### 7.3 Running Single-GPU / Debug Training
```bash
./.venv/bin/python train.py --config configs/train_config.yaml --epochs 1
```

### 7.4 Running Benchmark Evaluation
```bash
# Greedy decoding on test-clean:
./.venv/bin/python evaluate.py --checkpoint_dir checkpoints/best_model --split test-clean

# Full KenLM Beam Search decoding on test-clean:
./.venv/bin/python evaluate.py \
  --checkpoint_dir checkpoints/best_model \
  --split test-clean \
  --use_beam_search \
  --lm_path lm/4-gram.binary
```
*Outputs Created:* Saves `evaluation_results.json` (containing standard metrics, corpus-wide `global_alignment` breakdown, and full `utterance_stats` distributions across all 2,620 samples) and renders publication diagnostic figures (`evaluation_diagnostics.png`, `.pdf`, `.svg`).

### 7.5 Generating Thesis Visuals & Figures
```bash
# Render all architectural/methodological thesis figures:
./.venv/bin/python scripts/generate_thesis_visuals.py

# Render figures from training history:
./.venv/bin/python scripts/plot_results.py --history checkpoints/best_model/training_history.json

# Render figures from evaluation results (using full empirical distributions):
./.venv/bin/python scripts/plot_results.py --results evaluation_results.json

# Generate complete publication demo suite:
./.venv/bin/python scripts/plot_results.py --demo
```

### 7.6 Launching the Streamlit Interactive Assessment App
```bash
# Launch interactive Oral Reading Fluency assessment prototype:
./.venv/bin/python -m streamlit run streamlit/app.py --server.port 8501

# Or execute via the launcher script:
./run_streamlit.sh 8501
```
*Features:*
- Dual audio input: in-browser live recording (`st.audio_input`) or file upload (`st.file_uploader`).
- Calibrated Grade 1–3 benchmark reading passages with editable custom override and clean word counting (`streamlit/passages.py`).
- Audio signal conditioning (`streamlit/audio_processor.py`): 16 kHz mono downmixing, 80 Hz – 7.5 kHz Butterworth bandpass filter, stationary spectral noise suppression, and RMS volume normalization.
- Side-by-side original vs. denoised audio preview players.
- Real-time CNN-BiLSTM-CTC inference + KenLM 4-gram beam search decoding with **Passage Contextual Biasing** (hotword boost slider, default 8.0) to prevent out-of-passage hallucinations.
- Educational reading fluency scorecard: Articulation Accuracy (%), WPM, CWPM, error breakdown (Substitutions, Omissions, Insertions), standard primary proficiency tier rating (*Independent*, *Instructional*, *Frustration*), and color-coded word-by-word alignment tokens with punctuation & spelling variant normalization.

### 7.7 Inspecting Technical Investigation & Diagnostics Reports
When debugging live microphone performance, acoustic degradation, or preparing thesis defense materials, consult:
- **`REPORTS.md` (`REP-2026-ASR-ORF-001`):** Complete technical investigation of the Benchmark-to-Prototype performance gap ("Laboratory-to-Field Distribution Shift"), detailing electrical mains hum ($50/60\text{ Hz}$ harmonics), WebRTC MediaRecorder packet jitter, spectral subtraction musical noise, and the Three-Tier Hardening Roadmap.
- **`COMMANDS.md`:** Comprehensive cheat sheet for all execution commands, environment configurations, and GPU overrides.

---

## 8. Deployment Artifact: The Checkpoint Bundle

The directory `checkpoints/best_model/` contains everything required for downstream inference, zero-shot assessment, and the Phase 2 Streamlit web application:

| File | Purpose |
| :--- | :--- |
| `model_weights.pt` | Complete PyTorch state dict for Wav2Vec2 + CNN + BiLSTM + Dense + CTC head (422 MB) |
| `model_config.json` | Exact hyperparameters for model reconstruction (channels, layers, hidden dims) |
| `vocab.json` | Complete 29-token vocabulary mapping |
| `metrics.json` | Best recorded validation metrics (`wer_percent: 3.38`, `cer_percent: 1.01`) |
| `training_history.json` | Full history of losses, validation WER/CER, and learning rates per epoch |
| `training_curves.*` | Publication figures in PNG, PDF, and SVG |

### Minimal Python Inference Example for Agents:
```python
import json
import torch
from src.data.tokenizer import CharacterTokenizer
from src.model.hybrid_asr import HybridASRModel
from src.decoding.beam_search import CTCDecoder
from src.assessment.alignment import clean_token

# 1. Load tokenizer & config
tokenizer = CharacterTokenizer.load_vocab("checkpoints/best_model/vocab.json")
with open("checkpoints/best_model/model_config.json") as f:
    cfg = json.load(f)

# 2. Instantiate and load weights
model = HybridASRModel(
    backbone_name=cfg["backbone"]["name"],
    cnn_channels=cfg["cnn"]["channels"],
    cnn_projection_dim=cfg["cnn"]["projection_dim"],
    lstm_hidden_size=cfg["rnn"]["hidden_size"],
    lstm_num_layers=cfg["rnn"]["num_layers"],
    vocab_size=cfg["vocab_size"]
)
model.load_state_dict(torch.load("checkpoints/best_model/model_weights.pt", map_location="cpu"))
model.eval()

# 3. Setup decoder
decoder = CTCDecoder(tokenizer=tokenizer, lm_path="lm/4-gram.binary")

# 4. Transcribe audio tensor (1, Time_samples) with optional passage contextual biasing
target_passage = "The quick brown fox jumps over the lazy dog."
hotwords = list({clean_token(w) for w in target_passage.split() if clean_token(w)})

with torch.no_grad():
    log_probs = model(waveform_tensor)
    transcription = decoder.decode(
        log_probs,
        use_beam_search=True,
        hotwords=hotwords,
        hotword_weight=8.0
    )[0]
```

---

## 9. Extension Guidelines for Future Agents (Phase 2 & Beyond)

When building Phase 3 (Session Tracking & PDF Export) or modifying existing capabilities:

1. **Preserve Modularity:**
   - Keep model logic in `src/model/`, data logic in `src/data/`, decoding in `src/decoding/`, evaluation in `src/assessment/`, and app UI in `streamlit/`.
   - Never import Streamlit or web UI code into `src/`. Streamlit code should live under `streamlit/`.
2. **Never Break Temporal Resolution:**
   - Any modifications to the CNN stage must preserve the `(1, 2)` max-pooling stride. Never introduce temporal downsampling `(2, 2)` without updating CTC length calculation and sequence alignment logic.
3. **Reproducibility & Verification:**
   - After modifying any component under `src/` or `streamlit/`, run `./.venv/bin/python -m unittest discover tests/` and verify that **all 15 unit tests pass** before considering the work complete.
4. **Environment Cleanliness:**
   - Keep model checkpoints (`*.pt`), dataset audio, and language models (`*.binary`, `*.arpa`) out of git commits. Check `.gitignore` before committing.
5. **Observe Signal Conditioning Standards:**
   - Review [REPORTS.md](file:///home/kathlenelaresma/tsis/REPORTS.md) when touching audio processing. Live classroom recordings require multi-harmonic notch filtering (60 Hz mains hum), voice activity detection (VAD), and EBU R128 LUFS loudness normalization to match laboratory benchmark accuracy.
6. **Mandatory Documentation Synchronization Rule:**
   - **For every single change made to the workspace**, `AGENTS.md` **MUST be updated**. Whenever code is refactored, new tests are added, configurations are modified, or architectural features are extended, update the corresponding sections of `AGENTS.md` before concluding your task so that future agents and developers never work with stale documentation.

