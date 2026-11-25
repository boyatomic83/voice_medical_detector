# SPEC: Voice-Based Disease Screening

## 1. Goal

Build a machine-learning pipeline that screens for a disease (default: **Parkinson's disease**) from voice recordings. Compare a classic acoustic-feature baseline against self-supervised speech-model embeddings (WavLM, optionally the Whisper encoder). Package it as a clean, reproducible GitHub project with a small web demo.

This is a **screening research prototype, not a diagnostic tool**. All docs and the demo must say so.

### Non-goals
- Clinical-grade accuracy or any medical claims
- Training large models from scratch
- Real-time / mobile deployment

---

## 2. Environment

| Where | Purpose | Notes |
|---|---|---|
| **GitHub Codespace** (CPU only) | Write code, run unit tests, run the pipeline on a tiny data sample | No GPU. Keep installs light. |
| **Kaggle Notebook** (GPU T4, Internet ON) | Full feature extraction + training | Dataset attached as input at `/kaggle/input/<dataset-folder>/`. Outputs go to `/kaggle/working/`. |
| **GitHub** | Source of truth for code | Repo is public so Kaggle can `git clone` it. |

### Secrets
- Kaggle credentials come from the env vars `KAGGLE_USERNAME` and `KAGGLE_KEY` (set as Codespaces secrets).
- **Never** commit `kaggle.json`, tokens, or the dataset. Add `kaggle.json`, `data/`, `outputs/`, and `*.pt` to `.gitignore`.

### Dataset
- Kaggle dataset slug: `DATASET_SLUG` (set in `configs/default.yaml`; ask the user if not set).
- First step: inspect the dataset structure (`kaggle datasets files <slug>`, then download a small subset into `data/sample/`) and write `docs/DATASET.md` describing the files, labels, number of speakers, sample rate, and class balance.
- The pipeline must work out a **speaker/subject ID** for every recording. If the dataset has none, stop and tell the user, because a speaker-independent evaluation is impossible without one.

---

## 3. Repository structure

```
voice-disease-detection/
├── SPEC.md
├── README.md
├── requirements.txt            # pinned; CPU-friendly locally, works on Kaggle
├── configs/
│   └── default.yaml            # dataset slug, paths, sample rate, model names, seeds
├── src/
│   ├── __init__.py
│   ├── config.py               # load YAML + auto-detect Kaggle vs local paths
│   ├── data.py                 # scan dataset → manifest.csv (path, label, speaker_id)
│   ├── audio.py                # load, resample to 16 kHz mono, trim silence, normalize
│   ├── features_acoustic.py    # MFCC stats, jitter, shimmer, HNR, F0 stats (librosa + parselmouth)
│   ├── features_ssl.py         # WavLM / Whisper-encoder mean-pooled embeddings, cached to .npy
│   ├── split.py                # speaker-grouped stratified K-fold (StratifiedGroupKFold)
│   ├── models.py               # LogisticRegression, SVM, RandomForest/XGBoost, small MLP
│   ├── evaluate.py             # metrics, confusion matrix, ROC curve, results table
│   ├── explain.py              # feature importance / SHAP for acoustic model
│   └── train.py                # CLI entry point that runs the full pipeline
├── app/
│   └── demo.py                 # Gradio app: upload/record audio → screening score
├── notebooks/
│   └── kaggle_run.ipynb        # thin notebook: clone, install, run src/train.py
├── kaggle/
│   └── kernel-metadata.json    # for `kaggle kernels push`
├── tests/
│   ├── test_split.py           # assert no speaker appears in both train and test
│   ├── test_audio.py
│   └── test_features.py
├── docs/
│   └── DATASET.md
└── outputs/                    # gitignored; metrics, plots, models
```

---

## 4. Pipeline

### 4.1 Data manifest (`data.py`)
- Walk the dataset directory and build `manifest.csv` with columns `path, label, speaker_id, task` (task = e.g. vowel / reading / speech, if available).
- Print class counts and speaker counts per class.

### 4.2 Audio preprocessing (`audio.py`)
- Load as mono at 16 kHz, trim leading/trailing silence, normalize peak amplitude.
- Skip corrupt files with a logged warning instead of crashing.
- Clips longer than 30 s are chunked; chunk embeddings are averaged per recording.

### 4.3 Features
**A. Acoustic baseline (`features_acoustic.py`)**: per recording:
- MFCC (13) + deltas, summarized by mean and std
- F0 mean/std/range, jitter (local), shimmer (local), HNR (via `praat-parselmouth`)
- Output: one row per recording in `outputs/features_acoustic.csv`

**B. Self-supervised embeddings (`features_ssl.py`)**
- Default model: `microsoft/wavlm-base-plus`. Optional: `openai/whisper-base` encoder only.
- Mean-pool the last hidden layer (configurable layer index) into one vector per recording.
- Use the GPU if available; batch inference; `torch.no_grad()`.
- **Cache** embeddings to `outputs/embeddings_<model>.npy` + an index CSV. Skip extraction if the cache exists.

### 4.4 Splitting (`split.py`)
- `StratifiedGroupKFold(n_splits=5)` with `groups = speaker_id`. Fixed seed.
- Hard assertion: no `speaker_id` in both train and test of any fold.

### 4.5 Models (`models.py`)
- Run these on each feature set: Logistic Regression, SVM (RBF), Random Forest or XGBoost, and a small MLP (sklearn is fine).
- Standardize features inside a sklearn `Pipeline` that is fit on the training folds only (no leakage).
- Handle class imbalance with `class_weight="balanced"`.

### 4.6 Evaluation (`evaluate.py`)
- Per fold and mean ± std: **accuracy, balanced accuracy, F1, sensitivity (recall), specificity, ROC-AUC**.
- Save:
  - `outputs/results.csv` (feature set × model × metrics)
  - `outputs/results.md` (markdown table for the README)
  - `outputs/confusion_<best>.png`, `outputs/roc_<best>.png`
- Pick the best model by mean balanced accuracy.

### 4.7 Explainability (`explain.py`)
- For the acoustic baseline: permutation importance or SHAP and a top-15 feature bar plot.
- Save `outputs/feature_importance.png`.

### 4.8 Final model
- Retrain the best (feature set, model) on all data and save it to `outputs/model.joblib`, with its config, for the demo.

### 4.9 CLI (`train.py`)
```
python -m src.train --config configs/default.yaml [--sample N] [--features acoustic,wavlm,whisper] [--out outputs/]
```
- `--sample N` uses only N recordings per class (for quick CPU tests in Codespaces).
- Paths are detected automatically: if `/kaggle/input` exists, use Kaggle paths.

---

## 5. Kaggle integration
- `notebooks/kaggle_run.ipynb` cells:
  1. `!git clone https://github.com/<user>/voice-disease-detection && cd voice-disease-detection`
  2. `!pip install -q -r requirements.txt`
  3. `!python -m src.train --config configs/default.yaml --out /kaggle/working`
- `kaggle/kernel-metadata.json` sets `enable_gpu: true`, `enable_internet: true`, attaches `DATASET_SLUG` in `dataset_sources`, and points `code_file` at the notebook.
- Helper commands documented in the README (and usable by Claude Code from the Codespace):
  - `kaggle kernels push -p kaggle/`
  - `kaggle kernels status <user>/<kernel-slug>`
  - `kaggle kernels output <user>/<kernel-slug> -p outputs/`

---

## 6. Demo (`app/demo.py`)
- Gradio interface: upload or record audio → preprocess → extract the best model's features → show the screening probability and a short explanation.
- A clear disclaimer is visible on the page: *"Research prototype. Not a medical diagnosis."*
- Runs on CPU. Loads `outputs/model.joblib`.

---

## 7. Quality requirements
- Python 3.10+, type hints on public functions, docstrings, `logging` instead of print in `src/`.
- Global seeds set for reproducibility.
- `pytest` passes in the Codespace without the full dataset (use synthetic audio, e.g. generated sine waves plus noise, in tests).
- `ruff` clean.
- No hard-coded absolute paths outside `config.py`.

---

## 8. README contents
1. One-paragraph summary + disclaimer
2. Dataset description and citation
3. Method diagram (Mermaid) and short explanation
4. Results table (from `outputs/results.md`) + confusion matrix / ROC images
5. Key finding: baseline vs WavLM vs Whisper, and why speaker-independent splits matter
6. How to run: Codespace quick test, Kaggle full run, demo
7. Limitations (small dataset, recording conditions, not clinically validated)

---

## 9. Milestones (commit after each)

| # | Milestone | Done when |
|---|---|---|
| M1 | Repo scaffold, config, requirements, `.gitignore`, tests skeleton | `pytest` runs |
| M2 | Dataset inspected, `docs/DATASET.md`, manifest builder | Manifest has speaker IDs and correct labels for the sample |
| M3 | Audio preprocessing + acoustic features + speaker-grouped CV + baseline models | `--sample 20 --features acoustic` runs end-to-end on CPU |
| M4 | WavLM / Whisper embeddings with caching | `--sample 10 --features wavlm` runs on CPU |
| M5 | Evaluation outputs, plots, explainability, final model | All files in §4.6–4.8 produced |
| M6 | Kaggle notebook + kernel metadata; full GPU run | Full results downloaded into `outputs/` |
| M7 | Gradio demo + README with real results | Demo works on a sample file; README complete |

---

## 10. Acceptance criteria
- [ ] Speaker-leakage test passes for all folds
- [ ] Results table compares at least 2 feature sets × 3 models with mean ± std
- [ ] Full pipeline reproducible on Kaggle from a single notebook run
- [ ] Demo runs locally on CPU with a disclaimer shown
- [ ] README reports real numbers (never made up) and lists limitations