"""Assemble a self-contained Hugging Face Space (Gradio SDK) in deploy/hf_space/.

    python scripts/build_hf_space.py [--model outputs/model.joblib] [--out deploy/hf_space]

The best model is acoustic-only, so the Space needs no torch/transformers (fast CPU
start). scikit-learn is pinned to the local version because the model is a pickle.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "src/__init__.py",
    "src/audio.py",
    "src/features_acoustic.py",
    "app/__init__.py",
    "app/demo.py",
]

APP_PY = '''"""Hugging Face Space entry point. Research prototype. Not a medical diagnosis."""

from app.demo import build_app

demo = build_app("model.joblib")

if __name__ == "__main__":
    demo.launch()
'''

README = """---
title: Voice Parkinson's Screening (research prototype)
emoji: 🎙️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {gradio}
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: Research prototype. Not a medical diagnosis.
---

# Voice-based Parkinson's screening: research prototype

> ⚠️ **Research prototype. Not a medical diagnosis.** It has not been clinically validated.
> Do not use it to make health decisions.

Model: {feature_set} features + {model_name}, trained on the Italian Parkinson's Voice and
Speech dataset (Dimauro & Girardi, 2019, CC BY 4.0, doi:10.21227/aw6b-tg17): 24 people
with PD vs. 22 elderly healthy controls. Speaker-independent CV balanced accuracy:
{bal:.3f}. PD and control recordings come from different sessions, so part of this score
may reflect recording conditions rather than disease.

Code, methods and limitations: https://github.com/boyatomic83/voice_medical_detector
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="outputs/model.joblib")
    ap.add_argument("--out", default="deploy/hf_space")
    args = ap.parse_args()

    import gradio
    import joblib
    import sklearn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(ROOT / args.model)
    if bundle["feature_set"] != "acoustic":
        sys.exit("This builder ships only acoustic models (no torch in the Space).")

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    for f in FILES:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, out / f)
    shutil.copy(ROOT / args.model, out / "model.joblib")
    (out / "app.py").write_text(APP_PY)
    reqs = [
        f"scikit-learn=={sklearn.__version__}",
        "numpy>=1.26,<3",
        "scipy>=1.11,<2",
        "pandas>=2.1,<4",
        "librosa>=0.10,<2",
        "soundfile>=0.12,<1",
        "praat-parselmouth>=0.4.3,<0.5",
        "joblib>=1.3,<2",
        "PyYAML>=6.0,<7",
    ]
    (out / "requirements.txt").write_text("\n".join(reqs) + "\n")
    (out / "README.md").write_text(
        README.format(
            gradio=gradio.__version__,
            feature_set=bundle["feature_set"],
            model_name=bundle["model_name"],
            bal=bundle["cv_metrics"]["balanced_accuracy_mean"],
        )
    )
    print(f"Space bundle written to {out}")


if __name__ == "__main__":
    main()
