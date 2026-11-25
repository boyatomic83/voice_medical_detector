"""Gradio demo: upload or record a voice sample → screening probability.

    python -m app.demo [--model outputs/model.joblib]

Research prototype. Not a medical diagnosis.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np

from src.audio import chunk, preprocess
from src.features_acoustic import extract_one, praat_features

logger = logging.getLogger("demo")

DISCLAIMER = (
    "⚠️ **Research prototype. Not a medical diagnosis.** This model was trained on a small "
    "Italian dataset (Parkinson's disease vs. elderly healthy controls) and has not been "
    "clinically validated. Do not use it to make health decisions; consult a doctor."
)


class Screener:
    """Wraps the saved model bundle and computes the matching features for new audio."""

    def __init__(self, model_path: str | Path) -> None:
        self.bundle = joblib.load(model_path)
        self.cfg = self.bundle["config"]
        self.fs = self.bundle["feature_set"]
        self.sr = self.cfg["audio"]["sample_rate"]
        self._encoder = None

    def _features(self, y: np.ndarray) -> np.ndarray:
        if self.fs == "acoustic":
            feats = extract_one(y, self.sr, self.cfg)
            return np.array([[feats[c] for c in self.bundle["feature_columns"]]], dtype=float)
        if self._encoder is None:
            from src.features_ssl import SSLEncoder

            s = self.cfg["features"]["ssl"][self.fs]
            self._encoder = SSLEncoder(self.fs, s["model_name"], int(s["layer"]), self.sr)
        chunks = chunk(y, self.sr, self.cfg["audio"]["max_chunk_seconds"])
        vecs = np.vstack([self._encoder.embed([c]) for c in chunks])
        return vecs.mean(0, keepdims=True)

    def predict(self, path: str) -> tuple[float, dict[str, float]]:
        """Return (probability of the positive class, interpretable voice measures)."""
        y = preprocess(path, self.sr, self.cfg["audio"]["trim_top_db"])
        if y is None or y.size < self.sr // 2:
            raise ValueError("Could not read audio, or it is shorter than 0.5 s after trimming.")
        prob = float(self.bundle["model"].predict_proba(self._features(y))[0, 1])
        a = self.cfg["features"]["acoustic"]
        return prob, praat_features(y, self.sr, a["f0_min"], a["f0_max"])


def explain(prob: float, measures: dict[str, float], bundle: dict) -> str:
    """Short markdown explanation of a prediction."""
    pos = bundle["positive_label"]
    cv = bundle.get("cv_metrics", {})
    level = "higher" if prob >= 0.5 else "lower"
    fmt = {
        "f0_mean": ("Mean pitch (F0)", "{:.0f} Hz"),
        "f0_std": ("Pitch variability", "{:.1f} Hz"),
        "jitter_local": ("Jitter (local)", "{:.2%}"),
        "shimmer_local": ("Shimmer (local)", "{:.2%}"),
        "hnr": ("Harmonics-to-noise ratio", "{:.1f} dB"),
    }
    rows = "\n".join(
        f"| {name} | {pattern.format(measures[k]) if np.isfinite(measures[k]) else 'n/a'} |"
        for k, (name, pattern) in fmt.items()
    )
    bal = cv.get("balanced_accuracy_mean")
    perf = (
        f"In speaker-independent cross-validation this model ({bundle['feature_set']} + "
        f"{bundle['model_name']}) reached a balanced accuracy of {bal:.2f}."
        if bal is not None
        else ""
    )
    return (
        f"### Screening score: {prob:.0%}\n"
        f"The model rates this recording as **{level}** similarity to the {pos} recordings "
        f"it was trained on (threshold 50%). {perf}\n\n"
        "Voice measures computed from your recording (for context, not used as a diagnosis):\n\n"
        "| Measure | Value |\n|---|---|\n" + rows + "\n\n"
        "Works best with a sustained vowel ('aaaa' for ~5 s) recorded in a quiet room."
    )


def build_app(model_path: str | Path):
    """Create the Gradio Blocks app."""
    import gradio as gr

    screener = Screener(model_path)

    def run(audio_path: str | None) -> str:
        if not audio_path:
            return "Please upload or record audio first."
        try:
            prob, measures = screener.predict(audio_path)
        except ValueError as exc:
            return f"**Error:** {exc}"
        return explain(prob, measures, screener.bundle)

    with gr.Blocks(title="Voice screening prototype") as app:
        gr.Markdown("# Voice-based Parkinson's screening (research prototype)")
        gr.Markdown(DISCLAIMER)
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Voice sample")
        btn = gr.Button("Analyse", variant="primary")
        out = gr.Markdown()
        btn.click(run, inputs=audio, outputs=out)
        gr.Markdown(DISCLAIMER)
    return app


def main() -> None:
    from src.config import setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="outputs/model.joblib")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    setup_logging()
    build_app(args.model).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
