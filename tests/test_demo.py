"""Demo prediction path with a tiny acoustic model (no Gradio server started)."""

import joblib
import numpy as np
import soundfile as sf

from app.demo import Screener, explain
from src.config import load_config
from src.features_acoustic import extract_one
from src.models import build_model

from .conftest import SR, make_tone


def test_screener_predict(tmp_path):
    cfg = load_config()
    rows = [extract_one(make_tone(f, 0.6, seed=i), SR, cfg) for i, f in enumerate([120, 200] * 4)]
    cols = list(rows[0])
    X = np.array([[r[c] for c in cols] for r in rows])
    model = build_model("logreg").fit(X, np.array([1, 0] * 4))
    bundle = {
        "model": model,
        "feature_set": "acoustic",
        "model_name": "logreg",
        "feature_columns": cols,
        "positive_label": "PD",
        "negative_label": "HC",
        "config": cfg,
        "cv_metrics": {"balanced_accuracy_mean": 0.5},
    }
    joblib.dump(bundle, tmp_path / "m.joblib")
    wav = tmp_path / "x.wav"
    sf.write(wav, make_tone(130, 1.0), SR)

    s = Screener(tmp_path / "m.joblib")
    prob, measures = s.predict(str(wav))
    assert 0 <= prob <= 1
    text = explain(prob, measures, bundle)
    assert "Screening score" in text and "Hz" in text
