"""Refit the best model from downloaded Kaggle outputs with the LOCAL scikit-learn version.

The Kaggle-trained ``model.joblib`` is pickled with Kaggle's sklearn and may not load
cleanly elsewhere. Acoustic features are deterministic, so refitting the same pipeline
on the same features gives an equivalent model for the demo.

    python scripts/refit_local_model.py --kaggle-out outputs/kaggle --out outputs/model.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.models import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kaggle-out", default="outputs/kaggle")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs/model.joblib")
    args = ap.parse_args()
    k = Path(args.kaggle_out)
    info = json.loads((k / "model_config.json").read_text())
    fs, name = info["feature_set"], info["model_name"]
    if fs != "acoustic":
        raise SystemExit(f"Best feature set is {fs}; refit from embeddings_{fs}.npy instead.")
    cfg = load_config(args.config)
    man = pd.read_csv(k / "manifest.csv")
    feats = pd.read_csv(k / "features_acoustic.csv")
    data = man.merge(feats, on="path")
    cols = [c for c in feats.columns if c != "path"]
    pos = cfg["dataset"]["positive_label"]
    y = (data["label"] == pos).astype(int).to_numpy()
    model = build_model(name, int(cfg["seed"])).fit(data[cols].to_numpy(float), y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = joblib.load(k / "model.joblib")["cv_metrics"]
    bundle = {
        "model": model,
        "feature_set": fs,
        "model_name": name,
        "feature_columns": cols,
        "positive_label": pos,
        "negative_label": cfg["dataset"]["negative_label"],
        "config": cfg,
        "cv_metrics": cv,
        "n_train_recordings": len(y),
    }
    joblib.dump(bundle, args.out)
    print(f"Saved {fs}+{name} trained on {len(y)} recordings -> {args.out}")


if __name__ == "__main__":
    main()
