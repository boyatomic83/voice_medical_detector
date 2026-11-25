"""End-to-end smoke test of the pipeline on a synthetic dataset (acoustic features only)."""

import json

import joblib
import soundfile as sf
import yaml

from src.train import main

from .conftest import SR, make_tone

PD = "italian_parkinson/28 People with Parkinson's disease/1-5"
EHC = "italian_parkinson/22 Elderly Healthy Control"


def test_pipeline_end_to_end(tmp_path):
    data = tmp_path / "data"
    for i in range(5):
        for j, vowel in enumerate(["VA1", "VE1"]):
            for root, f0, noise in ((PD, 120, 0.08), (EHC, 200, 0.005)):
                p = data / root / f"Spk{i} X" / f"{vowel}x.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(p, make_tone(f0 + 5 * i, 0.6, noise=noise, seed=10 * i + j), SR)
    with open("configs/default.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["dataset"]["local_dir"] = str(data)
    cfg["models"] = ["logreg", "rf"]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"

    main(["--config", str(cfg_path), "--features", "acoustic", "--out", str(out)])

    for f in [
        "results.csv",
        "results.md",
        "results_per_fold.csv",
        "model.joblib",
        "feature_importance.png",
        "results_speaker_level.csv",
        "results_leaky_record_split.csv",
        "run_info.json",
    ]:
        assert (out / f).exists(), f
    assert list(out.glob("confusion_*.png")) and list(out.glob("roc_*.png"))
    bundle = joblib.load(out / "model.joblib")
    assert bundle["feature_set"] == "acoustic"
    assert json.loads((out / "run_info.json").read_text())["n_speakers"] == 10
