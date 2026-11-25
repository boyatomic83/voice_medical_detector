import numpy as np

from src.config import load_config
from src.features_acoustic import extract_one, praat_features

from .conftest import SR


def test_acoustic_feature_vector(tone):
    cfg = load_config()
    feats = extract_one(tone, SR, cfg)
    assert len(feats) == 13 * 4 + 6
    assert abs(feats["f0_mean"] - 150) < 5  # synthetic tone at 150 Hz
    assert feats["hnr"] > 10  # mostly harmonic
    assert 0 <= feats["jitter_local"] < 0.05


def test_praat_handles_noise_only():
    noise = np.random.default_rng(0).standard_normal(SR // 2).astype(np.float32) * 0.01
    feats = praat_features(noise, SR)
    assert set(feats) == {"f0_mean", "f0_std", "f0_range", "jitter_local", "shimmer_local", "hnr"}
