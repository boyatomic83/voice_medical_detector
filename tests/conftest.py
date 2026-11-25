"""Shared pytest fixtures: synthetic audio so tests never need the real dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000


def make_tone(
    freq: float = 150.0, seconds: float = 1.0, sr: int = SR, noise: float = 0.01, seed: int = 0
) -> np.ndarray:
    """Harmonic 'voice-like' tone plus noise, padded with silence on both ends."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    sig = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 6))
    sig = 0.3 * sig / np.max(np.abs(sig)) + noise * rng.standard_normal(t.size)
    pad = np.zeros(int(0.3 * sr))
    return np.concatenate([pad, sig, pad]).astype(np.float32)


@pytest.fixture
def tone() -> np.ndarray:
    return make_tone()
