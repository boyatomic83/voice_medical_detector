"""Audio loading and preprocessing: mono 16 kHz, silence trim, peak normalisation."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray | None:
    """Load ``path`` as mono float32 at ``sr``. Returns ``None`` (with a warning) on failure."""
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None
    if y.size == 0 or not np.isfinite(y).all():
        logger.warning("Empty or non-finite audio: %s", path)
        return None
    return y.astype(np.float32)


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Remove leading/trailing segments quieter than ``top_db`` below peak."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size else y


def normalize_peak(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale so that ``max(|y|) == peak`` (silent input is returned unchanged)."""
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y if m == 0.0 else (y * (peak / m)).astype(np.float32)


def chunk(y: np.ndarray, sr: int, max_seconds: float = 30.0) -> list[np.ndarray]:
    """Split ``y`` into consecutive chunks of at most ``max_seconds``.

    A trailing chunk shorter than 1 s is merged into the previous one.
    """
    size = int(max_seconds * sr)
    if y.size <= size:
        return [y]
    chunks = [y[i : i + size] for i in range(0, y.size, size)]
    if len(chunks) > 1 and chunks[-1].size < sr:
        tail = chunks.pop()
        chunks[-1] = np.concatenate([chunks[-1], tail])
    return chunks


def preprocess(path: str | Path, sr: int = 16000, top_db: float = 30.0) -> np.ndarray | None:
    """Load → trim → normalise. Returns ``None`` for unreadable files."""
    y = load_audio(path, sr)
    if y is None:
        return None
    return normalize_peak(trim_silence(y, top_db))
