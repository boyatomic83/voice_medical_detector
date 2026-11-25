"""Hand-crafted acoustic features: MFCC(+delta) statistics and Praat voice measures."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
from joblib import Parallel, delayed
from parselmouth.praat import call

from src.audio import preprocess

logger = logging.getLogger(__name__)


def mfcc_stats(y: np.ndarray, sr: int, n_mfcc: int = 13) -> dict[str, float]:
    """Mean and std of MFCCs and their deltas."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    n_frames = mfcc.shape[1]
    width = min(9, n_frames if n_frames % 2 else n_frames - 1)  # librosa needs odd width
    delta = librosa.feature.delta(mfcc, width=width) if width >= 3 else np.zeros_like(mfcc)
    out: dict[str, float] = {}
    for name, mat in (("mfcc", mfcc), ("dmfcc", delta)):
        for i in range(n_mfcc):
            out[f"{name}{i}_mean"] = float(mat[i].mean())
            out[f"{name}{i}_std"] = float(mat[i].std())
    return out


def praat_features(
    y: np.ndarray, sr: int, f0_min: float = 75.0, f0_max: float = 500.0
) -> dict[str, float]:
    """F0 statistics, local jitter, local shimmer and HNR via Praat (parselmouth).

    Values that Praat cannot compute (e.g. no voiced frames) are returned as NaN.
    """
    nan = float("nan")
    feats = dict.fromkeys(
        ["f0_mean", "f0_std", "f0_range", "jitter_local", "shimmer_local", "hnr"], nan
    )
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    try:
        pitch = snd.to_pitch(pitch_floor=f0_min, pitch_ceiling=f0_max)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        if f0.size:
            feats.update(
                f0_mean=float(f0.mean()),
                f0_std=float(f0.std()),
                f0_range=float(f0.max() - f0.min()),
            )
        pp = call(snd, "To PointProcess (periodic, cc)", f0_min, f0_max)
        feats["jitter_local"] = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        feats["shimmer_local"] = float(
            call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        )
        harm = call(snd, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
        feats["hnr"] = float(call(harm, "Get mean", 0, 0))
    except Exception as exc:  # Praat raises on degenerate input
        logger.debug("Praat features failed: %s", exc)
    return {k: (v if np.isfinite(v) else nan) for k, v in feats.items()}


def extract_one(y: np.ndarray, sr: int, cfg: dict) -> dict[str, float]:
    """All acoustic features for one preprocessed waveform."""
    a = cfg["features"]["acoustic"]
    return {
        **mfcc_stats(y, sr, a["n_mfcc"]),
        **praat_features(y, sr, a["f0_min"], a["f0_max"]),
    }


def _row(path: str, cfg: dict) -> dict[str, float] | None:
    sr = cfg["audio"]["sample_rate"]
    y = preprocess(path, sr, cfg["audio"]["trim_top_db"])
    if y is None or y.size < sr // 10:
        return None
    return {"path": path, **extract_one(y, sr, cfg)}


def extract_acoustic_features(
    manifest: pd.DataFrame, cfg: dict, out_csv: str | Path | None = None, n_jobs: int = -1
) -> pd.DataFrame:
    """Compute one feature row per recording (unreadable files are dropped with a warning)."""
    paths = manifest["path"].tolist()
    rows = Parallel(n_jobs=n_jobs)(delayed(_row)(p, cfg) for p in paths)
    ok = [r for r in rows if r is not None]
    if len(ok) < len(rows):
        logger.warning("Dropped %d unreadable/too-short recordings", len(rows) - len(ok))
    df = pd.DataFrame(ok)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logger.info("Saved acoustic features %s to %s", df.shape, out_csv)
    return df
