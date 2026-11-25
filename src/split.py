"""Speaker-grouped, stratified cross-validation folds."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


def make_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(train_idx, test_idx)`` pairs with no speaker in both sides.

    ``n_splits`` is reduced (with a warning) if a class has fewer speakers than folds.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    min_spk = min(len(np.unique(groups[y == c])) for c in np.unique(y))
    if min_spk < n_splits:
        logger.warning("Only %d speakers in smallest class; using %d folds", min_spk, min_spk)
        n_splits = min_spk
    if n_splits < 2:
        raise ValueError("Need at least 2 speakers per class for cross-validation")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    assert_no_speaker_leakage(folds, groups)
    return folds


def assert_no_speaker_leakage(
    folds: list[tuple[np.ndarray, np.ndarray]], groups: np.ndarray
) -> None:
    """Raise ``AssertionError`` if any speaker appears in train and test of a fold."""
    groups = np.asarray(groups)
    for i, (tr, te) in enumerate(folds):
        overlap = set(groups[tr]) & set(groups[te])
        assert not overlap, f"Fold {i}: speakers in both train and test: {sorted(overlap)}"
