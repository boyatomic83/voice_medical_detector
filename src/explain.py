"""Explainability for the acoustic baseline: permutation importance on held-out folds."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def grouped_permutation_importance(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Mean drop in ROC-AUC when permuting each feature, measured on unseen speakers."""
    imps = []
    for tr, te in folds:
        m = clone(model).fit(X[tr], y[tr])
        if len(np.unique(y[te])) < 2:
            continue
        r = permutation_importance(
            m, X[te], y[te], scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=-1
        )
        imps.append(r.importances_mean)
    arr = np.vstack(imps)
    return (
        pd.DataFrame(
            {"feature": feature_names, "importance_mean": arr.mean(0), "importance_std": arr.std(0)}
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance(imp: pd.DataFrame, path: str | Path, title: str, top: int = 15) -> None:
    """Horizontal bar plot of the ``top`` most important features."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(d) + 1.2))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#4C72B0")
    ax.set_xlabel("Mean ROC-AUC drop when permuted (held-out speakers)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
