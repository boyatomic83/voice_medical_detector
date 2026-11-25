"""Cross-validated evaluation: metrics per fold, mean ± std summary, tables."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

METRICS = ["accuracy", "balanced_accuracy", "f1", "sensitivity", "specificity", "roc_auc"]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Binary metrics with the positive class encoded as 1."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "roc_auc": auc,
    }


def cross_validate(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit a fresh clone per fold. Returns per-fold metrics and out-of-fold probabilities."""
    oof = np.full(len(y), np.nan)
    rows = []
    for i, (tr, te) in enumerate(folds):
        m = clone(model).fit(X[tr], y[tr])
        prob = m.predict_proba(X[te])[:, 1]
        oof[te] = prob
        rows.append({"fold": i, **compute_metrics(y[te], prob)})
    return pd.DataFrame(rows), oof


def summarize_folds(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of each metric per (feature_set, model)."""
    agg = per_fold.groupby(["feature_set", "model"])[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index().sort_values("balanced_accuracy_mean", ascending=False)


def results_markdown(summary: pd.DataFrame) -> str:
    """Markdown table with ``mean ± std`` cells."""
    header = "| Features | Model | " + " | ".join(METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    lines = [header, sep]
    for _, r in summary.iterrows():
        cells = [f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}" for m in METRICS]
        lines.append(f"| {r['feature_set']} | {r['model']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str], path, title: str):
    """Confusion matrix of pooled out-of-fold predictions at threshold 0.5."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = confusion_matrix(
        np.asarray(y_true), (np.asarray(y_prob) >= 0.5).astype(int), labels=[0, 1]
    )
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, path, title: str):
    """ROC curve of pooled out-of-fold probabilities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    RocCurveDisplay.from_predictions(
        np.asarray(y_true), np.asarray(y_prob), ax=ax, name="out-of-fold"
    )
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
