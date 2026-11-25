"""Classifier factory. Every model is a Pipeline so scaling is fit on training folds only."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

AVAILABLE = ("logreg", "svm", "rf", "mlp")


def _classifier(name: str, seed: int) -> ClassifierMixin:
    if name == "logreg":
        return LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=seed)
    if name == "svm":
        # SVC(probability=True) is deprecated in sklearn>=1.9; Platt scaling via
        # CalibratedClassifierCV is the documented replacement (works on older sklearn too).
        return CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=seed),
            method="sigmoid",
            cv=3,
            ensemble=False,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    if name == "mlp":
        # sklearn's MLP has no class_weight; classes are near-balanced (~56/44) here.
        return MLPClassifier(
            hidden_layer_sizes=(64,),
            alpha=1e-2,
            max_iter=1000,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {AVAILABLE}")


def build_model(name: str, seed: int = 42) -> Pipeline:
    """Impute NaNs (median) → standardise → classifier."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", _classifier(name, seed)),
        ]
    )


def build_models(names: list[str], seed: int = 42) -> dict[str, Pipeline]:
    """Build a dict ``name -> unfitted Pipeline``."""
    return {n: build_model(n, seed) for n in names}
