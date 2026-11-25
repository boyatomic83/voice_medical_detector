"""Configuration loading and environment-aware path resolution.

This is the only module allowed to contain absolute paths (Kaggle mount points).
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def on_kaggle() -> bool:
    """Return True when running inside a Kaggle notebook."""
    return KAGGLE_INPUT.exists()


def load_config(path: str | os.PathLike[str] = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML config and add resolved ``data_dir`` / ``out_dir`` entries.

    Relative paths are resolved against the project root. On Kaggle, the dataset
    directory is ``/kaggle/input/<folder>`` and outputs go to ``/kaggle/working``.
    """
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    cfg["data_dir"] = str(resolve_data_dir(cfg))
    out = Path(cfg["paths"]["out_dir"])
    if on_kaggle():
        out = KAGGLE_WORKING
    elif not out.is_absolute():
        out = PROJECT_ROOT / out
    cfg["out_dir"] = str(out)
    return cfg


def resolve_data_dir(cfg: dict[str, Any]) -> Path:
    """Return the dataset root for the current environment."""
    ds = cfg["dataset"]
    if on_kaggle():
        folder = ds.get("kaggle_folder") or str(ds["slug"]).split("/")[-1]
        candidates = [KAGGLE_INPUT / folder, KAGGLE_INPUT / "datasets" / str(ds["slug"])]
        # Kaggle sometimes mounts datasets under different layouts; take the first that exists.
        for cand in candidates:
            if cand.exists():
                return cand
        subdirs = [p for p in KAGGLE_INPUT.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
        return candidates[0]
    local = Path(ds["local_dir"])
    return local if local.is_absolute() else PROJECT_ROOT / local


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the acoustic baseline
        pass


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once."""
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
