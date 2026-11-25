"""Download a small, deterministic subset of the Kaggle dataset into data/sample/.

Usage:
    python scripts/download_sample.py [--speakers 6] [--tasks VA1,VI1,VU1,D1,PR1]

Auth: KAGGLE_API_TOKEN (or KAGGLE_TOKEN, or KAGGLE_USERNAME/KAGGLE_KEY) env vars.
Only the canonical tree ``italian_parkinson/<NN Group>/...`` is used. The dataset also
ships a duplicate copy under ``Italian_Parkinson_s_Voice_and_speech/`` and a
``.cache/`` folder; both are ignored.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, setup_logging  # noqa: E402
from src.data import group_and_speaker, is_canonical_wav  # noqa: E402

logger = logging.getLogger("download_sample")


def list_all_files(api, slug: str) -> list[str]:
    """Return every file path in the dataset (handles pagination)."""
    names: list[str] = []
    token = None
    while True:
        res = api.dataset_list_files(slug, page_token=token, page_size=200)
        names += [f.name for f in (res.files or [])]
        token = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
        if not token:
            return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--speakers", type=int, default=6, help="speakers per group")
    ap.add_argument("--tasks", default="VA1,VI1,VU1,D1,PR1,B1")
    args = ap.parse_args()
    setup_logging()

    if "KAGGLE_TOKEN" in os.environ and "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_TOKEN"]
    from kaggle.api.kaggle_api_extended import KaggleApi

    cfg = load_config(args.config)
    slug = cfg["dataset"]["slug"]
    out = Path(cfg["data_dir"])
    api = KaggleApi()
    api.authenticate()

    tasks = tuple(args.tasks.split(","))
    by_spk: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name in list_all_files(api, slug):
        if not is_canonical_wav(name):
            continue
        group, spk = group_and_speaker(name)
        if Path(name).name.upper().startswith(tasks):
            by_spk[(group, spk)].append(name)

    chosen: list[str] = []
    per_group: dict[str, int] = defaultdict(int)
    for (group, _spk), files in sorted(by_spk.items()):
        if per_group[group] < args.speakers:
            per_group[group] += 1
            chosen += sorted(files)
    logger.info("Downloading %d files (%s speakers per group)", len(chosen), dict(per_group))

    for name in chosen:
        dest = out / Path(name).parent
        if (dest / Path(name).name).exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        api.dataset_download_file(slug, name, path=str(dest), quiet=True)
    logger.info("Done -> %s", out)


if __name__ == "__main__":
    main()
