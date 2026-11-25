"""Scan the Italian Parkinson's Voice and Speech dataset and build ``manifest.csv``.

Layout (canonical tree only)::

    italian_parkinson/15 Young Healthy Control/<Name S>/<TASK><code>.wav
    italian_parkinson/22 Elderly Healthy Control/<NAME S>/<TASK><code>.wav
    italian_parkinson/28 People with Parkinson's disease/<range>/<Name S>/<TASK><code>.wav

Speaker IDs: ``<group>_<normalized folder name>``. In the PD group the same person
was recorded in more than one session folder (e.g. "Roberto R" under ``1-5`` and
``11-16``; same age in TAB_5.xlsx), so folders with the same normalized name *within a
group* are merged into one speaker. Names are never merged across groups ("Vito L"
in the elderly HC group (age 55) and in the PD group (age 70) are different people).
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path, PurePosixPath

import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)

GROUP_PATTERNS: dict[str, re.Pattern[str]] = {
    "YHC": re.compile(r"^\d+\s+young healthy control", re.I),
    "EHC": re.compile(r"^\d+\s+elderly healthy control", re.I),
    "PD": re.compile(r"^\d+\s+people with parkinson", re.I),
}
DUPLICATE_TREE = "italian_parkinson_s_voice_and_speech"
TASK_RE = re.compile(r"^(FB[12]|PR1|B[12]|D[12]|V[AEIOU][12])", re.I)
TASK_TYPES = {
    "V": "vowel",
    "B": "reading",
    "D": "ddk",
    "FB1": "phrases",
    "FB2": "words",
    "PR1": "words",
}
MANIFEST_COLUMNS = [
    "path",
    "label",
    "speaker_id",
    "task",
    "task_type",
    "group",
    "orig_sr",
    "duration_s",
]


def normalize_name(name: str) -> str:
    """Lower-case, strip accents and collapse non-alphanumerics to ``_``."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def _group_of(parts: tuple[str, ...]) -> tuple[str, int] | None:
    for i, part in enumerate(parts):
        for group, pat in GROUP_PATTERNS.items():
            if pat.match(part):
                return group, i
    return None


def is_canonical_wav(path: str) -> bool:
    """True for a .wav inside a group folder, excluding the duplicate tree and caches."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if not parts or not parts[-1].lower().endswith(".wav"):
        return False
    if any(p.startswith(".") or p.lower() == DUPLICATE_TREE for p in parts[:-1]):
        return False
    return _group_of(parts[:-1]) is not None


def group_and_speaker(path: str) -> tuple[str, str]:
    """Return ``(group, speaker_id)`` for a canonical recording path."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    found = _group_of(parts[:-1])
    if found is None:
        raise ValueError(f"No group folder in path: {path}")
    group, _ = found
    speaker_folder = parts[-2]
    return group, f"{group}_{normalize_name(speaker_folder)}"


def parse_task(filename: str) -> tuple[str, str]:
    """Return ``(task_code, task_type)`` from a filename such as ``VA1xxxx.wav``."""
    m = TASK_RE.match(filename)
    if not m:
        return "unknown", "unknown"
    code = m.group(1).upper()
    return code, TASK_TYPES.get(code, TASK_TYPES.get(code[0], "unknown"))


def build_manifest(
    data_dir: str | Path,
    groups: dict[str, str] | None = None,
    tasks: list[str] | None = None,
) -> pd.DataFrame:
    """Walk ``data_dir`` and return the manifest DataFrame.

    Args:
        data_dir: dataset root.
        groups: mapping group -> label (e.g. ``{"PD": "PD", "EHC": "HC"}``); groups
            not in the mapping are dropped. ``None`` keeps all with PD vs HC labels.
        tasks: optional list of task types (``vowel``, ``reading``, ...) to keep.
    """
    root = Path(data_dir)
    groups = groups or {"PD": "PD", "EHC": "HC", "YHC": "HC"}
    rows = []
    for wav in sorted(root.rglob("*")):
        if not wav.is_file():
            continue
        rel = wav.relative_to(root).as_posix()
        if not is_canonical_wav(rel):
            continue
        group, speaker = group_and_speaker(rel)
        if group not in groups:
            continue
        code, ttype = parse_task(wav.name)
        if tasks and ttype not in tasks:
            continue
        try:
            info = sf.info(str(wav))
            orig_sr, duration = int(info.samplerate), float(info.duration)
        except Exception as exc:  # corrupt / unreadable header
            logger.warning("Skipping unreadable file %s (%s)", wav, exc)
            continue
        rows.append(
            {
                "path": str(wav),
                "label": groups[group],
                "speaker_id": speaker,
                "task": code,
                "task_type": ttype,
                "group": group,
                "orig_sr": orig_sr,
                "duration_s": round(duration, 3),
            }
        )
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if df.empty:
        logger.warning("No recordings found under %s", root)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Recordings and speakers per label/group, logged and returned."""
    summary = (
        df.groupby(["label", "group"])
        .agg(
            recordings=("path", "size"),
            speakers=("speaker_id", "nunique"),
            sample_rates=("orig_sr", lambda s: ",".join(map(str, sorted(set(s))))),
            median_dur_s=("duration_s", "median"),
        )
        .reset_index()
    )
    logger.info("Manifest summary:\n%s", summary.to_string(index=False))
    return summary


def manifest_from_config(cfg: dict) -> pd.DataFrame:
    """Build the manifest using the ``dataset`` section of a loaded config."""
    ds = cfg["dataset"]
    return build_manifest(cfg["data_dir"], groups=ds.get("groups"), tasks=ds.get("tasks"))


def main() -> None:
    from src.config import load_config, setup_logging

    ap = argparse.ArgumentParser(description="Build manifest.csv")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = manifest_from_config(cfg)
    out = Path(args.out or cfg["out_dir"]) / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summarize(df)
    logger.info("Wrote %d rows to %s", len(df), out)


if __name__ == "__main__":
    main()
