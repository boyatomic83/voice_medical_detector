"""End-to-end pipeline CLI.

    python -m src.train --config configs/default.yaml [--sample N] \
        [--features acoustic,wavlm,whisper] [--out outputs/]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import load_config, set_seed, setup_logging
from src.data import manifest_from_config, summarize
from src.evaluate import (
    compute_metrics,
    cross_validate,
    plot_confusion,
    plot_roc,
    results_markdown,
    summarize_folds,
)
from src.explain import grouped_permutation_importance, plot_importance
from src.models import build_model, build_models
from src.split import make_folds

logger = logging.getLogger("train")

SSL_SETS = ("wavlm", "whisper")


def sample_manifest(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """Take ``n_per_class`` recordings per label (spread across speakers)."""
    parts = []
    for _, g in df.groupby("label"):
        # Round-robin over speakers so the sample covers as many speakers as possible.
        g = g.sample(frac=1.0, random_state=seed)
        g = g.assign(_r=g.groupby("speaker_id").cumcount()).sort_values(["_r", "speaker_id"])
        parts.append(g.head(n_per_class).drop(columns="_r"))
    return pd.concat(parts).sort_values("path").reset_index(drop=True)


def load_feature_set(name: str, manifest: pd.DataFrame, cfg: dict, out: Path) -> pd.DataFrame:
    """Return a DataFrame with a ``path`` column plus numeric feature columns."""
    if name == "acoustic":
        from src.features_acoustic import extract_acoustic_features

        return extract_acoustic_features(manifest, cfg, out / "features_acoustic.csv")
    if name in SSL_SETS:
        from src.features_ssl import extract_ssl_features

        return extract_ssl_features(manifest, cfg, name, out)
    raise ValueError(f"Unknown feature set {name!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Voice disease screening pipeline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sample", type=int, default=None, help="recordings per class")
    ap.add_argument("--features", default="acoustic,wavlm")
    ap.add_argument("--out", default=None, help="output dir (default: from config)")
    ap.add_argument(
        "--no-leaky",
        action="store_true",
        help="skip the record-level (speaker-leaking) comparison split",
    )
    return ap.parse_args(argv)


def _speaker_level(paths, y, oof, manifest: pd.DataFrame) -> dict:
    """Metrics after averaging out-of-fold probabilities per speaker."""
    d = pd.DataFrame({"path": paths, "y": y, "p": oof}).merge(
        manifest[["path", "speaker_id"]], on="path"
    )
    spk = d.groupby("speaker_id").agg(y=("y", "first"), p=("p", "mean"))
    return {"n_speakers": len(spk), **compute_metrics(spk["y"].to_numpy(), spk["p"].to_numpy())}


def run(args: argparse.Namespace) -> pd.DataFrame:
    """Run manifest → features → grouped CV → results/plots/model. Returns the summary."""
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    out = Path(args.out or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    manifest = manifest_from_config(cfg)
    if manifest.empty:
        raise SystemExit(f"No recordings found in {cfg['data_dir']}")
    if args.sample:
        manifest = sample_manifest(manifest, args.sample, seed)
    manifest.to_csv(out / "manifest.csv", index=False)
    summarize(manifest).to_csv(out / "manifest_summary.csv", index=False)

    ds = cfg["dataset"]
    pos, neg = ds["positive_label"], ds["negative_label"]
    feature_sets = [f.strip() for f in args.features.split(",") if f.strip()]
    per_fold, leaky, spk_rows = [], [], []
    store: dict[str, dict] = {}
    oof_all: dict[tuple[str, str], np.ndarray] = {}
    for fs in feature_sets:
        feats = load_feature_set(fs, manifest, cfg, out)
        data = manifest.merge(feats, on="path", how="inner")
        cols = [c for c in feats.columns if c != "path"]
        X = data[cols].to_numpy(dtype=np.float64)
        y = (data["label"] == pos).to_numpy(dtype=int)
        groups = data["speaker_id"].to_numpy()
        folds = make_folds(y, groups, cfg["cv"]["n_splits"], seed)
        store[fs] = {"X": X, "y": y, "folds": folds, "cols": cols, "paths": data["path"]}
        logger.info("[%s] X=%s, %d folds, %d speakers", fs, X.shape, len(folds), len(set(groups)))
        for name, model in build_models(cfg["models"], seed).items():
            fold_df, oof = cross_validate(model, X, y, folds)
            per_fold.append(fold_df.assign(feature_set=fs, model=name))
            oof_all[(fs, name)] = oof
            spk_rows.append(
                {"feature_set": fs, "model": name, **_speaker_level(data["path"], y, oof, manifest)}
            )
            logger.info(
                "[%s/%s] balanced_acc=%.3f auc=%.3f",
                fs,
                name,
                fold_df.balanced_accuracy.mean(),
                fold_df.roc_auc.mean(),
            )
            if not args.no_leaky:
                # Deliberately WRONG split (recordings, not speakers) to show leakage effect.
                skf = StratifiedKFold(len(folds), shuffle=True, random_state=seed)
                lk, _ = cross_validate(model, X, y, list(skf.split(X, y)))
                leaky.append(lk.assign(feature_set=fs, model=name))

    per_fold_df = pd.concat(per_fold, ignore_index=True)
    per_fold_df.to_csv(out / "results_per_fold.csv", index=False)
    summary = summarize_folds(per_fold_df)
    summary.to_csv(out / "results.csv", index=False)
    (out / "results.md").write_text(results_markdown(summary), encoding="utf-8")
    pd.DataFrame(spk_rows).to_csv(out / "results_speaker_level.csv", index=False)
    if leaky:
        lsum = summarize_folds(pd.concat(leaky, ignore_index=True))
        lsum.to_csv(out / "results_leaky_record_split.csv", index=False)
        (out / "results_leaky_record_split.md").write_text(results_markdown(lsum), "utf-8")

    best = summary.iloc[0]
    bfs, bmodel = best["feature_set"], best["model"]
    tag = f"{bfs}_{bmodel}"
    s = store[bfs]
    title = f"{bfs} + {bmodel} (out-of-fold, speaker-grouped CV)"
    plot_confusion(s["y"], oof_all[(bfs, bmodel)], [neg, pos], out / f"confusion_{tag}.png", title)
    plot_roc(s["y"], oof_all[(bfs, bmodel)], out / f"roc_{tag}.png", title)
    logger.info("Best: %s / %s (balanced acc %.3f)", bfs, bmodel, best["balanced_accuracy_mean"])

    if "acoustic" in store:
        a = store["acoustic"]
        best_ac = summary[summary.feature_set == "acoustic"].iloc[0]["model"]
        imp = grouped_permutation_importance(
            build_model(best_ac, seed), a["X"], a["y"], a["folds"], a["cols"], seed=seed
        )
        imp.to_csv(out / "feature_importance.csv", index=False)
        plot_importance(
            imp,
            out / "feature_importance.png",
            f"Acoustic baseline ({best_ac}): permutation importance",
        )

    final = build_model(bmodel, seed).fit(s["X"], s["y"])
    bundle = {
        "model": final,
        "feature_set": bfs,
        "model_name": bmodel,
        "feature_columns": s["cols"],
        "positive_label": pos,
        "negative_label": neg,
        "config": cfg,
        "cv_metrics": best.to_dict(),
        "n_train_recordings": len(s["y"]),
    }
    joblib.dump(bundle, out / "model.joblib")
    (out / "model_config.json").write_text(
        json.dumps(
            {k: v for k, v in bundle.items() if k not in ("model", "feature_columns")},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    run_info = {
        "sample": args.sample,
        "features": feature_sets,
        "n_recordings": len(manifest),
        "n_speakers": int(manifest.speaker_id.nunique()),
        "seed": seed,
        "best": {"feature_set": bfs, "model": bmodel},
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    logger.info("Results:\n%s", results_markdown(summary))
    return summary


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    run(parse_args(argv))


if __name__ == "__main__":
    main()
