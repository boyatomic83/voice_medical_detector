"""Self-supervised speech embeddings (WavLM, Whisper encoder), mean-pooled and cached.

Cache layout (in the output dir)::

    embeddings_<name>.npy        float32 [n_recordings, dim]
    embeddings_<name>_index.csv  path per row, plus model/layer used

Only recordings missing from the cache are extracted, so re-runs are free.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio import chunk, preprocess

logger = logging.getLogger(__name__)


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SSLEncoder:
    """Wrap a HuggingFace speech model and return one pooled vector per waveform chunk."""

    def __init__(self, kind: str, model_name: str, layer: int = -1, sr: int = 16000) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel, WhisperModel

        self.kind, self.layer, self.sr = kind, layer, sr
        self.device = _device()
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        if kind == "whisper":
            self.model = WhisperModel.from_pretrained(model_name).get_encoder()
        else:
            self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self._torch = torch
        logger.info("Loaded %s (%s) on %s", model_name, kind, self.device)

    def embed(self, chunks: list[np.ndarray]) -> np.ndarray:
        """Mean-pool hidden states of layer ``self.layer`` over valid (non-padded) frames."""
        torch = self._torch
        lengths = [len(c) for c in chunks]
        with torch.no_grad():
            if self.kind == "whisper":
                inp = self.extractor(chunks, sampling_rate=self.sr, return_tensors="pt")
                feats = inp.input_features.to(self.device)
                hs = self.model(feats, output_hidden_states=True).hidden_states[self.layer]
                # Whisper pads to 30 s; encoder emits 50 frames/s.
                n_valid = [max(1, min(hs.shape[1], math.ceil(n / self.sr * 50))) for n in lengths]
            else:
                inp = self.extractor(
                    chunks, sampling_rate=self.sr, return_tensors="pt", padding=True
                )
                values = inp.input_values.to(self.device)
                kwargs = {}
                if "attention_mask" in inp:
                    kwargs["attention_mask"] = inp["attention_mask"].to(self.device)
                out = self.model(values, output_hidden_states=True, **kwargs)
                hs = out.hidden_states[self.layer]
                out_len = self.model._get_feat_extract_output_lengths(torch.tensor(lengths))
                n_valid = [max(1, min(hs.shape[1], int(n))) for n in out_len]
            mask = torch.zeros(hs.shape[:2], device=hs.device)
            for i, n in enumerate(n_valid):
                mask[i, :n] = 1.0
            pooled = (hs * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return pooled.float().cpu().numpy()


def _cache_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    return out_dir / f"embeddings_{name}.npy", out_dir / f"embeddings_{name}_index.csv"


def load_cache(out_dir: Path, name: str, model_name: str, layer: int) -> pd.DataFrame | None:
    """Return cached embeddings as DataFrame(path, emb...) if they match model and layer."""
    npy, idx = _cache_paths(out_dir, name)
    if not (npy.exists() and idx.exists()):
        return None
    index = pd.read_csv(idx)
    if (index["model"] != model_name).any() or (index["layer"] != layer).any():
        logger.info("Cache %s was built with a different model/layer; ignoring it", npy)
        return None
    emb = np.load(npy)
    return _to_frame(index["path"].tolist(), emb, name)


def _to_frame(paths: list[str], emb: np.ndarray, name: str) -> pd.DataFrame:
    df = pd.DataFrame(emb, columns=[f"{name}_{i}" for i in range(emb.shape[1])])
    df.insert(0, "path", paths)
    return df


def extract_ssl_features(
    manifest: pd.DataFrame, cfg: dict, name: str, out_dir: str | Path
) -> pd.DataFrame:
    """Embeddings for every manifest path (cached). ``name`` is ``wavlm`` or ``whisper``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scfg = cfg["features"]["ssl"]
    model_name, layer = scfg[name]["model_name"], int(scfg[name]["layer"])
    sr, max_s = cfg["audio"]["sample_rate"], cfg["audio"]["max_chunk_seconds"]
    on_gpu = _device().type == "cuda"
    batch_size = int(scfg.get("batch_size", 8) if on_gpu else scfg.get("cpu_batch_size", 1))

    cached = load_cache(out_dir, name, model_name, layer)
    done = set(cached["path"]) if cached is not None else set()
    todo = [p for p in manifest["path"] if p not in done]
    logger.info("[%s] %d cached, %d to extract", name, len(done), len(todo))

    if todo:
        enc = SSLEncoder(name, model_name, layer, sr)
        # Flatten recordings into chunks, sort by length to minimise padding.
        items: list[tuple[str, np.ndarray]] = []
        for p in todo:
            y = preprocess(p, sr, cfg["audio"]["trim_top_db"])
            if y is None or y.size < sr // 10:
                logger.warning("Skipping unusable recording %s", p)
                continue
            items += [(p, c) for c in chunk(y, sr, max_s)]
        items.sort(key=lambda t: len(t[1]))
        per_rec: dict[str, list[np.ndarray]] = {}
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            vecs = enc.embed([c for _, c in batch])
            for (p, _), v in zip(batch, vecs, strict=True):
                per_rec.setdefault(p, []).append(v)
            if (i // batch_size) % 20 == 0:
                logger.info("[%s] %d/%d chunks", name, min(i + batch_size, len(items)), len(items))
        new_paths = sorted(per_rec)
        if new_paths:
            new = _to_frame(new_paths, np.stack([np.mean(per_rec[p], 0) for p in new_paths]), name)
            cached = new if cached is None else pd.concat([cached, new], ignore_index=True)
            _save_cache(cached, out_dir, name, model_name, layer)

    if cached is None:
        raise RuntimeError(f"No {name} embeddings could be extracted")
    return cached[cached["path"].isin(set(manifest["path"]))].reset_index(drop=True)


def _save_cache(df: pd.DataFrame, out_dir: Path, name: str, model_name: str, layer: int) -> None:
    npy, idx = _cache_paths(out_dir, name)
    np.save(npy, df.drop(columns="path").to_numpy(dtype=np.float32))
    pd.DataFrame({"path": df["path"], "model": model_name, "layer": layer}).to_csv(idx, index=False)
    logger.info("Cached %d %s embeddings to %s", len(df), name, npy)
