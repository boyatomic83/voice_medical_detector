"""SSL caching / chunk averaging, tested with a fake encoder (no model download)."""

import numpy as np
import pandas as pd
import soundfile as sf

import src.features_ssl as fssl
from src.config import load_config

from .conftest import SR, make_tone


class FakeEncoder:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def embed(self, chunks):
        FakeEncoder.calls += len(chunks)
        return np.stack([[len(c) / SR, float(np.mean(np.abs(c)))] for c in chunks])


def test_extract_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fssl, "SSLEncoder", FakeEncoder)
    cfg = load_config()
    cfg["audio"]["max_chunk_seconds"] = 1.0
    paths = []
    for i, secs in enumerate([0.5, 2.5]):
        p = tmp_path / f"a{i}.wav"
        sf.write(p, make_tone(seconds=secs, seed=i), SR)
        paths.append(str(p))
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"nope")
    man = pd.DataFrame({"path": paths + [str(bad)]})

    df = fssl.extract_ssl_features(man, cfg, "wavlm", tmp_path / "out")
    assert list(df.path) == paths  # unreadable file skipped
    assert df.shape[1] == 3
    assert FakeEncoder.calls == 1 + 2  # 2.5 s -> 1 s + 1.5 s (short tail merged)
    assert (tmp_path / "out" / "embeddings_wavlm.npy").exists()

    calls = FakeEncoder.calls
    df2 = fssl.extract_ssl_features(man.iloc[:2], cfg, "wavlm", tmp_path / "out")
    assert FakeEncoder.calls == calls  # served from cache
    np.testing.assert_allclose(df2.iloc[:, 1:].to_numpy(), df.iloc[:, 1:].to_numpy())

    cfg["features"]["ssl"]["wavlm"]["layer"] = 3  # different layer invalidates cache
    fssl.extract_ssl_features(man.iloc[:1], cfg, "wavlm", tmp_path / "out")
    assert FakeEncoder.calls > calls
