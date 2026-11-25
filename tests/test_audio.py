import numpy as np
import soundfile as sf

from src.audio import chunk, load_audio, normalize_peak, preprocess, trim_silence

from .conftest import SR, make_tone


def test_trim_and_normalize(tone):
    trimmed = trim_silence(tone, top_db=30)
    assert trimmed.size < tone.size
    assert abs(np.max(np.abs(normalize_peak(trimmed))) - 0.95) < 1e-5
    assert np.all(normalize_peak(np.zeros(10)) == 0)


def test_load_resamples_stereo_to_mono_16k(tmp_path):
    t = make_tone(seconds=0.5, sr=44100)
    p = tmp_path / "x.wav"
    sf.write(p, np.stack([t, t], axis=1), 44100)
    y = load_audio(p, SR)
    assert y.ndim == 1
    assert abs(y.size - t.size * SR / 44100) < 10


def test_corrupt_file_returns_none(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"garbage")
    assert preprocess(p) is None


def test_chunking():
    y = np.zeros(SR * 65)
    parts = chunk(y, SR, 30)
    assert [len(c) // SR for c in parts] == [30, 30, 5]
    assert sum(map(len, chunk(np.zeros(SR * 60 + 100), SR, 30))) == SR * 60 + 100
    assert len(chunk(np.zeros(SR * 60 + 100), SR, 30)) == 2  # tiny tail merged
