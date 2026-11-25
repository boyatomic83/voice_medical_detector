import numpy as np
import soundfile as sf

from src.data import build_manifest, group_and_speaker, is_canonical_wav, parse_task

PD = "italian_parkinson/28 People with Parkinson's disease"
EHC = "italian_parkinson/22 Elderly Healthy Control"


def test_canonical_filter():
    assert is_canonical_wav(f"{PD}/1-5/Roberto R/VA1rr49.wav")
    assert not is_canonical_wav(
        f"italian_parkinson/Italian_Parkinson_s_Voice_and_speech/{PD.split('/')[1]}/x/VA1.wav"
    )
    assert not is_canonical_wav(f".cache/huggingface/download/{PD}/1-5/R/VA1.wav")
    assert not is_canonical_wav(f"{PD}/TAB_5.xlsx")


def test_same_name_merged_within_group_not_across():
    assert group_and_speaker(f"{PD}/1-5/Roberto R/a.wav") == group_and_speaker(
        f"{PD}/11-16/Roberto R/b.wav"
    )
    assert (
        group_and_speaker(f"{PD}/17-28/Vito L/a.wav")[1]
        != group_and_speaker(f"{EHC}/VITO L/a.wav")[1]
    )
    assert group_and_speaker(f"{PD}/17-28/Nicolò C/a.wav")[1] == "PD_nicolo_c"


def test_parse_task():
    assert parse_task("VA1abc.wav") == ("VA1", "vowel")
    assert parse_task("FB1abc.wav") == ("FB1", "phrases")
    assert parse_task("B2abc.wav") == ("B2", "reading")
    assert parse_task("D1abc.wav") == ("D1", "ddk")
    assert parse_task("PR1abc.wav") == ("PR1", "words")
    assert parse_task("zzz.wav") == ("unknown", "unknown")


def test_build_manifest(tmp_path):
    sig = np.zeros(1600, dtype=np.float32)
    for rel in [
        f"{PD}/1-5/Roberto R/VA1x.wav",
        f"{PD}/11-16/Roberto R/VA1y.wav",
        f"{EHC}/ANNA B/VA1z.wav",
        "italian_parkinson/15 Young Healthy Control/L S/B1.wav",
    ]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, sig, 16000)
    (tmp_path / f"{EHC}/ANNA B/VE1broken.wav").write_bytes(b"not a wav")
    df = build_manifest(tmp_path, groups={"PD": "PD", "EHC": "HC"})
    assert len(df) == 3
    assert df.speaker_id.nunique() == 2
    assert set(df.label) == {"PD", "HC"}
    assert (df.orig_sr == 16000).all()
