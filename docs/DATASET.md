# Dataset: Italian Parkinson's Voice and Speech

- **Kaggle slug:** `ieeeai/italian-parkinsons-voice-and-speech` (a mirror of the IEEE DataPort original)
- **Original:** Dimauro, G. & Girardi, F. (2019). *Italian Parkinson's Voice and Speech*. IEEE DataPort. doi:[10.21227/aw6b-tg17](https://dx.doi.org/10.21227/aw6b-tg17)
- **Papers to cite:** Dimauro et al., *VoxTester…*, IEEE MeMeA 2016 (doi:10.1109/MeMeA.2016.7533761); Dimauro et al., *Assessment of Speech Intelligibility in Parkinson's Disease Using a Speech-To-Text System*, IEEE Access 5, 2017 (doi:10.1109/ACCESS.2017.2762475)
- **License:** CC BY 4.0
- **Language:** Italian

## Files

Counts come from listing every file with `kaggle datasets files` (2,521 entries).

```
README.md
italian_parkinson/
├── 15 Young Healthy Control/      15 speakers,  45 wav   (+ 15_YHC.xlsx, FILE_CODES.xlsx)
├── 22 Elderly Healthy Control/    22 speakers, 349 wav   (+ Tab_3.xlsx, FILE_CODES.xlsx)
├── 28 People with Parkinson's disease/
│   ├── 1-5/ 6-10/ 11-16/ 17-28/   28 session folders → 24 unique people, 437 wav (+ TAB_5.xlsx)
└── Italian_Parkinson_s_Voice_and_speech/   DUPLICATE copy of the three folders above (ignored)
.cache/huggingface/…                         HF download metadata (ignored)
```

The canonical tree holds 831 wav files, about 0.85 GB. `src/data.py` ignores the duplicate tree and `.cache/`.

## Labels

The class comes from the top-level group folder:

| Group folder | Group code | Label (default config) | Speakers | Recordings |
|---|---|---|---|---|
| 28 People with Parkinson's disease | `PD` | **PD** (positive) | 24 | 437 |
| 22 Elderly Healthy Control | `EHC` | **HC** (negative) | 22 | 349 |
| 15 Young Healthy Control | `YHC` | excluded by default | 15 | 45 |

**Class balance (PD vs EHC):** 24 vs 22 speakers, and 437 vs 349 recordings (56% PD).

The young controls are **excluded by default** (`dataset.groups` in `configs/default.yaml`):
- They are aged 19–29, while the PD and elderly-control groups are aged 40–80 (see the xlsx tables). Including them would let a model learn age.
- They only have tasks B1, B2 and PR1 (no vowels, DDK or phrases), so task type would also leak the class.

## Speaker / subject ID

Each recording sits in a per-person folder named "First-name Surname-initial". The pipeline sets `speaker_id = <group>_<normalized folder name>`.

- **Repeat sessions are merged.** In the PD group the same name appears in more than one session folder: Roberto R (1-5, 11-16), Nicola S (6-10, 11-16) and Vito S (6-10, 11-16, 17-28). TAB_5.xlsx gives each the same age and sex, so they are the same people recorded on different dates. They get one `speaker_id` each, which takes PD from 28 folders to 24 speakers and prevents cross-session leakage.
- **No merging across groups.** "Vito L" appears in both EHC and PD. One is a control and one has PD, and the recordings come from different sessions, so they are treated as two people and the IDs stay distinct.
- Filename codes (scrambled initials + age + sex + timestamp) are **not** reliable IDs: the age in a file code sometimes disagrees with the xlsx table.

## Tasks (filename prefix)

| Code | Task (from FILE_CODES.xlsx) | `task_type` |
|---|---|---|
| VA1/2, VE1/2, VI1/2, VO1/2, VU1/2 | sustained vowels a, e, i, o, u (2 repetitions) | `vowel` |
| B1, B2 | 1st and 2nd reading of the phonemically balanced text "Il ramarro della zia" | `reading` |
| D1, D2 | syllable repetition "pa" / "ta" (~5 s, DDK) | `ddk` |
| FB1 | phonemically balanced phrases | `phrases` |
| FB2 | phonemically balanced words | `words` |
| PR1 | word list (not documented in FILE_CODES; inferred from the bundled transcript `PR1….txt`) | `words` |

Every PD and EHC speaker has up to 16 recordings (a few are missing).

## Audio format

These figures come from the downloaded sample (84 files):

- Mono, 16-bit PCM WAV.
- **Mixed sample rates:** all EHC files in the sample are 16 kHz. PD files from the 2016 session folder `1-5` are **44.1 kHz**, and the 2017 folders (`11-16`, `17-28`) are 16 kHz. The pipeline resamples everything to 16 kHz mono. Recording sessions still differ between groups, though, which is a **channel/session confound** (see Limitations in the README). The full manifest records `orig_sr` for every file, so the full run can quantify this.
- Durations range from 5 s (DDK) to about 250 s (some readings). Clips longer than 30 s are chunked.

## Local sample (`data/sample/`, gitignored)

`python scripts/download_sample.py` downloads the first 6 speakers per group (sorted by folder name) and their tasks VA1, VI1, VU1, D1, PR1 and B1. That is 84 wav files, about 111 MB. With the default config the manifest has 72 rows: PD 36 recordings from 6 speakers, HC 36 from 6.
