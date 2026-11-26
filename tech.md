# tech.md: How This Project Was Built, and Everything You Need to Understand It

This is the learning guide for the **Voice-Based Parkinson's Disease Screening** project.
It covers the story from start to finish:
- what we built, and why each piece exists;
- every problem we hit and how we solved it;
- why we chose each tool over the alternatives;
- the medical and audio background you need;
- the machine-learning ideas that make or break a project like this;
- where to go next.

Read it top to bottom once, then use it as a reference.

> **Live demo:** https://voice-parkinsons-screening.onrender.com
> **Reminder:** this is a research prototype, *not* a medical diagnosis tool.

---

## Table of contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [The journey, step by step (M1 → M7 → live)](#2-the-journey-step-by-step)
3. [Every problem we hit and how we solved it](#3-every-problem-we-hit-and-how-we-solved-it)
4. [The tech stack: what, why, and why not something else](#4-the-tech-stack)
5. [Domain knowledge: Parkinson's disease and the voice](#5-domain-knowledge-parkinsons-disease-and-the-voice)
6. [Audio and signal processing, explained simply](#6-audio-and-signal-processing-explained-simply)
7. [Self-supervised speech models (WavLM, Whisper)](#7-self-supervised-speech-models)
8. [Machine learning concepts that matter here](#8-machine-learning-concepts-that-matter-here)
9. [Reading our results like a scientist](#9-reading-our-results-like-a-scientist)
10. [Software engineering lessons](#10-software-engineering-lessons)
11. [Deployment: from a script to a live website](#11-deployment)
12. [What more we can build](#12-what-more-we-can-build)
13. [Technologies worth learning next](#13-technologies-worth-learning-next)
14. [A learning path to get really good at this](#14-a-learning-path)
15. [Interview-style questions to test yourself](#15-test-yourself)
16. [Glossary](#16-glossary)

---

## 1. The project in one minute

**Goal:** given a voice recording, estimate whether it sounds more like a person with Parkinson's disease (PD) or a healthy person of similar age.

**What we built:**

```
WAV files ─► manifest (who, what label, which task)
          ─► clean audio (16 kHz, mono, silence trimmed, volume normalised)
          ─► 3 kinds of features:
               1. classic acoustic features (58 numbers: MFCCs, pitch, jitter, shimmer, HNR)
               2. WavLM embeddings (768 numbers from a pretrained speech model)
               3. Whisper-encoder embeddings (512 numbers)
          ─► 4 classifiers each (Logistic Regression, SVM, Random Forest, MLP)
          ─► speaker-independent 5-fold cross-validation
          ─► results tables, plots, feature importance, final model
          ─► Gradio web demo ─► Docker ─► Render (live website)
```

**Headline results** (real, from the Kaggle GPU run: 786 recordings, 46 speakers):

| Features | Best model | Balanced accuracy | ROC-AUC |
|---|---|---|---|
| Acoustic (58 hand-made features) | SVM | 0.977 ± 0.039 | 0.994 |
| WavLM | Logistic Regression | 0.921 ± 0.040 | 0.958 |
| Whisper encoder | Logistic Regression | 0.882 ± 0.085 | 0.924 |

**The most important lesson:** these numbers are *too good to trust blindly*. The patients and the healthy people were recorded in different sessions, so some of what the model learned may be "which recording session is this?" rather than "is this Parkinson's?". [Section 9](#9-reading-our-results-like-a-scientist) explains why this matters, and it's the kind of thinking that separates good ML engineers from average ones.

---

## 2. The journey, step by step

We followed `SPEC.md`, which split the work into seven milestones. Each one ended with tests (`pytest`), a lint check (`ruff`) and a commit.

### M1: Scaffold (the skeleton)

We created the folder structure, a config file (`configs/default.yaml`), `requirements.txt`, `.gitignore`, and a first test.

**Why start with the skeleton?** Every later step needs a place to live. The config file means no "magic numbers" are buried in code: sample rate, seed, model names and paths all sit in one YAML file. `src/config.py` also detects automatically whether we're running on Kaggle (`/kaggle/input` exists) or locally, so the *same code* runs in both places.

**Why `.gitignore` first?** To make sure secrets (`.env`, `kaggle.json`) and big data (`data/`, `outputs/`) can never be committed by accident. Leaking an API token to a public GitHub repo is one of the most common real-world security mistakes.

### M2: Understand the data before touching models

The prompt still had the placeholder `<PASTE_SLUG_HERE>`, so we had to pick a dataset. We searched Kaggle and found that **most "Parkinson's voice" datasets are tables of pre-computed numbers** (like the famous UCI/Oxford CSV with 195 rows). Those have no audio and no speaker IDs, so we couldn't extract our own features or do a fair evaluation. We chose **Italian Parkinson's Voice and Speech** (`ieeeai/italian-parkinsons-voice-and-speech`), which has real `.wav` files and one folder per person.

Then we *investigated* the data. This step found most of the project's important problems:
- a full duplicate copy of the dataset inside itself;
- the same patients appearing in several session folders;
- a young-control group that would leak age;
- patients recorded at 44.1 kHz while controls were at 16 kHz.

All of this is documented in `docs/DATASET.md` and in [section 3](#3-every-problem-we-hit-and-how-we-solved-it).

We wrote `src/data.py` to build `manifest.csv`: one row per recording with `path, label, speaker_id, task, task_type, group, orig_sr, duration_s`. We also wrote `scripts/download_sample.py` to download a small, deterministic sample (84 files, 111 MB), because the Codespace has no GPU and limited disk.

> **Lesson:** spend real time looking at your data. Almost every serious bug in ML is a data bug, not a model bug.

### M3: The classic baseline

- `src/audio.py`: load → convert to mono 16 kHz → trim silence → normalise peak volume → split long clips into 30 s chunks. Broken files are skipped with a warning instead of crashing the run.
- `src/features_acoustic.py`: 58 features per recording, computed with librosa (MFCCs) and Praat via parselmouth (pitch, jitter, shimmer, HNR).
- `src/split.py`: `StratifiedGroupKFold` so **no speaker is ever in both train and test**, plus a hard `assert` that checks it.
- `src/models.py`: four classifiers, each inside a scikit-learn `Pipeline` (fill missing values → standardise → classify).
- `src/evaluate.py`: accuracy, balanced accuracy, F1, sensitivity, specificity and ROC-AUC per fold, then mean ± std.
- `src/train.py`: the command-line entry point (`python -m src.train --sample 20 --features acoustic`).

**Why a baseline first?** You can't tell whether a fancy model is good unless you know what a simple one gets. Here the simple baseline ended up being the *best* model, which happens more often than people expect.

### M4: Pretrained speech models

`src/features_ssl.py` turns each recording into one vector using **WavLM** or the **Whisper encoder**, then **caches** the vectors to `.npy` files, so running again costs about 12 s instead of about 5 minutes.

### M5: Evaluation outputs and explainability

Confusion matrix, ROC curve, permutation feature importance, the final model saved to `model.joblib`, and two extra analyses we added because they make the project *honest*:
- **speaker-level results:** average each person's recordings into one decision;
- **a deliberately leaky split** (random by recording), to *measure* how much leakage inflates scores.

### M6: The full run on Kaggle

The Codespace has no GPU, so the full run happened on Kaggle (free T4 GPU). We wrote a thin notebook that clones the GitHub repo, installs requirements and runs one command. We pushed it with `kaggle kernels push`, polled with `kaggle kernels status`, and downloaded results with `kaggle kernels output`.

The first run **crashed** because of a real bug in our chunking code (see [3.9](#39-the-chunk-bug-that-crashed-kaggle)). We fixed it, added a better test, pushed again, and run 2 completed in about 9 minutes.

### M7: Demo and README

`app/demo.py` is a Gradio app: upload or record audio → screening score plus readable voice measures, with the disclaimer shown twice. The README reports **only real numbers** from the Kaggle run.

### After M7: going live

Hugging Face Spaces refused to host it (it now needs a paid plan), so we containerised the app with **Docker** and deployed it to **Render** for free. It's live at https://voice-parkinsons-screening.onrender.com.

---

## 3. Every problem we hit and how we solved it

This is the most valuable section. Real projects are mostly about solving problems like these.

### 3.1 No dataset slug

**Problem:** the instructions said `<PASTE_SLUG_HERE>`.
**Solution:** we searched Kaggle and evaluated candidates, then asked you to choose.
**Why not just take the most popular Parkinson's dataset?** The most-voted ones are the UCI/Oxford CSV, about 195 rows of numbers that someone else already extracted, with **no audio** and **no reliable speaker ID**. Without audio we couldn't build our feature pipeline or test WavLM/Whisper. Without speaker IDs we couldn't do a fair (speaker-independent) evaluation, and the spec explicitly says to stop if there's no speaker ID.

### 3.2 Kaggle credentials in a different format

**Problem:** the spec expected `KAGGLE_USERNAME` + `KAGGLE_KEY`, but your `.env` had a newer-style `KAGGLE_TOKEN`.
**Solution:** Kaggle CLI 2.x accepts a token through `KAGGLE_API_TOKEN`, so we export one from the other:
```bash
set -a; . ./.env; set +a; export KAGGLE_API_TOKEN="$KAGGLE_TOKEN"
```
We never printed the token. Treat tokens like passwords.

### 3.3 Different Python versions (3.14 locally, 3.12 on Kaggle)

**Problem:** the spec said "pinned requirements", but exact pins like `numpy==2.5.5` may not exist for both Python versions, and reinstalling packages on Kaggle can break its preinstalled GPU-enabled PyTorch.
**Solution:** version *ranges* (`numpy>=1.26,<3`), so pip mostly finds that Kaggle already satisfies them and installs nothing.
**Trade-off:** ranges are less reproducible than exact pins. A stronger setup would use a lock file per environment (see `uv` and `pip-tools` in [section 13](#13-technologies-worth-learning-next)).

### 3.4 The dataset contains a copy of itself

**Problem:** the Kaggle upload holds `italian_parkinson/...` **and** `italian_parkinson/Italian_Parkinson_s_Voice_and_speech/...`, the same files again with slightly different names, plus a `.cache/huggingface/` folder.
**Danger:** if you scan everything, every recording appears twice. A recording could then land in both train and test, which is perfect leakage.
**Solution:** `is_canonical_wav()` in `src/data.py` only accepts files under a real group folder and skips the duplicate tree and hidden folders. A unit test checks this.

### 3.5 "28 people with Parkinson's" is actually 24 people

**Problem:** the PD folder has 28 sub-folders, but "Roberto R", "Nicola S" and "Vito S" appear in two or three session folders each. The metadata spreadsheet (`TAB_5.xlsx`) lists them with the same age and sex, so they are the same people recorded on different days.
**Danger:** with folder-based IDs, Roberto's 2016 session could be in training and his 2017 session in testing. The model could recognise *Roberto* instead of *Parkinson's*.
**Solution:** `speaker_id = <group>_<normalised name>`, so same-name folders *within a group* merge into one speaker. PD becomes 24 speakers.
**The subtle part:** "Vito L" exists in **both** the control group and the PD group, and those are different people. So we merge only *within* a group, never across. Merging across groups would create a "speaker" with two labels.
**Rule of thumb:** when unsure, group *more* aggressively. Over-grouping only costs you a little data; under-grouping silently inflates your results.

### 3.6 Young controls would leak age

**Problem:** the dataset has 15 young healthy controls (ages 19–29), while patients and elderly controls are roughly 40–80. The young group also only did 3 of the 16 tasks.
**Danger:** a model could learn "old voice = PD" or "only did reading tasks = healthy". Both are shortcuts, not disease detection.
**Solution:** excluded by default (`dataset.groups: {PD: PD, EHC: HC}` in the config). You can switch them back on for experiments.

### 3.7 Different sample rates per group (the big confound)

**Problem:** some PD sessions were recorded at **44.1 kHz**, while every control was **16 kHz**: 160 PD recordings from 10 patients.
**Why it matters:** different sample rate means different recording sessions, probably a different microphone, room or software. A model could learn to recognise the *recording setup*.
**What we did:**
1. Resampled everything to 16 kHz, which removes the most obvious difference.
2. Recorded `orig_sr` in the manifest so it could be measured.
3. Ran a **sanity check** on the full data: the same CV using **only 16 kHz recordings** (17 PD vs 22 HC speakers). Acoustic + SVM still scored 0.988 balanced accuracy.

**What that tells us:** the sample rate itself isn't the shortcut. But other session differences (room, microphone, date) **can't be ruled out** with this dataset. We wrote this plainly in the README. Being honest about limitations is part of doing ML properly.

### 3.8 Small engineering problems

| Problem | Cause | Fix |
|---|---|---|
| `SVC(probability=True)` deprecation warning | scikit-learn 1.9 deprecates it | `CalibratedClassifierCV(SVC(...), method="sigmoid", ensemble=False)`, the documented replacement |
| MLP crashed on tiny sample folds | `early_stopping=True` holds out 10% for validation; 10% of 8 samples is 1 sample, which is too few for 2 classes | turned early stopping off |
| Process killed (exit 143) while extracting WavLM on CPU | batches of 8 × 30 s clips with attention needed more RAM than the 7 GB Codespace had | `cpu_batch_size: 1` on CPU; batch of 8 kept on GPU |
| Padding could distort WavLM embeddings | batched clips of different lengths get zero-padded | pass the `attention_mask`, mean-pool only over real frames, and sort clips by length so padding is small |
| The `PR1` task isn't documented | the dataset's code sheet lists every other task | read the bundled transcript `PR1….txt`: it's a word list, so `task_type = words` (marked "inferred" in the docs) |

### 3.9 The chunk bug that crashed Kaggle

This is the most educational bug in the project.

**The code (buggy):**
```python
if len(chunks) > 1 and chunks[-1].size < sr:
    chunks[-2] = np.concatenate([chunks[-2], chunks.pop()])
```
**The intent:** if the last chunk is shorter than 1 second, glue it onto the previous chunk.

**What actually happens:** Python evaluates the right-hand side first. `chunks.pop()` removes the last item, and *then* the assignment to `chunks[-2]` happens on the **shorter** list.
- With exactly 2 chunks, the list now has 1 item, `chunks[-2]` doesn't exist, and you get an **IndexError**. That crashed the Kaggle run.
- With 3 or more chunks, it **silently** wrote the merged data into the *wrong* slot. That's worse, because nothing crashes.

**Why the test didn't catch it:** the test checked that the *total length* was right, and the total length stays correct even when the data is scrambled.

**The fix:**
```python
tail = chunks.pop()
chunks[-1] = np.concatenate([chunks[-1], tail])
```
**The better test:** put in `np.arange(n)` (every sample is unique), then check that the chunks join back into *exactly* the original signal, for several lengths.

> **Lessons:** (1) avoid mutating a list and indexing it in the same expression; (2) test *content*, not only *shape or size*; (3) edge cases like "exactly 2 chunks" are where bugs live, so test boundary values.

### 3.10 A saved model that won't load elsewhere

**Problem:** Kaggle trained and pickled the model with scikit-learn **1.6.1**; the Codespace has **1.9.1**. Loading gave `InconsistentVersionWarning`, and pickles can break between versions.
**Solution:** `scripts/refit_local_model.py` retrains the *same* pipeline on the *same* features (downloaded from Kaggle) with the local scikit-learn version. The deployed Docker image pins `scikit-learn==1.9.1` to match.
**General rule:** a pickle is only safe to load with the same library versions. For long-lived models, use a format like ONNX or keep the training code and data so you can refit.

### 3.11 Hugging Face said "Payment Required"

**Problem:** creating a Gradio Space returned `402 Payment Required`. Free CPU hosting for Gradio and Docker Spaces now needs PRO.
**Options we weighed:**
- pay for PRO;
- a static Space, which would mean rewriting the Praat features in JavaScript, which wouldn't match the training features;
- a temporary `gradio.live` link;
- a free container host.

**Choice:** a Docker container on **Render**'s free tier: permanent, free, and it builds from GitHub.

### 3.12 Fitting in 512 MB of RAM

**Problem:** Render's free tier gives 512 MB. Our first local test peaked at 421 MB on a 60 s file, and a longer upload could crash the app.
**Solution:**
- No PyTorch in the deployed image. The best model is acoustic-only, so it isn't needed, and that saves hundreds of MB.
- Only the first **30 s** of an upload are analysed.
- Tested in Docker with `-m 512m`: idle 243 MB, and a 248 s upload peaked at 340 MB with no out-of-memory kill.
- The 140 KB model is committed in `models/model.joblib`, because Render builds from GitHub and `outputs/` is ignored.

---

## 4. The tech stack

### 4.1 The table

| Layer | We used | Why | Why not the alternative |
|---|---|---|---|
| Language | **Python** | The whole audio/ML ecosystem lives here | R lacks speech models; Julia has a smaller ecosystem |
| Audio I/O + MFCC | **librosa**, **soundfile** | Standard, well documented, handles resampling | torchaudio pulls in PyTorch (heavy for the demo); scipy.io.wavfile can't resample |
| Voice-quality measures | **Praat via parselmouth** | Praat is *the* reference tool in speech science; jitter, shimmer and HNR defined the clinical way | Writing jitter yourself is easy to get subtly wrong; openSMILE is a great alternative (see 13) |
| Classic ML | **scikit-learn** | Pipelines, grouped CV, calibration and metrics in one consistent API | XGBoost/LightGBM are great for big tables, but we have about 800 rows, where an SVM shines |
| Pretrained speech | **HF Transformers** (WavLM, Whisper) | One API to download and run thousands of models | SpeechBrain or fairseq are fine too, but Transformers is the most widely used |
| Deep learning runtime | **PyTorch** | What Transformers runs on; Kaggle has it with CUDA | TensorFlow has less speech-model support today |
| Config | **YAML** + a small loader | Human-readable, no code changes for experiments | Hydra is powerful but overkill for one config |
| Tests | **pytest** with **synthetic audio** | Tests run anywhere in seconds, with no dataset and no GPU | Testing on real data is slow and can't run in CI |
| Lint/format | **ruff** | One very fast tool replaces flake8 + isort + black | Several separate tools to configure |
| GPU compute | **Kaggle Notebooks** (T4) | Free GPU, the dataset is already there, and there's a CLI to automate it | Colab has no CLI push/poll workflow; paid clouds cost money |
| Demo UI | **Gradio** | An upload/record UI in about 20 lines, made for ML demos | Streamlit is good but more app-oriented; React + FastAPI is a lot more work |
| Packaging | **Docker** | Same environment everywhere; Render runs it directly | Bare Python hosting can break on system libraries |
| Hosting | **Render** (free) | Free, builds from GitHub, auto-deploys on push | Hugging Face needs PRO for Gradio; Fly.io and Cloud Run need a card |

### 4.2 Why the hand-made features beat the "smarter" models (here)

It's tempting to assume big pretrained models always win. They didn't here, and the reasons are worth understanding:

1. **Tiny data.** 46 speakers is very little. A 768-dimensional embedding plus a classifier can overfit. 58 carefully chosen features are a much smaller search space.
2. **The features were designed for this.** Jitter, shimmer and HNR were invented to measure voice disorders, so they encode decades of expert knowledge.
3. **We used the models "frozen" and averaged over time.** Averaging every frame of a recording into one vector throws away timing information. We also only used the *last* layer, while research often finds middle layers better for paralinguistics (voice quality, emotion, health).
4. **WavLM and Whisper were trained for other goals.** Whisper was trained to *transcribe words*, so it learns to *ignore* voice quality, which is exactly the signal we care about. That's one plausible reason it ranked last.

---

## 5. Domain knowledge: Parkinson's disease and the voice

### 5.1 What is Parkinson's disease?

Parkinson's disease is a **progressive neurodegenerative disorder**. Neurons that produce **dopamine** in a brain region called the *substantia nigra* gradually die. Dopamine helps control smooth, well-scaled movement, so losing it causes:
- **tremor** (shaking at rest);
- **bradykinesia** (slowness and smaller movements);
- **rigidity** (stiffness);
- **postural instability** (balance problems).

It mostly affects people over 60 (around 1% of people in that age group). It is diagnosed **clinically**, by a neurologist examining the person; there is no single blood test. That's why cheap, non-invasive screening tools are an active research area.

### 5.2 Why the voice?

Speaking is one of the most complex motor skills we have: about 100 muscles coordinated in milliseconds across:
- **respiration** (lungs push air);
- **phonation** (vocal folds vibrate);
- **articulation** (tongue, lips and jaw shape sounds);
- **resonance** (throat, mouth and nasal cavities).

Parkinson's affects this motor control, so speech changes. Speech changes are reported in most people with PD (figures of roughly 70–90% are commonly quoted), and they can appear early. The pattern is called **hypokinetic dysarthria**:

| Symptom | What you hear | What we can measure |
|---|---|---|
| **Hypophonia** | soft, quiet voice | intensity (loudness) |
| **Monopitch** | flat, robotic melody | low F0 standard deviation and range |
| **Breathy / hoarse voice** | air leaking, rough quality | low HNR, high jitter and shimmer |
| **Imprecise consonants** | slurred "p", "t", "k" | spectral features (MFCCs), DDK regularity |
| **Rate changes / rushes** | speech speeds up or stalls | timing and pause statistics |
| **Reduced articulation range** | vowels sound alike | vowel space area, formants |

### 5.3 The speech tasks in our dataset, and why each exists

| Task | Code | What it tests |
|---|---|---|
| **Sustained vowels** ("aaaa…") | VA, VE, VI, VO, VU | Pure **phonation**: the vocal folds alone. Ideal for jitter, shimmer and HNR because pitch should be steady. |
| **DDK: diadochokinesis** ("pa-pa-pa", "ta-ta-ta") | D1, D2 | Speed and regularity of **articulation**, i.e. how fast and evenly you can repeat a movement. |
| **Reading a balanced text** | B1, B2 | **Connected speech**: prosody, rate, pauses. "Phonemically balanced" means it contains all the language's sounds in natural proportions. |
| **Phrases / words** | FB1, FB2, PR1 | Articulation of specific sounds. |

### 5.4 Screening vs diagnosis, and why the disclaimer isn't just legal text

A **diagnosis** says "you have X". A **screen** says "you might want to get checked". Even a good screen can mislead when the disease is rare. Here's an example of why:

> Imagine a test with **95% sensitivity** and **95% specificity** used on 10,000 people over 60, where about **1%** (100 people) have PD.
> - It catches 95 of the 100 → **95 true positives**.
> - It wrongly flags 5% of the 9,900 healthy people → **495 false positives**.
> - So among the 590 people flagged, only 95 actually have PD: a **positive predictive value of about 16%**.

So even an "accurate" test mostly raises false alarms when the condition is rare. That's why medical AI must be tested on the real population it will be used on, and why our demo says *Research prototype. Not a medical diagnosis.* Real clinical deployment would also need regulatory approval (e.g. FDA in the US, CE marking in Europe) and prospective clinical studies.

### 5.5 Ethics you should always think about

- **Privacy:** voice is biometric data: it identifies people and can reveal health. Store as little as possible and get consent.
- **Bias:** a model trained on Italian speakers from one region may fail on other languages, accents, ages, sexes or microphones.
- **Harm from errors:** a false "you may have PD" causes anxiety; a false "you're fine" may delay care.
- **Transparency:** say clearly what the model can and cannot do (our README's Limitations section).

---

## 6. Audio and signal processing, explained simply

### 6.1 What a digital recording is

Sound is air pressure changing over time. A microphone turns it into a voltage, and the computer **samples** that voltage many times per second.
- **Sample rate:** samples per second. 16,000 Hz (16 kHz) is standard for speech models; 44,100 Hz is CD quality.
- **Nyquist rule:** a sample rate of *N* can only represent frequencies up to *N/2*. So 16 kHz audio contains up to 8 kHz, which is plenty for speech intelligibility.
- **Mono vs stereo:** we average to one channel, because voice analysis doesn't need stereo.
- **Bit depth:** 16-bit means each sample is one of 65,536 levels.

### 6.2 Our preprocessing, and why each step exists

1. **Resample to 16 kHz.** WavLM and Whisper expect exactly 16 kHz, and it puts all recordings on equal footing.
2. **Trim silence.** Leading and trailing silence carries no voice information and would distort averages (e.g. the mean pitch).
3. **Peak-normalise.** Scale so the loudest sample is 0.95, which removes "how close was the microphone" from the data. Note the trade-off: this also removes *real* loudness information (hypophonia). A future version could measure loudness before normalising.
4. **Chunk long clips.** Whisper only accepts up to 30 s, and long inputs use a lot of memory, so we split them and average the chunk embeddings.

### 6.3 The acoustic features, one by one

**Fundamental frequency (F0) = pitch.** How many times per second the vocal folds vibrate. Typical adult values are roughly 85–180 Hz for male voices and 165–255 Hz for female voices. We compute the **mean, standard deviation and range**. Low variability is "monopitch".

**Jitter (local).** Cycle-to-cycle variation in the *period* (timing) of vocal-fold vibration. A healthy voice is very regular; an unhealthy one wobbles. Example: periods of 8.00, 8.06, 7.95 ms → jitter is the average absolute difference between neighbouring periods divided by the average period. Values around 1% are often quoted as the healthy/pathological boundary in Praat.

**Shimmer (local).** The same idea, but for *amplitude* (loudness) from cycle to cycle. A threshold of around 3.8% is commonly quoted.

**HNR (harmonics-to-noise ratio).** How much of the sound is clean periodic tone (harmonics) versus noise, in dB. High HNR (e.g. 20 dB) means a clear voice; low HNR means breathy or hoarse. In our run, HNR was among the top 3 most important acoustic features.

**MFCCs (Mel-Frequency Cepstral Coefficients).** The most widely used features in speech processing. Intuitively, they describe the **shape of the spectrum**, which reflects the shape of your throat, mouth and tongue. Here's how they're made:
1. Cut audio into short frames of about 25 ms.
2. Take the FFT (which frequencies are present).
3. Group frequencies on the **Mel scale**, which mimics human hearing: we distinguish low pitches finely and high pitches coarsely.
4. Take the log (we hear loudness logarithmically).
5. Apply a DCT, which compresses the information into a few coefficients.

We keep 13 coefficients and their **deltas** (how fast they change over time), and summarise each with mean and standard deviation over the recording: 13 × 2 × 2 = **52 numbers**. Add 6 Praat numbers and you get **58 features**.

### 6.4 Praat: the speech scientist's microscope

Praat is free software made by phoneticians at the University of Amsterdam, and it's the reference tool in clinical voice research. **parselmouth** lets Python call Praat's real algorithms, so our jitter and shimmer are computed the same way as in published research. That's much better than writing our own approximation.

---

## 7. Self-supervised speech models

### 7.1 What "self-supervised" means

Labelling audio is expensive. **Self-supervised learning (SSL)** trains on huge amounts of *unlabelled* audio by inventing a task from the data itself. WavLM, for example, hides (masks) parts of the audio and learns to predict what was hidden, like a fill-in-the-blank exercise. To get good at that, the model has to learn what speech is made of: sounds, speakers, prosody and noise.

### 7.2 The two models we used

| | **WavLM Base+** (Microsoft, 2021) | **Whisper base** (OpenAI, 2022) |
|---|---|---|
| Training | Self-supervised masked prediction on ~94k hours of unlabelled speech, with added noise and overlapping speakers | *Weakly supervised*: trained to transcribe ~680k hours of web audio |
| Input | Raw waveform | 80-band log-Mel spectrogram, padded to 30 s |
| Output per frame | 768 numbers, 50 frames/s | 512 numbers (encoder), 50 frames/s |
| Good at | Speaker, emotion and paralinguistic tasks (strong on the SUPERB benchmark) | Robust speech recognition and language ID |

### 7.3 From "a sequence of frames" to "one vector"

A 10 s clip gives about 500 frames × 768 numbers. Classifiers need one fixed-length vector, so we use **mean pooling**: average over time. Two details:
- **Only average real frames, not padding.** In a batch, short clips are padded with zeros, and including those would drag the average. We build a mask and divide by the count of valid frames.
- **Whisper always pads to 30 s,** so we compute how many encoder frames are real (50 per second) and pool only those.

Mean pooling is simple and robust, but it throws away *time*. Better options are in [section 12](#12-what-more-we-can-build).

### 7.4 Which layer?

Each transformer layer captures different information. Lower layers are closer to acoustics, and higher layers are closer to words. We used the **last layer** (the config's `layer: -1`), which is the simple default. A great next experiment is to try every layer and pick the best with nested CV; for health tasks the middle layers are often better.

---

## 8. Machine learning concepts that matter here

### 8.1 Data leakage: the number one killer of medical ML papers

**Leakage** means information from the test set sneaks into training, so the score looks better than reality. The forms that matter here:

1. **Speaker leakage.** The same person's recordings are in both train and test. The model can memorise the *voice*, not the disease. **We measured this:** with a random record-level split, every model scored higher. For example, Whisper + MLP went from 0.878 to 0.937, and the fold-to-fold std shrank from ±0.096 to ±0.034, so it also looked *more certain*. That's a double lie.
2. **Preprocessing leakage.** Fitting the scaler on *all* data before splitting. Our `Pipeline` re-fits scaling and imputation inside each training fold.
3. **Duplicate leakage.** The same file twice, as with the dataset's duplicate tree.
4. **Confound or shortcut leakage.** Something correlated with the label that isn't the disease: age, recording session, microphone, sample rate. This is the hardest to remove, and it's our dataset's main open question.

### 8.2 Cross-validation, and the "grouped" part

- **Train/test split:** train on 80%, test on 20%. With only 46 speakers, one split is very noisy.
- **K-fold CV (k=5):** split into 5 parts and rotate which part is the test set. Every sample is tested exactly once. Report the **mean ± std** over folds.
- **Stratified:** each fold keeps the PD/HC ratio.
- **Grouped:** all recordings of a speaker go to the same fold. → `StratifiedGroupKFold(groups=speaker_id)`.

We also *assert* in code that no speaker crosses train and test. A rule that's checked by code is much stronger than one in a comment.

### 8.3 Pipelines

```python
Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("clf", SVC())])
```
A pipeline behaves like one model, so `fit()` on training data fits *all* steps on training data only. It prevents preprocessing leakage and makes the saved model self-contained: the demo calls one object.

- **Imputation:** Praat can return NaN (e.g. no voiced frames in a noisy clip), and median imputation fills the gaps.
- **Standardisation:** features have very different scales (F0 is about 150, jitter about 0.005). SVMs, logistic regression and MLPs care about scale; random forests don't.

### 8.4 The four classifiers, intuitively

| Model | Intuition | Strength | Weakness |
|---|---|---|---|
| **Logistic Regression** | Draws a straight line (hyperplane) between classes, outputs a probability | Simple, stable, interpretable, great with many features | Can't model curved boundaries |
| **SVM (RBF kernel)** | Finds the widest-margin boundary and can curve it with a kernel | Excellent on small, well-scaled data, which is why it won here | Slow on huge data; probabilities need calibration |
| **Random Forest** | Many decision trees vote | No scaling needed, handles interactions | Can overfit noise on tiny data; weaker on high-dimensional embeddings |
| **MLP** (small neural net) | Layers of weighted sums plus non-linearities | Flexible | Needs more data; sensitive to settings |

### 8.5 Class imbalance

We have 56% PD and 44% HC recordings, which is mild. `class_weight="balanced"` makes errors on the smaller class cost more. We report **balanced accuracy** (the average of sensitivity and specificity), so a model can't look good by always predicting the majority class.

### 8.6 Metrics: what each one tells you

Using the confusion matrix (PD = positive):

|  | Predicted PD | Predicted HC |
|---|---|---|
| **Actually PD** | True Positive (TP) | False Negative (FN) |
| **Actually HC** | False Positive (FP) | True Negative (TN) |

- **Sensitivity (recall)** = TP / (TP + FN): of the sick, how many did we catch? This matters most for screening.
- **Specificity** = TN / (TN + FP): of the healthy, how many did we correctly clear?
- **Accuracy:** overall correct, which can be misleading when classes are imbalanced.
- **Balanced accuracy** = (sensitivity + specificity) / 2. Our main metric.
- **F1:** the harmonic mean of precision and recall.
- **ROC-AUC:** the probability that a random PD recording gets a higher score than a random HC recording. 0.5 is a coin flip and 1.0 is perfect. It doesn't depend on the threshold.

### 8.7 Calibration

When a model says "80%", does that happen 80% of the time? SVMs don't natively output probabilities, so we use **Platt scaling** (a sigmoid fitted on top), via `CalibratedClassifierCV`. Calibration matters for a demo that shows a percentage.

### 8.8 Explainability: permutation importance

Shuffle one feature's values, which breaks its link with the label, and measure how much ROC-AUC drops. A big drop means the model relied on that feature. We did this **on held-out speakers in each fold**, because measuring on training data would show what the model *memorised*, not what *generalises*.

In our run the top features were `mfcc5_mean`, `hnr` and `shimmer_local`. Two of the top three are classic voice-quality measures, which fits the clinical picture. Caveat: when features are correlated (like the MFCCs), shuffling one lets the model lean on its "twins", so importance spreads out and looks small. **SHAP** is an alternative with different trade-offs.

### 8.9 Speaker-level decisions

A person, not a recording, gets screened. Averaging a speaker's recording probabilities gave 23/24 PD speakers and 22/22 controls correct for acoustic + SVM. But with 46 people, **one person changes the result by 2–4 percentage points**, so treat these numbers as rough.

---

## 9. Reading our results like a scientist

| Claim | Evidence | Confidence |
|---|---|---|
| Speaker leakage inflates scores | All 12 models scored higher under the leaky split | **High**, measured directly |
| Acoustic features beat frozen WavLM/Whisper *on this dataset* | 0.977 vs 0.921 vs 0.882, consistent across classifiers | **Medium**: one dataset, 46 speakers, last layer only |
| The model detects Parkinson's | 0.977 balanced accuracy | **Uncertain**: recording sessions differ between groups |
| The 44.1 vs 16 kHz difference isn't the shortcut | The 16 kHz-only subset still scores 0.988 | Medium, and other session differences remain |

**How to settle the open question:**
1. **External validation:** train here, test on a *different* dataset recorded elsewhere. If accuracy collapses, the model learned the dataset, not the disease.
2. **Session-matched data:** patients and controls recorded with the same microphone, room and protocol.
3. **Check whether you can predict the session itself:** among PD patients, can the features tell 2016 from 2017 sessions? If yes, recording conditions leave a strong fingerprint.
4. **Compare with published work,** keeping in mind that many papers on small datasets report very high accuracies. Be sceptical of any result above about 95% without external validation.

> **The mindset:** a surprisingly good result is a *bug report* until proven otherwise.

---

## 10. Software engineering lessons

1. **Config over code.** Every setting is in `configs/default.yaml`, so experiments don't need code edits.
2. **One code base, many environments.** `config.py` detects Kaggle vs local, and the same `src/train.py` runs everywhere.
3. **Test without the dataset.** `tests/conftest.py` generates a synthetic voice-like tone (150 Hz harmonics plus noise). Tests check things like "pitch of a 150 Hz tone ≈ 150 Hz" and "no speaker leaks". Eighteen tests run in about 25 s with no data and no GPU.
4. **Test invariants, not just shapes.** The chunk bug taught us this (§3.9).
5. **Cache expensive work.** Embeddings are cached and only missing ones are computed. A model or layer change invalidates the cache automatically, because a stale cache is a classic silent bug.
6. **Fail soft on bad inputs.** A corrupt WAV logs a warning instead of killing a 9-minute GPU run.
7. **Use `logging`, not `print`,** so you get timestamps and levels, and can filter.
8. **Seeds everywhere** for reproducibility: Python, NumPy and PyTorch.
9. **Never print or commit secrets.** `.env` is gitignored *and* excluded from Docker builds via `.dockerignore`.
10. **Write the README from real outputs only.** Every number is traceable to a file in `docs/results/`.

---

## 11. Deployment

### 11.1 What Docker gives us

A Docker **image** is a frozen, portable mini-computer: OS + Python + libraries + our code. A **container** is a running image. If it works in Docker on your laptop, it works the same on Render.

Our `Dockerfile`, line by line:
- `FROM python:3.12-slim`: a small Linux with Python.
- Install `ffmpeg`, so browser-recorded audio formats can be decoded.
- `pip install -r requirements-deploy.txt`: only what the demo needs (no PyTorch).
- Copy `src/`, `app/` and `models/model.joblib`.
- Run as a non-root user, which is safer.
- `CMD python -m app.demo`, which listens on `0.0.0.0:$PORT`. Render provides `PORT`, and `0.0.0.0` accepts outside connections (`127.0.0.1` would only accept local ones).

### 11.2 Render

`render.yaml` is a **blueprint**: infrastructure as code. Render reads it, builds the Dockerfile from GitHub, and redeploys on every push to `main` (`autoDeploy: true`).
- **Free-tier trade-offs:** 512 MB RAM, shared CPU, and the service **sleeps after about 15 minutes idle**, so the first request takes about a minute (a "cold start").
- **Upgrade paths:** a paid instance (no sleep), or Hugging Face PRO with our ready-made `scripts/build_hf_space.py`.

### 11.3 How we verified the deployment

1. Local Docker with `-m 512m` to copy Render's memory limit.
2. Real predictions through the app's API using `gradio_client`, which is the same path the browser uses.
3. Measured peak memory on the longest file (248 s → 340 MB).
4. After deploy: checked HTTP 200 and the disclaimer on the live URL, and ran a real upload against it.

---

## 12. What more we can build

Roughly in order of value.

### 12.1 Make the science stronger (most important)
1. **External validation** on another PD speech dataset. Well-known ones include **PC-GITA** (Colombian Spanish), **NeuroVoz** (Castilian Spanish) and **mPower** (smartphone recordings from a large US study). Most need a data-access request.
2. **Session/confound analysis:** predict the recording session from the features, and try **domain-adversarial training** to remove session information.
3. **Nested cross-validation** to tune hyperparameters (SVM `C` and `gamma`, the WavLM layer) without leaking into the test score.
4. **Per-task models and fusion:** separate models for vowels, DDK and reading, then combine them. Which task carries the most signal?
5. **Confidence intervals** with a bootstrap over speakers, which is more honest than ± std with 46 people.

### 12.2 Better features and models
1. **eGeMAPS via openSMILE:** a standardised set of 88 clinically motivated voice features, the research-community baseline.
2. **Layer-wise probing** of WavLM and Whisper, which usually finds a better layer than the last.
3. **Better pooling:** mean + std, attention pooling, or statistics over time.
4. **Fine-tuning** WavLM with a small classification head, with heavy regularisation because the data is tiny.
5. **Data augmentation:** noise, reverb, phone/codec simulation, so the model is robust to real-world microphones.
6. **Interpretable clinical features:** DDK rate and regularity, pause statistics, vowel space area, intensity before normalisation.
7. **Other models:** ECAPA-TDNN (SpeechBrain), HuBERT, wav2vec 2.0, BEATs.

### 12.3 Better product
1. **Recording quality check:** reject clips that are too quiet, clipped, too noisy or too short, with a helpful message.
2. **Guided recording flow:** "say 'aaa' for 5 s", then "say pa-ta-ka", then "read this sentence", and combine the tasks.
3. **Show *why*:** compare the user's jitter, shimmer and HNR against the ranges in the training data.
4. **Longitudinal tracking:** voice changes over months may matter more than one snapshot (with consent and privacy).
5. **A FastAPI backend + mobile app** (React Native or Flutter); ONNX models for on-device inference so audio never leaves the phone.
6. **Monitoring:** log (anonymised) score distributions to detect drift.

### 12.4 Other conditions studied with voice (research areas)
- ALS (motor speech), Alzheimer's/MCI (language and pauses), depression (prosody), dysphonia and vocal-fold lesions, respiratory conditions (cough analysis, which is controversial and shows why confounds matter), stuttering, and fatigue or intoxication.
Our pipeline's *structure* (manifest → features → grouped CV → honest evaluation → demo) transfers to all of them.

### 12.5 MLOps upgrades
- **GitHub Actions CI:** run pytest + ruff on every push.
- **MLflow or Weights & Biases:** track every experiment's config and metrics.
- **DVC:** version data and models alongside code.
- **uv / pip-tools lock files:** exact, reproducible environments per platform.
- **Model card:** a standard one-page document of intended use, data, metrics and limitations.

---

## 13. Technologies worth learning next

| Area | Technology | What it's for |
|---|---|---|
| Speech features | **openSMILE / eGeMAPS** | Standard paralinguistic feature sets |
| Speech toolkits | **SpeechBrain**, **NVIDIA NeMo**, **ESPnet** | Recipes for speaker, emotion and ASR models |
| Deep learning | **PyTorch Lightning**, **HF Trainer**, **Accelerate** | Clean training loops, multi-GPU |
| Tabular ML | **XGBoost**, **LightGBM**, **CatBoost**, **Optuna** | Strong tree models and hyperparameter search |
| Explainability | **SHAP**, **Captum** | Feature attributions for tables and neural nets |
| Serving | **FastAPI**, **ONNX Runtime**, **TorchScript**, **BentoML** | Fast, portable model APIs |
| Front-end | **Streamlit**, **React**, **Next.js** | Richer apps than Gradio |
| Infra | **Docker Compose**, **Kubernetes**, **GitHub Actions** | Multi-service apps, scaling, CI/CD |
| Experiment tracking | **MLflow**, **Weights & Biases** | Never lose a result again |
| Data versioning | **DVC**, **LakeFS** | Git-like history for datasets |
| Environments | **uv**, **conda**, **pip-tools** | Reproducible installs |
| Privacy | **Federated learning** (Flower), **differential privacy** (Opacus) | Learn from medical data without centralising it |

---

## 14. A learning path

**Stage 1: foundations (2–4 weeks)**
- Python, NumPy, pandas, matplotlib.
- Statistics: mean/std, distributions, sensitivity/specificity, Bayes' rule (the PPV example in §5.4).
- scikit-learn: pipelines, cross-validation, the metrics above.
*Exercise:* reproduce our acoustic baseline from `features_acoustic.csv` in a notebook.

**Stage 2: audio (2–3 weeks)**
- Sampling, FFT, spectrograms, Mel scale, MFCCs (the librosa docs and tutorials).
- Praat basics: open a recording, view the pitch track, measure jitter.
*Exercise:* record yourself saying "aaa" calmly, then with a shaky voice, and compare jitter, shimmer and HNR in Praat.

**Stage 3: deep learning for speech (4–6 weeks)**
- PyTorch basics, transformers and attention.
- Hugging Face course (the audio course specifically).
*Exercise:* probe every WavLM layer on our data and plot balanced accuracy per layer.

**Stage 4: rigour (ongoing)**
- Read about data leakage and shortcut learning in medical ML.
- Learn nested CV, bootstrap confidence intervals and external validation.
*Exercise:* the session-prediction experiment from §9.

**Stage 5: shipping (2 weeks)**
- Docker, FastAPI, CI with GitHub Actions, a model card.
*Exercise:* add a CI workflow to this repo that runs `pytest` and `ruff`.

---

## 15. Test yourself

1. Why is a random train/test split over *recordings* wrong for this dataset? What did we measure when we tried it?
2. "Vito L" appears in both the PD and control folders. Why don't we merge them, while we do merge "Roberto R"?
3. What's the difference between jitter and shimmer? Which voice symptom does each capture?
4. Why fit the `StandardScaler` inside the pipeline instead of on all the data first?
5. A test has 95% sensitivity and 95% specificity. Why can most of its positives still be wrong?
6. Why might Whisper's encoder be *worse* than WavLM for voice quality?
7. What does mean pooling throw away, and what could you use instead?
8. Our chunk test passed but the code was wrong. Why, and how did we change the test?
9. Why couldn't we load Kaggle's `model.joblib` safely in the Codespace, and what did we do?
10. Our score is 0.977. Give three reasons to be cautious and one experiment to test each.
11. Why does the deployed Docker image not include PyTorch?
12. Why must a server listen on `0.0.0.0` instead of `127.0.0.1` inside a container?

*(Every answer is in this document.)*

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **Balanced accuracy** | Average of sensitivity and specificity; robust to class imbalance |
| **Calibration** | Making predicted probabilities match real frequencies |
| **Cold start** | Delay when a sleeping server wakes for a request |
| **Confound** | A hidden variable linked to both the input and the label (e.g. recording session) |
| **Cross-validation** | Rotating train/test splits to use all data for evaluation |
| **DDK (diadochokinesis)** | Rapid syllable repetition ("pa-ta-ka") testing articulation |
| **Dysarthria** | A motor speech disorder; *hypokinetic* dysarthria is the Parkinson's type |
| **Embedding** | A vector of numbers that represents an input (here, a recording) |
| **F0** | Fundamental frequency, i.e. voice pitch |
| **HNR** | Harmonics-to-noise ratio, i.e. voice clarity |
| **Jitter / Shimmer** | Cycle-to-cycle instability of pitch period / amplitude |
| **Leakage** | Test information sneaking into training, inflating scores |
| **Manifest** | A table listing every data file with its metadata |
| **Mean pooling** | Averaging frame vectors over time into one vector |
| **MFCC** | Mel-frequency cepstral coefficients: a compact description of spectral shape |
| **Nyquist** | The highest representable frequency is half the sample rate |
| **Permutation importance** | How much the score drops when a feature is shuffled |
| **Pipeline** | A chain of preprocessing steps and a model, fitted together |
| **PPV** | Positive predictive value: of those flagged, the share who truly have the condition |
| **ROC-AUC** | Threshold-free ranking quality; 0.5 is chance, 1.0 is perfect |
| **Sensitivity / Specificity** | True positive rate / true negative rate |
| **Self-supervised learning** | Learning from unlabelled data by solving invented tasks |
| **Shortcut learning** | A model solving the task by an unintended cue |
| **StratifiedGroupKFold** | CV that keeps class ratios and never splits a group (speaker) across folds |

---

*Every number in this document comes from the real Kaggle run (kernel version 2) or from the 16 kHz-only check on its features. The raw files are in `docs/results/`.*
