# Building a sleep-stage classifier on the iSLEEPS stroke cohort

### A development report

---

## 1. What I set out to do

The goal was a five-class AASM sleep-stage classifier — Wake, N1, N2, N3, REM —
operating on 30-second epochs of overnight polysomnography from the iSLEEPS
dataset: 100 stroke patients, each with a raw EDF recording and a scoring
workbook, plus a `subject_description.xlsx` carrying 63 clinical columns per
subject. About 18 GB of signal in total.

I expected the modelling to be the hard part. It wasn't. Roughly two thirds of
the effort went into establishing that my data was what I thought it was, and
almost every serious problem I hit was an integrity or bookkeeping problem
wearing a modelling costume. This report is organised in the order things
actually happened, because the order matters: several of my early results were
wrong in ways that looked entirely plausible at the time.

---

## 2. The data integrity audit

### 2.1 What I found

I had downloaded the dataset in pieces over several sessions and suspected some
duplication. The audit turned up four distinct categories of damage.

**Byte-identical duplicate copies.** Nine files with `(1)` suffixes — three EDFs
(SN28, SN29, SN30) and six workbooks — created when a re-download collided with
an existing file. I confirmed each was byte-identical by MD5 before deleting it.

**Zero-byte files.** Six, from downloads that opened a file handle and wrote
nothing.

**HTML error pages saved with data extensions.** Four files — `SN43.edf`,
`SN5.edf`, `SN53.xlsx`, `SN95.xlsx` — each exactly 9,379 bytes and all sharing
the MD5 `c1f9838a645648cb3b25359f7890a288`. Opening one revealed a GitHub Pages
"404 — page not found" document. My loader would have read these as sleep
recordings.

**Silent chunk corruption — the one that mattered.** Five files where the
download manager had written some 25 MiB blocks twice and dropped others. I
found these by hashing each file in 25 MiB blocks and looking for repeats:

| file | blocks | unique | duplicated pairs |
|---|---|---|---|
| `SN7.edf` | 8 | 6 | (0,1), (3,4) |
| `SN98.edf` | 9 | 7 | (0,2), (3,4) |
| `SN43(1).edf` | 8 | 6 | (2,3), (5,6) |
| `SN58.edf` | 7 | 5 | (0,1), (3,5) |
| `SN5(1).edf` | 6 | 4 | (0,1), (1,2) — EDF header destroyed |

**`SN43(1).edf` and `SN58.edf` were exactly the right length.** Because one
block was duplicated and another dropped, the byte count came out correct and
the EDF header arithmetic — header length plus records times record size —
balanced perfectly. Every cheap integrity check I could think of passed on files
whose signal data was scrambled. Had I not hashed at block level, two corrupted
recordings would have gone straight into training, and I would never have known.

`SN7.edf` was recoverable: a second copy, `SN7(1).edf`, hashed to seven distinct
blocks and matched the reference file size exactly, so I promoted it. The other
four had no clean copy and were re-downloaded.

### 2.2 A download that reported success and delivered nothing

With the damaged files identified and deleted, I went back to india-data.org to
re-download the four corrupted recordings and the two missing workbooks. The
site's download dialogue ran to completion and reported **"Completed"**. Nothing
arrived on my local filesystem.

I spent a while assuming this was my problem — checking the destination folder,
browser download settings, disk permissions, whether something was quarantining
the files on arrival. None of it was the cause. The failure was on
india-data.org's side, and the only real fix available to me was to wait for
them to resolve it and then retry, which eventually worked.

Two things are worth recording about this. The first is practical: I had already
deleted the corrupted originals by the time I discovered the downloads weren't
landing, which left me temporarily holding neither a working copy nor a broken
one. Quarantining damaged files rather than deleting them outright — moving them
to a holding folder until replacements are verified — would have cost nothing and
avoided that gap. That is now how I handle it.

The second is thematic, and it is the same lesson the corrupted files taught in a
different costume: **a system reporting success is not evidence that the work
happened.** A downloader that says "Completed" while writing nothing to disk and
a downloader that writes a correctly-sized file full of duplicated blocks are the
same failure in two forms. In both cases the only trustworthy signal is
independent verification of the artefact itself — which is precisely what
`verify_downloads.py` exists to provide.

### 2.3 Outcome

Twenty files deleted, 709 MiB freed, six re-downloaded. Final state: 100
complete EDF + workbook pairs, and all 201 files passing structural validation
and the block-duplication check.

I wrote `verify_downloads.py` and kept it in the repository. It checks header
self-consistency, size arithmetic, *and* block-level duplication, because the
first two would have cleared `SN43` and `SN58`.

### 2.4 What this taught me

**File size is not an integrity check, and neither is a format-level sanity
check.** A corruption that inserts one block and drops another is invisible to
both. If a file arrived over a network and I intend to train on it, it needs a
content hash — and if I don't have a reference hash, internal block-level
self-consistency is the next best thing.

---

## 3. Three traps inside the dataset itself

With the files clean, I started building a loader. Three properties of the data
would each have silently corrupted my results.

### 3.1 Two channel-naming conventions

64 of the recordings name electrodes against the `M1`/`M2` references
(`C4:M1`, `E1:M2`); the other 36 use the older `A1`/`A2` nomenclature
(`C4:A1`, `EOG1:A2`). These are the same derivations — A1 and M1 are both the
left mastoid.

A literal string match for `C4:M1` finds it in 64 files. **There is no EEG
channel present in all 100 recordings under a single name.** Any pipeline that
selects channels by exact label and skips recordings missing them would quietly
train on 64% of the data while reporting success. After aliasing, seven channels
are genuinely universal: C3, C4, O1, O2, E1, E2 and chin EMG.

### 3.2 Ten recordings are labelled with the wrong subject IDs

`subject_description.xlsx` has exactly 100 rows. I expected SN1 through SN100.
What it actually contains is **SN1–SN10, then AN1–AN10, then SN21–SN100.**
There are no SN11–SN20 rows at all.

The EDF headers resolve it. `SN11.edf` carries the internal patient identifier
`SN1`, `SN12.edf` carries `SN2`, and so on through `SN20.edf` → `SN10`. The ten
files stored as SN11–SN20 are the AN1–AN10 subjects.

A naive join of filename to metadata row therefore mislabels ten recordings —
about 10% of the cohort — with another patient's clinical data. I handle this
with an explicit mapping table in `subjects.py` rather than renaming files.

### 3.3 SN15 and SN28 are the same recording

While looking for duplicate content I found that `SN15.edf` and `SN28.edf` are
the same length, with the same start time (21:37:11), the same 26,405 data
records, and the same 29 channels.

They differ in exactly **eight bytes** — the patient identifier field. The MD5
of everything after the 7,680-byte header is identical. All 29 channel labels
match. And the hypnograms in `SN15.xlsx` and `SN28.xlsx` agree on all 880
epochs.

The metadata explains why: AN5 (which lives on disk as SN15) and SN28 are the
same patient — 38-year-old male, auto driver, right MCA territory infarct, AHI
5.8. The same night appears twice under two identifiers. I drop SN15.

### 3.4 Twelve patients appear more than once

Generalising that check, I grouped the metadata rows on all 63 columns. **Twelve
groups cover 26 recording IDs**, sharing age, sex, occupation, diagnosis, and
AHI exactly — mostly the same patient recorded on two different nights:

```
SN2 = SN57 = SN59     SN4 = SN26            SN5 = AN7
AN2 = SN22            AN5 = SN28 = SN38     SN29 = SN46
SN30 = SN56           SN31 = SN61           SN49 = SN50
SN53 = SN89           SN74 = SN75           SN92 = SN93
```

So my 99 usable recordings come from only **86 unique patients**. Splitting
cross-validation folds by recording would put the same patient in both training
and test — and in the SN15/SN28 case, literally the same night's signal. Every
subsequent split in this project is grouped by patient.

### 3.5 The hypnogram is not where I assumed

Two further surprises in the label pipeline.

First, **the hypnogram is not aligned to the recording start.** It typically
begins a few seconds earlier, and both clocks are wall-clock times of day on
recordings that cross midnight. I place each epoch by its own timestamp with a
day-wrap correction rather than assuming a common origin.

Second, **sheet 1 of the workbook is not always the hypnogram.** For most
recordings it is (`Signal ID = SchlafProfil\profil`, stages in column B), but
`SN80.xlsx` leads with the *light sensor in lux*, carrying the sleep stage in a
separate third column. This is the source of the nonsense stage values I had
been seeing in my label counts — entries like `54`, `136`, `33`. Those were
illuminance readings.

Before I fixed this, SN80 produced **zero** usable epochs and several other
recordings were partially mislabelled. The parser now scans every sheet and
every column and keeps whichever column actually contains stage names. After the
fix, all 99 recordings align cleanly and the label vocabulary contains only the
five AASM stages plus `Artefact`, `A` and `Movement`, which I drop.

Final usable dataset: **93,937 scored epochs** across 99 recordings and 86
patients.

| stage | epochs | share |
|---|---|---|
| N2 | 39,198 | 41.7% |
| Wake | 25,921 | 27.6% |
| REM | 11,238 | 12.0% |
| N1 | 9,360 | 10.0% |
| N3 | 8,220 | 8.8% |

### 3.6 What this taught me

**Validate the label pipeline before the model pipeline.** All three of these
would have produced a model that trained, converged, and reported a plausible
number. Two of them — the AN mapping and the patient duplication — would have
made that number *better* than the truth. The dataset's own metadata file was
the thing that exposed both, and I only read it carefully because I was chasing
an unrelated size collision.

---

## 4. Choosing an approach

With clean data I had to decide what to build. I considered two families
seriously.

**Gradient-boosted trees on engineered features** — the approach I chose as the
baseline. Four reasons:

1. *The physics is already known.* AASM staging rules are defined in terms of
   band power and a small number of waveform properties: delta power for N3,
   sigma spindles for N2, chin atonia plus conjugate eye movement for REM, alpha
   dropout at the Wake/N1 boundary. A feature encoding "relative sigma power"
   starts at the answer; a CNN has to rediscover the Fourier transform from
   3,000 raw samples.
2. *Sample size.* 86 patients is small. The deep models in this literature —
   DeepSleepNet, U-Time, USleep — are trained on hundreds to thousands of
   nights.
3. *Compute.* Trees train on CPU in minutes.
4. *This cohort is pathological.* Stroke patients have focal slowing and
   asymmetry. Feature importances let me see which channels and bands the model
   leans on; a CNN gives an accuracy number and little else.

**A CNN on the raw signal** — built as the comparison, and I'll return to why it
earned its place despite losing.

I rejected linear models and SVMs because the decision boundaries here are
strongly interacting (REM is low EMG **and** eye movement **and** mixed
frequency), and a linear model needs those interactions hand-constructed. I
rejected random forests as strictly dominated by boosting on tabular biosignal
features at equal cost. I deferred an HMM/CRF layer to a later step, and
deferred pretrained EEG foundation models as too heavy to justify before a
baseline existed.

### 4.1 The features

Seven channels, resampled to 100 Hz. EEG and EOG band-passed 0.3–35 Hz; EMG
10–45 Hz, because its discriminative content is high-frequency muscle tone and
its low-frequency content is movement artefact.

Per 30-second epoch I compute **106 base features**: for each EEG channel, the
five relative band powers, log total power, five log band ratios, a slow/fast
ratio, spectral entropy, the 95% spectral edge frequency, Hjorth mobility and
complexity, and robust time-domain moments. For the EOG channels, band-limited
slow and fast power plus the **E1–E2 correlation** — REM bursts move the eyes in
opposite directions, so this goes sharply negative. For the EMG, log RMS and a
high/low band ratio, which is the atonia detector.

Then three context transforms, applied per recording:

- **Robust z-scoring** on the median and IQR, which removes between-subject and
  between-amplifier amplitude differences that would otherwise dominate the
  split criteria;
- **Centred rolling means over ±2 and ±7 epochs**, because a single 30-second
  epoch is genuinely ambiguous and human scorers look at neighbours too;
- **Position in the night**, because N3 concentrates early and REM late.

That gives **426 features** per epoch. The context transforms turned out to
matter enormously — more on that in section 7.

---

## 5. The bugs, and what each one cost me

This is the part I'd most want to read if someone else had written it. Four of
these produced *numbers* — plausible-looking numbers I could have reported.

### 5.1 A stale fold cache made my model look broken

I had built per-fold caching so interrupted training runs could resume. The
cache stored each fold's test indices and predictions.

I piloted on 20 patients (19,275 epochs), then scaled to all 99 recordings
(93,937 epochs). The full run reported **fold accuracies around 0.33** —
roughly chance — and announced that 74,662 epochs were unpredicted.

The cause: the fold files carried no identifier of which dataset they came from.
The pilot had cached test indices into a 19,275-row array; the full run happily
replayed them against 93,937 rows, addressing the wrong epochs entirely, and the
five cached folds "covered" only the old 19k.

The model was never the problem. The fix was a fingerprint — an MD5 over the
recording list, epoch count, fold count, seed, and hyper-parameters — stored in
each fold file and checked on load, with a mismatch triggering a silent
recompute. I verified it fires in both directions: identical settings reuse the
cache; a changed tree count forces a refit.

### 5.2 A fold mapping that tested on one recording

This one is worse, because it produced a *good* number.

My first full CNN run reported **accuracy 0.8115, κ 0.6912** — better than the
tree model. The fold line read `98 train / 1 test recordings ([SN1])`, and the
per-class supports summed to 716 epochs, which is exactly SN1's length. REM had
two epochs in it, and consequently a precision of 0.071.

The bug: I built folds with `GroupKFold` over a 99-element *per-recording* array
and then mapped the resulting indices through `searchsorted` on cumulative
*epoch* counts. Recording indices 0–98 nearly all fall inside SN1's first 716
epochs, so every fold collapsed to testing on SN1 alone. The model was being
evaluated on 0.76% of the data.

The fix builds folds once on epochs — exactly as the tree model does — then maps
each patient, and hence each recording, onto the fold its epochs landed in. I
verified against the real data that the recording folds now reproduce the tree
model's epoch partitions exactly: 79/20 train/test recordings per fold, ~18.9k
test epochs, identical to the tree partition on all five folds.

Corrected, the CNN scored **0.7349** — not 0.8115.

### 5.3 Dropped tail epochs

Found while fixing the above. My sequence windowing used
`range(0, n - seq_len + 1, seq_len)`, which never emits a final partial window.
The last ~16 epochs of every recording were never predicted and counted as
errors. I pulled the window logic into a testable function and verified full
coverage for every recording length.

### 5.4 The textbook correction that destroyed the model

Sleep is a strongly self-transitioning Markov chain. Measured on this data:

| stage | probability of staying |
|---|---|
| Wake | 0.908 |
| N1 | 0.711 |
| N2 | 0.930 |
| N3 | 0.913 |
| REM | 0.952 |

Both my models score epochs semi-independently, so Viterbi decoding over the
posteriors should help. The standard conversion from a classifier posterior to
an HMM emission likelihood is to divide by the class prior:
`P(x|s) ∝ P(s|x)/P(s)`.

My first implementation scored **0.299** — worse than always guessing N2.

The reason is that the textbook conversion assumes a classifier trained on the
natural class distribution. **Both of my models are trained with class weights**,
so their implied prior is already tilted toward the rare stages; dividing by the
prior again double-corrects and drives the decoder into N1 and N3. Using the
empirical class frequency cost 4 accuracy points. Using a badly scaled estimate
cost 46.

The fix was to use the log posteriors directly as emissions and scale the
transition term with a weight α. Not a proper HMM, but correct for a classifier
that is already contextual — and it works.

I also had to choose α honestly. Tuning it on the test fold is leakage, and I
had no posteriors on training data from a model that hadn't seen it — until I
realised I did: for any fold, the *training* recordings' out-of-fold posteriors
come from models that never saw them either. Selecting α per fold on those
touches nothing in the test fold. The procedure picks α = 0.15 almost
everywhere and recovers essentially the test-tuned optimum.

### 5.5 What these taught me

**A number that surprises me upward is a bug until proven otherwise.** The 0.8115
CNN result was the single most dangerous moment in the project, because it was
better than my baseline, arrived at a plausible point in development, and had a
completely coherent story attached to it. What exposed it was not the accuracy —
it was the fold line saying `1 test recordings` and REM having a support of 2.

The general lesson is to make the pipeline print things that *can't* be right
when it's wrong: fold sizes, per-class supports, coverage assertions. My
`compare.py` now asserts that the CNN's stored true labels match the canonical
ones before it compares anything, so a misalignment crashes rather than
producing a plausible table.

---

## 6. Results

Everything below is out-of-fold, five-fold cross-validation grouped by patient,
over all 93,937 epochs.

### 6.1 The two models

| model | accuracy | macro F1 | Cohen's κ |
|---|---|---|---|
| Gradient boosting (400 trees) | **0.7712** | **0.7070** | **0.6768** |
| CNN + BiGRU | 0.7219 | 0.6602 | 0.6144 |

Per-stage F1, CNN minus GBDT: Wake −0.039, N1 −0.023, N2 −0.034, N3 −0.046,
REM −0.093. **The tree model wins on every stage.**

I tested significance on the recording, not the epoch, because epochs within a
night are heavily correlated: the tree model is better on **74 of 99
recordings**, mean Δaccuracy +0.052, Wilcoxon signed-rank **p = 2.4 × 10⁻⁷**.
The epoch-level McNemar test gives p ≈ 3 × 10⁻²⁵⁰, but that number is
anti-conservative by construction and I don't quote it.

The paradigm argument held. A specific prediction I made did not: I expected the
BiGRU to beat the trees on stage *persistence*, and it didn't. N3→N2 leakage did
improve (28.1% → 20.4%) but N3 precision fell from 0.751 to 0.651, and REM got
worse overall. The CNN isn't modelling persistence better — it is simply less
biased toward the majority class, trading precision for recall on the rare
stages.

### 6.2 The ensemble

The two models agree on only **76.5%** of epochs, and at least one of them is
correct on **84.64%** — against 77.12% for the better single model. That 7.5
points of headroom is what made stacking worth trying.

| variant | accuracy | macro F1 | κ |
|---|---|---|---|
| gradient boosting | 0.7712 | 0.7070 | 0.6768 |
| gradient boosting + Viterbi | 0.7741 | 0.7058 | 0.6795 |
| CNN + BiGRU | 0.7219 | 0.6602 | 0.6144 |
| CNN + Viterbi | 0.7233 | 0.6593 | 0.6153 |
| ensemble (equal weight) | 0.7794 | 0.7137 | 0.6885 |
| **ensemble + Viterbi** | **0.7805** | **0.7125** | **0.6893** |

**Final model: the equal-weight ensemble plus Viterbi smoothing, at κ 0.6893.**

| stage | precision | recall | F1 | support |
|---|---|---|---|---|
| Wake | 0.832 | 0.860 | 0.846 | 25,921 |
| N1 | 0.433 | 0.325 | 0.372 | 9,360 |
| N2 | 0.800 | 0.861 | 0.830 | 39,198 |
| N3 | 0.735 | 0.735 | 0.735 | 8,220 |
| REM | 0.842 | 0.727 | 0.780 | 11,238 |

Confusion matrix, row-normalised (row = truth):

| | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| **Wake** | 86.0 | 7.9 | 5.1 | 0.1 | 0.9 |
| **N1** | 30.9 | 32.5 | 31.0 | 0.8 | 4.7 |
| **N2** | 2.8 | 3.6 | 86.1 | 5.2 | 2.2 |
| **N3** | 0.4 | 0.0 | 26.1 | 73.5 | 0.0 |
| **REM** | 4.0 | 4.7 | 18.4 | 0.2 | 72.7 |

Combining converted about 0.9 of the 7.5 available oracle points into real
accuracy. That ratio is normal — an oracle bound assumes a perfect arbiter
deciding which model to trust on each epoch.

**The CNN loses outright and is still worth keeping.** Removing it costs 0.012 κ.
Its value is not its accuracy; it is that its errors are different.

### 6.3 What the model is actually using

The top features are, reassuringly, the physiologically expected ones:

```
epoch_position                     0.952%
hours_from_start                   0.931%
EMG_log_iqr                        0.814%
EOG_corr_z_sm15                    0.676%
EMG_skew_z_sm15                    0.603%
EMG_kurtosis_z_sm15                0.564%
C4_kurtosis_z_sm15                 0.552%
```

Position in the night, chin EMG amplitude, and the E1–E2 eye-movement
correlation lead, followed by central-EEG spectral shape. Importance is spread
thinly — the top feature accounts for under 1% — because 426 correlated features
share credit. Two things stand out: the `_z_sm15` suffix dominates, meaning the
model overwhelmingly prefers the smoothed, per-recording-normalised versions of
each feature over the raw ones; and the two purely temporal features rank first
and second, which says the sleep architecture prior is doing real work.

---

## 7. Negative results worth recording

### 7.1 Early stopping is wrong for both models

Conventional practice says to early-stop on a validation split. I tested it and
it hurt both models, for the same reason.

**Gradient boosting.** Early stopping on log-loss, with a patient-disjoint
validation split and a refit on the full training fold once the tree count was
known, reliably stopped near 100 trees and cost about 0.008 κ against a fixed
400. Switching the stopping metric from log-loss to error rate barely changed
the tree count — which is the tell that the metric wasn't the problem.

**The CNN.** Against an earlier fixed 15-epoch run, on identical folds:

| fold | fixed 15 | early stopped | Δ | checkpoint chosen | stopped at |
|---|---|---|---|---|---|
| 1 | 0.7126 | 0.6868 | +0.0258 | epoch 5 | 15 |
| 2 | 0.7444 | 0.7293 | +0.0151 | epoch 6 | 16 |
| 3 | 0.7426 | 0.7307 | +0.0119 | epoch 4 | 14 |
| 4 | 0.7518 | 0.7414 | +0.0104 | epoch 18 | 28 |
| 5 | 0.7231 | 0.7216 | +0.0015 | epoch 16 | 26 |

Fixed wins **5 of 5 folds**, mean Δ +0.0129, overall κ 0.6351 → 0.6144.

Two causes, both traceable to cohort size. **The validation signal is noise**:
with roughly 12 held-out patients, validation κ swings between 0.467 and 0.600
*within a single fold* (sd 0.034), which is larger than the effect being
selected. Taking an argmax over that curve picks an early lucky spike — three of
five folds restored a checkpoint from before epoch 7, when training loss was
still around 0.63 and the model plainly underfit. And **it costs 15% of the
training recordings**, held out to generate the noisy signal in the first place.

**The lesson: at 86 patients there is no spare data for an inner validation
split.** Every budget decision — tree count, epoch count — belongs to the outer
cross-validation, which uses all patients, and should then be applied as a fixed
hyper-parameter. Both training scripts default to no early stopping.

This is the finding I'd most expect to generalise beyond this dataset.

### 7.2 The tree model has converged

Tripling from 400 to 1200 trees moves κ from 0.6768 to 0.6794 — **+0.0026 for 3×
the compute.** There is nothing left in tree count.

### 7.3 Viterbi helps far less than I expected

I predicted sequence decoding would be a major win, given self-transition
probabilities above 0.9 for four of five stages. It contributes +0.003 accuracy
on top of the ensemble.

The reason is informative: the ±2 and ±7 epoch rolling-mean features already
encode most of the available temporal persistence, so an explicit transition
model is largely redundant with them. It buys a little on REM and costs a little
on N1, which is transient by definition and gets smoothed away. In effect I had
already solved the problem in the feature engineering and didn't realise it.

---

## 8. Honest assessment of the result

κ 0.6893 is a fair result, not a strong one, and the comparison that matters is
against the right reference class. Published κ of 0.75–0.80 for automated sleep
staging is overwhelmingly on **healthy sleepers**. This is a stroke cohort:
focal slowing, asymmetry, and abnormal architecture are the norm, and both
automated and human scoring degrade on it.

Two error modes dominate, and they behave differently.

**N1 at F1 0.372 is close to a genuine ceiling.** 31% of true N1 is called Wake
and another 31% is called N2 — but N1 is the stage human scorers agree on least,
with published inter-rater κ frequently below 0.5. A model cannot reliably
exceed the consistency of the labels it was trained on.

**N3 at 0.735 and REM at 0.780 are not at a ceiling.** 26% of N3 leaks to N2 and
18% of REM leaks to N2. These are the recoverable losses.

I also want to be clear about what the headline number depends on. Grouping
folds by patient rather than by recording costs several points of apparent
accuracy — points that were never real. Reporting a recording-level split here
would have produced a better-looking and less honest result, and given that
twelve patients appear twice and one night appears under two IDs, the
difference is not hypothetical.

---

## 9. What I would do next

In order of expected payoff:

1. **Add the frontal channels F3/F4.** I excluded them for uniformity — they
   appear in only 87 of 100 recordings. But **AASM scores N3 primarily on
   frontal slow-wave activity**, where slow waves have their largest amplitude,
   and I am currently staging N3 from central and occipital electrodes only.
   N3 is my second-weakest stage with 26% leaking to N2. Adding F3/F4 with a
   missing-channel indicator for the 13 recordings that lack them is a targeted
   fix for a specific, diagnosed weakness. This is the clearest unexplored lever
   and I regret not testing it.
2. **Weight the ensemble.** Equal weighting is a placeholder and the tree model
   is clearly stronger; something like 0.6/0.4 is likely better. Choosing the
   weight honestly requires nested cross-validation, or accepting it as a fixed
   prior rather than a tuned parameter.
3. **Normalise the raw CNN input per subject**, matching the robust z-scoring
   the tree features get for free. This asymmetry is my best single hypothesis
   for why the CNN underperforms, and a better CNN means a better ensemble.
4. **Sweep the CNN epoch count against the outer CV** — 15 was chosen
   arbitrarily and is the last unjustified number in the pipeline.
5. **Fine-tune a pretrained sleep-staging model.** Models trained on thousands
   of nights across many cohorts are far more data-efficient to adapt than
   training a CNN from scratch on 86 patients, and represent the realistic path
   past κ 0.75.
6. **Prune features.** 426 is mostly redundant — each base feature appears raw,
   z-scored, and smoothed at two widths. Pruning by importance would cut
   training time with probably no accuracy cost.

The binding constraint on all of this is **86 patients**. Nothing on the list
changes that.

---

## 10. What I take away

**The dataset was the project.** Four categories of file corruption, two channel
naming conventions, ten mislabelled subject IDs, one duplicated recording,
twelve duplicated patients, an unaligned hypnogram, and a workbook whose first
sheet is a light sensor. Every one of these would have produced a model that
trained and reported a plausible number, and several would have made that number
look *better*. The modelling itself was comparatively routine.

**Cheap integrity checks give false confidence.** Two of my corrupted files had
the correct byte count and a self-consistent EDF header. Block-level hashing was
the only thing that caught them.

**Design the pipeline to fail loudly.** The bugs that cost me most time were the
ones that returned plausible numbers instead of errors. My defences now are
assertions on alignment, printed fold sizes and per-class supports, and a
dataset fingerprint on every cached artefact.

**Textbook corrections carry assumptions.** The posterior-over-prior conversion
is correct — for a classifier trained on the natural class distribution. Mine
weren't, and applying it anyway cost 46 accuracy points at worst.

**Small datasets invalidate standard practice.** Early stopping is close to
universal advice, and at 86 patients it made both of my models worse, measurably
and consistently. The general form of this is that any technique requiring a
held-out validation split is expensive when data is scarce, and the cost is paid
twice: once in the data withheld, once in the noise of the resulting signal.

**A model can lose and still be worth keeping.** The CNN is worse than the tree
model on every single stage and is part of my best result, because its errors
are different. I nearly discarded it.

---

## Appendix — reproducing this

```powershell
python dataset_tools\verify_downloads.py Dataset       # integrity first
python submission\extract_features.py --subjects all --raw
python submission\train_gbdt.py --folds 5 --save-proba
python submission\train_cnn.py  --folds 5 --epochs 15 --save-proba
python submission\compare.py                            # head-to-head
python submission\stack.py                              # final model
```

`stack.py` writes `results/stacked.npz` with the winning predictions.
Full design rationale is in `MODEL.md`; the data audit is in
`DATASET_AUDIT.md`; raw console output of the runs behind every number in
section 6 is in `console_runs.md`.
