# Sleep stage classification — pipeline and design rationale

Five-class AASM staging (Wake / N1 / N2 / N3 / REM) from 30-second epochs.

```
extract_features.py -> cache/*.npz     -> train_gbdt.py  (ensemble member 1)
                       cache/raw/*.npy  -> train_cnn.py   (ensemble member 2)
                                        -> stack.py       (BEST MODEL)
                                        -> compare.py     (analysis)
sleepstaging/          sleep_io · features · subjects · evaluate
```

---

## Step 1 — Decide what a "sample" is

One labelled example = one 30 s epoch. That is forced by the data: the scoring
workbooks give one stage per 30 s, which is the AASM standard. So the task is
sequence labelling over ~950 epochs per night, not whole-night classification.

## Step 2 — Find a montage that exists in every recording

This dataset does not have one consistent channel set. 64 recordings label
electrodes against `M1`/`M2` and 36 against `A1`/`A2`. These are the same
mastoid references under two naming conventions, but a literal string match
finds **no EEG channel present in all 100 files** and silently drops 36% of the
data. After aliasing, seven channels are universal:

| Role | Channels |
|---|---|
| EEG | C3, C4, O1, O2 |
| EOG | E1, E2 |
| EMG | chin EMG |

F3/F4 exist in only 87 recordings, so they are excluded rather than forcing an
imputation scheme. Everything is resampled to 100 Hz (sources are 128/256 Hz),
EEG/EOG band-passed 0.3–35 Hz, EMG 10–45 Hz — the chin EMG's discriminative
power is in high-frequency muscle tone, and its low-frequency content is
movement artefact.

## Step 3 — Align hypnograms to signal, defensively

The hypnogram is *not* aligned to the recording start: it typically begins a few
seconds earlier, and both clocks are wall-clock times that cross midnight. Each
epoch is therefore placed by its own timestamp, with a day-wrap correction.

The workbook layout is also not fixed. Sheet 1 is usually the hypnogram, but in
some recordings (SN80) sheet 1 is the **light sensor in lux**, with the stage in
a separate column — which is why raw label counts contain values like `54` and
`136`. The parser scans every sheet and every column and keeps whichever column
actually contains stage names. Before this fix one recording produced zero
usable epochs and others were partly mislabelled; after it, all 99 align and the
label vocabulary is clean.

**94,947 usable epochs**: N2 41.5%, Wake 27.9%, REM 11.8%, N1 10.0%, N3 8.8%.

## Step 4 — Split by patient, not by recording

99 recordings come from only **86 patients**. Twelve patients appear under more
than one ID with all 63 metadata columns identical, usually a second night;
SN15 and SN28 are byte-identical signal *and* hypnogram, so SN15 is dropped
outright. Splitting by file would put the same patient in train and test.
`GroupKFold` on patient ID is used everywhere. Expect this to cost several
points of accuracy versus a naive split — those points were never real.

---

## Step 5 — Choosing the paradigm

**Chosen baseline: gradient-boosted trees on engineered features.**

The reasoning, rather than the conclusion:

- **The physics is known.** AASM staging rules are literally defined in terms of
  band power and a handful of waveform properties: delta power for N3, sigma
  spindles for N2, chin atonia plus conjugate eye movement for REM, alpha
  dropout for the wake/N1 boundary. A feature that encodes "relative sigma
  power" starts at the answer. A CNN has to spend capacity rediscovering the
  Fourier transform from 3,000 raw samples.
- **Sample size favours it.** 86 patients is small. Deep models for sleep
  staging (DeepSleepNet, U-Time, USleep) are trained on hundreds to thousands of
  nights. Below roughly a hundred subjects, engineered features usually win, and
  they degrade more gracefully.
- **Compute.** The tree model trains on CPU in minutes. The CNN wants a GPU.
- **This cohort is atypical.** These are stroke patients: focal slowing and
  asymmetry are expected. Feature importances let you *see* which channels and
  bands the model leans on, which matters when the EEG is pathological. A CNN
  gives you an accuracy number and little else.

**Not chosen, and why:**

- *Plain logistic regression / SVM* — the decision boundaries here are strongly
  non-linear and interacting (REM = low EMG **and** eye movement **and** mixed
  frequency). Boosted trees model that natively; a linear model needs the
  interactions hand-built.
- *Random forest* — fine, but boosting consistently edges it out on tabular
  biosignal features at equal cost.
- *HMM / CRF on top* — a principled way to encode stage-transition structure.
  Deliberately skipped for v1 because smoothed context features capture most of
  the same benefit for a fraction of the complexity. This is the most promising
  next addition.
- *Pretrained foundation models for EEG* — real option now, but heavy and hard
  to justify before a baseline exists.

**Also built, for the comparison you asked for: a CNN + BiGRU.**

- Two parallel conv branches over the raw epoch (DeepSleepNet's design): a
  0.5 s kernel for transients like spindles and K-complexes, a 4 s kernel for
  slow rhythms. A single kernel size cannot resolve both.
- A bidirectional GRU over 20 consecutive epoch embeddings. Stages persist for
  minutes and AASM rules are explicitly contextual. Bidirectional is legitimate
  because whole nights are scored offline, not in real time.

## Step 6 — Features (106 base → 426 with context)

Per EEG channel: relative delta/theta/alpha/sigma/beta power, log total power,
five log band ratios, a slow/fast ratio, spectral entropy, 95% spectral edge,
Hjorth mobility and complexity, and robust time-domain moments. Per EOG:
band-limited slow/fast power plus the **E1–E2 correlation** — REM bursts move
the eyes in opposite directions, so this goes sharply negative. Per EMG: log
RMS and high/low band ratio, the atonia detector.

Then, per recording:

1. **Robust z-scoring** (median/IQR) — removes between-subject and
   between-amplifier amplitude differences that would otherwise dominate splits.
2. **Centred rolling means over ±2 and ±7 epochs** — a single 30 s epoch is
   genuinely ambiguous, and human scorers look at neighbours too.
3. **Position in the night** — N3 concentrates early, REM late.

## Step 7 — Results (all 99 recordings, 86 patients, 5-fold grouped CV)

| | accuracy | macro F1 | Cohen's κ |
|---|---|---|---|
| Gradient boosting | **0.7724** | **0.7064** | **0.6777** |

| stage | precision | recall | F1 | support |
|---|---|---|---|---|
| Wake | 0.820 | 0.854 | 0.837 | 25,921 |
| N1 | 0.413 | 0.326 | 0.364 | 9,360 |
| N2 | 0.793 | 0.851 | 0.821 | 39,198 |
| N3 | 0.751 | 0.711 | 0.730 | 8,220 |
| REM | 0.842 | 0.726 | 0.780 | 11,238 |

Confusion (row = truth, %):

|  | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| **Wake** | 85.4 | 7.9 | 5.9 | 0.1 | 0.7 |
| **N1** | 30.9 | 32.6 | 31.3 | 0.5 | 4.7 |
| **N2** | 3.7 | 4.2 | 85.1 | 4.7 | 2.2 |
| **N3** | 0.6 | 0.0 | 28.1 | 71.1 | 0.2 |
| **REM** | 4.4 | 5.4 | 17.5 | 0.2 | 72.6 |

Scaling from the 20-patient pilot to all 99 recordings moved κ from 0.631 to
0.678 and macro F1 from 0.645 to 0.706, with N1 F1 doubling (0.18 → 0.36) and
REM recall rising from 0.63 to 0.73. Fold-to-fold accuracy spans 0.764–0.783,
so the estimate is stable.

Top features are the physiologically expected ones — position in the night,
chin EMG amplitude, E1–E2 correlation, then central-EEG spectral shape.

**Reading this honestly.** κ ≈ 0.68 is respectable for a stroke cohort, whose
EEG is abnormal by definition; published κ of 0.75–0.80 is generally on healthy
sleepers. Two error modes dominate and both are structural:

- **N1** remains weak (F1 0.36), with 31% called Wake and 31% called N2. N1 is
  the stage human scorers agree on least — inter-rater κ for N1 is often below
  0.5 — so this is close to the practical ceiling without transition modelling.
- **N3 → N2 leakage (28%)** and **REM → N2 leakage (17.5%)** are the biggest
  recoverable losses. Both are stage-persistence failures: the model flips
  mid-run because it scores each epoch semi-independently.

## Step 8 — Head-to-head (identical folds, `compare.py`)

| | accuracy | macro F1 | Cohen's κ |
|---|---|---|---|
| **Gradient boosting** (400 trees) | **0.7712** | **0.7070** | **0.6768** |
| CNN + BiGRU | 0.7219 | 0.6602 | 0.6144 |

Per-stage F1 (CNN − GBDT): Wake −0.039, N1 −0.023, N2 −0.034, N3 −0.046,
REM −0.093. The trees win on every stage.

The models agree on 76.5% of epochs. On the independent unit — the recording,
not the epoch — the tree model is better on **74 of 99 recordings**, mean
Δaccuracy +0.052, **Wilcoxon p = 2.4e-7**. (Epoch-level McNemar reports
p ≈ 3e-250, but epochs within a night are heavily correlated, so that figure is
anti-conservative and should not be quoted.)

The oracle bound — at least one model correct — is **0.8464** against 0.7712 for
the better single model: 7.5 points of headroom, which motivates Step 9.

## Step 9 — Viterbi smoothing and stacking (`stack.py`)

| variant | accuracy | macro F1 | κ |
|---|---|---|---|
| gradient boosting | 0.7712 | 0.7070 | 0.6768 |
| gradient boosting + Viterbi | 0.7741 | 0.7058 | 0.6795 |
| CNN + BiGRU | 0.7219 | 0.6602 | 0.6144 |
| CNN + Viterbi | 0.7233 | 0.6593 | 0.6153 |
| ensemble (equal weight) | 0.7794 | 0.7137 | 0.6885 |
| **ensemble + Viterbi** | **0.7805** | **0.7125** | **0.6893** |

**The ensemble is the best model**, at κ 0.6893 against 0.6768 for the tree
model alone. That converts roughly 0.9 of the 7.5 available oracle points into
real accuracy, which is typical — oracle bounds assume a perfect arbiter.

The CNN is worth keeping despite losing outright: it is a useful *ensemble
member* because its errors differ from the tree model's on nearly a quarter of
epochs.

Viterbi contributes a further +0.003 accuracy on top of the ensemble. Two
implementation details mattered more than the idea itself:

1. **Do not divide the posterior by the class prior.** The textbook
   posterior→likelihood conversion assumes a classifier trained on the natural
   class distribution. Both models use class weights, so their implied prior is
   already tilted and dividing again double-corrects. Measured cost of getting
   this wrong: −4 accuracy points with an empirical prior, −46 with a
   badly-scaled one.
2. **Tune the transition weight without touching the test fold.** α is selected
   per fold on the *training* recordings' out-of-fold posteriors. It picks
   α = 0.15 almost everywhere.

The gain is small because the ±2 and ±7 epoch rolling-mean features already
encode most of the available temporal persistence.

## Step 10 — Training budgets: early stopping is wrong for BOTH models

This was tested rather than assumed, and the answer was the same in both cases
and opposite to the usual advice.

**Gradient boosting has converged at 400 trees.** Tripling to 1200 moves κ from
0.6768 to 0.6794 — +0.0026 for 3× the compute. Early stopping on an inner
validation split stops near 100 trees and *costs* about 0.008 κ.

**CNN: early stopping made it measurably worse.** Against the earlier fixed
15-epoch run, on identical folds:

| fold | fixed 15 | early stopped | Δ | checkpoint chosen | stopped at |
|---|---|---|---|---|---|
| 1 | 0.7126 | 0.6868 | +0.0258 | epoch 5 | 15 |
| 2 | 0.7444 | 0.7293 | +0.0151 | epoch 6 | 16 |
| 3 | 0.7426 | 0.7307 | +0.0119 | epoch 4 | 14 |
| 4 | 0.7518 | 0.7414 | +0.0104 | epoch 18 | 28 |
| 5 | 0.7231 | 0.7216 | +0.0015 | epoch 16 | 26 |

Fixed wins **5 of 5 folds**, mean Δ +0.0129, and overall κ 0.6351 → 0.6144.

Two causes, both traceable to cohort size:

- **The validation signal is noise.** With ~12 held-out patients, val κ swings
  0.467–0.600 within a single fold (sd 0.034) while the effect being selected is
  smaller than that. Taking an argmax over that curve picks an early lucky
  spike: three of five folds restored a checkpoint from before epoch 7, when
  training loss was still ~0.63 and clearly underfit.
- **It also costs 15% of the training recordings**, which are held out to
  produce the noisy signal in the first place.

**The general lesson for this dataset: at 86 patients there is no spare data for
an inner validation split.** Every budget decision — tree count, epoch count —
should be made against the outer cross-validation, which uses all patients, and
then applied as a fixed hyper-parameter. Both scripts default to no early
stopping for this reason (`--patience 0`).

## Step 11 — What to try next, in order of expected payoff

1. **Sweep the CNN epoch count against the outer CV** (15 vs 25 vs 40, no inner
   split, full training data). 15 was picked arbitrarily and is the last
   unjustified number in the pipeline.
2. **Weight the ensemble.** Equal weighting is a placeholder; the tree model is
   clearly stronger, so a 0.6/0.4 split is likely better. Choosing the weight
   honestly needs nested CV, or accept it as a fixed prior rather than tuned.
3. **Attack N1 directly** — still the weak stage at F1 0.372, with 31% called
   Wake and 31% called N2. `--class-weight balanced` trades overall accuracy for
   N1 recall; whether that is the right trade depends on the clinical question.
4. **Per-subject normalisation of the raw CNN input**, matching the robust
   z-scoring the tree features get for free. The most likely single reason the
   CNN underperforms.
5. **Prune features.** 426 is mostly redundant (each base feature appears raw,
   z-scored, and smoothed at two widths); pruning by importance would cut
   training time with probably no accuracy loss.

## Running it

```bash
python submission\extract_features.py --subjects all              # 99 recordings, ~5 min
python submission\extract_features.py --subjects all --raw        # + raw cache for the CNN (~4 GB)

python submission\train_gbdt.py --cache cache --folds 5 --save-proba              # 400 trees, no early stopping
python submission\train_gbdt.py --cache cache --folds 5 --save-proba \
                     --n-estimators 1200 --out results_1200            # convergence check
python submission\train_cnn.py  --cache cache --folds 5 --save-proba              # up to 60 epochs, early stopped
python submission\compare.py                                      # head-to-head + significance
python submission\stack.py                                        # Viterbi + ensemble
```

`train_gbdt.py` and `train_cnn.py` cache each fold, stamped with a fingerprint
of the recording list, epoch count, fold count, seed and hyper-parameters.
Change any of those and the affected folds recompute automatically — a fold
cached from a smaller run stores test indices into a shorter array, and
replaying it against more data addresses the wrong rows.

The CNN needs the raw cache and PyTorch; on Python 3.14 that means a 3.12
virtualenv.
