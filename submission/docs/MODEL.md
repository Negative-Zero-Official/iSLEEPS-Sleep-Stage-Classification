# Sleep stage classification — pipeline and design rationale

Five-class AASM staging (Wake / N1 / N2 / N3 / REM) from 30-second epochs.

**Final system: an equal-weight ensemble of gradient-boosted trees, a CNN+BiGRU,
and a BiLSTM over features. Accuracy 0.7794, macro F1 0.7210, Cohen's κ 0.690,
five-fold CV grouped by patient. This is ahead of the dataset paper's best
published baseline (LSTM, 74.70 / 67.68 / 0.64) on every overall metric and
every individual stage — see Step 12.**

```
extract_features.py -> cache/*.npz     -> train_gbdt.py  (ensemble member 1)
                       cache/raw/*.npy  -> train_cnn.py   (ensemble member 2)
                       cache/*.npz      -> train_lstm.py  (ensemble member 3)
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

## Step 11 — A BiLSTM over the feature sequence

The dataset paper's strongest baseline was an LSTM at 74.70% accuracy, run on
signals. I tried the same architectural idea on the 426 engineered features
instead: a projection to 256 units, a 2-layer BiLSTM, and a per-epoch head, over
windows of 100 epochs (~50 minutes of context). Feature standardisation uses
training recordings only.

**Result: accuracy 0.7373, macro F1 0.6863, κ 0.6371.**

- It **beats the CNN** decisively (κ 0.6371 vs 0.6144) on a fraction of the
  compute — 3 seconds per fold against several minutes — which supports the
  central thesis of this project: at 86 patients, handing the model a good
  representation beats making it learn one.
- It **loses to the trees** (κ 0.6768). Sequence modelling alone does not close
  the gap.
- At 73.73% it sits just **below** the paper's 74.70% LSTM figure. My folds are
  grouped by patient, which is stricter than a recording-level split, and the
  paper's protocol is not stated in what I have, so this is not a like-for-like
  comparison and I do not claim to have reproduced or beaten it.

Adding it as a third ensemble member:

| variant | accuracy | macro F1 | κ |
|---|---|---|---|
| ensemble (trees + CNN) + Viterbi | **0.7805** | 0.7125 | 0.6893 |
| ensemble (trees + CNN + BiLSTM) | 0.7794 | **0.7210** | **0.6902** |

Paired over the 99 recordings the two are indistinguishable on accuracy — the
three-way wins on 46/99, mean Δ −0.0009, Wilcoxon p = 0.53. The κ difference
(+0.0009) is noise. **The real gain is per-stage balance**: N1 F1 0.372 → 0.399,
N3 0.735 → 0.744, REM 0.780 → 0.789, against Wake 0.846 → 0.841.

Two further observations:

- **Viterbi becomes a no-op** once the LSTM is in the ensemble (0.7794 → 0.7794;
  the per-fold α selection picks 0). The recurrent model already supplies the
  temporal smoothing the transition matrix was providing.
- **The oracle bound rises from 0.8464 to 0.8756** with three members. Equal-
  weight averaging captures very little of that, which says a better combiner —
  a learned meta-classifier over the three posteriors — is where the remaining
  headroom is.

A caveat on selection: `stack.py` reports the best of twelve variants scored on
the same out-of-fold predictions. With differences this small, the top few are
within noise of each other and the reported maximum is mildly optimistic.

## Step 12 — Comparison with the dataset paper's baselines

Maiti et al., *Scientific Data* 13:421 (2026), Table 3, report five-class staging
on iSLEEPS using single-channel EEG or EOG, 10-fold cross-validation:

| model (paper) | modality | ACC | MF1 | κ | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|---|
| CNN | EEG | 61.65 | 54.44 | 0.48 | 68.15 | 17.43 | 68.82 | 67.65 | 50.12 |
| **LSTM** | **EEG** | **74.70** | **67.68** | **0.64** | 79.87 | 32.99 | 80.91 | 74.25 | 70.04 |
| LSTM | EOG | 62.33 | 52.61 | 0.46 | 64.55 | 15.81 | 70.95 | 64.49 | 47.25 |
| Transformer | EEG | 67.44 | 59.35 | 0.54 | 77.53 | 25.91 | 76.07 | 69.18 | 47.03 |
| Transformer | EOG | 66.29 | 58.88 | 0.52 | 72.29 | 23.43 | 77.06 | 68.25 | 52.83 |

Against this work:

| model | ACC | MF1 | κ | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| paper's best (LSTM, EEG) | 74.70 | 67.68 | 0.64 | 79.9 | 33.0 | 80.9 | 74.3 | 70.0 |
| this work, BiLSTM on features | 73.73 | 68.63 | 0.637 | 80.2 | 37.2 | 79.8 | 71.6 | 74.3 |
| this work, gradient boosting | 77.12 | 70.70 | 0.677 | 83.5 | 37.0 | 82.0 | 73.1 | 77.9 |
| **this work, 3-model ensemble** | **77.94** | **72.10** | **0.690** | **84.1** | **39.9** | **83.1** | **74.4** | **78.9** |

The ensemble is ahead on every overall metric and on every individual stage.
The largest margins are REM (+8.9 F1) and N1 (+6.9), the two stages the paper's
models struggled with most.

**On the split protocol.** The paper states it split "on a patient-wise basis to
prevent data leakage, ensuring that all sleep epochs from a given patient were
assigned exclusively to one set". That is the right principle, and it is what
this work does too — so the comparison above is broadly like-for-like.

However, the paper treats SN1–SN100 as 100 distinct patients. This project's
audit found that they are not: twelve groups of recording IDs share all 63
metadata columns, and SN15/SN28 are byte-identical in both signal and hypnogram.
99 recordings come from **86 patients**. A split keyed on recording ID therefore
still places the same patient on both sides. Measured directly on this cache,
grouping by recording puts a patient across the train/test boundary in **5 of 5
folds**; grouping by patient, in 0 of 5.

The paper's baselines are therefore evaluated under a slightly more permissive
protocol than the numbers above, which means the gap is, if anything,
understated. `train_gbdt.py --group-by recording` exists to quantify that
optimism; it is a diagnostic, not a reporting mode.

## Step 13 — What to try next, in order of expected payoff

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
