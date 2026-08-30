# Sleep stage classification on the iSLEEPS stroke cohort

Five-class AASM staging (Wake / N1 / N2 / N3 / REM) from 30-second epochs of
polysomnography, across 99 recordings from 86 patients.

## Result

**Final model: an equal-weight ensemble of three models — gradient-boosted trees
over engineered features, a CNN+BiGRU over raw signal, and a BiLSTM over the
feature sequence.**

| model | accuracy | macro F1 | Cohen's κ |
|---|---|---|---|
| gradient boosting | 0.7712 | 0.7070 | 0.6768 |
| gradient boosting + Viterbi | 0.7741 | 0.7058 | 0.6795 |
| CNN + BiGRU | 0.7219 | 0.6602 | 0.6144 |
| BiLSTM over features | 0.7373 | 0.6863 | 0.6371 |
| ensemble (trees + CNN) + Viterbi | **0.7805** | 0.7125 | 0.6893 |
| **ensemble (trees + CNN + BiLSTM)** | 0.7794 | **0.7210** | **0.6902** |

The last two are statistically indistinguishable on accuracy (paired over 99
recordings: 46/99, mean Δ −0.0009, Wilcoxon p = 0.53). The three-way ensemble is
preferred because it is clearly better on the minority stages — N1 +0.028,
N3 +0.009, REM +0.009 F1 — which is what macro F1 reflects. Pick the two-way if
raw accuracy is the only thing that matters.

Five-fold cross-validation, **grouped by patient**. Every number above is
out-of-fold. If you need a single model without the PyTorch dependency, use
gradient boosting + Viterbi at κ 0.6795.

Per-stage F1 for the three-way ensemble: Wake 0.841, N1 0.399, N2 0.831,
N3 0.744, REM 0.789.

**Against the dataset paper.** Maiti et al. (Sci Data 13:421, 2026) report a best
baseline of LSTM on single-channel EEG at ACC 74.70, MF1 67.68, κ 0.64. This
ensemble reaches **77.94 / 72.10 / 0.690**, ahead on every overall metric and on
every individual stage — largest margins on REM (+8.9 F1) and N1 (+6.9). Both
split patient-wise, but the paper treats SN1–SN100 as 100 distinct patients;
they are 86 (see `docs/DATASET_AUDIT.md`), so its protocol is marginally the
more permissive of the two.

## Pipeline

Run in order. Paths default relative to the repository root, so the working
directory does not matter.

```powershell
python submission\extract_features.py --subjects all --raw   # ~10 min, writes cache/
python submission\train_gbdt.py  --folds 5 --save-proba      # ~2 min
python submission\train_cnn.py   --folds 5 --epochs 15 --save-proba   # ~10 min, GPU
python submission\train_lstm.py  --folds 5 --save-proba              # ~10 min, GPU
python submission\compare.py                                 # head-to-head + significance
python submission\stack.py                                   # -> the best model
```

`stack.py` writes `results/stacked.npz` containing the winning predictions.
Both training scripts cache per fold and resume if interrupted.

## Files

| file | role |
|---|---|
| `extract_features.py` | EDF + hypnogram → 426 per-epoch features. The only slow step. |
| `train_gbdt.py` | Gradient-boosted trees. **Ensemble member 1.** |
| `train_cnn.py` | Two-branch 1D CNN + BiGRU on raw signal. **Ensemble member 2.** |
| `train_lstm.py` | BiLSTM over the feature sequence. **Ensemble member 3.** |
| `stack.py` | Viterbi smoothing + ensembling. **Produces the final model.** |
| `compare.py` | Head-to-head analysis with paired significance testing. |
| `sleepstaging/sleep_io.py` | Dependency-free EDF reader, hypnogram parser, channel aliasing. |
| `sleepstaging/features.py` | Spectral, Hjorth and time-domain features + temporal context. |
| `sleepstaging/subjects.py` | Recording → patient mapping. Prevents train/test leakage. |
| `sleepstaging/evaluate.py` | Shared folds, metrics, sequence windowing. |
| `tests/test_shapes.py` | Validates the CNN's tensor arithmetic (runs without PyTorch). |
| `tests/test_lstm_plumbing.py` | Validates fold, index and scaling bookkeeping against the real cache. |
| `docs/DEVELOPMENT_REPORT.md` | Development report: method, results, negative results, engineering. |
| `docs/MODEL.md` | Full design rationale, every result, every rejected alternative. |
| `docs/DATASET_AUDIT.md` | Data-integrity audit and the traps found in the dataset. |
| `docs/console_runs.md` | Raw console output of the runs the results table came from. |

Dependencies: numpy, scipy, scikit-learn, lightgbm. PyTorch only for
`train_cnn.py` — and only for the ensemble, not the single-model path.

## Three things that decide whether the numbers are real

1. **Split by patient, not by recording.** The 99 recordings come from only 86
   patients; twelve patients appear twice, and SN15/SN28 are byte-identical
   signal *and* hypnogram. Splitting by file leaks and inflates every metric.
2. **`subject_description.xlsx` has no SN11–SN20 rows.** It has AN1–AN10, which
   live on disk as SN11–SN20. Joining on filename mislabels ten recordings.
3. **Two channel-naming conventions.** 64 recordings use `M1`/`M2` references,
   36 use `A1`/`A2` for the same derivations. A literal string match finds no
   EEG channel common to all 100 files and silently drops 36% of the data.

## What was tried and rejected

Nothing in this folder is dead code — the CNN loses head-to-head but earns its
place as an ensemble member, because its errors differ from the tree model's on
nearly a quarter of epochs. What failed were *configurations*, all documented
with measurements in `docs/MODEL.md`:

- **Early stopping, on both models.** At 86 patients there is no spare data for
  an inner validation split. The GBDT loses ~0.008 κ; the CNN loses on 5 of 5
  folds (κ 0.6351 → 0.6144), because validation κ on ~12 held-out patients
  swings 0.47–0.60 and an argmax over that noise restores premature
  checkpoints. Both scripts default to `--patience 0`.
- **More trees.** 400 → 1200 moves κ by +0.0026 for 3× the compute. Converged.
- **Dividing posteriors by the class prior before Viterbi.** Textbook, and wrong
  here: both models train with class weights, so their implied prior is already
  tilted. Costs 4 accuracy points, or 46 if badly scaled.
- **Frontal channels F3/F4.** Excluded for uniformity (present in only 87 of
  100 recordings). Given that AASM scores N3 primarily on frontal slow-wave
  activity and N3 is the second-weakest stage, this is the most promising
  unexplored lever rather than a settled rejection.
