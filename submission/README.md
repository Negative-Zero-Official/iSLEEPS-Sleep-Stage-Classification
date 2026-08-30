# Sleep stage classification on the iSLEEPS stroke cohort

Five-class AASM staging (Wake / N1 / N2 / N3 / REM) from 30-second epochs of
polysomnography, across 99 recordings from 86 patients.

## Result

**Best model: an equal-weight ensemble of gradient-boosted trees and a
CNN+BiGRU, with Viterbi smoothing.**

| model | accuracy | macro F1 | Cohen's κ |
|---|---|---|---|
| gradient boosting | 0.7712 | 0.7070 | 0.6768 |
| gradient boosting + Viterbi | 0.7741 | 0.7058 | 0.6795 |
| CNN + BiGRU | 0.7219 | 0.6602 | 0.6144 |
| CNN + Viterbi | 0.7233 | 0.6593 | 0.6153 |
| ensemble (equal weight) | 0.7794 | 0.7137 | 0.6885 |
| **ensemble + Viterbi** | **0.7805** | **0.7125** | **0.6893** |

Five-fold cross-validation, **grouped by patient**. Every number above is
out-of-fold. If you need a single model without the PyTorch dependency, use
gradient boosting + Viterbi at κ 0.6795.

Per-stage F1 for the best model: Wake 0.846, N1 0.372, N2 0.830, N3 0.735,
REM 0.780.

## Pipeline

Run in order. Paths default relative to the repository root, so the working
directory does not matter.

```powershell
python submission\extract_features.py --subjects all --raw   # ~10 min, writes cache/
python submission\train_gbdt.py  --folds 5 --save-proba      # ~2 min
python submission\train_cnn.py   --folds 5 --epochs 15 --save-proba   # ~10 min, GPU
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
| `stack.py` | Viterbi smoothing + ensembling. **Produces the final model.** |
| `compare.py` | Head-to-head analysis with paired significance testing. |
| `sleepstaging/sleep_io.py` | Dependency-free EDF reader, hypnogram parser, channel aliasing. |
| `sleepstaging/features.py` | Spectral, Hjorth and time-domain features + temporal context. |
| `sleepstaging/subjects.py` | Recording → patient mapping. Prevents train/test leakage. |
| `sleepstaging/evaluate.py` | Shared folds, metrics, sequence windowing. |
| `tests/test_shapes.py` | Validates the CNN's tensor arithmetic (runs without PyTorch). |
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
