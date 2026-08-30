# Sleep Stage Classification: Implementation Notes

I built a five-class AASM classifier (W/N1/N2/N3/REM) on 30-second epochs across all 100 subjects.

## Data Integrity

The download arrived damaged in four ways: byte-identical duplicate copies, zero-byte files, HTML 404 pages saved with `.edf`/`.xlsx` extensions, and files where the downloader had written some 25 MiB blocks twice while dropping others. Two of those had the **correct byte count and a self-consistent EDF header**, so only block-level hashing caught them. Separately, india-data.org's download dialogue reported "Completed" while writing nothing to disk — a server-side fault I could only wait out.

## Traps in the Dataset

  - Two channel-naming conventions (**M1/M2** in 64 recordings, **A1/A2** in 36). No EEG channel matches by name across all 100 files; a literal match silently discards **36%** of the data.
  - The metadata has no SN11–SN20 rows — those files are the **AN1–AN10** subjects. Joining on filename mislabels ten recordings.
  - **SN15 and SN28 are the same recording**, differing in eight header bytes, with identical signal data and hypnogram.
  - 99 usable recordings come from only **86 unique patients**, so folds must be grouped by patient.
  - Sheet 1 of the workbook is not always the hypnogram — in SN80 it is the light sensor, in lux.

## Approach and Results

426 spectral and temporal features per epoch into gradient-boosted trees, compared against a CNN + BiGRU on the raw signal. Five-fold cross-validation grouped by patient, 93,937 epochs.

  - **Gradient boosting**: 77.12% accuracy, macro F1 0.7070, Cohen's κ 0.6768
  - **CNN + BiGRU**: 72.19% accuracy, macro F1 0.6602, κ 0.6144
  - **Ensemble + Viterbi (best)**: **78.05% accuracy**, macro F1 0.7125, **κ 0.6893**

The trees beat the CNN on 74 of 99 recordings (Wilcoxon p = 2.4e-7), but the CNN still earns its place in the ensemble because its errors differ on roughly a quarter of epochs. For reference, the baselines published with the dataset report LSTM 74.70%, Transformer 67.44% and CNN 61.65% — though I do not know their split protocol, so this is indicative rather than a like-for-like comparison.

## What Failed

Early stopping made **both** models worse. At 86 patients there is no spare data for an inner validation split: validation κ over ~12 held-out patients swings more than the effect being selected, and the CNN lost on 5 of 5 folds. Tree count converged at 400 (1200 trees gained κ 0.0026). The textbook posterior-over-prior conversion before Viterbi decoding collapsed accuracy to **29.9%**, because both models train with class weights and their implied prior is already tilted.
