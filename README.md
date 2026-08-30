# iSLEEPS Sleep Classification

| folder | contents |
|---|---|
| **`submission/`** | The model. Start at `submission/README.md`. |
| `dataset_tools/` | Dataset integrity: download verifier and the audit that found the traps. |
| `Dataset/` | 99 recordings (EDF + scoring workbook) plus `subject_description.xlsx`. |
| `cache/` | Generated per-epoch features and raw epochs. Rebuildable; not in version control. |
| `figures/` | Result figures and their regeneration script. Start at `figures/README.md`. |
| `results/`, `results_1200/` | Model outputs. `results_1200` is the tree-count convergence check. |

**Final model: a three-model ensemble — accuracy 0.7794, macro F1 0.7210,
Cohen's κ 0.690**, five-fold cross-validation grouped by patient. This is ahead
of the dataset paper's best published baseline (Maiti et al., Sci Data 13:421,
2026: LSTM at 74.70 / 67.68 / 0.64) on every overall metric and every individual
sleep stage.

Before trusting any downloaded copy of the dataset, run
`python dataset_tools\verify_downloads.py Dataset` — the original download
produced four files that were the correct size but internally scrambled, which
a size check passes. See `dataset_tools/DATASET_AUDIT.md`.
