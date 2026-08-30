# iSLEEPS Sleep Classification

| folder | contents |
|---|---|
| **`submission/`** | The model. Start at `submission/README.md`. |
| `dataset_tools/` | Dataset integrity: download verifier and the audit that found the traps. |
| `Dataset/` | 99 recordings (EDF + scoring workbook) plus `subject_description.xlsx`. |
| `cache/` | Generated per-epoch features and raw epochs. Rebuildable; not in version control. |
| `results/`, `results_1200/` | Model outputs. `results_1200` is the tree-count convergence check. |

**Best model: ensemble + Viterbi — accuracy 0.7805, macro F1 0.7125, Cohen's
κ 0.6893**, five-fold cross-validation grouped by patient.

Before trusting any downloaded copy of the dataset, run
`python dataset_tools\verify_downloads.py Dataset` — the original download
produced four files that were the correct size but internally scrambled, which
a size check passes. See `dataset_tools/DATASET_AUDIT.md`.
