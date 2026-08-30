# Figures

Regenerate all of them with:

```powershell
python submission\make_figures.py
```

Figure numbers match the order they appear in `submission/docs/DEVELOPMENT_REPORT.md`.

Nothing here recomputes a model. Every value is read from `results/*_fold*.npz`
or, for the baselines, transcribed once from Table 3 of Maiti et al.,
*Scientific Data* 13:421 (2026). The paper's numbers appear in exactly one place
in the script (`PAPER`), so they cannot drift between figures.

| figure | what it shows | why it earns a place |
|---|---|---|
| `fig1_overall_metrics.png` | Accuracy, macro F1 and κ for the three published baselines and the three models here | The headline comparison. Three panels rather than one, because κ is a 0–1 statistic and forcing it onto a percentage axis would misrepresent it |
| `fig2_per_recording_accuracy.png` | Distribution of accuracy over the 99 nights, per model | A single headline figure hides night-to-night variation, which is what a clinician would actually meet |
| `fig3_model_agreement.png` | Pairwise agreement between the three models | The models disagree on roughly a quarter of epochs, which is the precondition for ensembling to help at all |
| `fig4_confusion_matrix.png` | Row-normalised confusion for the final ensemble | Shows the two structural error modes, N3→N2 and REM→N2, that the overall metrics compress into one number |
| `fig5_per_stage_f1.png` | Per-stage F1 for all four systems | The overall numbers hide where the gains are: N1 and REM, the two stages the published models handled worst |
| `fig6_accuracy_by_severity.png` | Per-recording staging accuracy grouped by the patient's apnea severity | No new model and no new predictions: the same ensemble outputs as elsewhere, aggregated per recording and grouped by the AHI column of `subject_description.xlsx`. Around 85% of this cohort has sleep-disordered breathing, and apnea fragments sleep with frequent arousals, so a model validated here could plausibly be worse for exactly the patients it would be used on. It is not: Spearman ρ = −0.12 (p = 0.22), Kruskal–Wallis p = 0.39 |
| `fig7_f1_vs_support.png` | Per-stage F1 against how common the stage is | Shows that N1 is hard for reasons beyond rarity: it is more common than N3 and scores half as well |
| `fig8_hypnogram.png` | Expert versus predicted hypnogram for one night, with an error strip | The direct analogue of Figure 3 in the dataset paper. The recording shown is the one closest to the median accuracy, not the best one |

## Design notes

Colour is semantic and consistent across the set: **magenta is always the
dataset paper, blue is always this work.** Violet and orange appear only to
separate the three published baselines from one another. Apnea severity is an
ordered quantity, so it uses a single-hue ramp rather than categorical hues, and
the confusion matrix uses the same ramp because a cell value is a magnitude.

An earlier version paired the agreement matrix with a chart of the oracle upper
bound. The bound is a hypothetical that no combiner reaches, so stating it in the
text is clearer than drawing bars for accuracies nothing achieves; that panel was
cut.

Every palette was checked with a validator rather than by eye. That caught two
problems: a four-colour categorical set for the scatter plots failed the
normal-vision separation floor (ΔE 12.9, below the 15 threshold), and an
earlier scheme placed green next to orange, which fails badly under protanopia
(ΔE 3.2). Neither was visible to me on screen.
