#!/usr/bin/env python3
"""Regenerate every figure from the saved fold posteriors.

Nothing here recomputes a model. Every number plotted comes either from
results/*_fold*.npz (this work) or from Table 3 of the dataset paper, which is
transcribed once in PAPER below and never edited elsewhere.

    python submission/make_figures.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleepstaging"))
import evaluate as E          # noqa: E402
import stack as S             # noqa: E402
import subjects as SUB        # noqa: E402

STAGES = ["Wake", "N1", "N2", "N3", "REM"]

# --- palette -----------------------------------------------------------------
# Categorical hues assigned in fixed order, never cycled. Sequential encoding
# uses a single hue, light to dark.
# Validated with the palette checker, not chosen by eye. The scheme is
# semantic: magenta always means the dataset paper, blue always means this work.
# Violet and orange appear only to separate the three paper baselines from each
# other. Green is deliberately absent - green next to orange fails CVD
# separation badly (delta-E 3.2 for protanopia), the classic red/green trap.
PAPER_C, MINE_C = "#e87ba4", "#2a78d6"
P_CNN, P_TRF, P_LSTM = "#e87ba4", "#4a3aa7", "#eb6834"   # adjacent-pair CVD 24.7
ORACLE_C = "#eb6834"
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#104281"]   # ordered severity, one hue
C = [MINE_C, "#eb6834", "#1baf7a", "#eda100", PAPER_C, "#008300"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})

# --- the dataset paper's Table 3 (Maiti et al., Sci Data 13:421, 2026) -------
PAPER = {
    "CNN (paper)":         dict(acc=61.65, mf1=54.44, kappa=0.48,
                                f1=[68.15, 17.43, 68.82, 67.65, 50.12]),
    "Transformer (paper)": dict(acc=67.44, mf1=59.35, kappa=0.54,
                                f1=[77.53, 25.91, 76.07, 69.18, 47.03]),
    "LSTM (paper)":        dict(acc=74.70, mf1=67.68, kappa=0.64,
                                f1=[79.87, 32.99, 80.91, 74.25, 70.04]),
}


def save(fig, out, name):
    """Write via a local temp file, then move into place.

    Overwriting an existing PNG directly on a mounted filesystem intermittently
    fails with OSError 22, and a half-run leaves a stale figure behind that
    silently disagrees with the rest of the set. Writing locally and moving is
    atomic enough to avoid both.
    """
    import shutil, tempfile
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    fig.savefig(tmp)
    dest = os.path.join(out, name)
    if os.path.exists(dest):
        os.remove(dest)
    shutil.move(tmp, dest)
    plt.close(fig)


def load_everything(cache, results):
    d = E.load_cache(cache)
    y = d["y"]
    lens, rec_group, rec_name = [], [], []
    for p in sorted(glob.glob(os.path.join(cache, "*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        z = np.load(p, allow_pickle=True)
        lens.append(len(z["y"])); rec_group.append(str(z["patient"])); rec_name.append(str(z["rec"]))
    bounds = np.cumsum([0] + lens)
    rf, _ = E.recording_folds(d["groups"], rec_group, 5)
    P = {}
    for tag, pat, by_te in [("Gradient boosting", "gbdt", True),
                            ("CNN + BiGRU", "cnn", False),
                            ("BiLSTM", "lstm", True)]:
        arr = S.load_side(os.path.join(results, pat + "_fold{k}.npz"),
                          by_te, lens, bounds, rf, len(y))
        if arr is not None:
            P[tag] = arr
    P["Ensemble"] = sum(P.values()) / len(P)
    return y, P, lens, bounds, rec_name, rf


def metrics(y, pred):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    return dict(acc=100 * accuracy_score(y, pred),
                mf1=100 * f1_score(y, pred, average="macro"),
                kappa=cohen_kappa_score(y, pred),
                f1=[100 * f1_score(y == i, pred == i) for i in range(5)])


def bar_labels(ax, bars, fmt="{:.1f}", dx=0.0):
    for b in bars:
        w = b.get_width()
        ax.text(w + dx, b.get_y() + b.get_height() / 2, fmt.format(w),
                va="center", ha="left", fontsize=8, color=INK2)


# =============================================================================
def fig1_overall(mine, out):
    """Overall metrics, this work against the published baselines.

    Three panels rather than one: accuracy and macro F1 are percentages while
    kappa is a 0-1 agreement statistic, and putting them on a shared axis would
    misrepresent the comparison.
    """
    rows = [("CNN (paper)", PAPER_C), ("Transformer (paper)", PAPER_C),
            ("LSTM (paper)", PAPER_C), ("BiLSTM (this work)", MINE_C),
            ("Gradient boosting (this work)", MINE_C), ("Ensemble (this work)", MINE_C)]
    src = {**PAPER, "BiLSTM (this work)": mine["BiLSTM"],
           "Gradient boosting (this work)": mine["Gradient boosting"],
           "Ensemble (this work)": mine["Ensemble"]}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, key, title, xmax in zip(
            axes, ["acc", "mf1", "kappa"],
            ["Accuracy (%)", "Macro F1 (%)", "Cohen's κ"], [95, 95, 0.86]):
        vals = [src[n][key] for n, _ in rows]
        bars = ax.barh(range(len(rows)), vals, color=[c for _, c in rows], height=0.62)
        bar_labels(ax, bars, "{:.2f}" if key == "kappa" else "{:.1f}", xmax * 0.012)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([n for n, _ in rows] if ax is axes[0] else [])
        if ax is not axes[0]:
            ax.tick_params(axis="y", length=0)
        for k, b in enumerate(bars):          # the final model gets a visible ring
            if rows[k][0].startswith("Ensemble"):
                b.set_edgecolor(INK); b.set_linewidth(1.4)
        ax.invert_yaxis(); ax.set_xlim(0, xmax); ax.set_title(title)
        ax.xaxis.grid(True); ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=PAPER_C, label="Dataset paper (Maiti et al. 2026)"),
                        Patch(facecolor=MINE_C, label="This work"),
                        Patch(facecolor=MINE_C, edgecolor=INK, linewidth=1.4,
                              label="This work, final ensemble")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Five-class sleep staging on iSLEEPS: this work vs published baselines",
                 y=1.04, fontsize=11.5, fontweight="bold")
    save(fig, out, "fig1_overall_metrics.png")


def fig2_per_stage(mine, out):
    """Per-stage F1. The overall numbers hide that the gains are concentrated
    in the two stages the published models handled worst."""
    series = [("CNN (paper)", PAPER["CNN (paper)"]["f1"], P_CNN),
              ("Transformer (paper)", PAPER["Transformer (paper)"]["f1"], P_TRF),
              ("LSTM (paper)", PAPER["LSTM (paper)"]["f1"], P_LSTM),
              ("Ensemble (this work)", mine["Ensemble"]["f1"], MINE_C)]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    n, w = len(series), 0.2
    x = np.arange(5)
    for i, (name, vals, col) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        b = ax.bar(x + off, vals, w * 0.92, label=name, color=col)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():.0f}",
                    ha="center", fontsize=7.2, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(STAGES)
    ax.set_ylabel("F1 (%)"); ax.set_ylim(0, 100)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
    ax.set_title("Per-stage F1: every stage improves, most of all N1 and REM")
    save(fig, out, "fig5_per_stage_f1.png")


def fig3_confusion(y, pred, out):
    """Row-normalised confusion for the final ensemble. Sequential single hue:
    the cell value is a magnitude, not an identity."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred, labels=range(5))
    cmn = 100 * cm / cm.sum(1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if cmn[i, j] > 55 else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(STAGES)
    ax.set_yticks(range(5)); ax.set_yticklabels(STAGES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Expert score")
    ax.set_title("Confusion matrix, final ensemble (row %)")
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, shrink=0.75, label="% of true stage")
    save(fig, out, "fig4_confusion_matrix.png")


def fig4_per_recording(y, P, lens, bounds, out):
    """Per-recording accuracy. A single headline number hides how much the
    models vary night to night, which is what a clinician would actually meet."""
    order = ["CNN + BiGRU", "BiLSTM", "Gradient boosting", "Ensemble"]
    data = []
    for name in order:
        p = P[name].argmax(1)
        data.append(np.array([(p[bounds[i]:bounds[i + 1]] == y[bounds[i]:bounds[i + 1]]).mean()
                              for i in range(len(lens))]) * 100)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    cols = [MINE_C] * 4          # rows are separated and labelled; colour adds nothing
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.6),
                    whiskerprops=dict(color=AXIS), capprops=dict(color=AXIS),
                    boxprops=dict(edgecolor=AXIS, linewidth=0.8))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.28)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, cols), start=1):
        ax.scatter(vals, i + rng.uniform(-0.13, 0.13, len(vals)), s=11, color=c,
                   alpha=0.75, linewidths=0.5, edgecolors=SURFACE, zorder=3)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{n}\nmedian {np.median(v):.1f}%" for n, v in zip(order, data)])
    ax.set_xlabel("Accuracy on one recording (%)"); ax.set_xlim(15, 100)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("Per-recording accuracy across the 99 nights")
    save(fig, out, "fig2_per_recording_accuracy.png")


def fig5_complementarity(y, P, out):
    """How much the three models disagree with each other.

    This is the precondition for the ensemble helping at all: averaging models
    that agree everywhere gains nothing. The oracle-bound companion to this
    figure was cut - the bound is a hypothetical that no combiner reaches, and
    stating it in the text is clearer than drawing bars for accuracies nothing
    achieves.
    """
    names = ["Gradient boosting", "CNN + BiGRU", "BiLSTM"]
    short = ["Trees", "CNN", "BiLSTM"]
    preds = {n: P[n].argmax(1) for n in names}
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)

    # (a) pairwise agreement
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    M = np.full((3, 3), np.nan)
    for i in range(3):
        for j in range(3):
            if i != j:
                M[i, j] = 100 * (preds[names[i]] == preds[names[j]]).mean()
    ax.imshow(M, cmap=cmap, vmin=60, vmax=100)
    for i in range(3):
        for j in range(3):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=MUTED)
            else:
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=10.5,
                        color="#ffffff" if M[i, j] > 82 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(short)
    ax.set_yticks(range(3)); ax.set_yticklabels(short)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    ax.set_title("Pairwise agreement between models\n(% of the 93,937 epochs)")
    save(fig, out, "fig3_model_agreement.png")



def fig6_hypnogram(y, pred, lens, bounds, rec_name, out):
    """Expert versus predicted hypnogram for one night, the direct analogue of
    Figure 3 in the dataset paper. The recording shown is the one whose accuracy
    is closest to the median, so it is representative rather than flattering."""
    accs = np.array([(pred[bounds[i]:bounds[i + 1]] == y[bounds[i]:bounds[i + 1]]).mean()
                     for i in range(len(lens))])
    i = int(np.argsort(np.abs(accs - np.median(accs)))[0])
    sl = slice(bounds[i], bounds[i + 1])
    yt, yp = y[sl], pred[sl]
    hours = np.arange(len(yt)) * 30 / 3600
    # Plot order Wake, REM, N1, N2, N3 - the conventional hypnogram layout.
    order = [0, 4, 1, 2, 3]
    pos = {s: k for k, s in enumerate(order)}
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 5.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 3, 0.7]))
    for ax, series, title, col in [(axes[0], yt, "Expert scoring", INK2),
                                   (axes[1], yp, "Final ensemble", MINE_C)]:
        ax.step(hours, [pos[s] for s in series], where="post", linewidth=1.1, color=col)
        ax.set_yticks(range(5)); ax.set_yticklabels([STAGES[s] for s in order])
        ax.set_ylim(-0.5, 4.5); ax.invert_yaxis()
        ax.set_ylabel(title, fontsize=9, color=INK2)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    wrong = yt != yp
    axes[2].fill_between(hours, 0, wrong.astype(float), step="post", color=ORACLE_C, linewidth=0)
    axes[2].set_yticks([]); axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Errors", fontsize=9, color=INK2)
    axes[2].set_xlabel("Hours from recording start")
    axes[0].set_title(f"Hypnogram, recording {rec_name[i]} "
                      f"(accuracy {100*accs[i]:.1f}%, the median night)")
    save(fig, out, "fig8_hypnogram.png")


def fig7_support(y, mine, out):
    """Per-stage F1 against how common the stage is.

    The first version of this plotted F1 on a log support axis and claimed
    performance tracks abundance. It does, roughly, but N1 breaks it: N1 is more
    common than N3 and scores half as well. Plotting stages in order of
    abundance makes that exception the point of the figure rather than an
    inconvenience hidden by a trend line. Ordering by support also removes the
    marker collision the log axis produced, where N3's two values differ by 0.15
    and one marker sat on top of the other.
    """
    counts = np.bincount(y, minlength=5)
    order = np.argsort(counts)
    f1_mine = np.array(mine["Ensemble"]["f1"])[order]
    f1_paper = np.array(PAPER["LSTM (paper)"]["f1"])[order]
    labels = [STAGES[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(5)
    ax.vlines(x, f1_paper, f1_mine, color=AXIS, linewidth=1.4, zorder=2)
    ax.scatter(x, f1_paper, s=95, color=PAPER_C, marker="s", zorder=3,
               label="LSTM (dataset paper)")
    ax.scatter(x, f1_mine, s=105, color=MINE_C, zorder=4, label="Ensemble (this work)")
    for k in range(5):
        # When the two values are close the labels collide, so push them apart
        # vertically rather than letting one hide the other.
        gap = f1_mine[k] - f1_paper[k]
        dy_m, dy_p = (7, -12) if abs(gap) < 6 else (-3, -3)
        ax.annotate(f"{f1_mine[k]:.0f}", (x[k], f1_mine[k]), textcoords="offset points",
                    xytext=(13, dy_m), fontsize=8.5, color=MINE_C, fontweight="bold")
        ax.annotate(f"{f1_paper[k]:.0f}", (x[k], f1_paper[k]), textcoords="offset points",
                    xytext=(13, dy_p), fontsize=8.5, color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{counts[i]:,} epochs" for s, i in zip(labels, order)])
    ax.set_ylabel("F1 (%)"); ax.set_ylim(0, 100); ax.set_xlim(-0.5, 4.55)
    ax.yaxis.grid(True); ax.set_axisbelow(True); ax.legend(loc="lower right")
    ax.set_title("F1 broadly follows how common a stage is, and N1 breaks the pattern")
    save(fig, out, "fig7_f1_vs_support.png")


def fig8_severity(y, pred, lens, bounds, rec_name, dataset, out):
    """Is a night harder to stage when the patient's sleep apnea is worse?

    No new model and no new predictions. These are the same ensemble outputs
    plotted in Figures 1-3. For each of the 99 recordings we take the share of
    its 30-second epochs that were staged correctly, then group those 99 numbers
    by that patient's apnea severity, read from the AHI column of
    subject_description.xlsx.

    The question matters for this cohort specifically: around 85% of these
    patients have sleep-disordered breathing, and apnea fragments sleep with
    frequent arousals. If that fragmentation made staging harder, a model
    validated on this dataset would be quietly worse for exactly the patients it
    is most likely to be used on.
    """
    import re, zipfile
    from scipy import stats as st
    z = zipfile.ZipFile(os.path.join(dataset, "subject_description.xlsx"))
    ss = re.findall(r"<t[^>]*>(.*?)</t>", z.read("xl/sharedStrings.xml").decode("utf8"), re.S)
    sheet = z.read("xl/worksheets/sheet1.xml").decode("utf8", "ignore")
    ahi = {}
    for r in re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S)[1:]:
        cs = []
        for c in re.findall(r"<c[^>]*?(?:/>|>.*?</c>)", r, re.S):
            t = re.search(r't="(\w+)"', c); v = re.search(r"<v>(.*?)</v>", c, re.S)
            cs.append((ss[int(v.group(1))] if (t and t.group(1) == "s") else v.group(1)) if v else "")
        if not cs or not cs[0].strip():
            continue
        rid = SUB.AN_TO_FILE.get(cs[0].strip().replace(".edf", ""), cs[0].strip().replace(".edf", ""))
        try:
            ahi[rid] = float(cs[-1])
        except ValueError:
            pass

    def sev(a):
        return "Normal" if a < 5 else "Mild" if a < 15 else "Moderate" if a < 30 else "Severe"

    classes = ["Normal", "Mild", "Moderate", "Severe"]
    buckets = {c: [] for c in classes}
    all_ahi, all_acc = [], []
    for i, name in enumerate(rec_name):
        if name not in ahi:
            continue
        a = (pred[bounds[i]:bounds[i + 1]] == y[bounds[i]:bounds[i + 1]]).mean() * 100
        buckets[sev(ahi[name])].append(a)
        all_ahi.append(ahi[name]); all_acc.append(a)

    rho, p_rho = st.spearmanr(all_ahi, all_acc)
    H, p_kw = st.kruskal(*[np.array(buckets[c]) for c in classes])

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    rng = np.random.default_rng(1)
    for k, (cl, col) in enumerate(zip(classes, ORDINAL)):
        vals = np.array(buckets[cl])
        ax.scatter(k + rng.uniform(-0.16, 0.16, len(vals)), vals, s=26, color=col,
                   alpha=0.85, linewidths=0.5, edgecolors=SURFACE, zorder=3)
        ax.plot([k - 0.3, k + 0.3], [vals.mean()] * 2, color=INK, linewidth=2, zorder=4)
        ax.text(k, 96.5 if k == 0 else 96.5, f"n={len(vals)}   mean {vals.mean():.1f}%",
                ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{c}\n(AHI {r})" for c, r in
                        zip(classes, ["<5", "5-15", "15-30", "≥30"])])
    ax.set_xlabel("Patient's apnea severity", labelpad=8)
    ax.set_ylabel("Epochs of this recording staged correctly (%)")
    ax.set_ylim(50, 100)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.text(0.015, 0.035,
            f"no detectable relationship\n"
            f"Spearman ρ = {rho:+.2f}, p = {p_rho:.2f}\n"
            f"Kruskal–Wallis p = {p_kw:.2f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=INK2,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=SURFACE, edgecolor=GRID))
    ax.set_title("Staging accuracy does not degrade as apnea severity rises", pad=26)
    ax.text(0.5, 1.045,
            "One point = one recording, scored by the same ensemble as Figures 1–3. "
            "Black rule = group mean.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=INK2)
    save(fig, out, "fig6_accuracy_by_severity.png")
    return {c: (len(buckets[c]), float(np.mean(buckets[c]))) for c in classes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--dataset", default=os.path.join(ROOT, "Dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    y, P, lens, bounds, rec_name, rf = load_everything(a.cache, a.results)
    mine = {n: metrics(y, p.argmax(1)) for n, p in P.items()}
    ens = P["Ensemble"].argmax(1)

    print(f"{len(y):,} epochs | {len(lens)} recordings | models: {', '.join(P)}\n")
    print(f"  {'model':<22}{'ACC':>7}{'MF1':>8}{'kappa':>8}")
    for n, m in mine.items():
        print(f"  {n:<22}{m['acc']:>7.2f}{m['mf1']:>8.2f}{m['kappa']:>8.4f}")

    fig1_overall(mine, a.out);                                    print("\n  fig1 overall metrics")
    fig2_per_stage(mine, a.out);                                  print("  fig5 per-stage F1")
    fig3_confusion(y, ens, a.out);                                print("  fig4 confusion matrix")
    fig4_per_recording(y, P, lens, bounds, a.out);                print("  fig2 per-recording accuracy")
    fig5_complementarity(y, P, a.out);                            print("  fig3 model agreement")
    fig6_hypnogram(y, ens, lens, bounds, rec_name, a.out);        print("  fig8 hypnogram")
    fig7_support(y, mine, a.out);                                 print("  fig7 F1 vs support")
    sev = fig8_severity(y, ens, lens, bounds, rec_name, a.dataset, a.out)
    print("  fig6 accuracy by apnea severity ->", {k: f"n={v[0]}, {v[1]:.1f}%" for k, v in sev.items()})
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
