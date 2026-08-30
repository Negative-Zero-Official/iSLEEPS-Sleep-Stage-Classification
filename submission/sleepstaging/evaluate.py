"""Shared dataset loading, splitting and reporting, so both models are judged
on exactly the same folds and the same metrics."""
from __future__ import annotations

import glob
import os

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold

STAGES = ("Wake", "N1", "N2", "N3", "REM")


def load_cache(cache_dir, want_raw=False):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")),
                   key=lambda p: int(os.path.basename(p)[2:-4]))
    if not files:
        raise SystemExit(f"no cached recordings in {cache_dir}/ -- run extract_features.py first")
    X, y, groups, recs, raw, names = [], [], [], [], [], None
    for p in files:
        d = np.load(p, allow_pickle=True)
        rawp = os.path.join(cache_dir, "raw", os.path.basename(p).replace(".npz", ".npy"))
        if want_raw and not os.path.exists(rawp):
            continue
        X.append(d["X"]); y.append(d["y"])
        n = len(d["y"])
        groups.append(np.repeat(str(d["patient"]), n))
        recs.append(np.repeat(str(d["rec"]), n))
        if want_raw:
            raw.append(np.load(rawp, mmap_mode="r"))
        names = d["names"]
    if not X:
        raise SystemExit("no cached recordings contain raw epochs -- re-run with --raw")
    out = dict(X=np.concatenate(X), y=np.concatenate(y),
               groups=np.concatenate(groups), recs=np.concatenate(recs),
               names=[str(s) for s in names])
    if want_raw:
        out["raw"] = raw          # list of memmaps, concatenated lazily
        out["raw_lengths"] = [len(r) for r in raw]
    return out


def folds(groups, n_splits=5):
    """Group-aware CV. Splitting by patient rather than by recording is the whole
    point: 99 recordings come from only 86 patients, and several patients have a
    second night in the set."""
    n_splits = min(n_splits, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)), n_splits


def report(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {title} ===")
    print(f"  accuracy   {acc:.4f}")
    print(f"  macro F1   {mf1:.4f}")
    print(f"  Cohen kappa {kappa:.4f}")

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(5), zero_division=0)
    print(f"\n  {'stage':<7}{'prec':>8}{'recall':>8}{'F1':>8}{'support':>9}")
    for i, st in enumerate(STAGES):
        print(f"  {st:<7}{p[i]:>8.3f}{r[i]:>8.3f}{f[i]:>8.3f}{s[i]:>9,}")

    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    print(f"\n  confusion (row = truth, %)")
    print("           " + "".join(f"{s:>8}" for s in STAGES))
    for i, st in enumerate(STAGES):
        print(f"  {st:<7}" + "".join(f"{100*v:>8.1f}" for v in cmn[i]))
    return dict(accuracy=acc, macro_f1=mf1, kappa=kappa)


def window_starts(n, seq_len):
    """Start indices of `seq_len` windows that cover ALL n epochs.

    A plain range(0, n-seq_len+1, seq_len) silently drops the final partial
    window, leaving the last few epochs of every recording unpredicted.
    """
    if n <= seq_len:
        return [0]
    starts = list(range(0, n - seq_len + 1, seq_len))
    if starts[-1] + seq_len < n:
        starts.append(n - seq_len)      # final window, back-aligned to the end
    return starts


def recording_folds(epoch_groups, rec_group, n_splits=5):
    """Fold assignment at recording level that matches the epoch-level folds.

    Both models must be scored on identical partitions. GroupKFold balances by
    sample count, so splitting an array of 99 recordings and an array of 93,937
    epochs gives different partitions. We therefore build the folds once on
    epochs -- exactly as the tree model does -- then map each patient, and hence
    each recording, onto the fold its epochs landed in.
    """
    splits, n = folds(epoch_groups, n_splits)
    patient_fold = {}
    for k, (_, te) in enumerate(splits):
        for p in np.unique(np.asarray(epoch_groups)[te]):
            patient_fold[p] = k
    out = []
    for k in range(n):
        te_recs = [i for i, g in enumerate(rec_group) if patient_fold[g] == k]
        tr_recs = [i for i, g in enumerate(rec_group) if patient_fold[g] != k]
        out.append((tr_recs, te_recs))
    return out, n


def inner_split(group_labels, val_frac=0.15, seed=0):
    """Split a training set into train/validation *by patient*.

    Early stopping needs a held-out set, and it must not be the test fold or the
    stopping point is chosen on the data being reported. It also must not be a
    random slice of epochs: epochs from the same night are near-duplicates, so a
    random split leaks and the validation curve stops being informative. This
    partitions on the group label, like the outer CV.
    """
    uniq = np.unique(group_labels)
    rng = np.random.RandomState(seed)
    order = uniq.copy()
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_frac)))
    val_groups = set(order[:n_val].tolist())
    is_val = np.array([g in val_groups for g in group_labels])
    return ~is_val, is_val
