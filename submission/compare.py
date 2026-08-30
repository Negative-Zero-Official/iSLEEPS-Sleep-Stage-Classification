#!/usr/bin/env python3
"""Step 3: head-to-head comparison of the two models on identical folds.

The CNN saves predictions in fold-then-recording order while the tree model
saves them in canonical epoch order, so this script reconstructs the CNN's
ordering and *verifies* the alignment by checking its stored true labels
against the canonical ones before comparing anything.

    python compare.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleepstaging"))
import evaluate as E  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    d = E.load_cache(a.cache)
    y, groups, recs = d["y"], d["groups"], d["recs"]

    lens, rec_group, rec_name = [], [], []
    for p in sorted(glob.glob(os.path.join(a.cache, "*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        z = np.load(p, allow_pickle=True)
        lens.append(len(z["y"])); rec_group.append(str(z["patient"])); rec_name.append(str(z["rec"]))
    bounds = np.cumsum([0] + lens)

    gb = np.load(os.path.join(a.results, "gbdt_oof.npz"), allow_pickle=True)
    assert np.array_equal(gb["y"], y), "gbdt cache and feature cache disagree"
    gbdt = gb["pred"]

    rec_folds, n = E.recording_folds(groups, rec_group, 5)
    cnn = np.full(len(y), -1)
    for k, (_, te_recs) in enumerate(rec_folds, 1):
        f = os.path.join(a.results, f"cnn_fold{k}.npz")
        if not os.path.exists(f):
            raise SystemExit(f"missing {f} -- run train_cnn.py --folds 5")
        z = np.load(f, allow_pickle=True)
        pred, true, at = z["pred"], z["true"], 0
        for i in te_recs:
            m = lens[i]
            sl = slice(bounds[i], bounds[i + 1])
            # Alignment self-check: the stored true labels must match canonical.
            assert np.array_equal(true[at:at + m], y[sl]), f"CNN fold {k} misaligned at {rec_name[i]}"
            cnn[sl] = pred[at:at + m]
            at += m
        assert at == len(pred), f"fold {k}: {len(pred)-at} unconsumed predictions"
    assert (cnn >= 0).all(), "some epochs have no CNN prediction"
    print("alignment verified: both models scored on the same 93,937 epochs, same folds\n")

    from sklearn.metrics import cohen_kappa_score, f1_score
    print(f"{'':<22}{'accuracy':>10}{'macro F1':>10}{'kappa':>9}")
    for name, p in [("gradient boosting", gbdt), ("CNN + BiGRU", cnn)]:
        print(f"  {name:<20}{(p==y).mean():>10.4f}{f1_score(y,p,average='macro'):>10.4f}"
              f"{cohen_kappa_score(y,p):>9.4f}")

    print(f"\n  per-stage F1{'':<10}{'GBDT':>8}{'CNN':>8}{'delta':>9}")
    for i, st in enumerate(E.STAGES):
        a = f1_score(y == i, gbdt == i); b = f1_score(y == i, cnn == i)
        print(f"  {st:<22}{a:>8.3f}{b:>8.3f}{b-a:>+9.3f}")

    # --- paired significance -------------------------------------------------
    ga, ca = gbdt == y, cnn == y
    b = int((ga & ~ca).sum()); c = int((~ga & ca).sum())
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    print(f"\n  agreement between models: {(gbdt==cnn).mean():.1%}")
    print(f"  McNemar (epoch-level): GBDT-only-right {b:,}  CNN-only-right {c:,}  "
          f"chi2 {chi2:.1f}  p {stats.chi2.sf(chi2,1):.3g}")
    print("    (epochs within a night are correlated, so treat this as anti-conservative)")

    per = []
    for i in range(len(lens)):
        sl = slice(bounds[i], bounds[i + 1])
        per.append(((gbdt[sl] == y[sl]).mean(), (cnn[sl] == y[sl]).mean()))
    per = np.array(per)
    w = stats.wilcoxon(per[:, 0], per[:, 1])
    wins = int((per[:, 0] > per[:, 1]).sum())
    print(f"\n  per-recording paired test (n={len(per)} recordings, the independent unit):")
    print(f"    GBDT better on {wins}/{len(per)}   mean delta {np.mean(per[:,0]-per[:,1]):+.4f}"
          f"   Wilcoxon p {w.pvalue:.3g}")

    either = (ga | ca).mean()
    print(f"\n  oracle upper bound (either model right): {either:.4f} "
          f"vs {ga.mean():.4f} best single -> {either-ga.mean():+.4f} headroom from combining")


if __name__ == "__main__":
    main()
