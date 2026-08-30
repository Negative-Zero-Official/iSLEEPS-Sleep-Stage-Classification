#!/usr/bin/env python3
"""Step 4: Viterbi smoothing and model stacking on saved posteriors.

Both models score each epoch semi-independently, so their errors are largely
stage-persistence failures -- N3 and REM flipping to N2 mid-run. Real sleep is a
Markov chain with very strong self-transitions (an epoch is ~90% likely to match
its predecessor), and that structure is free information neither model uses.

Viterbi decoding imposes it: treat each model's posterior as a scaled emission
likelihood, estimate the transition matrix from the TRAINING recordings of each
fold only, and decode the most likely stage sequence for each night.

    python train_gbdt.py --cache cache --folds 5 --save-proba
    python train_cnn.py  --cache cache --folds 5 --epochs 15 --save-proba
    python stack.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleepstaging"))
import evaluate as E  # noqa: E402

K = 5


def transition_matrix(y_list, smoothing=1.0):
    A = np.full((K, K), smoothing)
    pi = np.full(K, smoothing)
    for y in y_list:
        pi[y[0]] += 1
        np.add.at(A, (y[:-1], y[1:]), 1)
    return A / A.sum(1, keepdims=True), pi / pi.sum()


def viterbi(log_emis, logA, logpi):
    n = len(log_emis)
    dp = np.empty((n, K)); bp = np.zeros((n, K), dtype=np.int8)
    dp[0] = logpi + log_emis[0]
    for t in range(1, n):
        m = dp[t - 1][:, None] + logA
        bp[t] = m.argmax(0)
        dp[t] = m.max(0) + log_emis[t]
    path = np.empty(n, dtype=np.int64)
    path[-1] = dp[-1].argmax()
    for t in range(n - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def load_side(pattern, order_by_te, lens, bounds, rec_folds, n_epochs):
    """Return posteriors in canonical epoch order, or None if not saved."""
    P = np.zeros((n_epochs, K), dtype=np.float32)
    seen = np.zeros(n_epochs, bool)
    for k, (_, te_recs) in enumerate(rec_folds, 1):
        f = pattern.format(k=k)
        if not os.path.exists(f):
            return None
        z = np.load(f, allow_pickle=True)
        if "proba" not in z:
            return None
        if order_by_te:                      # gbdt: explicit canonical indices
            P[z["te"]] = z["proba"]; seen[z["te"]] = True
        else:                                # cnn: fold-then-recording order
            at = 0
            for i in te_recs:
                m = lens[i]
                P[bounds[i]:bounds[i + 1]] = z["proba"][at:at + m]
                seen[bounds[i]:bounds[i + 1]] = True
                at += m
    return P if seen.all() else None


ALPHA_GRID = (0.0, 0.15, 0.25, 0.4, 0.6, 0.85, 1.2)


def _decode_recs(P, y, bounds, recs, logA, logpi, alpha):
    out = {}
    for i in recs:
        sl = slice(bounds[i], bounds[i + 1])
        emis = np.log(np.clip(P[sl], 1e-8, None))
        out[i] = P[sl].argmax(1) if alpha == 0 else viterbi(emis, alpha * logA, logpi)
    return out


def decode(P, y, lens, bounds, rec_folds, alpha="auto", verbose=False):
    """Fold-wise Viterbi smoothing of a classifier's posteriors.

    Emissions are the log posteriors directly rather than posterior/prior. The
    textbook conversion assumes a classifier trained on the natural class
    distribution; both models here are trained with class weights, so their
    implied prior is already tilted and dividing by any prior estimate
    double-corrects -- measured at roughly -4 accuracy points.

    `alpha` scales the transition term. With alpha="auto" it is chosen per fold
    on the TRAINING recordings, using their out-of-fold posteriors (which came
    from models that never saw them either), so the test fold is untouched.
    """
    from sklearn.metrics import cohen_kappa_score
    out = np.empty(len(y), dtype=np.int64)
    for tr_recs, te_recs in rec_folds:
        A, pi = transition_matrix([y[bounds[i]:bounds[i + 1]] for i in tr_recs])
        logA, logpi = np.log(A), np.log(pi)

        if alpha == "auto":
            best_a, best_k = 0.0, -np.inf
            for cand in ALPHA_GRID:
                dec = _decode_recs(P, y, bounds, tr_recs, logA, logpi, cand)
                yy = np.concatenate([y[bounds[i]:bounds[i + 1]] for i in tr_recs])
                pp = np.concatenate([dec[i] for i in tr_recs])
                k = cohen_kappa_score(yy, pp)
                if k > best_k:
                    best_a, best_k = cand, k
            a_use = best_a
            if verbose:
                print(f"      fold alpha={a_use} (train kappa {best_k:.4f})")
        else:
            a_use = float(alpha)

        for i, pred in _decode_recs(P, y, bounds, te_recs, logA, logpi, a_use).items():
            out[bounds[i]:bounds[i + 1]] = pred
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--alpha", default="auto",
                    help="transition weight; 'auto' tunes it per fold on training recordings")
    a = ap.parse_args()
    d = E.load_cache(a.cache)
    y = d["y"]
    lens, rec_group = [], []
    for p in sorted(glob.glob(os.path.join(a.cache, "*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        z = np.load(p, allow_pickle=True)
        lens.append(len(z["y"])); rec_group.append(str(z["patient"]))
    bounds = np.cumsum([0] + lens)
    rec_folds, _ = E.recording_folds(d["groups"], rec_group, 5)

    gb = load_side(a.results + "/gbdt_fold{k}.npz", True, lens, bounds, rec_folds, len(y))
    cn = load_side(a.results + "/cnn_fold{k}.npz", False, lens, bounds, rec_folds, len(y))
    ls = load_side(a.results + "/lstm_fold{k}.npz", True, lens, bounds, rec_folds, len(y))
    if gb is None:
        raise SystemExit("no gradient-boosting posteriors -- re-run train_gbdt.py with --save-proba")

    A_all, _ = transition_matrix([y[bounds[i]:bounds[i + 1]] for i in range(len(lens))])
    print("empirical stage self-transition probabilities (why smoothing helps):")
    for i, st in enumerate(E.STAGES):
        print(f"    {st:<6} stays {A_all[i, i]:.3f}")

    members = [("gradient boosting", gb)]
    if cn is not None:
        members.append(("CNN + BiGRU", cn))
    if ls is not None:
        members.append(("BiLSTM (features)", ls))
    missing = [n for n, p in [("CNN", cn), ("LSTM", ls)] if p is None]
    if missing:
        print(f"\n(no posteriors for: {', '.join(missing)} -- "
              f"re-run the corresponding script with --save-proba to include it)")

    print("\nchoosing the transition weight per fold on training recordings only:")
    variants = []
    for name, P in members:
        variants.append((name, P.argmax(1)))
        variants.append((name + " + Viterbi", decode(P, y, lens, bounds, rec_folds, a.alpha)))

    # Equal-weight ensembles. Averaging posteriors needs no tuning and cannot
    # overfit a held-out set, which matters at this cohort size.
    if len(members) >= 2:
        for combo in ([members] if len(members) == 2 else
                      [[members[0], m] for m in members[1:]] + [members]):
            names = [n for n, _ in combo]
            P = sum(p for _, p in combo) / len(combo)
            tag = "ensemble (" + " + ".join(
                n.split()[0].lower() for n in names) + ")"
            variants.append((tag, P.argmax(1)))
            variants.append((tag + " + Viterbi",
                             decode(P, y, lens, bounds, rec_folds, a.alpha)))

    from sklearn.metrics import cohen_kappa_score, f1_score
    print(f"\n{'':<44}{'accuracy':>10}{'macro F1':>10}{'kappa':>9}")
    for name, p in variants:
        print(f"  {name:<42}{(p==y).mean():>10.4f}{f1_score(y,p,average='macro'):>10.4f}"
              f"{cohen_kappa_score(y,p):>9.4f}")

    best = max(variants, key=lambda kv: cohen_kappa_score(y, kv[1]))
    E.report(y, best[1], f"best variant: {best[0]}")
    np.savez(os.path.join(a.results, "stacked.npz"), y=y, pred=best[1], name=best[0])
    print(f"\nsaved -> {os.path.join(a.results, 'stacked.npz')} ({best[0]})")


if __name__ == "__main__":
    main()
