#!/usr/bin/env python3
"""Step 2a: gradient-boosted trees on the per-epoch feature table.

    python train_gbdt.py --cache cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleepstaging"))
import sleepstaging.evaluate as E  # noqa: E402


def fit_with_early_stopping(model, Xf, yf, wf, Xv, yv, wv, patience, metric="multi_error"):
    """LightGBM renamed the eval-set arguments between versions; support both."""
    import inspect
    import lightgbm as lgb
    cb = [lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)]
    params = inspect.signature(model.fit).parameters
    if "eval_X" in params:
        model.fit(Xf, yf, sample_weight=wf, eval_X=Xv, eval_y=yv,
                  eval_sample_weight=[wv], eval_metric=metric, callbacks=cb)
    else:
        model.fit(Xf, yf, sample_weight=wf, eval_set=[(Xv, yv)],
                  eval_sample_weight=[wv], eval_metric=metric, callbacks=cb)
    return model.best_iteration_


def dataset_fingerprint(recs, n_epochs, a):
    """Identify the exact (data, split, hyper-parameter) combination a fold
    belongs to. Without this, a cached fold from a smaller run gets replayed
    against a larger dataset and its stored test indices silently address the
    wrong rows -- which looks like a catastrophically bad model rather than a
    stale cache."""
    key = json.dumps({
        "recordings": sorted(set(map(str, recs))),
        "n_epochs": int(n_epochs),
        "folds": a.folds, "seed": a.seed,
        "n_estimators": a.n_estimators, "class_weight": a.class_weight,
        "proba": bool(a.save_proba), "patience": a.patience, "val_frac": a.val_frac,
        "refit": not a.no_refit, "es_metric": a.es_metric, "group_by": a.group_by,
    }, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:16]


def load_fold(path, fingerprint):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    if str(z.get("fingerprint", "")) != fingerprint:
        return None
    return z


def make_model(n_classes, seed, n_estimators=600):
    """LightGBM when available; scikit-learn's histogram booster otherwise.

    They are the same algorithm family (histogram-binned gradient boosting), and
    HistGradientBoosting ships with scikit-learn, so the pipeline has no hard
    dependency on a wheel that may not exist for a very new Python."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.6, reg_lambda=1.0,
                              objective="multiclass", num_class=n_classes,
                              random_state=seed, n_jobs=-1, verbose=-1), "LightGBM", True
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=0.06,
                                              max_leaf_nodes=63, min_samples_leaf=40,
                                              l2_regularization=1.0, early_stopping=False,
                                              random_state=seed), "sklearn HistGradientBoosting", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="sqrt")
    ap.add_argument("--group-by", choices=["patient", "recording"], default="patient",
                    help="What defines a fold boundary. 'patient' is correct. 'recording' "
                         "reproduces the looser protocol used when duplicate patients are "
                         "not known about, and exists only to measure how much that inflates "
                         "the result -- do not report a number produced with it.")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stopping rounds; 0 (default) disables it. Measured on this "
                         "dataset, early stopping HURTS: the inner validation set is only "
                         "~12 patients, its error curve flattens near 100 trees, and stopping "
                         "there costs about 0.008 kappa versus simply training 400. Sweep "
                         "--n-estimators against the outer CV instead.")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of TRAINING patients held out to choose the stopping point")
    ap.add_argument("--es-metric", default="multi_error",
                    help="early-stopping metric. multi_logloss bottoms out well before "
                         "accuracy does on this task and stops ~4x too early")
    ap.add_argument("--no-refit", action="store_true",
                    help="skip refitting on the full training fold after early stopping")
    ap.add_argument("--only-fold", type=int, default=0, help="run just this fold (1-based); 0 = all")
    ap.add_argument("--save-proba", action="store_true",
                    help="also store per-class posteriors (needed for Viterbi smoothing and stacking)")
    a = ap.parse_args()

    d = E.load_cache(a.cache)
    X, y = d["X"], d["y"]
    groups = d["groups"] if a.group_by == "patient" else d["recs"]
    if a.group_by == "recording":
        print("WARNING: splitting by recording, not patient. 12 patients appear under more\n"
              "         than one recording ID, so the same patient will land in train and\n"
              "         test. This measures the optimism of that protocol; it is not a\n"
              "         result to report.\n")
    print(f"{X.shape[0]:,} epochs x {X.shape[1]} features | "
          f"{len(np.unique(d['recs']))} recordings | "
          f"{len(np.unique(d['groups']))} patients | grouping by {a.group_by}")
    print("class balance:", {E.STAGES[i]: int(c) for i, c in enumerate(np.bincount(y, minlength=5))})

    # N1 is ~10% of epochs and is the stage humans agree on least. Full inverse
    # frequency weighting over-corrects and costs more N2 accuracy than it buys
    # in N1, so sqrt weighting is the middle ground.
    counts = np.bincount(y, minlength=5).astype(float)
    if a.class_weight == "none":
        cw = np.ones(5)
    elif a.class_weight == "balanced":
        cw = counts.sum() / (5 * counts)
    else:
        cw = np.sqrt(counts.sum() / (5 * counts))
    print("class weights:", {E.STAGES[i]: round(float(w), 3) for i, w in enumerate(cw)})

    splits, n = E.folds(groups, a.folds)
    os.makedirs(a.out, exist_ok=True)
    fp = dataset_fingerprint(d["recs"], len(y), a)
    print(f"run fingerprint: {fp}\n")

    oof = np.full(len(y), -1)
    importances = None
    for k, (tr, te) in enumerate(splits, 1):
        path = os.path.join(a.out, f"gbdt_fold{k}.npz")
        cached = load_fold(path, fp)
        if cached is not None:
            oof[cached["te"]] = cached["pred"]
            imp = cached["imp"]
            importances = imp if importances is None else importances + imp
            print(f"  fold {k}/{n}: cached  acc {(cached['pred'] == y[cached['te']]).mean():.4f}")
            continue
        if a.only_fold and k != a.only_fold:
            continue

        import time as _t
        t0 = _t.time()
        model, backend, can_stop = make_model(5, a.seed, a.n_estimators)
        n_used = a.n_estimators
        if can_stop and a.patience:
            m_fit, m_val = E.inner_split(groups[tr], a.val_frac, a.seed)
            fit_i, val_i = tr[m_fit], tr[m_val]
            n_used = fit_with_early_stopping(
                model, X[fit_i], y[fit_i], cw[y[fit_i]],
                X[val_i], y[val_i], cw[y[val_i]], a.patience, a.es_metric) or a.n_estimators
            if not a.no_refit:
                # Early stopping had to hold out 15% of the training patients to
                # find the stopping point. Once we know it, refit on the whole
                # training fold so no data is wasted -- the tree count is a
                # hyper-parameter chosen on validation, the model is fit on all.
                model, _, _ = make_model(5, a.seed, n_used)
                model.fit(X[tr], y[tr], sample_weight=cw[y[tr]])
        else:
            model.fit(X[tr], y[tr], sample_weight=cw[y[tr]])
        proba = model.predict_proba(X[te]).astype(np.float32)
        pred = proba.argmax(1)
        oof[te] = pred
        imp = getattr(model, "feature_importances_", np.zeros(X.shape[1])).astype(float)
        extra = {"proba": proba} if a.save_proba else {}
        np.savez(path, te=te, pred=pred, imp=imp, fingerprint=fp, **extra)
        importances = imp if importances is None else importances + imp
        print(f"  fold {k}/{n} [{backend}]: {len(tr):>6,} train / {len(te):>6,} test  "
              f"acc {(pred == y[te]).mean():.4f}  trees {n_used}  ({_t.time()-t0:.0f}s)")

    if (oof < 0).any():
        print(f"\n{int((oof < 0).sum()):,} epochs still unpredicted -- run the remaining folds")
        return

    m = E.report(y, oof, f"Gradient boosting -- {n}-fold, grouped by patient")

    if importances is not None:
        print("\n  top 20 features")
        for i in np.argsort(importances)[::-1][:20]:
            print(f"    {d['names'][i]:<34}{importances[i]/importances.sum():>7.3%}")

    np.savez(os.path.join(a.out, "gbdt_oof.npz"), y=y, pred=oof,
             groups=groups, recs=d["recs"], **m)
    print(f"\nsaved -> {a.out}/gbdt_oof.npz")


if __name__ == "__main__":
    main()
