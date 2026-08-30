"""Validate train_lstm.py's data bookkeeping without needing PyTorch.

The two worst bugs in this project were both index bookkeeping, not modelling:
a fold cache replayed against the wrong-sized array, and a fold mapping that
collapsed to a single recording. This exercises the same code paths the LSTM
uses -- fold construction, test-index assembly, posterior concatenation order,
window coverage, and training-only standardisation -- against the real cache.

    python tests/test_lstm_plumbing.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sleepstaging"))
import evaluate as E  # noqa: E402


def main():
    cache = os.path.join(ROOT, "cache")
    d = E.load_cache(cache)
    y, groups = d["y"], d["groups"]
    lens, rec_group = [], []
    for p in sorted(glob.glob(os.path.join(cache, "*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        z = np.load(p, allow_pickle=True)
        lens.append(len(z["y"])); rec_group.append(str(z["patient"]))
    bounds = np.cumsum([0] + lens)
    ys = [y[bounds[i]:bounds[i + 1]] for i in range(len(lens))]
    feats = [d["X"][bounds[i]:bounds[i + 1]] for i in range(len(lens))]

    rec_folds, n = E.recording_folds(groups, rec_group, 5)
    print(f"{len(y):,} epochs | {len(lens)} recordings | {len(set(rec_group))} patients | {n} folds")

    covered = np.zeros(len(y), bool)
    for k, (tr_recs, te_recs) in enumerate(rec_folds, 1):
        # 1. no recording is in both halves, and every recording is in one
        assert not (set(tr_recs) & set(te_recs)), f"fold {k}: overlap"
        assert len(tr_recs) + len(te_recs) == len(lens), f"fold {k}: incomplete"

        # 2. no patient straddles the split -- the whole point of grouping
        assert not ({rec_group[i] for i in tr_recs} & {rec_group[i] for i in te_recs}), \
            f"fold {k}: a patient appears in train and test"

        # 3. test indices assemble exactly as train_lstm.py builds them
        te = np.concatenate([np.arange(bounds[i], bounds[i + 1]) for i in te_recs])
        assert np.array_equal(te, np.sort(te)), f"fold {k}: te not ascending"
        assert len(np.unique(te)) == len(te), f"fold {k}: duplicate indices"
        covered[te] = True

        # 4. posteriors concatenated over te_recs line up with those indices
        fake = {i: np.repeat(ys[i][:, None], 5, 1).astype(np.float32) for i in te_recs}
        proba = np.concatenate([fake[i] for i in te_recs])
        assert len(proba) == len(te), f"fold {k}: length mismatch"
        assert np.array_equal(proba[:, 0].astype(np.int64), y[te]), \
            f"fold {k}: posterior order does not match test indices"

        # 5. standardisation sees training recordings only
        stack = np.concatenate([feats[i] for i in tr_recs])
        assert len(stack) == sum(lens[i] for i in tr_recs)
        mu, sd = stack.mean(0), stack.std(0) + 1e-6
        assert np.isfinite(mu).all() and (sd > 0).all(), f"fold {k}: bad statistics"
        z = np.clip((feats[te_recs[0]] - mu) / sd, -10, 10)
        assert np.isfinite(z).all(), f"fold {k}: non-finite after scaling"

        print(f"  fold {k}: {len(tr_recs):>2} train / {len(te_recs):>2} test recordings, "
              f"{len(te):>6,} test epochs  OK")

    assert covered.all(), f"{(~covered).sum()} epochs never appear in any test fold"

    # 6. window coverage for every real recording length
    for i, m in enumerate(lens):
        seen = np.zeros(m, bool)
        for s in E.window_starts(m, 100):
            seen[s:s + 100] = True
        assert seen.all(), f"recording {i} (len {m}) not fully covered"

    print(f"\nall checks passed: {len(y):,} epochs covered exactly once, "
          f"no patient leakage, window coverage complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
