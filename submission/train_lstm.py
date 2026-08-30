#!/usr/bin/env python3
"""A BiLSTM over the engineered feature sequence.

The dataset paper's strongest baseline was an LSTM at 74.70% accuracy, run on
the signals. This takes the same architectural idea but feeds it the 426
per-epoch features instead, which is the representation that already beats that
baseline on its own. The intent is to combine the two things that have worked:
domain features for *what an epoch looks like*, and a recurrent model for *how a
night is structured*.

Why this is worth trying after the CNN lost:

  - The CNN had to learn a spectral representation from 3,000 raw samples with
    86 patients of data. This model gets that representation for free and spends
    all of its capacity on temporal structure.
  - The tree model scores each epoch semi-independently. Its two largest
    remaining error modes -- 26% of N3 and 18% of REM leaking to N2 -- are
    stage-persistence failures. An LSTM addresses precisely that.
  - Sequences are cheap here (426 numbers per epoch, not 7x3000 samples), so it
    can see 100 epochs of context -- roughly 50 minutes -- against the CNN's 20.

    python train_lstm.py --folds 5 --save-proba
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleepstaging"))
import evaluate as E  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise SystemExit("PyTorch is required for this script.\n"
                     "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
                     "If your Python is too new for a torch wheel, use a 3.12 virtualenv.")


class SeqDataset(Dataset):
    """Windows of `seq_len` epochs that never cross a recording boundary."""

    def __init__(self, feats, labels, seq_len):
        self.feats, self.labels, self.seq_len, self.index = feats, labels, seq_len, []
        for r, arr in enumerate(feats):
            for s in E.window_starts(len(arr), seq_len):
                self.index.append((r, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r, s = self.index[i]
        x = np.asarray(self.feats[r][s:s + self.seq_len], dtype=np.float32)
        t = np.asarray(self.labels[r][s:s + self.seq_len], dtype=np.int64)
        if len(x) < self.seq_len:                       # short recording: pad
            pad = self.seq_len - len(x)
            x = np.concatenate([x, np.zeros((pad, x.shape[1]), np.float32)])
            t = np.concatenate([t, np.full(pad, -100, dtype=np.int64)])
        return torch.from_numpy(x), torch.from_numpy(t)


class SleepLSTM(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2, n_classes=5, dropout=0.3):
        super().__init__()
        # A projection before the recurrence: 426 correlated inputs are a poor
        # thing to feed a gate directly, and this lets the model build a compact
        # mixture of them first.
        self.proj = nn.Sequential(
            nn.Linear(n_features, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout))
        self.rnn = nn.LSTM(256, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):                    # (B, L, F)
        z, _ = self.rnn(self.proj(x))
        return self.head(z)                  # (B, L, C)


def run_fold(tr_recs, te_recs, feats, ys, args, weights, device):
    # Standardise on TRAINING recordings only. The features are partly raw
    # magnitudes, and an LSTM needs them scaled; computing the statistics over
    # everything would leak the test fold's distribution into training.
    stack = np.concatenate([feats[i] for i in tr_recs])
    mu = stack.mean(0)
    sd = stack.std(0) + 1e-6
    del stack
    norm = [np.clip((f - mu) / sd, -10, 10).astype(np.float32) for f in feats]

    model = SleepLSTM(feats[0].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-100)
    dl = DataLoader(SeqDataset([norm[i] for i in tr_recs], [ys[i] for i in tr_recs], args.seq_len),
                    batch_size=args.batch, shuffle=True, num_workers=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        model.train(); tot = n = 0; t0 = time.time()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb).reshape(-1, 5), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        sched.step()
        print(f"    epoch {ep:>2}/{args.epochs}  loss {tot/max(n,1):.4f}  ({time.time()-t0:.0f}s)")

    model.eval()
    out = {}
    with torch.no_grad():
        for i in te_recs:
            ds = SeqDataset([norm[i]], [ys[i]], args.seq_len)
            pr = np.zeros((len(ys[i]), 5), dtype=np.float32)
            for j in range(len(ds)):
                xb, _ = ds[j]
                o = torch.softmax(model(xb.unsqueeze(0).to(device))[0], -1).cpu().numpy()
                _, s = ds.index[j]
                take = min(args.seq_len, len(ys[i]) - s)
                pr[s:s + take] = o[:take]
            out[i] = pr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--save-proba", action="store_true")
    a = ap.parse_args()

    d = E.load_cache(a.cache)
    y, groups = d["y"], d["groups"]
    lens, rec_group, rec_name = [], [], []
    for p in sorted(glob.glob(os.path.join(a.cache, "*.npz")),
                    key=lambda p: int(os.path.basename(p)[2:-4])):
        z = np.load(p, allow_pickle=True)
        lens.append(len(z["y"])); rec_group.append(str(z["patient"])); rec_name.append(str(z["rec"]))
    bounds = np.cumsum([0] + lens)
    feats = [d["X"][bounds[i]:bounds[i + 1]] for i in range(len(lens))]
    ys = [y[bounds[i]:bounds[i + 1]] for i in range(len(lens))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(y):,} epochs x {d['X'].shape[1]} features | {len(lens)} recordings | "
          f"{len(set(rec_group))} patients | device={device}")
    if device == "cpu":
        print("NOTE: no GPU detected; this will be slow.")

    counts = np.bincount(y, minlength=5).astype(float)
    weights = torch.tensor(np.sqrt(counts.sum() / (5 * counts)), dtype=torch.float32)

    # Same folds as every other model in this project.
    rec_folds, n = E.recording_folds(groups, rec_group, 5)
    fp = hashlib.md5(json.dumps({
        "recordings": sorted(map(str, rec_name)), "n_epochs": int(len(y)),
        "folds": 5, "epochs": a.epochs, "seq_len": a.seq_len,
        "lr": a.lr, "batch": a.batch, "proba": bool(a.save_proba),
    }, sort_keys=True).encode()).hexdigest()[:16]
    print(f"run fingerprint: {fp}")
    os.makedirs(a.out, exist_ok=True)

    oof = np.full(len(y), -1)
    for k, (tr_recs, te_recs) in enumerate(rec_folds[:a.folds], 1):
        path = os.path.join(a.out, f"lstm_fold{k}.npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            if str(z.get("fingerprint", "")) == fp:
                oof[z["te"]] = z["pred"]
                print(f"\n  fold {k}/{n}: cached  acc {(z['pred'] == y[z['te']]).mean():.4f}")
                continue
        print(f"\n  fold {k}/{n}: {len(tr_recs)} train / {len(te_recs)} test recordings, "
              f"{sum(lens[i] for i in te_recs):,} test epochs")
        t0 = time.time()
        pr = run_fold(tr_recs, te_recs, feats, ys, a, weights, device)
        te = np.concatenate([np.arange(bounds[i], bounds[i + 1]) for i in te_recs])
        proba = np.concatenate([pr[i] for i in te_recs]).astype(np.float32)
        pred = proba.argmax(1)
        oof[te] = pred
        extra = {"proba": proba} if a.save_proba else {}
        np.savez(path, te=te, pred=pred, fingerprint=fp, **extra)
        print(f"    fold accuracy {(pred == y[te]).mean():.4f}  ({time.time()-t0:.0f}s)")

    if (oof < 0).any():
        print(f"\n{int((oof < 0).sum()):,} epochs unpredicted -- run the remaining folds")
        return
    m = E.report(y, oof, f"BiLSTM over features -- {a.folds} of {n} folds, grouped by patient")
    np.savez(os.path.join(a.out, "lstm_oof.npz"), y=y, pred=oof, groups=groups, recs=d["recs"], **m)
    print(f"\nsaved -> {os.path.join(a.out, 'lstm_oof.npz')}")


if __name__ == "__main__":
    main()
