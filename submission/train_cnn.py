#!/usr/bin/env python3
"""Step 2b: a convolutional + recurrent network on the raw signal.

Architecture, and why each piece is there:

  Representation learner -- two parallel 1D conv branches over the raw 30 s
  epoch, the DeepSleepNet/TinySleepNet design. A short-kernel branch (0.5 s)
  resolves transient graphoelements such as spindles and K-complexes; a
  long-kernel branch (4 s) resolves slow rhythms such as delta. One kernel size
  cannot do both well, which is why the split exists rather than a single stack.

  Sequence learner -- a bidirectional GRU over 20 consecutive epoch embeddings.
  Sleep stages persist for minutes and the AASM rules are explicitly contextual
  (an epoch is N2 partly because of what preceded it). This is the neural
  equivalent of the smoothed context features the tree model gets, except it is
  learned rather than hand-specified, and bidirectional so it can use both past
  and future -- legitimate here because we score whole nights offline.

    python train_cnn.py --cache cache --folds 1 --epochs 15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleepstaging"))
import sleepstaging.evaluate as E  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise SystemExit("PyTorch is required for this script.\n"
                     "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                     "If your Python is too new for a torch wheel, use a 3.12 virtualenv.")


class SeqDataset(Dataset):
    """Non-overlapping windows of `seq_len` epochs, never crossing a recording."""

    def __init__(self, raw_list, y_list, seq_len):
        self.raw, self.y, self.seq_len, self.index = raw_list, y_list, seq_len, []
        for r, arr in enumerate(raw_list):
            for s in E.window_starts(len(arr), seq_len):
                self.index.append((r, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r, s = self.index[i]
        arr, lab = self.raw[r], self.y[r]
        x = np.asarray(arr[s:s + self.seq_len], dtype=np.float32)
        t = np.asarray(lab[s:s + self.seq_len], dtype=np.int64)
        if len(x) < self.seq_len:                       # short tail: edge-pad
            pad = self.seq_len - len(x)
            x = np.concatenate([x, np.repeat(x[-1:], pad, 0)])
            t = np.concatenate([t, np.full(pad, -100, dtype=np.int64)])
        return torch.from_numpy(x), torch.from_numpy(t)


def conv_branch(in_ch, kernel, stride, pool1, pool2, width=64):
    return nn.Sequential(
        nn.Conv1d(in_ch, width, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(width), nn.ReLU(),
        nn.MaxPool1d(pool1), nn.Dropout(0.3),
        nn.Conv1d(width, width * 2, 8, padding=4, bias=False),
        nn.BatchNorm1d(width * 2), nn.ReLU(),
        nn.Conv1d(width * 2, width * 2, 8, padding=4, bias=False),
        nn.BatchNorm1d(width * 2), nn.ReLU(),
        nn.MaxPool1d(pool2), nn.AdaptiveAvgPool1d(1),
    )


class SleepNet(nn.Module):
    def __init__(self, n_ch, fs=100, n_classes=5, hidden=128):
        super().__init__()
        self.fine = conv_branch(n_ch, fs // 2, fs // 16, 8, 4)     # transients
        self.coarse = conv_branch(n_ch, fs * 4, fs // 2, 4, 2)     # slow rhythms
        self.drop = nn.Dropout(0.5)
        self.rnn = nn.GRU(256, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):                       # x: (B, L, C, T)
        b, l, c, t = x.shape
        x = x.reshape(b * l, c, t)
        z = torch.cat([self.fine(x).squeeze(-1), self.coarse(x).squeeze(-1)], dim=1)
        z = self.drop(z).reshape(b, l, -1)
        z, _ = self.rnn(z)
        return self.head(z)                     # (B, L, n_classes)


def predict_recordings(model, recs, raw, ys, seq_len, device):
    """Windowed inference over whole recordings; returns per-epoch posteriors."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in recs:
            ds = SeqDataset([raw[i]], [ys[i]], seq_len)
            pr = np.zeros((len(ys[i]), 5), dtype=np.float32)
            for j in range(len(ds)):
                xb, _ = ds[j]
                o = torch.softmax(model(xb.unsqueeze(0).to(device))[0], -1).cpu().numpy()
                _, st = ds.index[j]
                take = min(seq_len, len(ys[i]) - st)
                pr[st:st + take] = o[:take]
            out.append(pr)
    return out


def run_fold(tr_recs, te_recs, raw, ys, args, weights, device, tr_groups):
    """Train one fold with early stopping on a patient-disjoint validation split.

    The stopping epoch is a hyper-parameter, so it cannot be chosen on the test
    fold. It also cannot be chosen on a random slice of training epochs -- epochs
    from the same night are near-duplicates -- so the split is by patient, like
    the outer CV. We monitor validation kappa rather than loss: on this task the
    loss keeps drifting while kappa plateaus, and kappa is what we report.
    """
    from sklearn.metrics import cohen_kappa_score
    import copy

    if args.patience:
        m_fit, m_val = E.inner_split(np.array(tr_groups), args.val_frac, 0)
        fit_recs = [r for r, keep in zip(tr_recs, m_fit) if keep]
        val_recs = [r for r, keep in zip(tr_recs, m_val) if keep]
    else:
        fit_recs, val_recs = tr_recs, []

    model = SleepNet(raw[0].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-100)
    dl = DataLoader(SeqDataset([raw[i] for i in fit_recs], [ys[i] for i in fit_recs], args.seq_len),
                    batch_size=args.batch, shuffle=True, num_workers=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_k, best_state, since = -np.inf, None, 0
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
        msg = f"    epoch {ep:>2}/{args.epochs}  loss {tot/max(n,1):.4f}"

        if val_recs:
            pr = predict_recordings(model, val_recs, raw, ys, args.seq_len, device)
            vy = np.concatenate([ys[i] for i in val_recs])
            vk = cohen_kappa_score(vy, np.concatenate(pr).argmax(1))
            msg += f"  val kappa {vk:.4f}"
            if vk > best_k + 1e-4:
                best_k, since = vk, 0
                best_state = copy.deepcopy(model.state_dict())
                msg += "  *"
            else:
                since += 1
        print(msg + f"  ({time.time()-t0:.0f}s)")
        if val_recs and args.patience and since >= args.patience:
            print(f"    early stop at epoch {ep}; best val kappa {best_k:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pr = predict_recordings(model, te_recs, raw, ys, args.seq_len, device)
    return (np.concatenate(pr).argmax(1),
            np.concatenate([ys[i] for i in te_recs]),
            np.concatenate(pr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--folds", type=int, default=1, help="CNN folds to run (5 = full CV)")
    ap.add_argument("--total-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60, help="maximum epochs; early stopping usually ends sooner")
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many epochs without val-kappa improvement; 0 disables")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of TRAINING patients held out to pick the stopping epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--save-proba", action="store_true",
                    help="also store per-class posteriors (needed for Viterbi smoothing and stacking)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()

    d = E.load_cache(a.cache, want_raw=True)
    lens = d["raw_lengths"]
    bounds = np.cumsum([0] + lens)
    ys = [d["y"][bounds[i]:bounds[i + 1]] for i in range(len(lens))]
    rec_group = [d["groups"][bounds[i]] for i in range(len(lens))]
    rec_name = [d["recs"][bounds[i]] for i in range(len(lens))]
    raw = d["raw"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{sum(lens):,} epochs | {len(lens)} recordings | {len(set(rec_group))} patients | device={device}")
    if device == "cpu":
        print("NOTE: no GPU detected. This will be slow -- consider --epochs 5 for a smoke test.")

    counts = np.bincount(d["y"], minlength=5).astype(float)
    weights = torch.tensor(np.sqrt(counts.sum() / (5 * counts)), dtype=torch.float32)

    # Fold the recordings the same way the tree model folds the epochs, so the
    # two models are scored on identical partitions.
    rec_folds, n = E.recording_folds(d["groups"], rec_group, a.total_folds)

    fp = hashlib.md5(json.dumps({
        "recordings": sorted(map(str, rec_name)), "n_epochs": int(sum(lens)),
        "folds": a.total_folds, "epochs": a.epochs, "seq_len": a.seq_len,
        "lr": a.lr, "batch": a.batch, "proba": bool(a.save_proba),
        "patience": a.patience, "val_frac": a.val_frac,
    }, sort_keys=True).encode()).hexdigest()[:16]
    print(f"run fingerprint: {fp}")
    os.makedirs(a.out, exist_ok=True)

    all_pred, all_true = [], []
    for k, (tr_recs, te_recs) in enumerate(rec_folds[:a.folds], 1):
        path = os.path.join(a.out, f"cnn_fold{k}.npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            if str(z.get("fingerprint", "")) == fp:
                all_pred.append(z["pred"]); all_true.append(z["true"])
                print(f"\n  fold {k}/{n}: cached  acc {(z['pred'] == z['true']).mean():.4f}")
                continue
        print(f"\n  fold {k}/{n}: {len(tr_recs)} train / {len(te_recs)} test recordings, "
              f"{sum(lens[i] for i in te_recs):,} test epochs")
        p, t, pr = run_fold(tr_recs, te_recs, raw, ys, a, weights, device,
                            [rec_group[i] for i in tr_recs])
        extra = {"proba": pr} if a.save_proba else {}
        np.savez(path, pred=p, true=t, fingerprint=fp, **extra)
        all_pred.append(p); all_true.append(t)
        print(f"    fold accuracy {(p == t).mean():.4f}")

    y_true, y_pred = np.concatenate(all_true), np.concatenate(all_pred)
    m = E.report(y_true, y_pred, f"CNN + BiGRU -- {a.folds} of {n} folds, grouped by patient")
    np.savez(os.path.join(a.out, "cnn_oof.npz"), y=y_true, pred=y_pred, **m)
    print(f"\nsaved -> {a.out}/cnn_oof.npz")


if __name__ == "__main__":
    main()
