#!/usr/bin/env python3
"""Step 1 of the pipeline: turn EDF + hypnogram into a per-epoch feature cache.

This is the only slow step. It is resumable -- each recording is cached to its
own .npz, so interrupting and re-running picks up where it left off.

    python extract_features.py --subjects pilot          # 20 patients, ~2 min
    python extract_features.py --subjects all            # 99 recordings
    python extract_features.py --subjects all --raw      # also cache raw epochs
                                                         # for the CNN (~4 GB)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleepstaging"))
import sleepstaging.features as F          # noqa: E402
import sleepstaging.sleep_io as io         # noqa: E402
import sleepstaging.subjects as S          # noqa: E402


def process(rec, dataset, cache, patient, want_raw):
    edf = os.path.join(dataset, rec + ".edf")
    xls = os.path.join(dataset, rec + ".xlsx")
    hdr = io.read_edf_header(edf)

    stages, times = io.read_hypnogram(xls)
    y, offs_sec = io.align_epochs(stages, times, hdr.start_seconds, hdr.duration)
    if len(y) == 0:
        raise RuntimeError("no epochs survived alignment")

    sig = {}
    for ch in io.CORE_CHANNELS:
        i = io.channel_index(hdr, ch)
        if i is None:
            raise RuntimeError(f"missing channel {ch}")
        x = F.resample_to(io.read_edf_channel(edf, hdr, i), hdr.fs[i])
        # EMG carries information at higher frequencies than EEG/EOG, and the
        # low-frequency drift there is movement artefact, so it gets its own band.
        sig[ch] = F.bandpass(x, 10, 45) if ch == "EMG" else F.bandpass(x, 0.3, 35)

    n_samples = min(len(v) for v in sig.values())
    offs = np.round(offs_sec * F.TARGET_FS).astype(np.int64)
    ok = (offs >= 0) & (offs + F.EPOCH_LEN <= n_samples)
    offs, y = offs[ok], y[ok]

    X, names = F.epoch_features(sig, offs)
    X, names = F.add_context(X, names)

    np.savez_compressed(os.path.join(cache, rec + ".npz"),
                        X=X, y=y, names=np.array(names), rec=rec, patient=patient)

    if want_raw:
        # Raw epochs live in their own uncompressed .npy: they are large and
        # incompressible, and zipping them costs far more time than it saves.
        rawdir = os.path.join(cache, "raw")
        os.makedirs(rawdir, exist_ok=True)
        idx = offs[:, None] + np.arange(F.EPOCH_LEN)[None, :]
        arr = np.stack([sig[c][idx] for c in io.CORE_CHANNELS], axis=1)
        # Robust per-recording scaling makes recordings comparable despite
        # different amplifiers; float16 halves the cache with no practical loss.
        scale = np.percentile(np.abs(arr), 95, axis=(0, 2), keepdims=True) + 1e-6
        np.save(os.path.join(rawdir, rec + ".npy"),
                np.clip(arr / scale, -10, 10).astype(np.float16))
    return len(y), X.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "Dataset"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--subjects", default="pilot", help="'pilot', 'all', or SN1,SN2,...")
    ap.add_argument("--n-pilot", type=int, default=20)
    ap.add_argument("--raw", action="store_true", help="also cache raw epochs for the CNN")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    desc = os.path.join(a.dataset, "subject_description.xlsx")
    recs, patient_of = S.usable_recordings(a.dataset, desc)
    if a.subjects == "pilot":
        recs = S.pilot_subset(recs, patient_of, a.n_pilot)
    elif a.subjects != "all":
        want = {s.strip() for s in a.subjects.split(",")}
        recs = [r for r in recs if r in want]

    os.makedirs(a.cache, exist_ok=True)
    print(f"{len(recs)} recordings -> {a.cache}/  (raw={a.raw})")
    t0, total = time.time(), 0
    for i, rec in enumerate(recs, 1):
        out = os.path.join(a.cache, rec + ".npz")
        raw_ok = (not a.raw) or os.path.exists(os.path.join(a.cache, "raw", rec + ".npy"))
        if os.path.exists(out) and raw_ok and not a.force:
            print(f"  [{i:>3}/{len(recs)}] {rec:<7} cached")
            continue
        t = time.time()
        try:
            n, d = process(rec, a.dataset, a.cache, patient_of[rec], a.raw)
            total += n
            print(f"  [{i:>3}/{len(recs)}] {rec:<7} {n:>5} epochs  {d:>4} features  {time.time()-t:5.1f}s")
        except Exception as e:
            print(f"  [{i:>3}/{len(recs)}] {rec:<7} FAILED: {e}")
    print(f"\ndone in {time.time()-t0:.0f}s, {total:,} new epochs")


if __name__ == "__main__":
    main()
