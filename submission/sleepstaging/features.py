"""Per-epoch feature extraction.

One 30 s epoch becomes a vector of interpretable, mostly scale-free descriptors:
spectral band powers and their ratios, spectral shape, Hjorth parameters and
robust time-domain statistics. Sleep stages are defined by exactly these
properties in the AASM scoring manual (delta for N3, sigma spindles for N2, REM
atonia in the chin EMG, slow eye movements for N1), so hand-built features start
much closer to the answer than raw voltage does.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis, skew

TARGET_FS = 100
EPOCH_SEC = 30
EPOCH_LEN = TARGET_FS * EPOCH_SEC

EEG_CH = ("C3", "C4", "O1", "O2")
EOG_CH = ("E1", "E2")
EMG_CH = "EMG"

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
         "sigma": (12, 16), "beta": (16, 30)}


def resample_to(x: np.ndarray, fs_in: float) -> np.ndarray:
    if abs(fs_in - TARGET_FS) < 1e-6:
        return x.astype(np.float32)
    from fractions import Fraction
    fr = Fraction(TARGET_FS / fs_in).limit_denominator(1000)
    return sps.resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)


def bandpass(x, lo, hi, fs=TARGET_FS, order=4):
    sos = sps.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


def _hjorth(x, axis=-1):
    d1 = np.diff(x, axis=axis)
    d2 = np.diff(d1, axis=axis)
    v0 = x.var(axis=axis) + 1e-12
    v1 = d1.var(axis=axis) + 1e-12
    v2 = d2.var(axis=axis) + 1e-12
    mob = np.sqrt(v1 / v0)
    return mob, np.sqrt(v2 / v1) / mob


def _psd(epochs):
    """Welch PSD per epoch. 4 s Hann windows, 50% overlap -> 0.25 Hz bins."""
    f, p = sps.welch(epochs, fs=TARGET_FS, nperseg=4 * TARGET_FS,
                     noverlap=2 * TARGET_FS, axis=-1)
    return f, p


def _spectral_feats(epochs, prefix, out, names):
    f, p = _psd(epochs)
    total = np.trapezoid(p, f, axis=-1) + 1e-12
    bp = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp[name] = np.trapezoid(p[..., m], f[m], axis=-1)
        out.append(bp[name] / total); names.append(f"{prefix}_rel_{name}")
    out.append(np.log(total)); names.append(f"{prefix}_log_total")

    eps = 1e-12
    for a, b in [("delta", "beta"), ("theta", "alpha"), ("sigma", "theta"),
                 ("delta", "alpha"), ("alpha", "beta")]:
        out.append(np.log((bp[a] + eps) / (bp[b] + eps))); names.append(f"{prefix}_log_{a}_{b}")
    slow = bp["delta"] + bp["theta"]
    fast = bp["alpha"] + bp["beta"] + bp["sigma"]
    out.append(np.log((slow + eps) / (fast + eps))); names.append(f"{prefix}_log_slow_fast")

    pn = p / (p.sum(axis=-1, keepdims=True) + eps)
    out.append(-(pn * np.log(pn + eps)).sum(axis=-1) / np.log(pn.shape[-1]))
    names.append(f"{prefix}_spec_entropy")
    csum = np.cumsum(p, axis=-1)
    csum /= csum[..., -1:] + eps
    out.append(f[np.argmax(csum >= 0.95, axis=-1)]); names.append(f"{prefix}_sef95")
    return bp, total


def _time_feats(epochs, prefix, out, names):
    out.append(np.log(epochs.std(axis=-1) + 1e-12)); names.append(f"{prefix}_log_std")
    q75, q25 = np.percentile(epochs, [75, 25], axis=-1)
    out.append(np.log(q75 - q25 + 1e-12)); names.append(f"{prefix}_log_iqr")
    out.append(skew(epochs, axis=-1)); names.append(f"{prefix}_skew")
    out.append(kurtosis(epochs, axis=-1)); names.append(f"{prefix}_kurtosis")
    mob, comp = _hjorth(epochs)
    out.append(mob); names.append(f"{prefix}_hjorth_mobility")
    out.append(comp); names.append(f"{prefix}_hjorth_complexity")


def epoch_features(sig: dict, offsets_samples: np.ndarray):
    """Build the base feature matrix for one recording.

    `sig` maps canonical channel name -> filtered 100 Hz signal.
    Returns (X, feature_names) with X shaped (n_epochs, n_features).
    """
    idx = offsets_samples[:, None] + np.arange(EPOCH_LEN)[None, :]
    out, names = [], []

    for ch in EEG_CH:
        e = sig[ch][idx]
        _spectral_feats(e, ch, out, names)
        _time_feats(e, ch, out, names)

    eog = {}
    for ch in EOG_CH:
        e = sig[ch][idx]
        eog[ch] = e
        f, p = _psd(e)
        total = np.trapezoid(p, f, axis=-1) + 1e-12
        for lo, hi, tag in [(0.3, 2, "slow"), (2, 5, "fast")]:
            m = (f >= lo) & (f < hi)
            out.append(np.trapezoid(p[..., m], f[m], axis=-1) / total)
            names.append(f"{ch}_rel_{tag}")
        _time_feats(e, ch, out, names)

    a, b = eog["E1"], eog["E2"]
    az, bz = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    denom = np.sqrt((az ** 2).sum(-1) * (bz ** 2).sum(-1)) + 1e-12
    # REM bursts move the eyes in opposite directions, so E1/E2 anti-correlate.
    out.append((az * bz).sum(-1) / denom); names.append("EOG_corr")

    emg = sig[EMG_CH][idx]
    out.append(np.log(np.sqrt((emg ** 2).mean(-1)) + 1e-12)); names.append("EMG_log_rms")
    f, p = _psd(emg)
    total = np.trapezoid(p, f, axis=-1) + 1e-12
    for lo, hi, tag in [(10, 20, "low"), (20, 45, "high")]:
        m = (f >= lo) & (f < hi)
        out.append(np.trapezoid(p[..., m], f[m], axis=-1) / total)
        names.append(f"EMG_rel_{tag}")
    _time_feats(emg, "EMG", out, names)

    X = np.column_stack([np.asarray(c, dtype=np.float32) for c in out])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def _rolling_mean(X, k):
    """Centred moving average over +/-k epochs, edge-padded."""
    if k == 0:
        return X
    pad = np.pad(X, ((k, k), (0, 0)), mode="edge")
    ker = np.ones(2 * k + 1, dtype=np.float32) / (2 * k + 1)
    return np.stack([np.convolve(pad[:, j], ker, mode="valid")
                     for j in range(X.shape[1])], axis=1).astype(np.float32)


def add_context(X, names):
    """Robust per-recording scaling plus temporal context.

    Two things matter here. Per-recording scaling removes between-subject and
    between-device amplitude differences, which otherwise dominate the split
    criteria. Temporal smoothing encodes the fact that sleep stages come in runs
    of minutes -- a single 30 s epoch is genuinely ambiguous, and human scorers
    also look at neighbouring epochs.
    """
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    Z = (X - med) / (iqr + 1e-9)

    parts = [X, Z, _rolling_mean(Z, 2), _rolling_mean(Z, 7)]
    out_names = (list(names)
                 + [f"{n}_z" for n in names]
                 + [f"{n}_z_sm5" for n in names]
                 + [f"{n}_z_sm15" for n in names])

    n = X.shape[0]
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1))[:, None]
    hours = (np.arange(n, dtype=np.float32) * EPOCH_SEC / 3600.0)[:, None]
    parts += [pos, hours]
    out_names += ["epoch_position", "hours_from_start"]

    return np.nan_to_num(np.hstack(parts).astype(np.float32)), out_names
