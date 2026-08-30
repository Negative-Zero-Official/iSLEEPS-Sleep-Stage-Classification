"""Reading the iSLEEPS dataset: EDF signals, hypnograms, and patient grouping.

Deliberately dependency-light: numpy only. EDF and xlsx are both parsed directly
so the pipeline does not need mne or openpyxl.
"""
from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- channels ---
# 36 of the 100 recordings use the older A1/A2 reference nomenclature and 64 use
# M1/M2. They are the same derivations (A1==M1, A2==M2, both mastoids), so we
# normalise to a single canonical name. Without this, no EEG channel appears in
# all 100 files and you silently lose 36 recordings.
CHANNEL_ALIASES = {
    "C3:M2": "C3", "C3:A2": "C3",
    "C4:M1": "C4", "C4:A1": "C4",
    "O1:M2": "O1", "O1:A2": "O1",
    "O2:M1": "O2", "O2:A1": "O2",
    "E1:M2": "E1", "EOG1:A2": "E1",
    "E2:M2": "E2", "EOG2:A2": "E2",
    "EMG": "EMG", "Chin 1": "EMG",
    "EMG2": "EMG2", "Chin 2": "EMG2",
    "F3:M2": "F3", "F3:A2": "F3",
    "F4:M1": "F4", "F4:A1": "F4",
}

# Present in all 100 recordings. F3/F4 are only in 87, so they are excluded.
CORE_CHANNELS = ("C3", "C4", "O1", "O2", "E1", "E2", "EMG")

STAGES = ("Wake", "N1", "N2", "N3", "REM")
STAGE_TO_INT = {s: i for i, s in enumerate(STAGES)}
EPOCH_SEC = 30


# --------------------------------------------------------------------- EDF ---
@dataclass
class EdfHeader:
    patient: str
    start_seconds: float          # seconds since midnight
    n_records: int
    record_duration: float
    labels: list
    fs: list                      # sampling rate per signal
    phys_min: list
    phys_max: list
    dig_min: list
    dig_max: list
    header_bytes: int

    @property
    def duration(self) -> float:
        return self.n_records * self.record_duration


def read_edf_header(path: str) -> EdfHeader:
    with open(path, "rb") as f:
        h = f.read(256)
        n_sig = int(h[252:256])
        s = f.read(256 * n_sig)

    def field(off, width):
        return [s[off * n_sig + i * width: off * n_sig + (i + 1) * width].decode("latin1").strip()
                for i in range(n_sig)]

    hh, mm, ss = (int(x) for x in h[176:184].decode().split("."))
    dur = float(h[244:252])
    n_samp = [int(x) for x in field(216, 8)]
    return EdfHeader(
        patient=h[8:88].decode("latin1").strip(),
        start_seconds=hh * 3600 + mm * 60 + ss,
        n_records=int(h[236:244]),
        record_duration=dur,
        labels=field(0, 16),
        fs=[n / dur for n in n_samp],
        phys_min=[float(x) for x in field(104, 8)],
        phys_max=[float(x) for x in field(112, 8)],
        dig_min=[float(x) for x in field(120, 8)],
        dig_max=[float(x) for x in field(128, 8)],
        header_bytes=int(h[184:192]),
    )


def read_edf_channel(path: str, hdr: EdfHeader, index: int) -> np.ndarray:
    """Read one signal, scaled to physical units, as float32.

    Uses a memmap + strided view so we touch only the columns we need instead of
    loading a ~180 MB recording into RAM for one channel.
    """
    n_samp = [int(round(f * hdr.record_duration)) for f in hdr.fs]
    per_record = sum(n_samp)
    start = sum(n_samp[:index])
    raw = np.memmap(path, dtype="<i2", mode="r", offset=hdr.header_bytes,
                    shape=(hdr.n_records, per_record))
    dig = np.asarray(raw[:, start:start + n_samp[index]], dtype=np.float32).ravel()
    dmin, dmax = hdr.dig_min[index], hdr.dig_max[index]
    pmin, pmax = hdr.phys_min[index], hdr.phys_max[index]
    scale = (pmax - pmin) / (dmax - dmin) if dmax != dmin else 1.0
    return (dig - dmin) * scale + pmin


def channel_index(hdr: EdfHeader, canonical: str):
    for i, lab in enumerate(hdr.labels):
        if CHANNEL_ALIASES.get(lab) == canonical:
            return i
    return None


# --------------------------------------------------------------- hypnogram ---
_CELL = re.compile(r"<c[^>]*?(?:/>|>.*?</c>)", re.S)
_ROW = re.compile(r"<row[^>]*>(.*?)</row>", re.S)


def _cell_values(row_xml: str, shared: list) -> dict:
    """Return {column_letter: value} for one row."""
    out = {}
    for c in _CELL.findall(row_xml):
        ref = re.search(r'r="([A-Z]+)\d+"', c)
        val = re.search(r"<v>(.*?)</v>", c, re.S)
        if not ref or not val:
            continue
        typ = re.search(r't="(\w+)"', c)
        raw = val.group(1)
        out[ref.group(1)] = shared[int(raw)] if (typ and typ.group(1) == "s") else raw
    return out


def read_hypnogram(path: str):
    """Return (stage_names, epoch_start_seconds_of_day) from a scoring workbook.

    The layout is not fixed. Sheet 1 is usually the hypnogram
    (`Signal ID = SchlafProfil\\profil`, stages in column B), but in some
    recordings sheet 1 is a different channel entirely -- SN80 leads with the
    light sensor in lux and carries the stage in a separate "Sleep stage"
    column. So rather than trusting position, we scan every sheet and every
    column and keep whichever column actually contains stage names.
    """
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        shared = re.findall(r"<t[^>]*>(.*?)</t>",
                            z.read("xl/sharedStrings.xml").decode("utf8"), re.S)

    best = None
    for entry in z.namelist():
        if "/worksheets/sheet" not in entry:
            continue
        rows = _ROW.findall(z.read(entry).decode("utf8", "ignore"))
        parsed, start = [], None
        for i, r in enumerate(rows):
            cells = _cell_values(r, shared)
            if start is None:
                if cells.get("A") == "Time":
                    start = i + 1
                continue
            parsed.append(cells)
        if start is None or not parsed:
            continue

        cols = {c for row in parsed[:200] for c in row if c != "A"}
        for col in cols:
            stages, times, hits = [], [], 0
            for row in parsed:
                try:
                    serial = float(row.get("A", ""))
                except (TypeError, ValueError):
                    continue
                val = row.get(col, "")
                stages.append(val)
                times.append((serial % 1.0) * 86400.0)
                hits += val in STAGE_TO_INT
            if hits and (best is None or hits > best[0]):
                best = (hits, stages, np.asarray(times, dtype=np.float64))

    if best is None:
        return [], np.zeros(0)
    return best[1], best[2]


def align_epochs(stages, epoch_times, edf_start_seconds, edf_duration):
    """Map scored epochs onto sample offsets in the recording.

    Both clocks are seconds-of-day, and recordings cross midnight, so a negative
    delta of more than 12 h is a day wrap rather than an epoch before the start.
    """
    delta = epoch_times - edf_start_seconds
    delta = np.where(delta < -43200, delta + 86400, delta)
    keep = (delta >= 0) & (delta + EPOCH_SEC <= edf_duration)
    keep &= np.array([s in STAGE_TO_INT for s in stages])
    idx = np.nonzero(keep)[0]
    return (np.array([STAGE_TO_INT[stages[i]] for i in idx], dtype=np.int64),
            delta[idx])
