"""Recording -> patient mapping.

Two traps in this dataset, both of which inflate scores if ignored:

1. `subject_description.xlsx` has no SN11..SN20 rows. It has AN1..AN10 instead,
   and the EDF headers confirm the mapping (SN11.edf carries header id "SN1",
   i.e. AN1). Joining on filename mislabels ten recordings.

2. Several patients appear under more than one recording ID -- same age, sex,
   occupation, diagnosis and AHI across all 63 metadata columns, usually a
   second night. SN15 and SN28 are the extreme case: byte-identical signal data
   and an identical hypnogram. Splitting by recording therefore puts the same
   patient in train and test, which quietly inflates every metric.
"""
from __future__ import annotations

import os
import re
import zipfile
from collections import defaultdict

# The ten AN rows are stored on disk as SN11..SN20.
AN_TO_FILE = {f"AN{i}": f"SN{i + 10}" for i in range(1, 11)}

# SN15 and SN28 are the same recording (identical signal payload and hypnogram).
# Keep one; keeping SN28 because its EDF header id matches its filename.
EXACT_DUPLICATE_RECORDINGS = {"SN15"}


def _read_rows(path):
    z = zipfile.ZipFile(path)
    shared = re.findall(r"<t[^>]*>(.*?)</t>",
                        z.read("xl/sharedStrings.xml").decode("utf8"), re.S)
    rows = []
    for r in re.findall(r"<row[^>]*>(.*?)</row>",
                        z.read("xl/worksheets/sheet1.xml").decode("utf8", "ignore"), re.S):
        cells = []
        for c in re.findall(r"<c[^>]*?(?:/>|>.*?</c>)", r, re.S):
            t = re.search(r't="(\w+)"', c)
            v = re.search(r"<v>(.*?)</v>", c, re.S)
            cells.append((shared[int(v.group(1))] if (t and t.group(1) == "s") else v.group(1))
                         if v else "")
        rows.append(cells)
    return rows


def build_patient_map(description_xlsx: str):
    """Return {recording_id: patient_id} plus the metadata row per recording."""
    rows = _read_rows(description_xlsx)
    meta, order = {}, []
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        rec = r[0].strip().replace(".edf", "")
        rec = AN_TO_FILE.get(rec, rec)
        meta[rec] = r[1:]
        order.append(rec)

    # Identical metadata across every column => same patient.
    groups = defaultdict(list)
    for rec in order:
        groups[tuple(meta[rec])].append(rec)

    patient_of, pid = {}, {}
    for key in groups:
        members = sorted(groups[key])
        name = members[0]
        pid[key] = name
        for m in members:
            patient_of[m] = name
    return patient_of, meta


def usable_recordings(dataset_dir: str, description_xlsx: str):
    """Recordings that have both files, are not exact duplicates, and have a patient."""
    patient_of, _ = build_patient_map(description_xlsx)
    recs = []
    for n in range(1, 101):
        rid = f"SN{n}"
        if rid in EXACT_DUPLICATE_RECORDINGS:
            continue
        if not (os.path.exists(os.path.join(dataset_dir, rid + ".edf"))
                and os.path.exists(os.path.join(dataset_dir, rid + ".xlsx"))):
            continue
        if rid not in patient_of:
            continue
        recs.append(rid)
    return recs, patient_of


def pilot_subset(recs, patient_of, n_patients=20, seed=0):
    """One recording per patient, deterministic, for a fast end-to-end run."""
    import random
    by_patient = defaultdict(list)
    for r in recs:
        by_patient[patient_of[r]].append(r)
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    return sorted((sorted(by_patient[p])[0] for p in patients[:n_patients]),
                  key=lambda s: int(s[2:]))
