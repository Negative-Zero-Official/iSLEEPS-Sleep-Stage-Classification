# iSLEEPS Dataset — integrity audit & cleanup

Audit date: 2026-08-30. Scope: `Dataset/` (100 subjects, EDF + XLSX pairs).

## 1. What was removed (20 files, 709 MiB freed)

| Category | Files |
|---|---|
| Byte-identical duplicate copies (md5-verified) | `SN28(1).edf`, `SN29(1).edf`, `SN30(1).edf`, `SN1(1).xlsx`, `SN10(1).xlsx`, `SN11(1).xlsx`, `SN20(1).xlsx`, `SN27(1).xlsx`, `SN28(1).xlsx` |
| Zero-byte files | `SN43(2).edf`, `SN98(1).edf`, `SN53(1).xlsx`, `SN53(2).xlsx`, `SN95(1).xlsx`, `SN95(2).xlsx` |
| GitHub Pages 404 HTML saved with a data extension (all 9,379 B, same md5) | `SN43.edf`, `SN5.edf`, `SN53.xlsx`, `SN95.xlsx` |
| Corrupt EDF with a verified clean replacement | `SN7.edf` (corrupt) deleted; `SN7(1).edf` promoted to `SN7.edf` |

## 2. Quarantined — corrupt, no clean copy (`Dataset/_corrupt_redownload/`)

The downloader wrote some 25 MiB chunks twice and dropped others, so these files are
the right length but internally scrambled — unrecoverable without a re-download.

| File | Chunks | Damage |
|---|---|---|
| `SN43(1).edf` | 8 (6 unique) | chunks 2=3, 5=6; passes a naive size check |
| `SN58.edf` | 7 (5 unique) | chunks 0=1, 3=5; passes a naive size check |
| `SN98.edf` | 9 (7 unique) | chunks 0=2, 3=4; 25 MiB over length |

`SN5.edf` was re-downloaded from the Sample Version on 30 Aug and verified clean (119.21 MiB,
header patient `SN5`, size check OK, 5/5 unique blocks, hypnogram delta +7.7). Its corrupt
predecessor `SN5(1).edf` has been deleted.

**Still to re-download:** `SN43.edf`, `SN58.edf`, `SN98.edf`, `SN53.xlsx`, `SN95.xlsx` —
none of these are in the Sample Version (which covers SN1–SN40 only), so they must come
from the full release.

After re-downloading, verify each EDF with: header length == 256 + 256*n_signals, and
file size == header + n_records * record_size. That check alone would **not** have caught
`SN43(1).edf` or `SN58.edf` — also hash the file in 25 MiB blocks and confirm all blocks differ.

## 3. Current state

- 97 EDF + 98 XLSX + `subject_description.xlsx`; **95 of 100 subjects have a complete pair**
- Every remaining file passes structural validation and the duplicate-chunk check
- No remaining `(1)`/`(2)` filenames

> **Note on repeat downloads.** Re-pulling the Sample Version only ever re-fetches files you
> already have; every collision lands as `SNx(1).xlsx`/`SNx(1).edf`. Two rounds of these have
> been removed (14 files total), all confirmed byte-identical to the originals first.

## 4. "Sample Version" files — keep them

The 81-file Sample Version is a **strict subset of this same dataset**, not a separate
lower-quality release. Local `SN1`–`SN40` byte sizes match the Sample Version listing exactly
for 38 of 40 files (the two exceptions were the corrupt `SN5` and `SN7` downloads above), and
`subject_description.xlsx` covers all of them as one cohort. Deleting them would remove 40%
of the subjects. **Nothing was deleted on Sample Version grounds** — only the redundant
second copies, which were byte-identical.

## 5. Data-quality issues in the source dataset (NOT fixed — decide before training)

### 5a. `SN11`–`SN20` are the `AN1`–`AN10` rows

`subject_description.xlsx` contains 100 rows: `SN1`–`SN10`, **`AN1`–`AN10`**, `SN21`–`SN100`.
There are no `SN11`–`SN20` rows. The EDF headers resolve the mapping:

| File | EDF header patient | Metadata row |
|---|---|---|
| `SN11.edf` … `SN20.edf` | `SN1` … `SN10` | `AN1.edf` … `AN10.edf` |

**A naive join of filename → metadata row silently mislabels these 10 subjects.**
Consider renaming `SN{n+10}` → `AN{n}` , or add an explicit mapping in your loader.

### 5b. `SN15` and `SN28` are the same recording

- EDF signal data is **byte-identical** (md5 of everything after the 7,680-byte header matches);
  the only difference is 8 bytes of the patient-ID field (`SN5` vs `SN28`)
- Same start time (21.37.11), same 26,405 records, same 29 channels, identical channel labels
- The hypnograms in `SN15.xlsx` and `SN28.xlsx` are identical across all 880 epochs
- Metadata confirms it: `AN5` and `SN28` are the same patient (38 M, auto driver, Rt MCA
  territory infarct, AHI 5.8)

**Drop one of them, or the same night appears in both train and test.**

### 5c. 26 subject IDs map to only 12 distinct patients

These groups have all 63 metadata columns identical — same patient, different nights
(except `AN5`/`SN28`, which is the same night twice):

```
SN2  = SN57 = SN59      SN4  = SN26           SN5  = AN7 (=SN17.edf)
AN2 (=SN12.edf) = SN22  AN5 (=SN15.edf) = SN28 = SN38
SN29 = SN46             SN30 = SN56           SN31 = SN61
SN49 = SN50             SN53 = SN89           SN74 = SN75
SN92 = SN93
```

So 100 IDs cover roughly **86 unique patients**. Split **by patient, not by file**, or your
validation scores will be optimistic through subject leakage.

### 5d. Minor

`SN9`, `SN24`, `SN30`, `SN36`, `SN38`, `SN42` have hypnogram lengths a few epochs off the usual
offset from recording duration (scoring started/stopped slightly outside the recording).
Normal for PSG data — just clip to the shorter of the two when aligning.
