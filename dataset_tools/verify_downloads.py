#!/usr/bin/env python3
"""Verify iSLEEPS dataset files after downloading.

Checks each EDF for (a) a self-consistent header, (b) file size == header + records,
and (c) no duplicated 25 MiB blocks -- the last one matters because the download
corruption seen on this dataset produced files of the CORRECT SIZE with repeated
blocks, which a size check alone will pass.

Usage:  python verify_downloads.py [Dataset]          # check everything
        python verify_downloads.py Dataset SN43.edf SN58.edf   # check specific files
"""
import sys, os, glob, hashlib, zipfile

BLOCK = 26214400  # 25 MiB -- the downloader's chunk size


def check_edf(path):
    sz = os.path.getsize(path)
    if sz < 100_000:
        return "FAIL", f"only {sz} bytes (failed download / HTML error page)"
    with open(path, "rb") as f:
        head = f.read(256)
        try:
            hb, nrec, nsig = int(head[184:192]), int(head[236:244]), int(head[252:256])
        except ValueError:
            return "FAIL", "unreadable EDF header"
        if hb != 256 + 256 * nsig:
            return "FAIL", f"header length {hb} != {256 + 256*nsig}"
        f.seek(256)
        sig = f.read(256 * nsig)
        try:
            rec = sum(int(sig[216*nsig + i*8: 216*nsig + (i+1)*8]) for i in range(nsig)) * 2
        except ValueError:
            return "FAIL", "unreadable signal header"
        expected = hb + nrec * rec
        if expected != sz:
            d = sz - expected
            return "FAIL", f"size {sz:,} vs expected {expected:,} ({d:+,} bytes)"
        blocks, off = [], 0
        while off < sz:
            f.seek(off)
            blocks.append(hashlib.md5(f.read(min(BLOCK, sz - off))).hexdigest())
            off += BLOCK
    if len(set(blocks)) != len(blocks):
        seen, dups = {}, []
        for i, h in enumerate(blocks):
            if h in seen:
                dups.append(f"{seen[h]}={i}")
            seen[h] = i
        return "FAIL", f"duplicated 25MiB blocks ({', '.join(dups)}) - re-download"
    patient = head[8:88].decode("latin1").strip().split()[0]
    return "OK", f"{sz/1048576:.2f} MiB, {nsig} ch, {nrec/30:.0f} epochs, header id {patient}"


def check_xlsx(path):
    if os.path.getsize(path) < 10_000:
        return "FAIL", f"only {os.path.getsize(path)} bytes"
    try:
        z = zipfile.ZipFile(path)
        rows = z.read("xl/worksheets/sheet1.xml").decode("utf8", "ignore").count("<row ")
        return "OK", f"{os.path.getsize(path)/1024:.0f} KB, {rows} rows"
    except Exception as e:
        return "FAIL", f"not a readable xlsx ({type(e).__name__})"


def main():
    args = sys.argv[1:]
    root = args[0] if args and os.path.isdir(args[0]) else "Dataset"
    names = args[1:] if args and os.path.isdir(args[0]) else args
    if not names:
        names = sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(root, "*.edf")) + glob.glob(os.path.join(root, "*.xlsx")))
    failed = 0
    for n in names:
        p = os.path.join(root, n)
        if not os.path.exists(p):
            print(f"  MISSING  {n}")
            failed += 1
            continue
        status, msg = check_edf(p) if n.lower().endswith(".edf") else check_xlsx(p)
        print(f"  {status:<8} {n:<16} {msg}")
        failed += status == "FAIL"
    print(f"\n{len(names) - failed}/{len(names)} passed"
          + ("" if not failed else f"  --  {failed} need re-downloading"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
