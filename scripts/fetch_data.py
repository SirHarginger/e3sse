#!/usr/bin/env python3
"""
E3-SSE data acquisition (protocol v3.0, Section 5.2, Table 6; Week 1 / gate G0).

Downloads every open dataset the protocol needs, resumably, into data/raw/,
verifies checksums where the host publishes them, records SHA-256 for all
files in data/manifest.json, and makes raw files read-only (Section 11).

Usage
  python fetch_data.py --dry-run             # sizes only, downloads nothing
  python fetch_data.py --tier 1              # small files needed for G0 (~0.3 GB)
  python fetch_data.py --tier 1 2            # + MPLiTrj (~4.3 GB zipped, ~18 GB unzipped)
  python fetch_data.py --tier 1 2 3          # + FPMD archive (19.8 GiB + 7.9 GiB OPTIMADE)
  python fetch_data.py --all                 # tiers 1-4: everything (~50 GB download)
  python fetch_data.py --only nebDFT2k       # a single item

Re-running is safe: completed files are skipped, partial files are resumed.
Requires: Python >= 3.9, requests, tqdm
"""
import argparse
import datetime
import hashlib
import json
import shutil
import stat
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

LITRAJ = "https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj"
MCLOUD = "https://archive.materialscloud.org/records/4da2r-xag86/files"
MCLOUD2 = "https://archive.materialscloud.org/records/5zenj-34e64/files"   # doi:10.24435/materialscloud:xm-46
OBELIX_COMMIT = "4eaac889809dd489e3bae468c45d5c0e02b9147a"                 # pinned 26 Nov 2025 commit
OBELIX = f"https://raw.githubusercontent.com/NRC-Mila/OBELiX/{OBELIX_COMMIT}/data/downloads"

# name, url, tier, unzip, expected md5 (None if host publishes none), protocol role
ITEMS = [
    # ---- Tier 1: small, needed for G0 checks, Study 2a/4 and Study 5 pool
    dict(name="nebDFT2k", url=f"{LITRAJ}/nebDFT2k.zip", tier=1, unzip=True, md5=None,
         role="Ground-truth DFT-NEB barriers; conformal units (Studies 2, 4)"),
    dict(name="BVEL13k", url=f"{LITRAJ}/BVEL13k.zip", tier=1, unzip=True, md5=None,
         role="Prospective pool, percolation prefilter (Study 5, gates 0-1)"),
    dict(name="nebBVSE122k", url=f"{LITRAJ}/nebBVSE122k.zip", tier=1, unzip=True, md5=None,
         role="BVSE hop barriers; rate-limiting hop (Study 5, gate 1)"),
    dict(name="FPMD_README.txt", url=f"{MCLOUD}/README.txt?download=1", tier=1, unzip=False,
         md5="3adef433ac13a114c8d9d8f95571d917", role="FPMD archive documentation (G0)"),
    dict(name="FPMD_files_description.md", url=f"{MCLOUD}/files_description.md?download=1", tier=1,
         unzip=False, md5="d14acbfd3f123eccf6c317be3acc2ad7", role="FPMD archive documentation (G0)"),
    dict(name="FPMD_optimade.yaml", url=f"{MCLOUD}/optimade.yaml?download=1", tier=1, unzip=False,
         md5="ef6ee3ae9c9e66c5bda16bdedda0dfc6", role="FPMD archive documentation (G0)"),
    # ---- Tier 2: DFT frames for estimating s (Study 1, leakage rule Section 3.6)
    dict(name="MPLiTrj_subsample", url=f"{LITRAJ}/MPLiTrj_subsample.zip", tier=2, unzip=True, md5=None,
         role="Fast pilot of s estimator (Study 1 pilot)"),
    dict(name="MPLiTrj", url=f"{LITRAJ}/MPLiTrj.zip", tier=2, unzip=True, md5=None,
         role="Full DFT frames for leakage-free s (Study 1, 2)"),
    dict(name="MPLiTrj_raw", url=f"{LITRAJ}/MPLiTrj_raw.zip", tier=2, unzip=False, md5=None,
         role="Undocumented in README; may carry per-hop provenance needed by leakage rule. "
              "Kept zipped until inspected."),
    # ---- Tier 3: large FPMD archive (Study 3)
    dict(name="FPMD_screening.aiida", url=f"{MCLOUD}/screening.aiida?download=1", tier=3, unzip=False,
         md5="99de1623cf3f1b73e0055e6fe0e45ebe",
         role="FPMD trajectories/forces/diffusion (Study 3). Read with aiida-core."),
    dict(name="FPMD_optimade.jsonl.gz", url=f"{MCLOUD}/optimade.jsonl.gz?download=1", tier=3, unzip=False,
         md5="47486a87ffb461ae51b3d696fe40584b",
         role="Optional: per-timestep structures via OPTIMADE (likely no forces; check before use)"),
    # ---- Tier 4: follow-up FPMD screen [34] and experimental sanity-check data [64, 65]
    dict(name="FPMD2026_README.txt", url=f"{MCLOUD2}/README.txt?download=1", tier=4, unzip=False,
         md5="f095df76465d8cbfa7fb1680485dd381", role="Follow-up FPMD archive documentation [34]"),
    dict(name="FPMD2026_structures.aiida", url=f"{MCLOUD2}/fpmd_structures.aiida?download=1", tier=4,
         unzip=False, md5="14c51c3917335cb116124047dec4f369", role="Follow-up FPMD starting structures [34]"),
    dict(name="FPMD2026_screening_Li7NbO6.aiida", url=f"{MCLOUD2}/fpmd_screening_Li7NbO6.aiida?download=1",
         tier=4, unzip=False, md5="f78e4ba790f31ef7ba18dc2fb4ec3df1",
         role="Full screening provenance for one structure [34]"),
    dict(name="FPMD2026_trajectories.aiida", url=f"{MCLOUD2}/fpmd_trajectories.aiida?download=1", tier=4,
         unzip=False, md5="88ea15082b6afda3160a205296a75221",
         role="Follow-up FPMD trajectories, PBEsol (optional Study 3 panel extension) [34]"),
    dict(name="OBELiX_all.csv", url=f"{OBELIX}/all.csv", tier=4, unzip=False,
         md5="4d2f77129ed415efa5e7f26264f38f7f", role="Experimental RT conductivities, 599 entries [65]"),
    dict(name="OBELiX_train.csv", url=f"{OBELIX}/train.csv", tier=4, unzip=False,
         md5="e642b05158a9cb3eb0ca2563cc147258", role="OBELiX official train split [65]"),
    dict(name="OBELiX_test.csv", url=f"{OBELIX}/test.csv", tier=4, unzip=False,
         md5="c6d9bda6fb44f9d63fad2df72ce58ae5", role="OBELiX official test split [65]"),
    dict(name="OBELiX_cifs", url=f"{OBELIX}/all_cifs.zip", tier=4, unzip=True,
         md5="76ae8014765d17ca003b77bef321a038", role="OBELiX CIFs, 321 structures [65]"),
    dict(name="LiverpoolIonics", url=None, tier=4, unzip=False, md5=None, manual=True,
         role="Liverpool Ionics Dataset, 820 entries [64]; MANUAL download (terms must be accepted)"),
]

MANUAL_HELP = {
    "LiverpoolIonics": (
        "1. Open https://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html in a browser\n"
        "  2. Read and accept the terms of use, download the CSV\n"
        "  3. Put the file (any name ending .csv) in  {manual}/\n"
        "  4. Re-run: python fetch_data.py --only LiverpoolIonics"),
}

UA = {"User-Agent": "E3-SSE-data-fetch/1.0 (research; contact in manifest)",
      "Accept-Encoding": "identity"}   # byte-exact files: no transparent gzip, sizes match content-length


def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n is None:
            return "unknown"
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def remote_size(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=30, headers=UA)
        if r.ok and r.headers.get("content-length"):
            return int(r.headers["content-length"])
    except requests.RequestException:
        pass
    return None


def hash_file(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest: Path, total):
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if total and have >= total:          # stale or oversized partial file: start over
        part.unlink()
        have = 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, stream=True, allow_redirects=True, timeout=60, headers=headers) as r:
        if have and r.status_code == 200:      # server ignored Range: restart
            have = 0
        elif r.status_code not in (200, 206):
            r.raise_for_status()
        mode = "ab" if have else "wb"
        with open(part, mode) as f, tqdm(total=total, initial=have, unit="B", unit_scale=True,
                                         unit_divisor=1024, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if total and part.stat().st_size != total:
        raise IOError(f"{dest.name}: size {part.stat().st_size} != expected {total}; re-run to resume")
    part.rename(dest)


def make_readonly(path: Path):
    for p in [path] + (list(path.rglob("*")) if path.is_dir() else []):
        if p.is_file():
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data", help="project data directory (default: ./data)")
    ap.add_argument("--tier", type=int, nargs="+", default=[1])
    ap.add_argument("--only", nargs="+", help="download only these item names")
    ap.add_argument("--all", action="store_true", help="all tiers (1-4)")
    ap.add_argument("--dry-run", action="store_true", help="report sizes and exit")
    ap.add_argument("--keep-zip", action="store_true", help="keep .zip after unzipping")
    args = ap.parse_args()

    root = Path(args.root)
    zips, raw = root / "downloads", root / "raw"
    zips.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if args.all:
        args.tier = [1, 2, 3, 4]
    manual_dir = root / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    items = [i for i in ITEMS if (i["name"] in args.only if args.only else i["tier"] in args.tier)]
    if not items:
        sys.exit("Nothing selected.")

    sizes = {i["name"]: (remote_size(i["url"]) if i["url"] else None) for i in items}
    print(f"\n{'item':34s} {'tier':>4s} {'size':>10s}  role")
    for i in items:
        size = "manual" if i.get("manual") else human(sizes[i["name"]])
        print(f"{i['name']:34s} {i['tier']:>4d} {size:>10s}  {i['role']}")
    known = sum(s for s in sizes.values() if s)
    print(f"\nTotal (known sizes, compressed): {human(known)}")
    if args.dry_run:
        return

    failures = []
    for i in items:
        name = i["name"]
        if i.get("manual"):
            found = sorted(manual_dir.glob("*.csv"))
            if not found:
                print(f"[MANUAL] {name} not found. To add it:\n  " + MANUAL_HELP[name].format(manual=manual_dir))
                manifest[name] = dict(status="manual_pending", role=i["role"])
                failures.append(name + " (manual)")
                manifest_path.write_text(json.dumps(manifest, indent=2))
                continue
            src = found[0]
            dest = raw / f"{name}.csv"
            if not dest.exists():
                dest.write_bytes(src.read_bytes())
                make_readonly(dest)
            manifest[name] = dict(source="manual download (terms accepted by user)", original_filename=src.name,
                                  role=i["role"], bytes=dest.stat().st_size, md5=hash_file(dest, "md5"),
                                  sha256=hash_file(dest, "sha256"), status="ok",
                                  registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
            print(f"[manual ok] {name} registered from {src.name}")
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue
        is_zip = i["url"].split("?")[0].endswith(".zip")
        dest = zips / (name + ".zip") if is_zip else raw / name
        done_marker = raw / name if (is_zip and i["unzip"]) else dest
        try:
            if name in manifest and manifest[name].get("status") == "ok" and done_marker.exists():
                print(f"[skip] {name} already verified")
                continue
            if not dest.exists():
                download(i["url"], dest, sizes[name])
            print(f"[hash] {name}")
            md5 = hash_file(dest, "md5")
            if i["md5"] and md5 != i["md5"]:
                raise IOError(f"{name}: md5 mismatch ({md5} != {i['md5']}); delete and re-download")
            sha = hash_file(dest, "sha256")
            entry = dict(url=i["url"], tier=i["tier"], role=i["role"], bytes=dest.stat().st_size,
                         md5=md5, md5_published=i["md5"], sha256=sha,
                         downloaded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
            if is_zip and i["unzip"]:
                target = raw / name
                if not target.exists():
                    print(f"[unzip] {name}")
                    tmp = raw / (name + ".unzipping")
                    with zipfile.ZipFile(dest) as z:
                        bad = z.testzip()
                        if bad:
                            raise IOError(f"{name}: corrupt member {bad}")
                        z.extractall(tmp)
                    # archives usually hold one top-level folder (its name may differ,
                    # e.g. README shows "MPLiTraj/"); normalise to raw/<name> for litraj.load_data
                    kids = [k for k in tmp.iterdir() if not k.name.startswith("__MACOSX")]
                    if len(kids) == 1 and kids[0].is_dir():
                        kids[0].rename(target)
                        shutil.rmtree(tmp)
                    else:
                        tmp.rename(target)
                entry["files"] = sum(1 for p in target.rglob("*") if p.is_file())
                make_readonly(target)
                if not args.keep_zip:
                    dest.unlink()
                    entry["zip_deleted"] = True
            elif not is_zip or not i["unzip"]:
                make_readonly(dest)
            entry["status"] = "ok"
            manifest[name] = entry
        except Exception as e:  # keep going; log failure
            print(f"[FAIL] {name}: {e}")
            manifest[name] = dict(url=i["url"], status="failed", error=str(e))
            failures.append(name)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nManifest: {manifest_path}")
    if failures:
        sys.exit(f"Failed: {', '.join(failures)} (re-run to resume)")
    print("All selected items verified.")


if __name__ == "__main__":
    main()
