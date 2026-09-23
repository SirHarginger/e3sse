#!/usr/bin/env bash
# E3-SSE project bootstrap: creates the project, installs dependencies, checks
# connectivity and disk, then downloads ALL data in the background and runs the G0 check.
#
#   bash e3sse_bootstrap.sh                       # project in /srv/ben/e3sse
#   PROJECT_DIR=/other/path bash e3sse_bootstrap.sh
#
# Safe to re-run: finished downloads are skipped, partial ones resume.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/srv/ben/e3sse}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
NEED_GB="${NEED_GB:-75}"
SESSION="e3sse_data"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"/{scripts,logs} "$DATA_ROOT"/{raw,downloads,manual}
cd "$PROJECT_DIR"

say "Writing scripts"
cat > scripts/fetch_data.py <<'E3SSE_EOF'
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
import argparse, datetime, hashlib, json, os, stat, sys, zipfile
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
        part.unlink(); have = 0
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
                        import shutil; shutil.rmtree(tmp)
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
E3SSE_EOF
cat > scripts/check_g0.py <<'E3SSE_EOF'
#!/usr/bin/env python3
"""
E3-SSE gate G0 data verification (protocol v3.0, Section 6, Table 9; Section 5.3).

Run after `fetch_data.py --tier 1 2`. Reads data/raw/ and writes data/G0_report.md
answering the questions the protocol depends on:

  nebDFT2k   Q1 do relaxed NEB images carry per-image DFT energies AND forces?
             Q2 does max-min image energy reproduce the published barrier column?
             Q3 split, transition-metal / Hubbard-U counts (primary analysis size)
             Q4 hops per structure (feasibility of the leakage rule, Sec. 3.6)
  MPLiTrj    Q5 which per-frame metadata exist (material_id? edge_id? image index?)
             Q6 coverage: nebDFT2k materials that have MPLiTrj frames
  FPMD       Q7 archive documentation present (content must then be checked by hand)

Usage: python check_g0.py [--root data] [--mplitrj MPLiTrj_subsample|MPLiTrj] [--max-frames 200000]
Requires: ase, pandas, numpy, tqdm
"""
import argparse, collections, re
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import iread, read
from tqdm import tqdm

# Elements with Hubbard U in Materials Project / MPtrj (applied in oxide & fluoride chemistries)
MP_U = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}
D_BLOCK = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
              "Hf Ta W Re Os Ir Pt Au Hg".split())


def elements_of(chemsys):
    return set(re.findall(r"[A-Z][a-z]?", str(chemsys)))


def check_nebdft2k(folder: Path, out):
    out.append("## nebDFT2k\n")
    idx_path = folder / "nebDFT2k_index.csv"
    if not idx_path.exists():
        out.append(f"**MISSING** {idx_path}\n")
        return None
    idx = pd.read_csv(idx_path)
    out.append(f"- Hops: **{len(idx)}** (protocol expects 1,681); columns: `{', '.join(idx.columns)}`")
    if "material_id" in idx:
        out.append(f"- Structures: **{idx.material_id.nunique()}** (expects 876)")
    if "_split" in idx:
        out.append(f"- Split counts: {idx._split.value_counts().to_dict()}")

    # Q3 TM / U
    if "chemsys" in idx:
        els = idx.chemsys.map(elements_of)
    else:
        els = None
    # Q1/Q2 per-image data
    n_img, has_E, has_F, barrier, fmax_saddle, problems = [], 0, 0, [], [], []
    for eid in tqdm(idx.edge_id, desc="nebDFT2k images"):
        p = folder / f"{eid}_relaxed.xyz"
        if not p.exists():
            problems.append(f"missing {p.name}"); barrier.append(np.nan); n_img.append(0); continue
        imgs = read(p, index=":")
        n_img.append(len(imgs))
        E, F = [], []
        for a in imgs:
            try:
                E.append(a.get_potential_energy())
            except Exception:
                E.append(np.nan)
            try:
                F.append(a.get_forces())
            except Exception:
                F.append(None)
        if all(np.isfinite(E)):
            has_E += 1
            barrier.append(max(E) - min(E))
        else:
            barrier.append(np.nan)
        if all(f is not None for f in F):
            has_F += 1
            # max atomic force on interior images (NEB convergence check, fmax=0.1 eV/A claimed)
            interior = F[1:-1] or F
            fmax_saddle.append(max(np.linalg.norm(f, axis=1).max() for f in interior))
        if els is None:
            els_i = set(imgs[0].get_chemical_symbols())
            idx.loc[idx.edge_id == eid, "_els"] = ",".join(sorted(els_i))
    if els is None:
        els = idx["_els"].fillna("").map(lambda s: set(s.split(",")))
    N = len(idx)
    out.append("\n### Q1 per-image DFT data (G0 pass condition)")
    out.append(f"- Hops whose relaxed images all have energies: **{has_E}/{N}**")
    out.append(f"- Hops whose relaxed images all have forces: **{has_F}/{N}**")
    out.append(f"- Images per hop: {dict(sorted(collections.Counter(n_img).items()))}")
    if fmax_saddle:
        fs = np.array(fmax_saddle)
        out.append(f"- Max |F| on interior images: median {np.median(fs):.3f}, 95th pct {np.percentile(fs,95):.3f} eV/Å "
                   f"(protocol assumes fmax≈0.1); note CI-NEB fmax refers to NEB-projected forces, raw forces may be larger")
    g0_q1 = has_E == N and has_F == N
    out.append(f"- **Q1 verdict: {'PASS' if g0_q1 else 'FAIL/PARTIAL — see contingency G0'}**")

    out.append("\n### Q2 barrier definition")
    idx["Eb_maxmin"] = barrier
    cand = [c for c in idx.columns if c.lower() in ("em", "em_dft", "barrier", "e_m", "migration_barrier")]
    if cand:
        c = cand[0]
        d = (idx["Eb_maxmin"] - idx[c]).abs()
        out.append(f"- Published column `{c}` vs max−min image energy: median |Δ| {d.median():.4f} eV, "
                   f"max {d.max():.4f} eV, exact (<1 meV) for {(d < 1e-3).sum()}/{d.notna().sum()}")
        out.append(f"- **Q2 verdict: {'definition confirmed' if (d < 1e-3).mean() > 0.99 else 'DEFINITION DIFFERS — inspect before Study 2'}**")
    else:
        out.append("- No barrier column recognised; inspect index columns manually")

    out.append("\n### Q3 primary-analysis eligibility (Section 5.3)")
    has_U = els.map(lambda s: bool(s & MP_U))
    has_TM = els.map(lambda s: bool(s & D_BLOCK))
    idx["has_MP_U_element"], idx["has_d_block"] = has_U, has_TM
    out.append(f"- Hops without MP Hubbard-U elements: **{(~has_U).sum()}**; without any d-block element: **{(~has_TM).sum()}**")
    if "_split" in idx:
        out.append(f"- d-block-free by split: {idx[~has_TM]._split.value_counts().to_dict()}")
    out.append("- Decision needed: protocol text says 'transition metals *or* U elements'; the d-block-free set is the stricter reading")

    out.append("\n### Q4 hops per structure (leakage rule feasibility)")
    if "material_id" in idx:
        hps = idx.groupby("material_id").size()
        out.append(f"- Distribution: {dict(sorted(collections.Counter(hps).items()))}")
        out.append(f"- Structures with a single hop: **{(hps == 1).sum()}** — for these, only endpoint-relaxation or "
                   f"non-nebDFT2k MPLiTrj frames can give leakage-free material-level s")
    if problems:
        out.append(f"\n- File problems ({len(problems)}): {problems[:10]}")
    idx.drop(columns=[c for c in ("_els",) if c in idx]).to_csv(folder.parent.parent / "nebDFT2k_g0_table.csv", index=False)
    return idx


def check_mplitrj(folder: Path, name: str, neb_idx, max_frames, out):
    out.append(f"\n## {name}\n")
    files = sorted(folder.glob("*.xyz"))
    if not files:
        out.append(f"**MISSING** xyz files in {folder}")
        return
    out.append(f"- Files: {[f.name for f in files]}")
    info_keys, arrays_keys, calc_ok = collections.Counter(), collections.Counter(), 0
    mats, edges, n = set(), set(), 0
    for f in files:
        for a in tqdm(iread(f, index=":", format="extxyz"), desc=f.name):
            info_keys.update(a.info.keys()); arrays_keys.update(a.arrays.keys())
            try:
                a.get_forces(); a.get_potential_energy(); calc_ok += 1
            except Exception:
                pass
            for k in ("material_id", "mp_id"):
                if k in a.info: mats.add(a.info[k])
            for k in ("edge_id", "traj_id", "path_id"):
                if k in a.info: edges.add(a.info[k])
            n += 1
            if n >= max_frames:
                break
        if n >= max_frames:
            break
    out.append(f"- Frames scanned: {n} (cap {max_frames}); with energy+forces: {calc_ok}")
    out.append(f"- `atoms.info` keys: {dict(info_keys)}")
    out.append(f"- `atoms.arrays` keys: {dict(arrays_keys)}")
    out.append("\n### Q5 provenance for the leakage rule")
    if edges:
        out.append(f"- Hop identifiers present ({len(edges)} distinct). **Leakage rule enforceable at hop level.**")
    elif mats:
        out.append(f"- Only material identifiers ({len(mats)} distinct). Hop-level exclusion NOT possible from these files; "
                   "check MPLiTrj_raw.zip, or match frames to nebDFT2k images by geometry.")
    else:
        out.append("- **No material or hop identifiers found.** Leakage rule not enforceable from these files; "
                   "inspect MPLiTrj_raw.zip next (fetch_data.py keeps it zipped).")
    if neb_idx is not None and mats and "material_id" in neb_idx:
        cov = neb_idx.material_id.isin(mats)
        out.append(f"\n### Q6 coverage\n- nebDFT2k hops whose material has frames here: **{cov.sum()}/{len(neb_idx)}**")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--mplitrj", default="MPLiTrj_subsample", choices=["MPLiTrj_subsample", "MPLiTrj"])
    ap.add_argument("--max-frames", type=int, default=200_000)
    args = ap.parse_args()
    raw = Path(args.root) / "raw"
    out = ["# E3-SSE gate G0 report\n"]
    neb = check_nebdft2k(raw / "nebDFT2k", out)
    check_mplitrj(raw / args.mplitrj, args.mplitrj, neb, args.max_frames, out)
    out.append("\n## FPMD archive (Materials Cloud 10.24435/materialscloud:vg-ya)\n")
    for f in ("FPMD_README.txt", "FPMD_files_description.md"):
        p = raw / f
        out.append(f"### {f}\n```\n{p.read_text(errors='replace') if p.exists() else 'MISSING'}\n```")
    out.append("- Q7 manual checks: are per-material D(T) tables and FPMD forces in `screening.aiida`? "
               "What pseudopotentials/cutoffs were used? (s for Study 3 must be estimated against the FPMD reference, "
               "not LiTraj PBE, see Table 3.)")
    rep = Path(args.root) / "G0_report.md"
    rep.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nWritten: {rep}")


if __name__ == "__main__":
    main()
E3SSE_EOF
cat > README_data.md <<'E3SSE_EOF'
# E3-SSE data acquisition

```bash
pip install requests tqdm ase pandas numpy
python fetch_data.py --dry-run --all    # sizes only
python fetch_data.py --all              # everything (resumable; re-run if interrupted)
python check_g0.py                      # data/G0_report.md
```

| Tier | Item | Size | Protocol role | Checksum |
|---|---|---|---|---|
| 1 | nebDFT2k | 65 MB | Ground-truth barriers, conformal units (Studies 2, 4) | SHA-256 recorded |
| 1 | BVEL13k | 10 MB | Prospective pool (Study 5) | SHA-256 recorded |
| 1 | nebBVSE122k | 191 MB | Rate-limiting hop prefilter (Study 5) | SHA-256 recorded |
| 1 | FPMD README / description / yaml | <1 MB | FPMD archive documentation | md5 verified |
| 2 | MPLiTrj_subsample | 492 MB | Pilot of *s* estimator | SHA-256 recorded |
| 2 | MPLiTrj | 3.8 GB (16 GB unzipped) | Leakage-free *s* (Studies 1–2) | SHA-256 recorded |
| 2 | MPLiTrj_raw | 6.9 GB (kept zipped) | Possible hop provenance | SHA-256 recorded |
| 3 | FPMD screening.aiida | 19.8 GiB | FPMD reference, PBE (Study 3) [6] | md5 verified |
| 3 | FPMD optimade.jsonl.gz | 7.9 GiB | Per-timestep structures | md5 verified |
| 4 | FPMD2026 README + structures + Li7NbO6 provenance | ~0.5 GiB | Follow-up screen documentation [34] | md5 verified |
| 4 | FPMD2026 trajectories.aiida | 15.9 GiB | Follow-up FPMD, PBEsol (optional Study 3 panel) [34] | md5 verified |
| 4 | OBELiX all/train/test CSV + 321 CIFs | <1 MB | Experimental sanity check [65] | md5 verified, commit-pinned |
| 4 | Liverpool Ionics Dataset | <1 MB | Experimental sanity check [64] | **manual download**, SHA-256 recorded |

**Disk:** about 70 GB free before starting (about 25 GB for tiers 1–2, already done, plus about 44 GiB for tiers 3–4).

**Liverpool Ionics Dataset (manual):** open https://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html,
accept the terms of use, download the CSV into `data/manual/`, then run
`python fetch_data.py --only LiverpoolIonics`. The script copies it into `data/raw/` and records it in the manifest.

**Layout:** `data/downloads/` (transient zips), `data/raw/` (read-only), `data/manual/` (hand-downloaded
originals), `data/manifest.json` (URL, md5, SHA-256, UTC time, status for every item).

**Licences and citations:** LiTraj (Dembitskiy et al., npj Comput. Mater. 2025; structures from Materials
Project, CC BY 4.0); FPMD archives CC BY 4.0 (Kahle et al., EES 2020, doi:10.24435/materialscloud:vg-ya;
Thakur et al., EES 2026, doi:10.24435/materialscloud:xm-46); OBELiX (Therrien et al., Digital Discovery 2026;
MIT-licensed repository; most CIFs carry deliberately added noise); Liverpool Ionics Dataset
(Hargreaves et al., npj Comput. Mater. 2023; use under the terms accepted on download).
E3SSE_EOF
[ -f .gitignore ] || printf 'data/\nlogs/\n.venv/\n__pycache__/\n*.part\n' > .gitignore

say "Python environment"
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "Using active virtualenv: $VIRTUAL_ENV"
elif [ -n "${CONDA_PREFIX:-}" ]; then
  echo "Using active conda env: $CONDA_PREFIX"
else
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "Created/using $PROJECT_DIR/.venv"
fi
PY="$(command -v python)"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q requests tqdm ase pandas numpy
"$PY" - <<'E3SSE_EOF'
import sys, requests, tqdm, ase, pandas, numpy
print(f"python {sys.version.split()[0]}  ase {ase.__version__}  pandas {pandas.__version__}  numpy {numpy.__version__}")
E3SSE_EOF

say "Disk space at $DATA_ROOT"
AVAIL_GB=$(df -BG --output=avail "$DATA_ROOT" | tail -1 | tr -dc '0-9')
echo "Available: ${AVAIL_GB} GB (need about ${NEED_GB} GB)"
if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
  echo "ERROR: not enough space. Free space or set DATA_ROOT to a larger volume, e.g."
  echo "  DATA_ROOT=/bigdisk/e3sse_data bash $0"
  exit 1
fi

say "Checking that data hosts are reachable (no download yet)"
"$PY" scripts/fetch_data.py --root "$DATA_ROOT" --dry-run --all | tee logs/dry_run.txt
if grep -E '^(nebDFT2k|MPLiTrj|FPMD_screening|FPMD2026_traj|OBELiX_all)' logs/dry_run.txt | grep -q unknown; then
  echo
  echo "WARNING: some hosts returned no size (blocked by firewall/proxy?). Rows marked 'unknown' above."
  echo "The download will be attempted anyway; failures are logged and can be retried."
fi

say "Starting full download + G0 check in the background"
cat > scripts/run_download.sh <<E3SSE_EOF
#!/usr/bin/env bash
# Full data download followed by the G0 check (generated by e3sse_bootstrap.sh)
cd "$PROJECT_DIR"
"$PY" scripts/fetch_data.py --root "$DATA_ROOT" --all
"$PY" scripts/check_g0.py --root "$DATA_ROOT" --mplitrj MPLiTrj_subsample
echo
echo "FINISHED at \$(date -u)"
"$PY" - <<'PYEOF'
import json
m = json.load(open("$DATA_ROOT/manifest.json"))
for k, v in m.items():
    print(f"{k:36s} {v['status']}")
bad = [k for k, v in m.items() if v["status"] != "ok"]
print("\nALL DATA OK" if not bad else f"\nNOT OK: {bad}  (re-run: bash $PROJECT_DIR/scripts/run_download.sh)")
PYEOF
E3SSE_EOF
chmod +x scripts/run_download.sh
LOG="$PROJECT_DIR/logs/download_$(date -u +%Y%m%dT%H%M%SZ).log"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "A download session is already running. Attach with: tmux attach -t $SESSION"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash scripts/run_download.sh 2>&1 | tee '$LOG'; exec bash"
  echo "Running in tmux session '$SESSION'."
  echo "  Watch:   tmux attach -t $SESSION     (detach: Ctrl-b then d)"
else
  nohup bash scripts/run_download.sh > "$LOG" 2>&1 &
  echo "Running in background (PID $!)."
fi
echo "  Log:     tail -f $LOG"
echo "  Report:  $DATA_ROOT/G0_report.md   (written when downloads finish)"
echo
echo "Manual step still needed: Liverpool Ionics Dataset (terms must be accepted in a browser)."
echo "  1. Download CSV from https://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html"
echo "  2. Copy it to $DATA_ROOT/manual/"
echo "  3. $PY $PROJECT_DIR/scripts/fetch_data.py --root $DATA_ROOT --only LiverpoolIonics"
