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
