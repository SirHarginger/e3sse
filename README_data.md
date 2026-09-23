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
