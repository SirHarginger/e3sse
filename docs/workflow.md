# E3-SSE Development Workflow

The project uses the same repository name everywhere:

- Local VS Code workspace: `e3sse`
- GitHub: `github.com/SirHarginger/e3sse`
- Server: `/srv/ben/e3sse`

## Workflow

1. Develop and test locally in VS Code.
2. Commit work to a feature branch.
3. Push to GitHub.
4. Merge approved work into `main`.
5. Pull `main` on the server.
6. Run full-data calculations on the server.
7. Keep raw data, logs, scratch files and generated outputs outside Git.

## Gate G0 on the server

After pushing and merging the validated commit, run:

```bash
cd /srv/ben/e3sse
git status
git pull --ff-only
git rev-parse HEAD
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src python scripts/check_g0.py --config configs/server.json --workers 8 --resume
```

`--workers` (default: `audit.workers` in the JSON config; `1` = serial) sets a
bounded process pool. The work units are each non-MPLiTrj dataset, each MPLiTrj
source file (streamed; one file per worker), and then the MPLiTrj provenance
sample. Only the parent writes checkpoints (`datasets/`, `checkpoints/MPLiTrj/`,
`linkage/`) and the canonical `audit.json`, each atomically. The worker count is
operational: it does not change the run ID, the checkpoints, or any scientific
output, so serial and parallel runs are interchangeable under `--resume`. If a
unit crashes, completed checkpoints are kept, `failures.json` is written, any
earlier `audit.json` for the run is moved to `audit.previous.json`, and the
command exits non-zero. The server has 80 cores
but about 12 GiB of free RAM: start at 8 workers, and try 12 only if memory stays
stable. MPLiTrj runtime is bounded by its largest file
(`execution.longest_unit` in the report).

The command reads `/srv/ben/e3sse/data/raw` and, read-only,
`/srv/ben/e3sse/data/downloads/MPLiTrj_raw.zip`, and writes only below the
configured `outputs_root`. It requires `ase` in the environment. Re-running with `--resume` reuses completed
per-dataset checkpoints only when their input fingerprint matches.

## MPLiTrj frame-to-hop provenance index

G0 reports `same_hop_exclusion_supported = true` only against a valid, complete
index built by:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src python scripts/build_mplitrj_provenance.py \
  --config configs/server.json --workers 8 --resume
```

The build reads `data/raw/MPLiTrj/*.xyz` and `data/downloads/MPLiTrj_raw.zip`
read-only and writes under `outputs/provenance/mplitrj/<index_id>/`:

- `manifest.json`: identities, hashes, counts, ordering evidence, and the nebDFT2k cross-check;
- `edges.json`: frame counts per mapped edge;
- `<split>/material-bucket-XX.json`: one record per flattened frame. `XX` is the first two hex digits of `sha256(material_id)`.

Matching is per material. It requires identical atomic numbers in order, a cell
within 1e-4 Å, and every atom within 1e-4 Å after minimum-image wrapping.
Frame statuses are `mapped_unique`, `mapped_hop_unique`, `ambiguous` (candidates
in more than one hop, so no hop is asserted), `unmapped`, and `error`.

`index_id` covers the input identities, the mapping code and the tolerances.
Changed inputs, code or tolerances make an index stale, and G0 then reverts to
false. G0 also re-verifies every shard's SHA-256.

Resume works per unit: per-file scans, the archive hash, and per-bucket mapping.
Checkpoints are keyed by the Git SHA, so a new commit rebuilds the units in place.

Safeguards. These keep `same_hop_exclusion_supported` false rather than
guessing:

- a hop directory containing files other than step, source, target or traj_init files;
- an unparseable edge ID;
- a duplicate archive member;
- a flattened scan error;
- an unreadable raw member (the whole material is marked `error`);
- a material whose raw members exceed `audit.provenance_max_material_raw_bytes` (default 1 GiB, marked `error`).

G0 re-hashes the indexed inputs (`audit.provenance_verify_input_sha256`) on every
run. Content changes that preserve size and mtime therefore still invalidate the
index.
