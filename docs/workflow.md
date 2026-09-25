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
