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
PYTHONPATH=src python scripts/check_g0.py --config configs/server.json --resume
```

The command reads `/srv/ben/e3sse/data/raw` and, read-only,
`/srv/ben/e3sse/data/downloads/MPLiTrj_raw.zip`, and writes only below the
configured `outputs_root`. It requires `ase` in the environment. Re-running with `--resume` reuses completed
per-dataset checkpoints only when their input fingerprint matches.
