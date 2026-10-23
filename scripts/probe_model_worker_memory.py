#!/usr/bin/env python3
"""Measure one frozen model's per-worker peak memory for execution planning.

Operational measurement only: a synthetic Li2O supercell, no LiTraj data, no
outcome, and no scientific result.  Run it in the model's pinned environment.  The
JSON report is written atomically to
``<output>/<canonical_name>/<fingerprint-prefix>.json``; ``--resume`` reuses it only
when the probe identity (model, package version, threads, size) is unchanged.  Pass
the report to production runners as ``--worker-memory-report`` for ``--workers auto``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from e3sse.execution.resource_planning import (
    WORKER_MEMORY_REPORT_SCHEMA,
    observe_host_resources,
)
from e3sse.execution.sharded_execution import (
    canonical_sha256,
    ensure_safe_output_dir,
    publish_json_once,
)
from e3sse.execution.worker_memory_probe import (
    measure_worker_memory,
    probe_identity,
)


def _model_entry(config: dict, name: str) -> dict:
    for group in config["protocol"]["model_groups"].values():
        for model in group:
            if model.get("canonical_name") == name:
                return model
    raise SystemExit(f"model {name!r} is not in the frozen registry")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="canonical_name in the registry")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--min-atoms", required=True, type=int,
                        help="largest production frame size, from the frozen input manifest")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--evaluations", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protected-root", required=True, type=Path, action="append",
                        help="immutable source-data root(s); output may not overlap them")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = _model_entry(config, args.model)
    package_version = importlib.metadata.version(model["package_name"])
    if package_version != model["package_version"]:
        raise SystemExit("installed package version differs from the frozen registry")
    identity = probe_identity(
        model,
        min_atoms=args.min_atoms,
        threads_per_worker=args.threads_per_worker,
        evaluations=args.evaluations,
        package_version=package_version,
    )
    fingerprint = canonical_sha256(identity)
    output = ensure_safe_output_dir(args.output, args.protected_root)
    path = output / args.model / f"{fingerprint[:16]}.json"
    if path.exists():
        if not args.resume:
            raise SystemExit(f"{path} exists; use --resume")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("fingerprint_sha256") != fingerprint or canonical_sha256(
            stored.get("identity")
        ) != fingerprint:
            raise SystemExit("stale worker-memory report: identity changed")
        print(f"verified existing report {path}")
        return 0

    repo = Path(__file__).resolve().parents[1]
    git = ["git", "-C", str(repo)]
    measurement = measure_worker_memory(
        model,
        args.checkpoint,
        min_atoms=args.min_atoms,
        threads_per_worker=args.threads_per_worker,
        evaluations=args.evaluations,
    )
    report = {
        "schema_version": WORKER_MEMORY_REPORT_SCHEMA,
        "classification": "operational_resource_measurement_not_scientific",
        "identity": identity,
        "fingerprint_sha256": fingerprint,
        "measurement": measurement,
        "host": observe_host_resources().as_json(),
        "git_sha": subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output([*git, "status", "--porcelain"], text=True)),
        "python": sys.version,
    }
    publish_json_once(path, report, root=output)
    print(json.dumps({"model": args.model, **measurement}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
