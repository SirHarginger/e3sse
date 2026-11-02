#!/usr/bin/env python3
"""Run the E3-SSE production pipeline as one resumable command.

    python scripts/run_e3sse_production.py --config configs/production_execution.json \\
        --stage all --resume

Stages (``configs/production_execution.json``) run in dependency order.  ``--stage X``
runs X plus every dependency that has not completed yet.  ``--dry-run`` prints the
resolved DAG, inputs, model runtimes, predicted resource allocation, output paths,
blocking evidence and the commands that would run.  It writes nothing, reads no
outcome and launches no computation.  Resources are discovered from the process
(affinity mask, MemAvailable); ``--max-cpus`` and ``--max-memory-gib`` can only
narrow them.

Exit codes: 0 every requested stage completed or was verified; 2 blocked (missing
evidence, missing runner, deferred, or a blocked dependency); 3 validation failure;
4 infrastructure failure; 1 usage or configuration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from e3sse.production.orchestrator import (
    DEFAULT_ADAPTERS,
    StaleStageError,
    dry_run_plan,
    execute,
    exit_code,
    load_context,
)
from e3sse.production.stage_graph import StageGraphError

REPOSITORY = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        default=REPOSITORY / "configs" / "production_execution.json")
    parser.add_argument("--stage", default="all", help="a stage name or 'all'")
    parser.add_argument("--resume", action="store_true",
                        help="reuse verified completed stages and shard checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cpus", type=int, default=None,
                        help="use at most this many of the schedulable CPUs")
    parser.add_argument("--max-memory-gib", type=float, default=None,
                        help="treat at most this much memory as available")
    args = parser.parse_args(argv)
    try:
        ctx = load_context(REPOSITORY, args.config, adapters=DEFAULT_ADAPTERS,
                           max_cpus=args.max_cpus, max_memory_gib=args.max_memory_gib,
                           resume=args.resume, dry_run=args.dry_run)
        if args.dry_run:
            plan = dry_run_plan(ctx, args.stage, DEFAULT_ADAPTERS)
            print(json.dumps(plan, indent=2, sort_keys=True, default=str))
            return 0
        results = execute(ctx, args.stage, DEFAULT_ADAPTERS)
    except (StageGraphError, StaleStageError, FileExistsError, ValueError) as exc:
        print(f"production run refused: {exc}", file=sys.stderr)
        return 1
    summary = [{"stage": r.stage, "outcome": r.outcome, "detail": r.detail} for r in results]
    print(json.dumps({"run_id": ctx.run_id, "run_directory": str(ctx.run_dir),
                      "stages": summary}, indent=2))
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
