#!/usr/bin/env python3
"""Run the reusable, read-only E3-SSE Gate G0 dataset audit.

Production (bounded workers, one BLAS thread each):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    python scripts/check_g0.py --config configs/server.json --workers 8 --resume
"""

from __future__ import annotations

import argparse
import sys

from e3sse.data.dataset_audit import G0ExecutionError, concise_summary, run_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes (1 = serial); overrides audit.workers in the JSON config",
    )
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be a positive integer")
    try:
        report, path = run_audit(
            args.config,
            resume=args.resume,
            workers=args.workers,
            progress=lambda message: print(message, flush=True),
        )
    except G0ExecutionError as exc:
        print(f"Gate G0 incomplete: {exc}", file=sys.stderr)
        print("Completed units are checkpointed; rerun with --resume.", file=sys.stderr)
        return 1
    print(concise_summary(report, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
