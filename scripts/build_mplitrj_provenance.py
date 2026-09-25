#!/usr/bin/env python3
"""Build the full, resumable MPLiTrj frame-to-hop provenance index (read-only inputs).

Production (bounded workers, one BLAS thread each):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    python scripts/build_mplitrj_provenance.py --config configs/server.json --workers 8 --resume
"""

from __future__ import annotations

import argparse
import sys

from e3sse.data.dataset_audit import G0ExecutionError
from e3sse.data.mplitrj_provenance import build_provenance_index


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
        manifest, path = build_provenance_index(
            args.config,
            resume=args.resume,
            workers=args.workers,
            progress=lambda message: print(message, flush=True),
        )
    except G0ExecutionError as exc:
        print(f"Provenance index incomplete: {exc}", file=sys.stderr)
        print("Completed units are checkpointed; rerun with --resume.", file=sys.stderr)
        return 1
    counts = manifest["counts"]
    print(
        f"MPLiTrj provenance index {manifest['index_id']}: records={counts['record_count']}, "
        f"mapped={counts['mapped_count']} (unique={counts['mapped_unique_count']}, "
        f"hop_unique={counts['mapped_hop_unique_count']}), ambiguous={counts['ambiguous_count']}, "
        f"unmapped={counts['unmapped_count']}, error={counts['error_count']}, "
        f"coverage={counts['coverage_fraction']:.6f}"
    )
    print(f"complete_coverage={manifest['complete_coverage']}; blockers={manifest['coverage_blockers']}")
    for name, ordering in manifest["ordering"].items():
        print(f"ordering {name}: {ordering['conclusion']}")
    print(f"Manifest: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
