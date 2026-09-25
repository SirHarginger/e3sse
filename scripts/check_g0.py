#!/usr/bin/env python3
"""Run the reusable, read-only E3-SSE Gate G0 dataset audit."""

from __future__ import annotations

import argparse

from e3sse.data.dataset_audit import concise_summary, run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report, path = run_audit(args.config, resume=args.resume)
    print(concise_summary(report, path))


if __name__ == "__main__":
    main()
