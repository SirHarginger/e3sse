#!/usr/bin/env python3
"""Discover frozen checkpoint bytes and publish a verified create-once inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e3sse.execution.sharded_execution import publish_json_once
from e3sse.production.checkpoints import build_inventory, load_inventory
from e3sse.production.model_runtimes import frozen_models


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="configs/protocol_freeze.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, action="append", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = frozen_models(config)
    if args.output.exists():
        if not args.resume:
            raise SystemExit(f"{args.output} exists; use --resume")
        load_inventory(args.output, models)
        print(f"verified existing checkpoint inventory {args.output}")
        return 0
    inventory = build_inventory(
        models=models,
        smoke_pins=config["inputs"]["production_evidence"]["model_smoke_reports"],
        search_roots=args.search_root,
    )
    publish_json_once(args.output, inventory, root=args.output.parent)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
