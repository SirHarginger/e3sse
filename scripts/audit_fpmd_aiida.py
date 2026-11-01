#!/usr/bin/env python3
"""Inventory FPMD AiiDA archives from a dedicated isolated environment."""

from __future__ import annotations

import argparse
import sys

from e3sse.data.fpmd_aiida_inventory import build_inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        inventory, path = build_inventory(args.config, resume=args.resume)
    except Exception as exc:
        print(f"FPMD AiiDA inventory failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"FPMD AiiDA inventory: {inventory['status']} ({inventory['inventory_id']})")
    print(f"JSON inventory: {path}")
    return 0 if inventory["status"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
