#!/usr/bin/env python3
"""Validate Phase B inputs and materialize the E3-SSE protocol freeze."""

from __future__ import annotations

import argparse
import sys

from e3sse.protocol import ProtocolFreezeError, materialize_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON protocol-freeze configuration")
    parser.add_argument("--resume", action="store_true", help="reuse only an exact verified run")
    parser.add_argument(
        "--git-sha", default=None, help="reviewed commit SHA; must equal HEAD of a clean tree"
    )
    args = parser.parse_args(argv)
    try:
        protocol, path = materialize_protocol(
            args.config, resume=args.resume, git_sha=args.git_sha
        )
    except ProtocolFreezeError as exc:
        print(f"Protocol materialization blocked: {exc}", file=sys.stderr)
        return 1
    print(f"Protocol ID: {protocol['protocol_id']}")
    print(f"Protocol: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
