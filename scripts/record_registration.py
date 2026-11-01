#!/usr/bin/env python3
"""Record the OSF registration of a frozen protocol as a separate addendum.

Writes ``registration_<status>.json`` next to the frozen ``protocol.json``; the
protocol file and its content-derived ID are never modified.  An existing record for
the same status is never overwritten: ``--resume`` verifies that it is identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from e3sse.protocol.contracts import validate_pilot_amendment
from e3sse.protocol.registration import build_registration_addendum, protocol_file_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path, help="frozen protocol.json")
    parser.add_argument(
        "--status", required=True, choices=("pending_submission", "submitted", "registered")
    )
    parser.add_argument("--osf-id", default=None, help="OSF registration identifier, as issued")
    parser.add_argument("--doi", default=None)
    parser.add_argument("--submitted-at-utc", default=None, help="RFC 3339, ending in Z")
    parser.add_argument("--visibility", default=None, choices=("embargoed", "public"))
    parser.add_argument(
        "--pilot-amendment", type=Path, default=None, help="pilot amendment deposited by this update"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    raw = args.protocol.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    if hashlib.sha256(raw).hexdigest() != protocol_file_sha256(protocol):
        raise SystemExit(f"{args.protocol} is not byte-identical to the builder's output")
    pilot_id = None
    if args.pilot_amendment is not None:
        amendment = json.loads(args.pilot_amendment.read_text(encoding="utf-8"))
        validate_pilot_amendment(amendment, protocol)
        pilot_id = amendment["amendment_id"]
    addendum = build_registration_addendum(
        protocol,
        status=args.status,
        osf_registration_id=args.osf_id,
        doi=args.doi,
        submitted_at_utc=args.submitted_at_utc,
        visibility=args.visibility,
        pilot_amendment_id=pilot_id,
    )
    suffix = f"_pilot-{pilot_id[:12]}" if pilot_id else ""
    path = args.protocol.parent / f"registration_{args.status}{suffix}.json"
    if path.exists():
        if not args.resume:
            raise SystemExit(f"{path} exists; use --resume")
        if json.loads(path.read_text(encoding="utf-8")) != addendum:
            raise SystemExit("existing registration record differs; refusing to overwrite")
        print(path)
        return 0
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(addendum, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.link(temporary, path)
    os.unlink(temporary)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
