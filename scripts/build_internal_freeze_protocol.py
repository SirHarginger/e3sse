#!/usr/bin/env python3
"""Materialize the reliability-gated internal-freeze protocol.

Reads the frozen parent ``protocol.json`` of the registration-gated historical
protocol, byte-verified against the SHA-256 pinned in the configuration.  Validates
the internal-freeze configuration and writes
``<output_root>/<protocol_id>/internal_freeze_protocol.json`` plus
``resolved_config.json``.  The parent protocol directory is only read.  An existing
artifact is never overwritten; ``--resume`` verifies that it is byte-identical.

An artifact with blocking open items is written as ``candidate_unfrozen`` and grants
no outcome access.  Production materialization requires a clean worktree at the
reviewed commit (``--git-sha``) and evidence that no Study-1 outcome was read.

Subcommand ``pilot-record`` binds a validated pilot amendment to a frozen
internal-freeze protocol.  It replaces the OSF update that deposited the amendment
under the parent governance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from e3sse.protocol.contracts import validate_pilot_amendment
from e3sse.protocol.freeze import (
    ProtocolFreezeError,
    _atomic_json,
    _validate_repository_state,
    canonical_sha256,
)
from e3sse.protocol.internal_freeze import (
    InternalFreezeError,
    build_internal_freeze_protocol,
    build_internal_pilot_record,
)
from e3sse.protocol.registration import protocol_file_sha256

REPOSITORY = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def _publish(path: Path, value: dict, *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise ProtocolFreezeError(f"{path} exists; use --resume to verify it")
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ProtocolFreezeError(f"existing {path.name} differs; refusing to overwrite")
        return
    _atomic_json(path, value)


def _load_parent(config: dict, override: Path | None) -> dict:
    path = override or Path(config["parent_protocol"]["directory"]) / "protocol.json"
    parent, raw = _read_json(path)
    if hashlib.sha256(raw).hexdigest() != protocol_file_sha256(parent):
        raise InternalFreezeError(f"{path} is not byte-identical to the freeze builder output")
    return parent


def outcome_access_evidence(study1_config: Path) -> dict:
    """Check the 'no outcome read' declaration against every Study-1 run record.

    The Study-1 output root comes from the canonical Study-1 configuration, never from
    a typed path.  Every protocol subdirectory below it is scanned for per-invocation
    outcome-access records and run records; any such record falsifies the declaration.
    A missing root is recorded explicitly (no Study-1 run has ever written output).
    """

    config = json.loads(study1_config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    if not root.exists():
        return {"checked": True, "study1_output_root": str(root), "root_exists": False,
                "n_entries": 0, "listing_sha256": canonical_sha256([])}
    if not root.is_dir():
        raise InternalFreezeError(f"Study-1 output root {root} is not a directory")
    found = sorted(str(p.relative_to(root)) for pattern in
                   ("*/inference/*/outcome_access/*.json", "*/inference/*/study1_runs/*.json")
                   for p in root.glob(pattern))
    if found:
        raise InternalFreezeError(
            f"outcome_access_declaration is falsified by {len(found)} record(s), e.g. {found[0]}")
    listing = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    return {"checked": True, "study1_output_root": str(root), "root_exists": True,
            "n_entries": len(listing), "listing_sha256": canonical_sha256(listing)}


def command_build(args: argparse.Namespace) -> int:
    config, _ = _read_json(args.config)
    parent = _load_parent(config, args.parent_protocol)
    if not args.synthetic and not args.git_sha:
        raise InternalFreezeError("production materialization needs --git-sha of the reviewed commit")
    evidence = outcome_access_evidence(args.study1_config)
    mode = "synthetic" if args.synthetic else "production"
    git_sha, branch, dirty = _validate_repository_state(
        REPOSITORY, expected_sha=args.git_sha, require_clean=mode == "production"
    )
    internal_protocol = build_internal_freeze_protocol(parent, config, git_sha=git_sha, materialization_mode=mode)
    root = args.output_root or Path(config["output_root"])
    data_root = Path(json.loads(args.server_config.read_text(encoding="utf-8"))["paths"]["data_root"])
    if root.resolve() == data_root.resolve() or data_root.resolve() in root.resolve().parents:
        raise InternalFreezeError("the output root may not lie under the raw data root")
    directory = root / internal_protocol["protocol_id"]
    _publish(directory / "internal_freeze_protocol.json", internal_protocol, resume=args.resume)
    _publish(
        directory / "resolved_config.json",
        {"config": config, "config_sha256": canonical_sha256(config), "git_sha": git_sha,
         "branch": branch, "worktree_dirty": dirty, "outcome_access_evidence": evidence},
        resume=args.resume,
    )
    print(f"Internal-freeze protocol ID: {internal_protocol['protocol_id']}")
    print(f"Freeze status: {internal_protocol['freeze_status']}")
    if internal_protocol["freeze_blockers"]:
        print(f"Blocking open items: {', '.join(internal_protocol['freeze_blockers'])}")
    print(f"Internal-freeze protocol: {directory / 'internal_freeze_protocol.json'}")
    return 0


def command_pilot_record(args: argparse.Namespace) -> int:
    internal_protocol, _ = _read_json(args.internal_protocol)
    parent, raw = _read_json(args.parent_protocol)
    if hashlib.sha256(raw).hexdigest() != protocol_file_sha256(parent):
        raise InternalFreezeError("parent protocol.json is not byte-identical")
    amendment, _ = _read_json(args.pilot_amendment)
    validate_pilot_amendment(amendment, parent)
    record = build_internal_pilot_record(internal_protocol, parent, amendment)
    path = args.internal_protocol.parent / f"internal_pilot_record_{amendment['amendment_id'][:12]}.json"
    _publish(path, record, resume=args.resume)
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="materialize the internal-freeze protocol")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--parent-protocol", type=Path, default=None,
                       help="parent protocol.json (default: config parent directory)")
    build.add_argument("--git-sha", default=None, help="reviewed commit; must equal HEAD")
    build.add_argument("--output-root", type=Path, default=None)
    build.add_argument("--study1-config", type=Path,
                       default=REPOSITORY / "configs" / "study1_softening_atlas.json",
                       help="canonical Study-1 configuration (its output_root is scanned)")
    build.add_argument("--server-config", type=Path,
                       default=REPOSITORY / "configs" / "server.json")
    build.add_argument("--synthetic", action="store_true",
                       help="test materialization; never grants outcome access")
    build.add_argument("--resume", action="store_true")
    build.set_defaults(handler=command_build)
    pilot = commands.add_parser("pilot-record", help="bind a pilot amendment to the internal-freeze protocol")
    pilot.add_argument("--internal-freeze-protocol", dest="internal_protocol", type=Path, required=True)
    pilot.add_argument("--parent-protocol", type=Path, required=True)
    pilot.add_argument("--pilot-amendment", type=Path, required=True)
    pilot.add_argument("--resume", action="store_true")
    pilot.set_defaults(handler=command_pilot_record)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ProtocolFreezeError, InternalFreezeError, ValueError, OSError) as exc:
        print(f"Internal-freeze protocol blocked: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
