#!/usr/bin/env python3
"""Outcome-free Gate-1 structural population audit at the frozen d = 3.

Completes the structural bookkeeping after the Gate-1 linkage audit: the unique
normalized chemical systems among the nebDFT2k materials whose own rate-limiting hop
is a nebDFT2k hop, then the structure rule per chemical system and the published-
split classes (development / P / Q candidates).  Rules:
``e3sse.screening.gate1_populations``.

Reads exactly the identifier and BVSE-covariate columns of the Gate-1 linkage audit,
through its own verified readers: the G0 identity of every CSV is checked and DFT
outcome columns are refused.  It also reads the frozen partition membership and
TM/Hubbard-U assignment from the frozen parent protocol directory.  The output goes
once to ``<outputs_root>/gate1_linkage/gate1_structural_population_audit-<fp>.json``
(the last printed line is its path).
``--resume`` verifies that an existing audit has the same input and code identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from e3sse.config import load_config
from e3sse.execution.sharded_execution import (
    canonical_sha256,
    ensure_safe_output_dir,
    publish_json_once,
)
from e3sse.screening.gate1_populations import structural_populations
from e3sse.softening import study1_inputs as inputs

REPOSITORY = Path(__file__).resolve().parents[1]
CODE_FILES = ("scripts/audit_gate1_structural_populations.py",
              "scripts/audit_rate_limiting_hop_linkage.py",
              "src/e3sse/production/runners/gate1_tables.py",
              "src/e3sse/screening/gate1_populations.py",
              "src/e3sse/screening/rate_limiting_hop.py",
              "src/e3sse/protocol/partitions.py",
              "src/e3sse/softening/study1_inputs.py")


def _linkage_audit_module():
    """The executed linkage audit's own readers (G0 verification, forbidden columns)."""

    path = REPOSITORY / "scripts" / "audit_rate_limiting_hop_linkage.py"
    spec = importlib.util.spec_from_file_location("audit_rate_limiting_hop_linkage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPOSITORY), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="configs/server.json")
    parser.add_argument("--protocol-config", type=Path, required=True,
                        help="configs/protocol_freeze.json (pins the G0 input manifest)")
    parser.add_argument("--study1-config", type=Path, required=True,
                        help="configs/study1_softening_atlas.json (frozen parent protocol)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="audit directory (default <outputs_root>/gate1_linkage)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    linkage = _linkage_audit_module()
    config = load_config(args.config)  # rejects outputs under the data root
    data_root = Path(config["paths"]["data_root"])
    output_dir = ensure_safe_output_dir(
        args.output_dir or Path(config["paths"]["outputs_root"]) / "gate1_linkage", [data_root])
    protocol_config = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    manifest = protocol_config["inputs"]["g0_input_manifest"]
    if _sha256(Path(manifest["path"])) != manifest["sha256"]:
        raise SystemExit("G0 input manifest SHA-256 differs from the pinned value")
    g0_inputs = json.loads(Path(manifest["path"]).read_text(encoding="utf-8")).get("inputs", {})
    study1, _ = inputs.load_study1_config(args.study1_config)
    protocol_dir = Path(study1["frozen_protocol"]["directory"])
    protocol = inputs.load_frozen_protocol(study1, directory=protocol_dir)
    from e3sse.production.runners.gate1_tables import linkage_tables

    try:
        tables = linkage_tables(data_root / "raw", g0_inputs)
    except linkage.AuditError as exc:
        raise SystemExit(f"structural population audit stopped: {exc}") from exc
    provenance = tables.pop("_provenance")
    identity = {
        "g0_input_manifest_sha256": manifest["sha256"],
        "inputs": {name: item["sha256"] for name, item in provenance.items()},
        "membership_sha256": protocol["partition_definition"]["partition_universe"][
            "membership_sha256"],
        "frozen_protocol_id": protocol["protocol_id"],
        "code_sha256": {name: _sha256(REPOSITORY / name) for name in CODE_FILES},
    }
    # One audit per identity: a code or input change writes a new file, never a collision.
    path = output_dir / f"gate1_structural_population_audit-{canonical_sha256(identity)[:16]}.json"
    if path.exists():
        if not args.resume:
            raise SystemExit(f"{path} exists; audits are never overwritten (use --resume)")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("identity") != identity:
            raise SystemExit(f"{path} has a different input or code identity; inspect it")
        print(f"verified existing audit {path}")
        print(path)
        return 0
    from e3sse.protocol.chemistry import primary_exclusion_elements

    frozen_elements = sorted(protocol["inclusion_exclusion_rules"]["primary_tm_hubbard_u"][
        "excluded_elements"])
    if frozen_elements != primary_exclusion_elements():
        raise SystemExit("frozen TM/Hubbard-U excluded_elements differ from the code rule")
    records = inputs.load_membership(protocol_dir, protocol)
    partition = inputs.load_partition_manifest(protocol_dir, protocol)
    report = structural_populations(
        nebbvse_rows=tables["nebBVSE122k"], bvel_rows=tables["BVEL13k"],
        nebdft_rows=tables["nebDFT2k"], membership_records=records,
        primary_assignment=partition["primary_population"]["assignment_by_chemical_system"],
        excluded_elements=protocol["inclusion_exclusion_rules"]["primary_tm_hubbard_u"][
            "excluded_elements"],
    )
    # Denylist assertion; the real guard is the REQUIRED column allowlist of the readers.
    report["outcome_columns_read"] = any(set(p["columns_read"]) & linkage.FORBIDDEN_COLUMNS
                                         for p in provenance.values())
    report["identity"] = identity
    report["inputs"] = provenance
    report["provenance"] = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unavailable"),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": list(sys.argv) if argv is None else
        ["audit_gate1_structural_populations.py", *argv],
    }
    publish_json_once(path, report, root=output_dir)
    print(json.dumps(report["counts"], sort_keys=True))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
