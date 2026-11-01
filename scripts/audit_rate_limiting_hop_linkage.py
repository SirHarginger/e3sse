#!/usr/bin/env python3
"""Read-only Gate-1 linkage audit: BVEL13k <-> nebBVSE122k <-> nebDFT2k identifiers.

Canonical bindings only, no new paths:
* ``--config`` (``configs/server.json``) gives ``paths.data_root``; raw data are
  ``<data_root>/raw`` and files are found with ``dataset_audit.discover_datasets``.
* ``--protocol-config`` (``configs/protocol_freeze.json``) pins the G0 input manifest.
  Its SHA-256 is verified, and every CSV read must match its G0 identity (path and
  SHA-256), otherwise the audit stops.
* Output goes to ``<outputs_root>/gate1_linkage/rate_limiting_hop_linkage_audit.json``
  and is never overwritten.

Per dataset, exactly one CSV with the required columns is read.  Only these columns
are extracted: BVEL13k ``material_id``, ``chemsys``; nebBVSE122k ``material_id``,
``edge_id``, ``chemsys``, ``em_bvse`` (BVSE covariate); nebDFT2k ``material_id``,
``edge_id``, ``chemsys``.  A forbidden-column list asserts that no DFT outcome column
is ever extracted.  Chemical systems are normalised by the frozen normaliser.  A
material with any unparseable, prefix-mismatched or conflicting row is ineligible as
a whole (``ambiguous_edges``); it is never analysed on a partial graph.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from e3sse.config import load_config
from e3sse.data.dataset_audit import discover_datasets
from e3sse.protocol.partitions import normalize_chemical_system
from e3sse.screening.rate_limiting_hop import parse_edge_id, rate_limiting_hop

AUDIT_SCHEMA = "e3sse.rate-limiting-hop-linkage-audit"
REQUIRED = {
    "BVEL13k": ("material_id", "chemsys"),
    "nebBVSE122k": ("material_id", "edge_id", "chemsys", "em_bvse"),
    "nebDFT2k": ("material_id", "edge_id", "chemsys"),
}
FORBIDDEN_COLUMNS = frozenset({"em_dft", "em", "barrier", "e_dft"})
DIMENSIONS = (1, 2, 3)
CODE_FILES = ("scripts/audit_rate_limiting_hop_linkage.py",
              "src/e3sse/screening/rate_limiting_hop.py")


class AuditError(RuntimeError):
    """The audit cannot proceed without guessing; it stops instead."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle), [])


def select_index(dataset: str, files: list[Path]) -> Path:
    """Exactly one CSV carrying the required columns; anything else stops the audit."""

    wanted = set(REQUIRED[dataset])
    matches = [p for p in files if p.suffix.lower() == ".csv" and wanted <= set(_header(p))]
    if len(matches) != 1:
        raise AuditError(f"{dataset}: expected exactly one CSV with {sorted(wanted)}, "
                         f"found {[str(p) for p in matches]}")
    return matches[0]


def read_columns(path: Path, wanted: tuple[str, ...]) -> tuple[list[dict[str, str]], int]:
    if set(wanted) & FORBIDDEN_COLUMNS:
        raise AuditError(f"refusing to extract outcome columns {sorted(set(wanted) & FORBIDDEN_COLUMNS)}")
    rows, short = [], 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = [header.index(c) for c in wanted]
        for row in reader:
            if len(row) < len(header):
                short += 1
                continue
            rows.append({c: row[i] for c, i in zip(wanted, index, strict=True)})
    return rows, short


def verify_against_g0(path: Path, digest: str, g0_inputs: dict) -> None:
    identities = [item for items in g0_inputs.values() if isinstance(items, list)
                  for item in items if isinstance(item, dict)]
    match = [i for i in identities if Path(i.get("path", "")).resolve() == path.resolve()]
    if len(match) != 1:
        raise AuditError(f"{path}: not exactly one identity in the G0 input manifest")
    if match[0].get("sha256") not in (None, digest):
        raise AuditError(f"{path}: SHA-256 differs from the G0 input manifest")


def _chemsys(value: str) -> str | None:
    try:
        return normalize_chemical_system(value)
    except ValueError:
        return None


def audit(raw_root: Path, g0_inputs: dict) -> dict:
    inventory = discover_datasets(raw_root)
    tables, provenance = {}, {}
    for dataset, columns in REQUIRED.items():
        path = select_index(dataset, inventory.get(dataset, []))
        digest = _sha256(path)
        verify_against_g0(path, digest, g0_inputs)
        rows, short = read_columns(path, columns)
        tables[dataset] = rows
        provenance[dataset] = {"path": str(path), "sha256": digest, "rows": len(rows),
                               "short_rows_skipped": short, "columns_read": list(columns)}
    outcome_columns_read = any(set(p["columns_read"]) & FORBIDDEN_COLUMNS
                               for p in provenance.values())

    rows_by_material: dict[str, list[dict]] = defaultdict(list)
    ambiguous: dict[str, str] = {}
    exact_duplicates = 0
    seen: dict[str, float] = {}
    for row in tables["nebBVSE122k"]:
        material, edge = row["material_id"], row["edge_id"]
        try:
            parsed = parse_edge_id(edge)
            barrier = float(row["em_bvse"])
        except ValueError:
            ambiguous.setdefault(material, "unparseable_edge_or_barrier")
            continue
        if parsed["material_id"] != material:
            ambiguous.setdefault(material, "edge_prefix_mismatch")
            continue
        if edge in seen:
            if seen[edge] != barrier:
                ambiguous.setdefault(material, "conflicting_duplicate_edge")
            else:
                exact_duplicates += 1
            continue
        seen[edge] = barrier
        rows_by_material[material].append({"edge_id": edge, "em": barrier})

    def analyse(materials):
        result = {}
        for d in DIMENSIONS:
            statuses, ties, chosen = Counter(), 0, {}
            for material in materials:
                if material in ambiguous:
                    statuses["ambiguous_edges"] += 1
                    continue
                r = rate_limiting_hop(rows_by_material.get(material, []), d)
                statuses[r["status"]] += 1
                if r["status"] == "percolating":
                    ties += len(r["tied_hops"]) > 1
                    chosen[material] = r["rate_limiting_edge_ids"]
            result[str(d)] = {"status_counts": dict(sorted(statuses.items())), "tie_cases": ties,
                              "eligible": len(chosen), "_chosen": chosen}
        return result

    bvel_ids = [r["material_id"] for r in tables["BVEL13k"]]
    bvel_materials = sorted(set(bvel_ids))
    dft_materials = sorted({r["material_id"] for r in tables["nebDFT2k"]})
    dft_edges = {r["edge_id"] for r in tables["nebDFT2k"]}
    pool, calibration = analyse(bvel_materials), analyse(dft_materials)
    for d in calibration:
        chosen = calibration[d].pop("_chosen")
        calibration[d]["rate_limiting_hop_in_nebDFT2k"] = sum(
            any(e in dft_edges for e in ids) for ids in chosen.values())
        pool[d].pop("_chosen")
    systems = {name: {r["material_id"]: _chemsys(r["chemsys"]) for r in tables[name]}
               for name in REQUIRED}
    inconsistent = sorted(
        m for m in set(systems["BVEL13k"]) & set(systems["nebBVSE122k"])
        if systems["BVEL13k"][m] != systems["nebBVSE122k"][m])
    bvel_systems = Counter(s for s in systems["BVEL13k"].values() if s)
    dft_systems = set(filter(None, systems["nebDFT2k"].values()))
    return {
        "schema": AUDIT_SCHEMA,
        "outcome_columns_read": outcome_columns_read,
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "inputs": provenance,
        "pool": {"total_candidate_structures": len(bvel_materials),
                 "duplicate_material_rows": len(bvel_ids) - len(bvel_materials),
                 "linked": sum(m in rows_by_material or m in ambiguous for m in bvel_materials),
                 "unlinked": sum(m not in rows_by_material and m not in ambiguous
                                 for m in bvel_materials),
                 "chemical_systems": len(bvel_systems),
                 "unparseable_chemical_systems": sum(s is None for s in systems["BVEL13k"].values()),
                 "structures_per_chemical_system_max": max(bvel_systems.values(), default=0),
                 "by_dimension": pool},
        "calibration_side": {"nebDFT2k_materials": len(dft_materials),
                             "nebDFT2k_edges": len(dft_edges),
                             "nebDFT2k_edges_in_nebBVSE122k": len(dft_edges & set(seen)),
                             "by_dimension": calibration},
        "overlap": {"materials_in_pool_and_nebDFT2k": len(set(bvel_materials) & set(dft_materials)),
                    "chemical_systems_in_pool_and_nebDFT2k": len(set(bvel_systems) & dft_systems)},
        "edge_quality": {"ambiguous_materials": len(ambiguous),
                         "ambiguous_reasons": dict(Counter(ambiguous.values())),
                         "exact_duplicate_rows": exact_duplicates, "unique_edges": len(seen),
                         "chemsys_inconsistent_materials": len(inconsistent)},
    }


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="configs/server.json")
    parser.add_argument("--protocol-config", type=Path, required=True,
                        help="configs/protocol_freeze.json (pins the G0 input manifest)")
    args = parser.parse_args(argv)
    config = load_config(args.config)  # also rejects outputs under data roots
    data_root = Path(config["paths"]["data_root"])
    output_dir = Path(config["paths"]["outputs_root"]) / "gate1_linkage"
    path = output_dir / "rate_limiting_hop_linkage_audit.json"
    if path.exists():
        raise SystemExit(f"{path} exists; audits are never overwritten")
    protocol = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    manifest = protocol["inputs"]["g0_input_manifest"]
    manifest_path = Path(manifest["path"])
    if _sha256(manifest_path) != manifest["sha256"]:
        raise SystemExit("G0 input manifest SHA-256 differs from the pinned value")
    g0_inputs = json.loads(manifest_path.read_text(encoding="utf-8")).get("inputs", {})
    try:
        report = audit(data_root / "raw", g0_inputs)
    except AuditError as exc:
        raise SystemExit(f"linkage audit stopped: {exc}") from exc
    repo = Path(__file__).resolve().parents[1]
    report["provenance"] = {
        "git_sha": _git(repo, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(repo, "status", "--porcelain")),
        "code_sha256": {f: _sha256(repo / f) for f in CODE_FILES},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": ["audit_rate_limiting_hop_linkage.py", *(argv or [])],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
