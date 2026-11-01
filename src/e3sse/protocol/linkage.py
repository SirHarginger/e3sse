"""Outcome-free linkage contract for the LiTraj evaluation universe."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from e3sse.data.litraj_provenance import parse_edge_id

from .partitions import CHEMICAL_SYSTEM_ALGORITHM, normalize_chemical_system

NEBDFT2K_LINKAGE_SCHEMA = "e3sse.nebdft2k.partition-linkage.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPLITS = {"train", "val", "test"}
_ELIGIBILITY = {"included", "excluded_preoutcome"}


class EvaluationLinkageError(ValueError):
    """Evaluation linkage cannot prove the required grouping invariants."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_nebdft2k_linkage(
    payload: Mapping[str, Any],
    *,
    g0_run_id: str,
    g0_input_identity_digest: str,
    _skip_declared_hash: bool = False,
) -> list[dict[str, Any]]:
    """Validate an identifier-only linkage without reading barrier outcomes."""

    if payload.get("schema_version") != NEBDFT2K_LINKAGE_SCHEMA:
        raise EvaluationLinkageError("incompatible nebDFT2k linkage schema")
    if payload.get("contains_outcomes") is not False:
        raise EvaluationLinkageError("nebDFT2k partition linkage must be outcome-free")
    if (
        payload.get("source_g0_run_id") != g0_run_id
        or payload.get("source_input_identity_digest") != g0_input_identity_digest
        or payload.get("chemical_system_algorithm") != CHEMICAL_SYSTEM_ALGORITHM
    ):
        raise EvaluationLinkageError("nebDFT2k linkage is not bound to the selected G0 inputs")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise EvaluationLinkageError("nebDFT2k linkage records must be non-empty")

    normalized: list[dict[str, Any]] = []
    seen_edges: set[str] = set()
    material_facts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        if not isinstance(record, Mapping):
            raise EvaluationLinkageError("nebDFT2k linkage record must be an object")
        edge_id = record.get("edge_id")
        material_id = record.get("material_id")
        split = record.get("published_split")
        eligibility = record.get("eligibility_status")
        source_sha = record.get("source_sha256")
        if not isinstance(edge_id, str) or not edge_id or edge_id in seen_edges:
            raise EvaluationLinkageError("nebDFT2k edge identifiers must be unique and non-empty")
        if not isinstance(material_id, str) or parse_edge_id(edge_id) != material_id:
            raise EvaluationLinkageError("nebDFT2k edge/material identity mismatch")
        if split not in _SPLITS:
            raise EvaluationLinkageError("nebDFT2k published split must be train, val, or test")
        if eligibility not in _ELIGIBILITY:
            raise EvaluationLinkageError("invalid pre-outcome eligibility status")
        if eligibility == "excluded_preoutcome" and not record.get("exclusion_reason"):
            raise EvaluationLinkageError("pre-outcome exclusion requires a reason")
        if not isinstance(source_sha, str) or _SHA256.fullmatch(source_sha) is None:
            raise EvaluationLinkageError("nebDFT2k source identity requires lowercase SHA-256")
        system = normalize_chemical_system(record.get("chemical_system_id", ""))
        material_facts[material_id].add((system, split))
        seen_edges.add(edge_id)
        normalized.append(
            {
                "record_kind": "nebdft2k_hop",
                "frame_id": None,
                "hop_id": edge_id,
                "material_id": material_id,
                "chemical_system_id": system,
                "published_split": split,
                "eligibility_status": eligibility,
                "exclusion_reason": record.get("exclusion_reason"),
                "source_identifier": record.get("source_identifier"),
                "source_sha256": source_sha,
            }
        )
    conflicts = {
        material: sorted(facts) for material, facts in material_facts.items() if len(facts) != 1
    }
    if conflicts:
        raise EvaluationLinkageError(
            f"nebDFT2K materials have conflicting chemical system/split facts: {conflicts}"
        )
    declared_count = payload.get("record_count")
    if declared_count != len(normalized) or (
        not _skip_declared_hash and payload.get("records_sha256") != _sha(normalized)
    ):
        raise EvaluationLinkageError("nebDFT2k linkage count or records hash mismatch")
    return normalized


def derive_nebdft2k_linkage(
    identities: Iterable[Mapping[str, Any]],
    *,
    data_root: Path,
    g0_run_id: str,
    g0_input_identity_digest: str,
) -> dict[str, Any]:
    """Read only the G0-pinned identifier index and discard all outcome columns."""

    candidates: list[tuple[Path, Mapping[str, Any], list[str]]] = []
    root = data_root.resolve()
    for identity in identities:
        identifier = identity.get("identifier")
        if not isinstance(identifier, str):
            raise EvaluationLinkageError("G0 nebDFT2k identity lacks an identifier")
        path = Path(identity.get("path", root / identifier)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise EvaluationLinkageError(
                "nebDFT2k source escapes the configured data root"
            ) from exc
        if not path.is_file():
            raise EvaluationLinkageError(f"missing G0-pinned nebDFT2k source: {identifier}")
        stat = path.stat()
        if stat.st_size != identity.get("bytes") or stat.st_mtime_ns != identity.get("mtime_ns"):
            raise EvaluationLinkageError(f"nebDFT2k source identity changed: {identifier}")
        if path.suffix.lower() != ".csv":
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if "edge_id" in header:
            candidates.append((path, identity, header))
    if len(candidates) != 1:
        raise EvaluationLinkageError(
            "expected exactly one G0-pinned nebDFT2k CSV index containing edge_id"
        )
    path, identity, header = candidates[0]
    required = {"edge_id", "material_id", "chemsys", "_split"}
    if not required <= set(header):
        raise EvaluationLinkageError(
            f"nebDFT2k index lacks identifier columns: {sorted(required - set(header))}"
        )
    source_sha = _file_sha256(path)
    if identity.get("sha256") not in {None, source_sha}:
        raise EvaluationLinkageError("nebDFT2k index SHA-256 differs from G0")
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "edge_id": row.get("edge_id"),
                    "material_id": row.get("material_id"),
                    "chemical_system_id": row.get("chemsys"),
                    "published_split": row.get("_split"),
                    "eligibility_status": "included",
                    "exclusion_reason": None,
                    "source_identifier": identity["identifier"],
                    "source_sha256": source_sha,
                }
            )
    preliminary = {
        "schema_version": NEBDFT2K_LINKAGE_SCHEMA,
        "contains_outcomes": False,
        "source_g0_run_id": g0_run_id,
        "source_input_identity_digest": g0_input_identity_digest,
        "chemical_system_algorithm": CHEMICAL_SYSTEM_ALGORITHM,
        "derivation": "identifier_columns_only_v1",
        # Column names only, as evidence for whether convergence flags are provided
        # (Research Design 5.3); no non-identifier value is read or copied.
        "source_columns": list(header),
        "record_count": len(records),
        "records": records,
    }
    normalized = validate_nebdft2k_linkage(
        {**preliminary, "records_sha256": "pending"},
        g0_run_id=g0_run_id,
        g0_input_identity_digest=g0_input_identity_digest,
        _skip_declared_hash=True,
    )
    preliminary["records_sha256"] = _sha(normalized)
    return preliminary


def combine_partition_universe(
    mplitrj_records: Iterable[Mapping[str, Any]],
    nebdft2k_records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join static-study units and prove cross-dataset split/system propagation."""

    mpl = [{"record_kind": "mplitrj_frame", **dict(record)} for record in mplitrj_records]
    neb = [dict(record) for record in nebdft2k_records]
    facts_by_material: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in [*mpl, *neb]:
        facts_by_material[str(record["material_id"])].add(
            (normalize_chemical_system(record["chemical_system_id"]), record["published_split"])
        )
    conflicts = {
        material: sorted(facts) for material, facts in facts_by_material.items() if len(facts) != 1
    }
    if conflicts:
        raise EvaluationLinkageError(
            f"cross-dataset material split/system propagation failed: {conflicts}"
        )
    combined = [*mpl, *neb]
    counts = Counter(record["record_kind"] for record in combined)
    evidence = {
        "schema_version": "e3sse.partition-universe.v1",
        "counts": {
            "mplitrj_frames": counts["mplitrj_frame"],
            "mplitrj_raw_hops": len({record["hop_id"] for record in mpl}),
            "nebdft2k_hops": counts["nebdft2k_hop"],
            "materials": len(facts_by_material),
            "chemical_systems": len(
                {normalize_chemical_system(record["chemical_system_id"]) for record in combined}
            ),
        },
        "cross_dataset_material_conflicts": 0,
        "membership_sha256": _sha(combined),
    }
    return combined, evidence
