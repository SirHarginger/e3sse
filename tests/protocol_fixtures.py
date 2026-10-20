"""Shared synthetic fixtures for protocol tests (no scientific data)."""

from __future__ import annotations

from e3sse.protocol.contracts import CANONICAL_UNITS, inference_record_key
from e3sse.protocol.decisions import PROTOCOL_ITEM_CATEGORIES


def canonical_units(*, verified: bool = True) -> dict:
    """Unit contract: frozen canonical units plus (synthetic) source-unit evidence."""

    return {
        name: {
            "canonical_analysis_unit": unit,
            "production_source_unit_evidence": {
                "source_unit": unit if verified else None,
                "conversion": "identity" if verified else None,
                "status": "verified" if verified else "unresolved",
                "evidence": "synthetic schema fixture" if verified else None,
            },
        }
        for name, unit in CANONICAL_UNITS.items()
    }


def resolved_register() -> list[dict]:
    """Every register item resolved with evidence (paper items cite the design)."""

    return [
        {
            "id": identifier,
            "resolution_category": category,
            "blocking": False,
            "resolution": {
                "status": "resolved",
                "evidence": "Research Design 5.4 fixture citation"
                if category == "resolve_from_paper"
                else "registration fixture",
            },
        }
        for identifier, category in PROTOCOL_ITEM_CATEGORIES.items()
    ]

_SYSTEM_BY_ROLE = {"training_or_calibration": "Li-O", "test": "Li-S"}
ASSIGNMENT = {system: role for role, system in _SYSTEM_BY_ROLE.items()}


def with_systems(records):
    """Give fixture frames the chemical system whose frozen side matches their label."""

    return [
        {**record, "chemical_system_id": _SYSTEM_BY_ROLE[record["partition_role"]]}
        for record in records
    ]


_FROZEN_FIELDS = (
    "frame_id",
    "hop_id",
    "material_id",
    "chemical_system_id",
    "published_split",
    "mapping_status",
    "raw_file_kind",
    "origin_evidence_status",
    "origin_evidence",
)


def frozen_universe(records):
    """Frozen membership rows plus a protocol stub whose digest and repeat 0 bind them."""

    import hashlib
    import json
    from pathlib import Path

    membership = [{field: record.get(field) for field in _FROZEN_FIELDS} for record in records]
    digest = hashlib.sha256(
        json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config = json.loads(Path("configs/protocol_freeze.json").read_text(encoding="utf-8"))
    support = next(
        item for item in config["protocol"]["unresolved_items"]
        if item["id"] == "estimator_minimum_support"
    )
    protocol = {
        "protocol_id": "f" * 64,
        "unresolved_items": [support],
        "partition_definition": {
            "partition_universe": {"membership_sha256": digest},
            "repeated_partitions": {
                "partitions": [{"repeat_index": 0, "assignment_by_chemical_system": ASSIGNMENT}]
            },
        },
    }
    return protocol, membership


def complete_inference_record(record: dict) -> dict:
    """Fill the v1 contract fields that depend on other fields of a fixture record."""

    record = dict(record)
    record.setdefault("analysis_population", "primary")
    record.setdefault("atomic_numbers_sha256", "9" * 64)
    record.setdefault("raw_energy_kind", "total")
    record.setdefault("raw_energy", record["energy"])
    record.setdefault("energy_reference_scheme", "model_native")
    record.setdefault("energy_corrections_applied", [])
    record.setdefault("attempt_id", f"attempt-{record['frame_id']}")
    record.setdefault("attempt_number", 1)
    model = dict(record["model_provenance"])
    model["reference_compatibility"] = dict(record["reference_compatibility"])
    record["model_provenance"] = model
    record["record_key"] = inference_record_key(
        protocol_id=record["protocol_id"],
        frame_id=record["frame_id"],
        weights_sha256=model["weights_sha256"],
        config_hash=record["config_hash"],
    )
    return record
