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


def unit_evidence_artifact(
    evidence: dict, *, inventory_sha256: str, files: list[str] | None = None
) -> dict:
    """The exact wrapper ``scripts/collect_unit_evidence.py`` writes to evidence.json."""

    return {
        "evidence": {"schema_version": "e3sse.unit-evidence.v2", **evidence},
        "fingerprint": {
            "schema": "e3sse.unit-evidence.v2",
            "git_sha": "f" * 40,
            "inventory_sha256": inventory_sha256,
            "files": sorted(files or ["/srv/host/data/MPLiTrj_train.xyz"]),
            "train_nonpath_frame_indices": [0, 3, 7],
        },
        "python": "3.11.16 (main) [GCC]",
    }


def production_evidence(models: list[dict], root=None, unit_evidence=None) -> dict:
    """Pinned production evidence binding one passed smoke report to each model.

    With ``root`` the reports, locks and unit evidence are written as real files so
    materialization can re-hash them; otherwise digests are placeholders.
    """

    import hashlib
    import json
    from pathlib import Path

    def pin(name: str, content: dict) -> dict:
        if root is None:
            return {"path": f"/evidence/{name}", "sha256": "9" * 64}
        path = Path(root) / "evidence" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, sort_keys=True))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    environments: dict[str, dict] = {}
    reports: dict[str, dict] = {}
    for model in models:
        environment = f"env-{model['package_name']}-{model['package_version']}"
        entry = environments.setdefault(environment, {
            "environment_lock": pin(f"{environment}.txt", {"lock": environment}),
            "python_version": "3.11.16",
            "pinned_packages": {model["package_name"]: model["package_version"],
                                "numpy": "1.26.4"},
            "lock_recorded_pins": {},
            "environment_variables": {},
            "models": [],
        })
        entry["models"].append(model["canonical_name"])
        report = {
            "fingerprint": {
                "model": model,
                "checkpoint_sha256": model["weights_sha256"],
                "package_version": model["package_version"],
                "adapter_version": model["adapter_version"],
                "smoke_test_version": "e3sse-unit-smoke-v3",
                "git_sha": "b" * 40,
                "git_dirty": False,
                "dependency_versions": {"numpy": "1.26.4"},
            },
            "result": {"passed": True},
            "python": "3.11.16 (main) [GCC]",
        }
        reports[model["canonical_name"]] = {
            **pin(f"{model['canonical_name']}.json", report),
            "runtime_environment": environment,
        }
    return {
        "unit_evidence": pin(
            "unit_evidence.json",
            unit_evidence or unit_evidence_artifact({}, inventory_sha256="5" * 64),
        ),
        "runtime_environments": environments,
        "model_smoke_reports": reports,
    }
