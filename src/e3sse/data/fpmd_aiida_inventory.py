"""Read-only inventory of FPMD AiiDA archives through an isolated interpreter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

from e3sse.config import load_config

SCHEMA_VERSION = "e3sse.fpmd.aiida.inventory.v1"
DEFAULT_REQUIRED_CAPABILITIES = (
    "structures_recoverable",
    "trajectories_recoverable",
    "temperature_metadata_recoverable",
    "diffusion_reference_recoverable",
    "msd_reference_recoverable",
    "provenance_linkage_recoverable",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _self_digest(value: dict[str, Any]) -> str:
    return _digest({key: item for key, item in value.items() if key != "self_digest"})


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve committed policy and guarantee all derived paths are safe."""

    project_root = Path(config["_config_path"]).parent.parent
    output_root = Path(config["paths"]["outputs_root"])
    scratch_root = Path(config["paths"]["scratch_root"])
    raw_root = Path(config["paths"]["data_root"]) / "raw"
    supplied = config.get("fpmd_aiida", {})
    settings = {
        "inventory_path": str(
            (output_root / supplied.get("inventory_path", "fpmd_aiida/current.json")).resolve()
        ),
        "profile_root": str(
            (scratch_root / supplied.get("profile_root", "fpmd_aiida/profiles")).resolve()
        ),
        "migration_root": str(
            (scratch_root / supplied.get("migration_root", "fpmd_aiida/migrated")).resolve()
        ),
        "requirements_lock": str(
            (project_root / supplied.get(
                "requirements_lock", "requirements/aiida-archive.lock"
            )).resolve()
        ),
        "python_version": str(supplied.get("python_version", "3.12")),
        "aiida_version": str(supplied.get("aiida_version", "2.9.2")),
        "example_limit": int(supplied.get("example_limit", 10)),
        "allow_derived_migration": bool(supplied.get("allow_derived_migration", True)),
        "required_archives": sorted(
            supplied.get("required_archives", []), key=lambda item: item["name"]
        ),
        "required_capabilities": sorted(
            supplied.get("required_capabilities", DEFAULT_REQUIRED_CAPABILITIES)
        ),
    }
    for key in ("profile_root", "migration_root"):
        path = Path(settings[key])
        if not _within(path, scratch_root) or _within(path, raw_root):
            raise ValueError(f"{key} must be below scratch_root and outside immutable raw data")
    inventory_path = Path(settings["inventory_path"])
    if not _within(inventory_path, output_root) or _within(inventory_path, raw_root):
        raise ValueError("FPMD inventory must be below outputs_root and outside immutable raw data")
    if settings["example_limit"] < 1:
        raise ValueError("fpmd_aiida.example_limit must be positive")
    return settings


def policy_record(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: settings[key]
        for key in (
            "python_version",
            "aiida_version",
            "example_limit",
            "allow_derived_migration",
            "required_archives",
            "required_capabilities",
        )
    }


def prepare_isolated_aiida_path(config: dict[str, Any]) -> Path:
    """Set AIIDA_PATH before importing AiiDA; never select a user profile."""

    root = Path(settings_from_config(config)["profile_root"])
    root.mkdir(parents=True, exist_ok=True)
    os.environ["AIIDA_PATH"] = str(root)
    return root


def environment_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Identity of the dedicated interpreter and exact dependency lock."""

    settings = settings_from_config(config)
    try:
        import aiida  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("aiida-core is not importable in the inventory environment") from exc
    lock = Path(settings["requirements_lock"])
    if not lock.is_file():
        raise RuntimeError(f"AiiDA requirements lock is missing: {lock}")
    identity = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "aiida_core": aiida.__version__,
        "requirements_lock": str(lock),
        "requirements_lock_sha256": _sha256(lock),
    }
    if not identity["python"].startswith(f"{settings['python_version']}."):
        raise RuntimeError(
            f"inventory requires Python {settings['python_version']}.x, got {identity['python']}"
        )
    if identity["aiida_core"] != settings["aiida_version"]:
        raise RuntimeError(
            f"inventory requires aiida-core {settings['aiida_version']}, "
            f"got {identity['aiida_core']}"
        )
    identity["environment_id"] = _digest(identity)
    return identity


def _bounded(values: Iterable[Any], limit: int, provenance: str) -> dict[str, Any]:
    ordered = sorted(values, key=lambda item: json.dumps(item, sort_keys=True))
    return {
        "count": len(ordered),
        "example_limit": limit,
        "examples": ordered[:limit],
        "provenance": provenance,
    }


MAX_TEMPERATURE_K = 10_000.0
DIFFUSION_PAIR_RTOL = 1.0e-6
DIFFUSION_PAIR_ATOL = 1.0e-30
_DIFFUSION_FIELD = re.compile(
    r"^(?P<prefix>.+\.diffusion_(?P<stat>mean|sem|std))_(?P<unit>SI|cm2_s)$",
    flags=re.IGNORECASE,
)
_MSD_ARRAY = re.compile(
    r"^msd(?:_isotropic_[A-Za-z0-9]+_(?:\d+|mean|sem|std))?$",
    flags=re.IGNORECASE,
)


def _representative_values(values: Iterable[float], limit: int) -> list[float]:
    ordered = sorted({float(value) for value in values})
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indices]


def validate_temperature_array(
    values: Any, declared_unit: Any, *, example_limit: int
) -> dict[str, Any]:
    """Validate one stored temperature array without inspecting trajectory coordinates."""

    import numpy as np

    array = np.asarray(values)
    errors: list[str] = []
    if (
        array.size == 0
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        errors.append("temperatures array must be non-empty, real and numeric")
        numeric = np.asarray([], dtype=float)
    else:
        numeric = array.astype(float, copy=False).reshape(-1)
        if not np.all(np.isfinite(numeric)):
            errors.append("temperatures array contains non-finite values")

    unit_key = declared_unit.strip().lower() if isinstance(declared_unit, str) else None
    conversions: dict[str, float] = {
        "k": 0.0,
        "kelvin": 0.0,
        "kelvins": 0.0,
        "c": 273.15,
        "°c": 273.15,
        "degc": 273.15,
        "celsius": 273.15,
    }
    if unit_key not in conversions:
        status = "unit_unresolved" if not errors else "invalid"
        errors.append("temperature unit is missing or unrecognized")
        values_k = np.asarray([], dtype=float)
    else:
        values_k = numeric + conversions[unit_key]
        if values_k.size and (
            np.any(values_k <= 0.0) or np.any(values_k > MAX_TEMPERATURE_K)
        ):
            errors.append(
                f"temperature values must satisfy 0 < T <= {MAX_TEMPERATURE_K:g} K"
            )
        status = "recoverable" if not errors else "invalid"

    return {
        "status": status,
        "recoverable": status == "recoverable",
        "declared_unit": declared_unit,
        "canonical_unit": "K" if unit_key in conversions else None,
        "dtype": str(array.dtype),
        "value_count": int(array.size),
        "min_K": float(np.min(values_k)) if values_k.size and not errors else None,
        "max_K": float(np.max(values_k)) if values_k.size and not errors else None,
        "representative_K": (
            _representative_values(values_k.tolist(), example_limit)
            if values_k.size and not errors
            else []
        ),
        "errors": errors,
    }



def _aggregate_temperature_records(
    records: list[dict[str, Any]], candidate_count: int, example_limit: int
) -> dict[str, Any]:
    valid = [record for record in records if record["recoverable"]]
    invalid = [record for record in records if not record["recoverable"]]
    representative = [
        value for record in valid for value in record["representative_K"]
    ]
    recoverable = bool(valid) and not invalid
    if recoverable:
        status = "recoverable"
    elif any(record["status"] == "invalid" for record in invalid):
        status = "invalid"
    elif any(record["status"] == "unit_unresolved" for record in invalid):
        status = "unit_unresolved"
    else:
        status = "missing"
    return {
        "status": status,
        "recoverable": recoverable,
        "candidate_trajectory_count": candidate_count,
        "sampled_trajectory_count": len(records),
        "valid_trajectory_count": len(valid),
        "validation_scope": "bounded deterministic UUID-ordered sample",
        "value_count": sum(record["value_count"] for record in valid),
        "min_K": min((record["min_K"] for record in valid), default=None),
        "max_K": max((record["max_K"] for record in valid), default=None),
        "representative_K": _representative_values(representative, example_limit),
        "samples": records[:example_limit],
        "errors": [
            f"{record['uuid']}: {error}"
            for record in invalid
            for error in record["errors"]
        ][:example_limit],
    }

def _flatten_values(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_values(child, path))
    else:
        flattened[prefix] = value
    return flattened


def validate_diffusion_attributes(
    attributes: dict[str, Any], *, example_limit: int
) -> dict[str, Any]:
    """Validate stored diffusion scalars whose names explicitly encode their units."""

    fields: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    recognized_count = 0
    for path, value in sorted(_flatten_values(attributes).items()):
        match = _DIFFUSION_FIELD.fullmatch(path)
        if not match:
            continue
        recognized_count += 1
        if not isinstance(value, Real) or isinstance(value, bool):
            errors.append(f"{path}: diffusion value is not a real scalar")
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            errors.append(f"{path}: diffusion value must be finite and non-negative")
            continue
        key = (match.group("prefix").lower(), match.group("stat").lower())
        fields.setdefault(key, {})[match.group("unit").lower()] = {
            "field": path,
            "value": number,
        }

    valid_fields: list[dict[str, Any]] = []
    valid_mean_count = 0
    for (prefix, stat), representations in sorted(fields.items()):
        valid_fields.extend(representations.values())
        if stat == "mean":
            valid_mean_count += 1
        if "si" in representations and "cm2_s" in representations:
            expected_cm2_s = representations["si"]["value"] * 1.0e4
            observed_cm2_s = representations["cm2_s"]["value"]
            if not math.isclose(
                observed_cm2_s,
                expected_cm2_s,
                rel_tol=DIFFUSION_PAIR_RTOL,
                abs_tol=DIFFUSION_PAIR_ATOL,
            ):
                errors.append(
                    f"{prefix}: inconsistent SI/cm2_s pair "
                    f"({observed_cm2_s} != {expected_cm2_s})"
                )

    if valid_mean_count and not errors:
        status = "recoverable"
    elif recognized_count:
        status = "invalid"
        if not valid_mean_count:
            errors.append("recognized diffusion fields contain no valid mean")
    else:
        status = "missing"
    return {
        "status": status,
        "recoverable": status == "recoverable",
        "recognized_field_count": recognized_count,
        "valid_mean_count": valid_mean_count,
        "unit_convention": {"SI": "m^2/s", "cm2_s": "cm^2/s"},
        "conversion": "cm^2/s = (m^2/s) * 1e4",
        "fields": _bounded(valid_fields, example_limit, "stored AiiDA node attributes"),
        "errors": errors[:example_limit],
    }


def _aggregate_diffusion_records(
    records: list[dict[str, Any]], example_limit: int
) -> dict[str, Any]:
    evidence = [
        validate_diffusion_attributes(record, example_limit=example_limit)
        for record in records
    ]
    errors = [error for item in evidence for error in item["errors"]]
    valid_mean_count = sum(item["valid_mean_count"] for item in evidence)
    recognized_count = sum(item["recognized_field_count"] for item in evidence)
    invalid = any(item["status"] == "invalid" for item in evidence)
    recoverable = valid_mean_count > 0 and not invalid
    return {
        "status": "recoverable" if recoverable else ("invalid" if invalid else "missing"),
        "recoverable": recoverable,
        "records_inspected": len(records),
        "recognized_field_count": recognized_count,
        "valid_mean_count": valid_mean_count,
        "unit_convention": {"SI": "m^2/s", "cm2_s": "cm^2/s"},
        "conversion": "cm^2/s = (m^2/s) * 1e4",
        "errors": errors[:example_limit],
    }


def _msd_unit(attributes: dict[str, Any], array_name: str) -> tuple[Any, str | None]:
    for key in (
        f"units|{array_name}",
        f"unit|{array_name}",
        f"{array_name}_unit",
        "units|msd",
        "unit|msd",
        "msd_unit",
    ):
        if key in attributes:
            return attributes[key], key
    return None, None


def _canonical_squared_length_unit(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("²", "^2").replace("å", "angstrom")
    aliases = {
        "a^2": "angstrom^2",
        "ang^2": "angstrom^2",
        "angstrom^2": "angstrom^2",
        "angstrom**2": "angstrom^2",
        "m^2": "m^2",
        "m**2": "m^2",
        "cm^2": "cm^2",
        "cm**2": "cm^2",
        "nm^2": "nm^2",
        "nm**2": "nm^2",
    }
    return aliases.get(normalized)


def validate_msd_arrays(
    arrays: dict[str, Any], attributes: dict[str, Any], *, example_limit: int
) -> dict[str, Any]:
    """Validate stored MSD arrays while keeping unresolved units explicit."""

    import numpy as np

    valid: list[dict[str, Any]] = []
    unresolved: list[str] = []
    errors: list[str] = []
    recognized = [
        (name, value)
        for name, value in sorted(arrays.items())
        if _MSD_ARRAY.fullmatch(name)
    ]
    for name, values in recognized:
        array = np.asarray(values)
        if (
            array.size == 0
            or not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.complexfloating)
        ):
            errors.append(f"{name}: MSD array must be non-empty, real and numeric")
            continue
        numeric = array.astype(float, copy=False).reshape(-1)
        if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
            errors.append(f"{name}: MSD values must be finite and non-negative")
            continue
        declared_unit, unit_source = _msd_unit(attributes, name)
        canonical_unit = _canonical_squared_length_unit(declared_unit)
        if canonical_unit is None:
            unresolved.append(name)
            continue
        valid.append(
            {
                "array": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "declared_unit": declared_unit,
                "canonical_unit": canonical_unit,
                "unit_source": unit_source,
                "min": float(np.min(numeric)),
                "max": float(np.max(numeric)),
            }
        )

    numeric_count = len(valid) + len(unresolved)
    if errors:
        status = "invalid"
    elif unresolved:
        status = "unit_unresolved"
    elif valid:
        status = "recoverable"
    else:
        status = "missing"
    return {
        "status": status,
        "recoverable": status == "recoverable",
        "numeric_arrays_recoverable": numeric_count > 0 and not errors,
        "recognized_array_count": len(recognized),
        "valid_arrays": _bounded(valid, example_limit, "stored AiiDA arrays"),
        "unresolved_unit_arrays": _bounded(
            unresolved, example_limit, "numeric MSD arrays lacking unit provenance"
        ),
        "errors": errors[:example_limit],
    }


def _aggregate_msd_records(records: list[dict[str, Any]], example_limit: int) -> dict[str, Any]:
    errors = [error for item in records for error in item["errors"]]
    unresolved_examples = [
        name
        for item in records
        for name in item["unresolved_unit_arrays"]["examples"]
    ]
    unresolved_count = sum(
        item["unresolved_unit_arrays"]["count"] for item in records
    )
    valid_count = sum(item["valid_arrays"]["count"] for item in records)
    candidate_count = sum(item["recognized_array_count"] for item in records)
    payload_inspected = sum(item.get("payload_arrays_inspected", 0) for item in records)
    numeric_recoverable = any(item["numeric_arrays_recoverable"] for item in records)
    if errors:
        status = "invalid"
    elif unresolved_count:
        status = "unit_unresolved"
    elif valid_count:
        status = "recoverable"
    else:
        status = "missing"
    return {
        "status": status,
        "recoverable": status == "recoverable",
        "numeric_arrays_recoverable": numeric_recoverable and not errors,
        "records_inspected": len(records),
        "candidate_array_count": candidate_count,
        "payload_arrays_inspected": payload_inspected,
        "payload_arrays_uninspected": candidate_count - payload_inspected,
        "validation_scope": "all array-name/unit metadata; bounded array payload sample",
        "valid_array_count": valid_count,
        "unresolved_unit_arrays": _bounded(
            unresolved_examples,
            example_limit,
            "MSD arrays lacking explicit unit provenance",
        ),
        "errors": errors[:example_limit],
    }


def _reference_capabilities(
    *,
    structure_count: int,
    trajectory_count: int,
    temperature_recoverable: bool,
    diffusion_recoverable: bool,
    msd_recoverable: bool,
    direct_links: int,
    process_links: int,
) -> dict[str, bool]:
    """Derive recoverability flags from observed archive data."""

    return {
        "structures_recoverable": structure_count > 0,
        "trajectories_recoverable": trajectory_count > 0,
        "temperature_metadata_recoverable": temperature_recoverable,
        "diffusion_reference_recoverable": diffusion_recoverable,
        "msd_reference_recoverable": msd_recoverable,
        "provenance_linkage_recoverable": direct_links > 0 or process_links > 0,
    }


def _query_group_members(group_id: int, group_label: str, limit: int) -> dict[str, Any]:
    from aiida import orm  # type: ignore[import-not-found]

    query = orm.QueryBuilder()
    query.append(orm.Group, filters={"id": group_id}, tag="group")
    query.append(
        orm.Node,
        with_group="group",
        project=["uuid", "node_type", "label", "attributes"],
        tag="member",
    )
    query.order_by({"member": ["uuid"]})
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    diffusion_records: list[dict[str, Any]] = []
    msd_records: list[dict[str, Any]] = []
    msd_payload_arrays_inspected = 0
    total = 0
    for uuid, node_type, label, attributes in query.iterall(batch_size=500):
        total += 1
        node_type = str(node_type)
        counts[node_type] += 1
        reference_group = any(
            token in group_label.lower() for token in ("diffusion", "msd", "mean square")
        )
        if reference_group and isinstance(attributes, dict):
            if "diffusion" in group_label.lower():
                diffusion_records.append(attributes)
            array_names = sorted(
                key.removeprefix("array|")
                for key in attributes
                if key.startswith("array|")
                and _MSD_ARRAY.fullmatch(key.removeprefix("array|"))
            )
            if array_names:
                remaining = max(0, limit - msd_payload_arrays_inspected)
                payload_names = array_names[:remaining]
                try:
                    arrays: dict[str, Any] = {}
                    if payload_names:
                        node = orm.load_node(uuid=str(uuid))
                        arrays = {name: node.get_array(name) for name in payload_names}
                    evidence = validate_msd_arrays(
                        arrays, attributes, example_limit=limit
                    )
                    msd_payload_arrays_inspected += len(payload_names)
                except Exception as exc:
                    evidence = {
                        "status": "invalid",
                        "recoverable": False,
                        "numeric_arrays_recoverable": False,
                        "recognized_array_count": len(array_names),
                        "valid_arrays": _bounded([], limit, "stored AiiDA arrays"),
                        "unresolved_unit_arrays": _bounded(
                            [], limit, "numeric MSD arrays lacking unit provenance"
                        ),
                        "errors": [f"{type(exc).__name__}: {exc}"],
                    }
                metadata_unresolved = [
                    name
                    for name in array_names
                    if _canonical_squared_length_unit(_msd_unit(attributes, name)[0])
                    is None
                ]
                evidence["recognized_array_count"] = len(array_names)
                evidence["payload_arrays_inspected"] = len(payload_names)
                evidence["unresolved_unit_arrays"] = _bounded(
                    metadata_unresolved,
                    limit,
                    "MSD arrays lacking explicit unit provenance",
                )
                if metadata_unresolved and evidence["status"] != "invalid":
                    evidence["status"] = "unit_unresolved"
                    evidence["recoverable"] = False
                msd_records.append(evidence)
        if len(examples) < limit:
            examples.append({"uuid": str(uuid), "node_type": node_type, "label": str(label)})
    return {
        "member_count": total,
        "node_type_counts": dict(sorted(counts.items())),
        "member_examples": examples,
        "diffusion_reference": _aggregate_diffusion_records(diffusion_records, limit),
        "msd_reference": _aggregate_msd_records(msd_records, limit),
    }


def _query_link_count(
    structure_types: list[str], trajectory_types: list[str], *, through_process: bool
) -> int:
    from aiida import orm  # type: ignore[import-not-found]

    if not structure_types or not trajectory_types:
        return 0
    query = orm.QueryBuilder()
    query.append(
        orm.Node, filters={"node_type": {"in": structure_types}}, tag="structure"
    )
    if through_process:
        query.append(orm.ProcessNode, with_incoming="structure", tag="process")
        query.append(
            orm.Node,
            filters={"node_type": {"in": trajectory_types}},
            with_incoming="process",
        )
    else:
        query.append(
            orm.Node,
            filters={"node_type": {"in": trajectory_types}},
            with_incoming="structure",
        )
    return query.count()


def _query_loaded_archive(example_limit: int) -> dict[str, Any]:
    """Query metadata plus bounded named temperature/MSD arrays; coordinates are never loaded."""

    from aiida import orm  # type: ignore[import-not-found]

    node_types: Counter[str] = Counter()
    label_hits: dict[str, list[dict[str, str]]] = {"diffusion": [], "msd": []}
    query = orm.QueryBuilder()
    query.append(orm.Node, project=["uuid", "node_type", "label"], tag="node")
    query.order_by({"node": ["uuid"]})
    for uuid, node_type, label in query.iterall(batch_size=1000):
        node_type = str(node_type)
        label = str(label or "")
        node_types[node_type] += 1
        lowered = label.lower()
        if "diffusion" in lowered and len(label_hits["diffusion"]) < example_limit:
            label_hits["diffusion"].append({"uuid": str(uuid), "label": label})
        if ("msd" in lowered or "mean square" in lowered) and len(
            label_hits["msd"]
        ) < example_limit:
            label_hits["msd"].append({"uuid": str(uuid), "label": label})

    structure_types = sorted(key for key in node_types if "structure" in key.lower())
    trajectory_types = sorted(key for key in node_types if "trajectory" in key.lower())
    groups: list[dict[str, Any]] = []
    group_query = orm.QueryBuilder()
    group_query.append(orm.Group, project=["id", "label", "type_string"], tag="group")
    group_query.order_by({"group": ["label", "id"]})
    for group_id, label, type_string in group_query.iterall(batch_size=100):
        groups.append(
            {
                "label": str(label),
                "type_string": str(type_string),
                **_query_group_members(int(group_id), str(label), example_limit),
            }
        )
    groups.sort(key=lambda item: (item["label"], item["type_string"]))

    temperature_candidates = 0
    temperature_records: list[dict[str, Any]] = []
    material_identifiers: set[str] = set()
    for node_type in trajectory_types:
        trajectory_query = orm.QueryBuilder()
        trajectory_query.append(
            orm.Node,
            filters={"node_type": node_type},
            project=["uuid", "attributes", "extras"],
            tag="trajectory",
        )
        trajectory_query.order_by({"trajectory": ["uuid"]})
        for uuid, attributes, extras in trajectory_query.iterall(batch_size=200):
            if isinstance(attributes, dict) and "array|temperatures" in attributes:
                temperature_candidates += 1
                if len(temperature_records) < example_limit:
                    try:
                        node = orm.load_node(uuid=str(uuid))
                        evidence = validate_temperature_array(
                            node.get_array("temperatures"),
                            attributes.get("units|temperatures"),
                            example_limit=example_limit,
                        )
                    except Exception as exc:
                        evidence = {
                            "status": "invalid",
                            "recoverable": False,
                            "declared_unit": attributes.get("units|temperatures"),
                            "canonical_unit": None,
                            "dtype": None,
                            "value_count": 0,
                            "min_K": None,
                            "max_K": None,
                            "representative_K": [],
                            "errors": [f"{type(exc).__name__}: {exc}"],
                        }
                    temperature_records.append({"uuid": str(uuid), **evidence})
            for metadata in (attributes, extras):
                if isinstance(metadata, dict):
                    for key in ("material_id", "structure_id", "mp_id"):
                        if metadata.get(key) not in (None, ""):
                            material_identifiers.add(str(metadata[key]))

    temperature_summary = _aggregate_temperature_records(
        temperature_records, temperature_candidates, example_limit
    )
    diffusion_candidates = [
        group for group in groups if "diffusion" in group["label"].lower() and group["member_count"]
    ]
    diffusion_groups = [
        group
        for group in diffusion_candidates
        if group["diffusion_reference"]["recoverable"]
    ]
    diffusion_invalid = [
        group
        for group in diffusion_candidates
        if group["diffusion_reference"]["status"] == "invalid"
    ]
    diffusion_recoverable = bool(diffusion_groups) and not diffusion_invalid

    msd_candidates = [
        group
        for group in groups
        if group["msd_reference"]["status"] != "missing"
        or "msd" in group["label"].lower()
    ]
    msd_groups = [
        group for group in msd_candidates if group["msd_reference"]["recoverable"]
    ]
    msd_unresolved = [
        group
        for group in msd_candidates
        if group["msd_reference"]["status"] == "unit_unresolved"
    ]
    msd_invalid = [
        group
        for group in msd_candidates
        if group["msd_reference"]["status"] == "invalid"
    ]
    msd_recoverable = bool(msd_groups) and not msd_unresolved and not msd_invalid
    try:
        direct_links = _query_link_count(structure_types, trajectory_types, through_process=False)
        process_links = _query_link_count(structure_types, trajectory_types, through_process=True)
        link_error = None
    except Exception as exc:
        direct_links = process_links = 0
        link_error = f"{type(exc).__name__}: {exc}"

    structure_count = sum(node_types[key] for key in structure_types)
    trajectory_count = sum(node_types[key] for key in trajectory_types)
    return {
        "group_labels": [group["label"] for group in groups],
        "groups": groups[:example_limit],
        "group_count": len(groups),
        "node_type_counts": dict(sorted(node_types.items())),
        "structure_count": structure_count,
        "trajectory_count": trajectory_count,
        "temperature_metadata": temperature_summary,
        "diffusion_references": {
            "status": "recoverable" if diffusion_recoverable else (
                "invalid" if diffusion_invalid else "missing"
            ),
            "recoverable": diffusion_recoverable,
            "candidate_group_count": len(diffusion_candidates),
            "recoverable_group_count": len(diffusion_groups),
            "node_label_examples": label_hits["diffusion"],
            "group_evidence": [
                {
                    "label": group["label"],
                    **group["diffusion_reference"],
                }
                for group in diffusion_candidates[:example_limit]
            ],
        },
        "msd_references": {
            "status": "recoverable" if msd_recoverable else (
                "invalid" if msd_invalid else (
                    "unit_unresolved" if msd_unresolved else "missing"
                )
            ),
            "recoverable": msd_recoverable,
            "numeric_arrays_recoverable": any(
                group["msd_reference"]["numeric_arrays_recoverable"]
                for group in msd_candidates
            ),
            "candidate_group_count": len(msd_candidates),
            "recoverable_group_count": len(msd_groups),
            "node_label_examples": label_hits["msd"],
            "group_evidence": [
                {"label": group["label"], **group["msd_reference"]}
                for group in msd_candidates[:example_limit]
            ],
        },
        "provenance_linkage": {
            "direct_structure_to_trajectory_links": direct_links,
            "process_mediated_structure_to_trajectory_paths": process_links,
            "query_error": link_error,
        },
        "material_identifiers": _bounded(
            material_identifiers, example_limit, "trajectory node attributes/extras"
        ),
        "capabilities": _reference_capabilities(
            structure_count=structure_count,
            trajectory_count=trajectory_count,
            temperature_recoverable=temperature_summary["recoverable"],
            diffusion_recoverable=diffusion_recoverable,
            msd_recoverable=msd_recoverable,
            direct_links=direct_links,
            process_links=process_links,
        ),
    }


def _query_path(path: Path, example_limit: int) -> dict[str, Any]:
    from aiida.manage import get_manager  # type: ignore[import-not-found]
    from aiida.storage.sqlite_zip.backend import SqliteZipBackend  # type: ignore[import-not-found]

    manager = get_manager()
    profile = SqliteZipBackend.create_profile(path)
    manager.load_profile(profile, allow_switch=True)
    try:
        return {
            "archive_schema_version": SqliteZipBackend.get_current_archive_version(path),
            **_query_loaded_archive(example_limit),
        }
    finally:
        manager.unload_profile()


def query_archive_read_only(
    path: Path, *, environment: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    """Query source directly, or a derived migrated copy under scratch."""

    try:
        queried = _query_path(path, settings["example_limit"])
        return {
            "query_status": "success",
            "queried_path_kind": "immutable_source",
            "environment_id": environment["environment_id"],
            **queried,
        }
    except Exception as direct_exc:
        if not settings["allow_derived_migration"]:
            raise RuntimeError(f"direct read-only query failed: {direct_exc}") from direct_exc
        from aiida.storage.sqlite_zip.backend import (
            SqliteZipBackend,  # type: ignore[import-not-found]
        )
        from aiida.storage.sqlite_zip.migrator import migrate  # type: ignore[import-not-found]

        source_sha = _sha256(path)
        migration_dir = Path(settings["migration_root"]) / source_sha
        migration_dir.mkdir(parents=True, exist_ok=True)
        migrated = migration_dir / path.name
        if not migrated.exists():
            temporary = migrated.with_suffix(migrated.suffix + ".tmp")
            migrate(path, temporary, SqliteZipBackend.version_head(), force=False)
            temporary.replace(migrated)
        queried = _query_path(migrated, settings["example_limit"])
        return {
            "query_status": "success",
            "queried_path_kind": "derived_migrated_copy",
            "environment_id": environment["environment_id"],
            "direct_query_error": f"{type(direct_exc).__name__}: {direct_exc}",
            "migration": {
                "source_sha256": source_sha,
                "derived_path": str(migrated),
                "derived_sha256": _sha256(migrated),
                "target_schema_version": SqliteZipBackend.version_head(),
            },
            **queried,
        }


def _archive_policy(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in settings["required_archives"]}


def aggregate_capabilities(
    archive_results: list[dict[str, Any]], settings: dict[str, Any]
) -> dict[str, bool]:
    policy = _archive_policy(settings)
    required = [item for item in archive_results if policy[item["name"]].get("required", True)]
    capabilities = {
        "aiida_query_available": bool(required)
        and all(item.get("query_status") == "success" for item in required)
    }
    for name in settings["required_capabilities"]:
        capabilities[name] = any(
            item.get("query_status") == "success"
            and item.get("query", {}).get("capabilities", {}).get(name, False)
            for item in required
        )
    return capabilities


def aggregate_family_capabilities(
    archive_results: list[dict[str, Any]], settings: dict[str, Any]
) -> dict[str, dict[str, bool]]:
    """Keep legacy and modern archive evidence distinguishable."""

    families = sorted({item.get("family", "unspecified") for item in archive_results})
    return {
        family: aggregate_capabilities(
            [item for item in archive_results if item.get("family") == family],
            {
                **settings,
                "required_archives": [
                    {**item, "required": True}
                    for item in settings["required_archives"]
                    if item.get("family", "unspecified") == family
                ],
            },
        )
        for family in families
    }


def build_inventory(
    config_path: str | Path,
    *,
    resume: bool = False,
    query_adapter: Callable[..., dict[str, Any]] = query_archive_read_only,
) -> tuple[dict[str, Any], Path]:
    """Build a compact resumable inventory without mutating source archives."""

    started_at = _utc_now()
    config = load_config(config_path)
    settings = settings_from_config(config)
    prepare_isolated_aiida_path(config)
    environment = environment_identity(config)
    raw_root = Path(config["paths"]["data_root"]) / "raw"
    policy = _archive_policy(settings)
    archives = [raw_root / name for name in sorted(policy)]
    missing_required = [
        path.name
        for path in archives
        if policy[path.name].get("required", True) and not path.is_file()
    ]
    if missing_required:
        raise FileNotFoundError(
            f"required FPMD archives missing: {', '.join(missing_required)}"
        )
    identities = []
    for path in archives:
        present = path.is_file()
        identities.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size if present else None,
                "sha256": _sha256(path) if present else None,
                "family": policy[path.name].get("family", "unspecified"),
                "required": bool(policy[path.name].get("required", True)),
                "source_present": present,
            }
        )
    code_sha = _sha256(Path(__file__))
    inventory_inputs = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "configuration": {
            "path": config["_config_path"],
            "sha256": _sha256(Path(config["_config_path"])),
        },
        "seed": config.get("runtime", {}).get("seed"),
        "implementation_sha256": code_sha,
        "environment": environment,
        "policy": policy_record(settings),
        "archives": identities,
    }
    inventory_id = _digest(inventory_inputs)[:20]
    inventory_root = Path(config["paths"]["outputs_root"]) / "fpmd_aiida" / inventory_id
    current_path = Path(settings["inventory_path"])
    final_path = inventory_root / "inventory.json"
    if resume and final_path.is_file():
        candidate = json.loads(final_path.read_text(encoding="utf-8"))
        if (
            candidate.get("inventory_id") == inventory_id
            and candidate.get("self_digest") == _self_digest(candidate)
            and not candidate.get("errors")
        ):
            _atomic_json(current_path, candidate)
            return candidate, final_path

    results: list[dict[str, Any]] = []
    for identity, path in zip(identities, archives, strict=True):
        fingerprint = _digest(
            {
                "identity": identity,
                "environment_id": environment["environment_id"],
                "implementation_sha256": code_sha,
                "policy": policy[path.name],
            }
        )
        checkpoint = inventory_root / "archives" / f"{path.name}.json"
        result = None
        if resume and checkpoint.is_file():
            candidate = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                candidate.get("_fingerprint") == fingerprint
                and candidate.get("query_status") == "success"
            ):
                result = candidate
        if result is None:
            if not path.is_file():
                result = {
                    **identity,
                    "query_status": "missing",
                    "environment_id": environment["environment_id"],
                    "errors": ["optional archive is not present"],
                }
            else:
                try:
                    query = query_adapter(path, environment=environment, settings=settings)
                    result = {**identity, **query, "query": query}
                except Exception as exc:
                    result = {
                        **identity,
                        "query_status": "error",
                        "environment_id": environment["environment_id"],
                        "errors": [f"{type(exc).__name__}: {exc}"],
                    }
            result["_fingerprint"] = fingerprint
            _atomic_json(checkpoint, result)
        results.append(result)

    capabilities = aggregate_capabilities(results, settings)
    family_capabilities = aggregate_family_capabilities(results, settings)
    blockers = [
        name
        for name in ("aiida_query_available", *settings["required_capabilities"])
        if not capabilities.get(name, False)
    ]
    inventory = {
        **inventory_inputs,
        "inventory_id": inventory_id,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "status": "pass" if not blockers else "partial",
        "capabilities": capabilities,
        "family_capabilities": family_capabilities,
        "blockers": blockers,
        "archives": results,
        "warnings": [
            f"{item['name']}: {error}"
            for item in results
            if not policy[item["name"]].get("required", True)
            for error in item.get("errors", [])
        ],
        "errors": [
            f"{item['name']}: {error}"
            for item in results
            if policy[item["name"]].get("required", True)
            for error in item.get("errors", [])
        ],
        "unresolved_issues": blockers,
    }
    inventory["self_digest"] = _self_digest(inventory)
    _atomic_json(final_path, inventory)
    _atomic_json(current_path, inventory)
    return inventory, final_path


def evaluate_inventory(config: dict[str, Any], archive_paths: list[Path]) -> dict[str, Any]:
    """Validate generated evidence before ordinary Gate G0 can trust it."""

    settings = settings_from_config(config)
    path = Path(settings["inventory_path"])
    if not path.is_file():
        return {"status": "missing", "reason": f"inventory not found: {path}"}
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "reason": f"inventory unreadable: {exc}"}
    recorded_environment = inventory.get("environment", {})
    environment_body = {
        key: value for key, value in recorded_environment.items() if key != "environment_id"
    }
    checks: list[tuple[bool, str]] = [
        (
            recorded_environment.get("environment_id") == _digest(environment_body),
            "AiiDA environment identity mismatch",
        ),
        (inventory.get("schema_version") == SCHEMA_VERSION, "inventory schema mismatch"),
        (inventory.get("self_digest") == _self_digest(inventory), "inventory self-digest mismatch"),
        (inventory.get("git_sha") == _git("rev-parse", "HEAD"), "inventory Git SHA mismatch"),
        (inventory.get("git_branch") == _git("branch", "--show-current"), "inventory branch mismatch"),
        (
            inventory.get("git_dirty") == bool(_git("status", "--porcelain")),
            "inventory Git dirty-state mismatch",
        ),
        (
            inventory.get("configuration", {}).get("sha256")
            == _sha256(Path(config["_config_path"])),
            "inventory configuration mismatch",
        ),
        (
            inventory.get("implementation_sha256") == _sha256(Path(__file__)),
            "inventory implementation mismatch",
        ),
        (inventory.get("policy") == policy_record(settings), "inventory scientific policy mismatch"),
    ]
    lock = Path(settings["requirements_lock"])
    checks.append((lock.is_file(), "requirements lock missing"))
    if lock.is_file():
        checks.extend(
            [
                (
                    inventory.get("environment", {}).get("requirements_lock_sha256")
                    == _sha256(lock),
                    "AiiDA requirements lock mismatch",
                ),
                (
                    inventory.get("environment", {}).get("aiida_core") == settings["aiida_version"],
                    "AiiDA version mismatch",
                ),
                (
                    str(inventory.get("environment", {}).get("python", "")).startswith(
                        f"{settings['python_version']}."
                    ),
                    "AiiDA Python version mismatch",
                ),
            ]
        )
    by_name = {item.get("name"): item for item in inventory.get("archives", [])}
    expected = _archive_policy(settings)
    checks.append((set(by_name) == set(expected), "required archive set mismatch"))
    if set(by_name) == set(expected):
        recomputed_capabilities = aggregate_capabilities(inventory["archives"], settings)
        recomputed_families = aggregate_family_capabilities(inventory["archives"], settings)
        recomputed_blockers = [
            name
            for name in ("aiida_query_available", *settings["required_capabilities"])
            if not recomputed_capabilities.get(name, False)
        ]
        checks.extend(
            [
                (
                    inventory.get("capabilities") == recomputed_capabilities,
                    "inventory capability claims mismatch archive evidence",
                ),
                (
                    inventory.get("family_capabilities") == recomputed_families,
                    "inventory family capability claims mismatch archive evidence",
                ),
                (
                    inventory.get("blockers") == recomputed_blockers,
                    "inventory blockers mismatch archive evidence",
                ),
                (
                    inventory.get("status") == ("pass" if not recomputed_blockers else "partial"),
                    "inventory status mismatch archive evidence",
                ),
            ]
        )
    source_by_name = {path.name: path for path in archive_paths if path.suffix == ".aiida"}
    for name in sorted(expected):
        source = source_by_name.get(name)
        record = by_name.get(name, {})
        required = expected[name].get("required", True)
        source_present = source is not None and source.is_file()
        checks.extend(
            [
                (not required or source_present, f"{name}: required source archive missing"),
                (
                    not required or record.get("query_status") == "success",
                    f"{name}: required archive query not successful",
                ),
                (
                    source_present
                    or (
                        record.get("query_status") == "missing"
                        and record.get("sha256") is None
                    ),
                    f"{name}: absent optional archive record mismatch",
                ),
            ]
        )
        if source is not None and source.is_file():
            checks.extend(
                [
                    (
                        _within(source, Path(config["paths"]["data_root"]) / "raw"),
                        f"{name}: source path outside raw root",
                    ),
                    (record.get("sha256") == _sha256(source), f"{name}: archive hash mismatch"),
                ]
            )
    failures = [message for passed, message in checks if not passed]
    if failures:
        return {
            "status": "invalid",
            "reason": failures[0],
            "failures": failures,
            "path": str(path),
        }
    return {
        "status": "valid",
        "path": str(path),
        "inventory_id": inventory["inventory_id"],
        "self_digest": inventory["self_digest"],
        "environment": inventory["environment"],
        "capabilities": inventory.get("capabilities", {}),
        "blockers": inventory.get("blockers", []),
        "family_capabilities": inventory.get("family_capabilities", {}),
        "warnings": inventory.get("warnings", []),
        "archives": [
            {
                "name": item["name"],
                "family": item["family"],
                "sha256": item["sha256"],
                "query_status": item["query_status"],
                "archive_schema_version": item.get("query", {}).get(
                    "archive_schema_version"
                ),
            }
            for item in inventory["archives"]
        ],
    }
