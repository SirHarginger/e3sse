"""Read-only, schema-first Gate G0 dataset auditing."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from e3sse.config import load_config

SCHEMA_VERSION = "e3sse.g0.audit.v1"
DATASETS = ("nebDFT2k", "MPLiTrj", "FPMD", "BVEL13k", "nebBVSE122k")
IDENTIFIER_FIELDS = {
    "chemical_system": ("chemical_system", "chemsys"),
    "material": ("material_id", "mp_id"),
    "structure": ("structure_id",),
    "hop": ("hop_id", "edge_id"),
    "neb_path": ("neb_path_id", "path_id"),
    "frame": ("frame_id", "image_id", "image_index"),
    "published_split": ("split", "_split", "dataset_split"),
}
ENERGY_FIELDS = ("energy", "free_energy")
FORCE_FIELDS = ("forces",)
STRESS_FIELDS = ("stress", "stresses", "virial")
BARRIER_FIELDS = ("em", "em_dft", "barrier", "e_m", "migration_barrier")
UNIT_FIELDS = {
    "energy": ("energy_unit", "energy_units"),
    "forces": ("force_unit", "force_units", "forces_unit", "forces_units"),
    "stress": ("stress_unit", "stress_units", "virial_unit", "virial_units"),
}


def utc_now() -> str:
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_output_root(config: dict[str, Any]) -> Path:
    raw_root = (Path(config["paths"]["data_root"]) / "raw").resolve()
    output_root = Path(config["paths"]["outputs_root"]).resolve()
    try:
        output_root.relative_to(raw_root)
    except ValueError:
        return output_root
    raise ValueError("Audit output root must not be inside immutable raw data")


def _files_for_dataset(raw_root: Path, dataset: str) -> list[Path]:
    if dataset == "MPLiTrj":
        full = raw_root / "MPLiTrj"
        sample = raw_root / "MPLiTrj_subsample"
        root = full if full.exists() and any(path.is_file() for path in full.rglob("*")) else sample
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    elif dataset == "FPMD":
        files = [p for p in raw_root.glob("FPMD*") if p.is_file()]
    else:
        root = raw_root / dataset
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    return sorted(files, key=lambda path: path.as_posix())


def discover_datasets(raw_root: Path) -> dict[str, list[Path]]:
    """Return deterministic file inventories without modifying source data."""

    return {name: _files_for_dataset(raw_root, name) for name in DATASETS}


def _file_identity(path: Path, raw_root: Path, hash_limit: int) -> dict[str, Any]:
    stat = path.stat()
    item: dict[str, Any] = {
        "path": str(path),
        "identifier": path.relative_to(raw_root).as_posix(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= hash_limit:
        item["sha256"] = _sha256(path)
    else:
        item["sha256"] = None
        item["hash_note"] = f"not computed: file exceeds {hash_limit} byte practical limit"
    return item


def _new_identifier_summary() -> dict[str, dict[str, Any]]:
    return {
        canonical: {
            "source_fields": [],
            "total_records": 0,
            "non_null_records": 0,
            "missing_records": 0,
            "alias_conflicts": 0,
            "distinct_values": [],
        }
        for canonical in IDENTIFIER_FIELDS
    }


def _observe_identifiers(
    record: dict[str, Any],
    summary: dict[str, dict[str, Any]],
    values: dict[str, Counter[str]],
) -> None:
    for canonical, aliases in IDENTIFIER_FIELDS.items():
        summary[canonical]["total_records"] += 1
        present = [name for name in aliases if record.get(name) not in (None, "")]
        if not present:
            summary[canonical]["missing_records"] += 1
            continue
        observed = {str(record[name]) for name in present}
        if len(observed) > 1:
            summary[canonical]["alias_conflicts"] += 1
        summary[canonical]["non_null_records"] += 1
        summary[canonical]["source_fields"] = sorted(
            set(summary[canonical]["source_fields"]) | set(present)
        )
        values[canonical][str(record[present[0]])] += 1


def _audit_csv(
    path: Path,
    identifiers: dict[str, dict[str, Any]],
    identifier_values: dict[str, Counter[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "columns": [], "rows": 0}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        result["columns"] = list(reader.fieldnames)
        nulls = Counter()
        numeric = {field: {"non_null": 0, "numeric": 0} for field in BARRIER_FIELDS}
        for row in reader:
            result["rows"] += 1
            nulls.update(key for key, value in row.items() if value in (None, ""))
            _observe_identifiers(row, identifiers, identifier_values)
            for field in BARRIER_FIELDS:
                if row.get(field) not in (None, ""):
                    numeric[field]["non_null"] += 1
                    try:
                        float(row[field])
                    except ValueError:
                        pass
                    else:
                        numeric[field]["numeric"] += 1
        result["null_counts"] = dict(sorted(nulls.items()))
        result["barrier_fields"] = {
            field: counts for field, counts in numeric.items() if counts["non_null"]
        }
    return result


def _parse_extxyz_header(line: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for token in shlex.split(line):
        if "=" in token:
            key, value = token.split("=", 1)
            metadata[key] = value
    return metadata


def _iter_extxyz(path: Path) -> Iterable[tuple[dict[str, str], list[str], set[str]]]:
    """Yield metadata, property names, and species while validating frame shape."""

    with path.open("r", encoding="utf-8") as handle:
        while True:
            count_line = handle.readline()
            if not count_line:
                return
            if not count_line.strip():
                continue
            try:
                atom_count = int(count_line)
            except ValueError as exc:
                raise ValueError(f"invalid atom count {count_line.strip()!r}") from exc
            header_line = handle.readline()
            if not header_line:
                raise ValueError("missing extxyz header")
            metadata = _parse_extxyz_header(header_line)
            property_spec = metadata.get("Properties", "species:S:1:pos:R:3")
            parts = property_spec.split(":")
            if len(parts) % 3:
                raise ValueError(f"invalid Properties specification {property_spec!r}")
            properties = parts[::3]
            widths = [int(value) for value in parts[2::3]]
            expected_columns = sum(widths)
            species: set[str] = set()
            for atom_index in range(atom_count):
                atom_line = handle.readline()
                if not atom_line:
                    raise ValueError(f"truncated frame at atom {atom_index}")
                columns = atom_line.split()
                if len(columns) != expected_columns:
                    raise ValueError(
                        f"atom {atom_index} has {len(columns)} columns; expected {expected_columns}"
                    )
                species.add(columns[0])
            yield metadata, properties, species


def _audit_extxyz(
    path: Path,
    identifiers: dict[str, dict[str, Any]],
    identifier_values: dict[str, Counter[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "frames": 0,
        "info_fields": [],
        "array_fields": [],
        "calculator_fields": [],
        "energy_frames": 0,
        "force_frames": 0,
        "stress_frames": 0,
        "chemical_symbols": [],
        "unit_metadata": {},
        "energy_values": [],
    }
    info_fields: set[str] = set()
    array_fields: set[str] = set()
    calc_fields: set[str] = set()
    species: set[str] = set()
    units: dict[str, set[str]] = defaultdict(set)
    for frame_index, (metadata, properties, frame_species) in enumerate(_iter_extxyz(path)):
        result["frames"] += 1
        _observe_identifiers(metadata, identifiers, identifier_values)
        info_fields.update(metadata)
        for key, value in metadata.items():
            if "unit" in str(key).lower():
                units[str(key)].add(str(value))
        array_fields.update(properties)
        species.update(frame_species)
        calc_fields.update(key for key in (*ENERGY_FIELDS, *STRESS_FIELDS) if key in metadata)
        if any(key in metadata for key in ENERGY_FIELDS):
            result["energy_frames"] += 1
            for key in ENERGY_FIELDS:
                if key in metadata:
                    try:
                        result["energy_values"].append(float(metadata[key]))
                    except ValueError:
                        pass
                    break
        if any(key in properties for key in FORCE_FIELDS):
            result["force_frames"] += 1
        if any(key in metadata or key in properties for key in STRESS_FIELDS):
            result["stress_frames"] += 1
    result["info_fields"] = sorted(info_fields)
    result["array_fields"] = sorted(array_fields)
    result["calculator_fields"] = sorted(calc_fields)
    result["chemical_symbols"] = sorted(species)
    result["unit_metadata"] = {
        key: sorted(values) for key, values in sorted(units.items())
    }
    return result


def _metadata_keywords(text: str) -> list[str]:
    terms = (
        "material",
        "structure",
        "trajectory",
        "temperature",
        "time",
        "timestep",
        "diffusion",
        "energy",
        "force",
        "stress",
        "reference",
    )
    lowered = text.lower()
    return [term for term in terms if re.search(rf"\b{term}\w*\b", lowered)]


def _audit_text_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "format": "text",
        "path": str(path),
        "lines": len(text.splitlines()),
        "metadata_keywords": _metadata_keywords(text),
    }


def _flatten_keys(value: Any, prefix: str = "", *, limit: int = 2_000) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.append(name)
            if len(keys) >= limit:
                break
            keys.extend(_flatten_keys(child, name, limit=limit - len(keys)))
    elif isinstance(value, list):
        for child in value[:20]:
            keys.extend(_flatten_keys(child, prefix, limit=limit - len(keys)))
            if len(keys) >= limit:
                break
    return keys[:limit]


def _audit_yaml_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    keys = sorted(set(_flatten_keys(value)))
    return {
        "format": "yaml",
        "path": str(path),
        "keys": keys,
        "metadata_keywords": _metadata_keywords(" ".join(keys)),
    }


def _audit_jsonl_gz(
    path: Path,
    identifiers: dict[str, dict[str, Any]],
    identifier_values: dict[str, Counter[str]],
    *,
    sample_records: int = 100,
) -> dict[str, Any]:
    keys: set[str] = set()
    sampled = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if sampled >= sample_records:
                break
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                keys.update(value)
                _observe_identifiers(value, identifiers, identifier_values)
            sampled += 1
    return {
        "format": "jsonl.gz",
        "path": str(path),
        "sampled_records": sampled,
        "sample_limit": sample_records,
        "top_level_keys": sorted(keys),
        "metadata_keywords": _metadata_keywords(" ".join(keys)),
    }


def _read_index_hops(files: list[Path]) -> tuple[dict[str, dict[str, str]], str | None]:
    for path in files:
        if path.suffix.lower() != ".csv":
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            hop_field = next((field for field in IDENTIFIER_FIELDS["hop"] if field in fields), None)
            if hop_field is None:
                continue
            return {
                row[hop_field]: row
                for row in reader
                if row.get(hop_field) not in (None, "")
            }, hop_field
    return {}, None


def _neb_summary(files: list[Path], schemas: list[dict[str, Any]]) -> dict[str, Any]:
    index, hop_field = _read_index_hops(files)
    extxyz = {
        Path(item["path"]): item for item in schemas if item.get("format") == "extxyz"
    }
    per_hop: dict[str, dict[str, Any]] = {}
    for hop, row in sorted(index.items()):
        matched = [path for path in extxyz if path.stem == hop or path.stem.startswith(f"{hop}_")]
        initial = [path for path in matched if "initial" in path.stem.lower()]
        relaxed = [path for path in matched if "relaxed" in path.stem.lower()]
        image_files = relaxed or matched
        frames = sum(extxyz[path]["frames"] for path in image_files)
        energy_frames = sum(extxyz[path]["energy_frames"] for path in image_files)
        force_frames = sum(extxyz[path]["force_frames"] for path in image_files)
        energy_values = [
            value for path in image_files for value in extxyz[path].get("energy_values", [])
        ]
        barrier_field = next(
            (field for field in BARRIER_FIELDS if row.get(field) not in (None, "")), None
        )
        barrier_check: dict[str, Any] = {
            "published_field": barrier_field,
            "image_energy_range_available": len(energy_values) == frames and frames > 0,
        }
        if barrier_field and barrier_check["image_energy_range_available"]:
            try:
                published = float(row[barrier_field])
            except ValueError:
                barrier_check["published_numeric"] = False
            else:
                barrier_check["published_numeric"] = True
                barrier_check["absolute_difference"] = abs(
                    (max(energy_values) - min(energy_values)) - published
                )
        per_hop[hop] = {
            "initial_files": [str(path) for path in initial],
            "relaxed_files": [str(path) for path in relaxed],
            "image_files": [str(path) for path in image_files],
            "images": frames,
            "energy_images": energy_frames,
            "force_images": force_frames,
            "barrier_check": barrier_check,
        }
    return {
        "index_hop_field": hop_field,
        "indexed_hops": len(index),
        "per_hop": per_hop,
        "missing_initial_hops": [hop for hop, item in per_hop.items() if not item["initial_files"]],
        "missing_relaxed_hops": [hop for hop, item in per_hop.items() if not item["relaxed_files"]],
        "complete_energy_hops": sum(
            item["images"] > 0 and item["energy_images"] == item["images"]
            for item in per_hop.values()
        ),
        "complete_force_hops": sum(
            item["images"] > 0 and item["force_images"] == item["images"]
            for item in per_hop.values()
        ),
    }


def audit_dataset(
    dataset: str,
    files: list[Path],
    identities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit observed schemas while isolating malformed records as errors."""

    started = utc_now()
    identifiers = _new_identifier_summary()
    identifier_values: dict[str, Counter[str]] = defaultdict(Counter)
    schemas: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for path in files:
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                schemas.append({"format": "csv", **_audit_csv(path, identifiers, identifier_values)})
            elif suffix in {".xyz", ".extxyz"}:
                schemas.append(
                    {"format": "extxyz", **_audit_extxyz(path, identifiers, identifier_values)}
                )
            elif suffix in {".txt", ".md"}:
                schemas.append(_audit_text_metadata(path))
            elif suffix in {".yaml", ".yml"}:
                schemas.append(_audit_yaml_metadata(path))
            elif path.name.lower().endswith(".jsonl.gz"):
                schemas.append(_audit_jsonl_gz(path, identifiers, identifier_values))
            else:
                schemas.append({"format": suffix.lstrip(".") or "unknown", "path": str(path)})
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    for canonical, item in identifiers.items():
        item["distinct_values"] = sorted(identifier_values[canonical])
        item["distinct_count"] = len(identifier_values[canonical])
        item["value_counts"] = dict(sorted(identifier_values[canonical].items()))
        item["repeated_values"] = {
            value: count for value, count in item["value_counts"].items() if count > 1
        }

    missing_ids = [key for key, item in identifiers.items() if not item["source_fields"]]
    unresolved = [f"No verified {key} identifier was observed." for key in missing_ids]
    if not files:
        unresolved.insert(0, f"{dataset} is not available under the configured raw-data root.")
    if dataset == "FPMD":
        unresolved.append(
            "Unsupported FPMD archive internals require AiiDA-aware server inspection; "
            "no trajectory analysis was performed."
        )
    units: dict[str, Any] = {
        "energy": {"values": [], "verified": False},
        "forces": {"values": [], "verified": False},
        "stress": {"values": [], "verified": False},
    }
    observed_unit_sets: dict[str, set[str]] = defaultdict(set)
    for schema in schemas:
        for field, values in schema.get("unit_metadata", {}).items():
            observed_unit_sets[field].update(values)
    observed_units = {
        field: sorted(values) for field, values in sorted(observed_unit_sets.items())
    }
    unit_fields = sorted(observed_units)
    if unit_fields:
        warnings.append(f"Unit metadata fields require semantic validation: {unit_fields}")
    classified_fields: set[str] = set()
    for target, aliases in UNIT_FIELDS.items():
        for field in aliases:
            if field in observed_units:
                classified_fields.add(field)
                units[target]["values"] = sorted(
                    set(units[target]["values"]) | set(observed_units[field])
                )
    units["unclassified"] = {
        field: observed_units[field] for field in unit_fields if field not in classified_fields
    }

    record_counts = {
        "files": len(files),
        "csv_rows": sum(item.get("rows", 0) for item in schemas),
        "frames": sum(item.get("frames", 0) for item in schemas),
        "chemical_systems": identifiers["chemical_system"]["distinct_count"],
        "materials": identifiers["material"]["distinct_count"],
        "structures": identifiers["structure"]["distinct_count"],
        "hops": identifiers["hop"]["distinct_count"],
        "neb_paths": identifiers["neb_path"]["distinct_count"],
    }
    splits = identifiers["published_split"]["distinct_values"]
    linkage = {
        key: identifiers[key]
        for key in ("chemical_system", "material", "structure", "hop", "neb_path", "frame")
    }
    dataset_specific = _neb_summary(files, schemas) if dataset == "nebDFT2k" else {}
    if dataset == "MPLiTrj" and files:
        selected = "MPLiTrj_subsample" if "MPLiTrj_subsample" in files[0].parts else "MPLiTrj"
        dataset_specific["selected_variant"] = selected
        if selected == "MPLiTrj_subsample":
            warnings.append("Full MPLiTrj was absent; audited MPLiTrj_subsample without merging variants.")
    identifiers_complete = all(
        item["source_fields"]
        and item["missing_records"] == 0
        and item["alias_conflicts"] == 0
        for item in identifiers.values()
    )
    if dataset_specific.get("missing_initial_hops") or dataset_specific.get("missing_relaxed_hops"):
        unresolved.append("One or more indexed NEB hops lack observed initial or relaxed files.")
    if not identifiers_complete:
        unresolved.append(
            "Leakage-relevant identifiers are missing or conflicting for one or more observed records."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "started_at": started,
        "completed_at": utc_now(),
        "status": "error" if errors else ("missing" if not files else ("partial" if unresolved else "complete")),
        "input_paths": [item["path"] for item in identities],
        "input_identifiers": [item["identifier"] for item in identities],
        "input_hashes": {item["identifier"]: item["sha256"] for item in identities},
        "record_counts": record_counts,
        "schema_summary": schemas,
        "dataset_summary": dataset_specific,
        "units": units,
        "split_summary": {
            "published_labels": splits,
            "counts": identifiers["published_split"]["value_counts"],
            "verified_semantics": False,
        },
        "linkage_summary": linkage,
        "validation_checks": {
            "read_only_source": True,
            "all_files_parseable": not errors,
            "leakage_identifier_dimensions_complete": identifiers_complete,
        },
        "warnings": warnings,
        "errors": errors,
        "unresolved_issues": unresolved,
    }


def _configuration_record(config: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(config["_config_path"])
    snapshot = {key: value for key, value in config.items() if key != "_config_path"}
    return {
        "path": str(config_path),
        "sha256": _sha256(config_path),
        "resolved": snapshot,
    }


def _implementation_record() -> dict[str, str]:
    paths = {
        "dataset_audit": Path(__file__).resolve(),
        "config_loader": Path(load_config.__code__.co_filename).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _cross_dataset_linkage(dataset_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {item["dataset"]: item for item in dataset_reports}
    neb = by_name["nebDFT2k"]["linkage_summary"]
    mpli = by_name["MPLiTrj"]["linkage_summary"]
    result: dict[str, Any] = {}
    for dimension in ("chemical_system", "material", "structure", "hop", "neb_path"):
        left = set(neb[dimension]["distinct_values"])
        right = set(mpli[dimension]["distinct_values"])
        result[dimension] = {
            "nebDFT2k_count": len(left),
            "MPLiTrj_count": len(right),
            "intersection_count": len(left & right),
            "intersection_values": sorted(left & right),
            "verified_semantics": False,
        }
    return result


def run_audit(config_path: str | Path, *, resume: bool = False) -> tuple[dict[str, Any], Path]:
    """Run or resume the Gate G0 audit and return the report and its path."""

    config = load_config(config_path)
    raw_root = Path(config["paths"]["data_root"]) / "raw"
    output_root = _safe_output_root(config)
    hash_limit = int(config.get("audit", {}).get("hash_max_bytes", 10 * 1024 * 1024))
    discovered = discover_datasets(raw_root)
    identities = {
        name: [_file_identity(path, raw_root, hash_limit) for path in paths]
        for name, paths in discovered.items()
    }
    configuration = _configuration_record(config)
    implementation = _implementation_record()
    git_sha = _git("rev-parse", "HEAD")
    run_inputs = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha,
        "configuration_sha256": configuration["sha256"],
        "implementation": implementation,
        "inputs": identities,
    }
    run_id = _fingerprint(run_inputs)[:16]
    run_root = output_root / "g0" / run_id
    final_path = run_root / "audit.json"
    if resume and final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        datasets_reusable = all(
            item.get("status") != "error" and not item.get("errors")
            for item in existing.get("datasets", [])
        )
        if (
            existing.get("run_id") == run_id
            and existing.get("status") in {"complete", "partial"}
            and not existing.get("errors")
            and datasets_reusable
        ):
            return existing, final_path

    started = utc_now()
    dataset_reports = []
    for name in DATASETS:
        dataset_fingerprint = _fingerprint(identities[name])
        checkpoint = run_root / "datasets" / f"{name}.json"
        report = None
        if resume and checkpoint.exists():
            candidate = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                candidate.get("_input_fingerprint") == dataset_fingerprint
                and candidate.get("status") in {"complete", "partial", "missing"}
                and not candidate.get("errors")
            ):
                report = candidate
        if report is None:
            report = audit_dataset(name, discovered[name], identities[name])
            report["_input_fingerprint"] = dataset_fingerprint
            _atomic_json(checkpoint, report)
        dataset_reports.append(report)

    unresolved = [
        f"{item['dataset']}: {issue}"
        for item in dataset_reports
        for issue in item["unresolved_issues"]
    ]
    has_errors = any(item["errors"] for item in dataset_reports)
    status = "complete" if not unresolved and not has_errors else "partial"
    cross_linkage = _cross_dataset_linkage(dataset_reports)
    report = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha,
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "configuration": configuration,
        "implementation": implementation,
        "run_id": run_id,
        "started_at": started,
        "completed_at": utc_now(),
        "status": status,
        "dataset": "Gate G0 catalog",
        "input_paths": [item["path"] for group in identities.values() for item in group],
        "input_identifiers": [
            item["identifier"] for group in identities.values() for item in group
        ],
        "input_hashes": {
            item["identifier"]: item["sha256"] for group in identities.values() for item in group
        },
        "record_counts": {
            item["dataset"]: item["record_counts"] for item in dataset_reports
        },
        "schema_summary": {
            item["dataset"]: item["schema_summary"] for item in dataset_reports
        },
        "units": {item["dataset"]: item["units"] for item in dataset_reports},
        "split_summary": {
            item["dataset"]: item["split_summary"] for item in dataset_reports
        },
        "linkage_summary": {
            item["dataset"]: item["linkage_summary"] for item in dataset_reports
        },
        "validation_checks": {
            item["dataset"]: item["validation_checks"] for item in dataset_reports
        },
        "warnings": [
            f"{item['dataset']}: {warning}"
            for item in dataset_reports
            for warning in item["warnings"]
        ],
        "errors": [
            f"{item['dataset']}: {error}"
            for item in dataset_reports
            for error in item["errors"]
        ],
        "unresolved_issues": unresolved,
        "datasets": dataset_reports,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "command": " ".join(sys.argv),
        "seed": config.get("runtime", {}).get("seed"),
    }
    report["linkage_summary"]["nebDFT2k_to_MPLiTrj"] = cross_linkage
    _atomic_json(final_path, report)
    return report, final_path


def concise_summary(report: dict[str, Any], path: Path) -> str:
    """Render the human-facing summary from the canonical JSON report."""

    lines = [f"Gate G0: {report['status']} ({report['run_id']})"]
    for item in report["datasets"]:
        counts = item["record_counts"]
        lines.append(
            f"- {item['dataset']}: {item['status']}; files={counts['files']}, "
            f"rows={counts['csv_rows']}, frames={counts['frames']}"
        )
    lines.append(f"Unresolved issues: {len(report['unresolved_issues'])}")
    lines.append(f"JSON report: {path}")
    return "\n".join(lines)
