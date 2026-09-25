"""Read-only, schema-first Gate G0 dataset auditing.

Every dataset is audited against the requirements of its scientific role rather
than a universal identifier checklist.  Each identifier dimension and capability
carries a ``requirement`` (required / optional / not_applicable) and an
``availability`` (source_provided / derived / unavailable).  Only a required
capability that is genuinely unresolved makes a dataset ``partial``;
``not_applicable`` is never counted as missing.  Large diagnostics are emitted as
exact counts with bounded, deterministic examples.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ase
import numpy as np
import yaml
from ase import Atoms
from ase.io import iread

from e3sse.config import load_config
from e3sse.data import litraj_provenance
from e3sse.data.litraj_provenance import BoundedExamples, FrameGeometry

SCHEMA_VERSION = "e3sse.g0.audit.v2"
DATASETS = ("nebDFT2k", "MPLiTrj", "FPMD", "BVEL13k", "nebBVSE122k")

REQUIRED = "required"
OPTIONAL = "optional"
NOT_APPLICABLE = "not_applicable"
SOURCE_PROVIDED = "source_provided"
DERIVED = "derived"
UNAVAILABLE = "unavailable"

IDENTIFIER_FIELDS = {
    "chemical_system": ("chemsys", "chemical_system"),
    "material": ("material_id", "mp_id"),
    "structure": ("structure_id",),
    "hop": ("edge_id", "hop_id"),
    "neb_path": ("neb_path_id", "path_id"),
    "frame": ("frame_id", "image_id", "image_index"),
    "published_split": ("_split", "split", "dataset_split"),
}
# Dimensions whose full value sets are kept for cross-dataset linkage.
LINKAGE_DIMENSIONS = ("chemical_system", "material", "hop")

# ASE moves extxyz energies, forces and stresses into SinglePointCalculator
# results, leaving atoms.info empty; look there first.  Only the exact total
# energy, per-atom forces and cell stress count: free_energy, per-atom
# "stresses" and "virial" are different quantities and are reported only as
# observed calculator result keys.
QUANTITY_FIELDS = {
    "energy": ("energy",),
    "forces": ("forces",),
    "stress": ("stress",),
}
XYZ_SUFFIXES = {".xyz", ".extxyz"}
DOC_SUFFIXES = {".txt", ".md"}
YAML_SUFFIXES = {".yaml", ".yml"}
SPLIT_FILENAME_PATTERN = re.compile(r"_(?P<split>train|val|test)\.(?:ext)?xyz$", re.IGNORECASE)
NEB_FILE_PATTERN = re.compile(r"^(?P<edge_id>.+)_(?P<kind>init|relaxed)\.xyz$")

DEFAULT_SETTINGS: dict[str, Any] = {
    "hash_max_bytes": 10 * 1024 * 1024,
    "example_limit": 10,
    "mplitrj_raw_archive": "downloads/MPLiTrj_raw.zip",
    "provenance_sample_materials": 2,
    "provenance_inspect_hops": 3,
    "optimade_line_limit": 200,
    "provenance_max_members_per_material": 2000,
    "provenance_max_frames_per_material": 500,
    # Operational only: never part of run IDs, fingerprints or scientific output.
    "workers": 1,
}
OPERATIONAL_SETTINGS = ("workers",)

# requirement, rationale, and (optionally) the capability that reports its blocker.
ROLES: dict[str, dict[str, Any]] = {
    "nebDFT2k": {
        "role": "DFT NEB barrier reference and conformal units",
        "identifiers": {
            "chemical_system": (REQUIRED, "chemsys groups hops for grouped partitioning"),
            "material": (REQUIRED, "material_id links hops to MPLiTrj frames"),
            "structure": (OPTIONAL, "no separate structure identifier is published"),
            "hop": (REQUIRED, "edge_id is the unit of barrier correction"),
            "neb_path": (NOT_APPLICABLE, "edge_id already identifies the hop's single NEB path"),
            "frame": (OPTIONAL, "derived <edge_id>:<init|relaxed>:<frame-index>"),
            "published_split": (REQUIRED, "_split is the published partition"),
        },
    },
    "MPLiTrj": {
        "role": "DFT frames for softening estimation",
        "identifiers": {
            "chemical_system": (REQUIRED, "derived from frame species for grouped partitioning"),
            "material": (REQUIRED, "material-level leakage exclusion"),
            "structure": (NOT_APPLICABLE, "frames are identified by source file and index"),
            "hop": (
                REQUIRED,
                "same-hop leakage exclusion for barrier correction",
                "same_hop_exclusion_supported",
            ),
            "neb_path": (NOT_APPLICABLE, "an edge_id would identify the NEB path"),
            "frame": (REQUIRED, "derived <source-file>:<zero-based-frame-index>"),
            "published_split": (REQUIRED, "derived from the immutable source filename"),
        },
    },
    "FPMD": {
        "role": "finite-temperature transport reference",
        "identifiers": {
            "chemical_system": (NOT_APPLICABLE, "archive internals need AiiDA queries"),
            "material": (OPTIONAL, "documentation-level only until AiiDA queries run"),
            "structure": (NOT_APPLICABLE, "stored as AiiDA nodes"),
            "hop": (NOT_APPLICABLE, "transport reference has no NEB hops"),
            "neb_path": (NOT_APPLICABLE, "transport reference has no NEB paths"),
            "frame": (NOT_APPLICABLE, "trajectory frames live inside AiiDA nodes"),
            "published_split": (NOT_APPLICABLE, "no train/val/test role"),
        },
    },
    "BVEL13k": {
        "role": "prospective screening pool",
        "identifiers": {
            "chemical_system": (OPTIONAL, "useful for grouping candidates"),
            "material": (REQUIRED, "identifies each candidate structure"),
            "structure": (OPTIONAL, "material_id is the candidate key"),
            "hop": (NOT_APPLICABLE, "percolation barriers are per structure"),
            "neb_path": (NOT_APPLICABLE, "no NEB paths"),
            "frame": (NOT_APPLICABLE, "no NEB optimisation frames"),
            "published_split": (OPTIONAL, "not used for prospective screening"),
        },
    },
    "nebBVSE122k": {
        "role": "BVSE hop prefilter",
        "identifiers": {
            "chemical_system": (OPTIONAL, "useful for grouping candidates"),
            "material": (REQUIRED, "links hops to candidate materials"),
            "structure": (OPTIONAL, "material_id is the candidate key"),
            "hop": (REQUIRED, "edge_id identifies the rate-limiting-hop candidate"),
            "neb_path": (OPTIONAL, "edge_id identifies the BVSE path"),
            "frame": (NOT_APPLICABLE, "no arbitrary frame identifier is needed"),
            "published_split": (OPTIONAL, "not used for prospective screening"),
        },
    },
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


def aiida_query_available() -> bool:
    """AiiDA archives can only be queried with aiida-core importable."""

    return importlib.util.find_spec("aiida") is not None


def _environment_capabilities() -> dict[str, Any]:
    return {
        "ase_version": ase.__version__,
        "aiida_core_importable": aiida_query_available(),
        "verdi_executable": shutil.which("verdi") is not None,
    }


# --------------------------------------------------------------------------- discovery


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


def _file_identity(path: Path, data_root: Path, hash_limit: int) -> dict[str, Any]:
    stat = path.stat()
    try:
        identifier = path.relative_to(data_root).as_posix()
    except ValueError:
        identifier = path.as_posix()
    item: dict[str, Any] = {
        "path": str(path),
        "identifier": identifier,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= hash_limit:
        item["sha256"] = _sha256(path)
    else:
        item["sha256"] = None
        item["hash_note"] = f"not computed: file exceeds {hash_limit} byte practical limit"
    return item


def _inputs_summary(identities: list[dict[str, Any]], example_limit: int) -> dict[str, Any]:
    return {
        "file_count": len(identities),
        "total_bytes": sum(item.get("bytes", 0) for item in identities),
        "sha256_computed": sum(item.get("sha256") is not None for item in identities),
        "sha256_skipped": sum(item.get("sha256") is None for item in identities),
        "identity_digest": _fingerprint(identities),
        "example_limit": example_limit,
        "examples": [item["identifier"] for item in identities[:example_limit]],
        "manifest": "inputs.json",
    }


# --------------------------------------------------------------------------- primitives


def _source_value(record: dict[str, Any], aliases: Iterable[str]) -> tuple[str | None, str | None, bool]:
    """Return (value, field, alias_conflict) from the first populated alias."""

    present = [(name, str(record[name])) for name in aliases if record.get(name) not in (None, "")]
    if not present:
        return None, None, False
    return present[0][1], present[0][0], len({value for _, value in present}) > 1


def split_from_filename(name: str) -> str | None:
    """Published split encoded in a LiTraj source filename, e.g. MPLiTrj_train.xyz."""

    match = SPLIT_FILENAME_PATTERN.search(name)
    return match.group("split").lower() if match else None


def chemical_system(symbols: Iterable[str]) -> str:
    """LiTraj chemsys convention: sorted unique element symbols joined by '-'."""

    return "-".join(sorted(set(symbols)))


def locate_quantities(atoms: Atoms) -> dict[str, str | None]:
    """Where each scientific quantity was found, e.g. 'calc.results:forces', or None."""

    results = getattr(atoms.calc, "results", None) or {}
    containers = (("calc.results", results), ("atoms.info", atoms.info), ("atoms.arrays", atoms.arrays))
    return {
        quantity: next(
            (
                f"{label}:{name}"
                for label, container in containers
                for name in names
                if container.get(name) is not None
            ),
            None,
        )
        for quantity, names in QUANTITY_FIELDS.items()
    }


def _iter_frames(path: Path) -> Iterable[Atoms]:
    return iread(str(path), index=":", format="extxyz")


class IdentifierTracker:
    """Per-dimension identifier coverage with role requirement and value provenance."""

    def __init__(
        self,
        dimension: str,
        requirement: str,
        note: str,
        example_limit: int,
        *,
        keep_values: bool,
    ) -> None:
        self.dimension = dimension
        self.requirement = requirement
        self.note = note
        self.limit = example_limit
        self.total = 0
        self.with_value = 0
        self.missing = BoundedExamples(example_limit)
        self.conflicts = BoundedExamples(example_limit)
        self.sources: Counter[str] = Counter()
        self.kinds: set[str] = set()
        self.values: Counter[str] | None = Counter() if keep_values else None

    def observe(
        self,
        value: str | None,
        value_source: str | None,
        availability: str,
        label: str,
        *,
        conflict: bool = False,
    ) -> None:
        self.total += 1
        if value is None:
            self.missing.add(label)
            return
        self.with_value += 1
        self.sources[str(value_source)] += 1
        self.kinds.add(availability)
        if conflict:
            self.conflicts.add(label)
        if self.values is not None:
            self.values[value] += 1

    @property
    def availability(self) -> str:
        if not self.with_value:
            return UNAVAILABLE
        return SOURCE_PROVIDED if SOURCE_PROVIDED in self.kinds else DERIVED

    @property
    def status(self) -> str:
        if self.requirement == NOT_APPLICABLE:
            return NOT_APPLICABLE
        if not self.with_value:
            return UNAVAILABLE
        if self.missing.count or self.conflicts.count:
            return "incomplete"
        return "satisfied"

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requirement": self.requirement,
            "availability": self.availability,
            "status": self.status,
            "note": self.note,
            "value_sources": dict(sorted(self.sources.items())),
            "total_records": self.total,
            "records_with_value": self.with_value,
            # not_applicable dimensions are never reported as missing.
            "missing": None
            if self.requirement == NOT_APPLICABLE
            else self.missing.to_json("records without a value"),
            "alias_conflicts": self.conflicts.to_json("records whose aliases disagree"),
        }
        if self.values is None:
            result["distinct_count"] = self.with_value
            result["distinct_note"] = "unique by construction; values not retained"
        else:
            repeated = BoundedExamples(self.limit)
            for value, count in sorted(self.values.items()):
                if count > 1:
                    repeated.add({"value": value, "records": count})
            result["distinct_count"] = len(self.values)
            result["value_examples"] = sorted(self.values)[: self.limit]
            result["repeated_values"] = repeated.to_json("values seen in more than one record")
        return result


    def to_state(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "requirement": self.requirement,
            "note": self.note,
            "limit": self.limit,
            "total": self.total,
            "with_value": self.with_value,
            "missing": self.missing.to_state(),
            "conflicts": self.conflicts.to_state(),
            "sources": dict(sorted(self.sources.items())),
            "kinds": sorted(self.kinds),
            "values": None if self.values is None else dict(sorted(self.values.items())),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> IdentifierTracker:
        tracker = cls(
            state["dimension"],
            state["requirement"],
            state["note"],
            state["limit"],
            keep_values=state["values"] is not None,
        )
        tracker.total = state["total"]
        tracker.with_value = state["with_value"]
        tracker.missing = BoundedExamples.from_state(state["missing"])
        tracker.conflicts = BoundedExamples.from_state(state["conflicts"])
        tracker.sources = Counter(state["sources"])
        tracker.kinds = set(state["kinds"])
        if state["values"] is not None:
            tracker.values = Counter(state["values"])
        return tracker

    def absorb(self, other: IdentifierTracker) -> None:
        """Merge a later work unit's observations, as if seen after this one's."""

        self.total += other.total
        self.with_value += other.with_value
        self.missing.absorb(other.missing)
        self.conflicts.absorb(other.conflicts)
        self.sources.update(other.sources)
        self.kinds |= other.kinds
        if self.values is not None and other.values is not None:
            self.values.update(other.values)


def _trackers(dataset: str, example_limit: int, keep: Iterable[str]) -> dict[str, IdentifierTracker]:
    keep_set = set(keep)
    trackers = {}
    for dimension, spec in ROLES[dataset]["identifiers"].items():
        trackers[dimension] = IdentifierTracker(
            dimension, spec[0], spec[1], example_limit, keep_values=dimension in keep_set
        )
    return trackers


def _observe_source(tracker: IdentifierTracker, record: dict[str, Any], label: str) -> None:
    value, field, conflict = _source_value(record, IDENTIFIER_FIELDS[tracker.dimension])
    tracker.observe(value, field, SOURCE_PROVIDED, label, conflict=conflict)


def _capability(
    requirement: str,
    available: bool,
    detail: str,
    *,
    value_source: str | None = None,
) -> dict[str, Any]:
    status = NOT_APPLICABLE if requirement == NOT_APPLICABLE else (
        "available" if available else UNAVAILABLE
    )
    item: dict[str, Any] = {
        "requirement": requirement,
        "available": bool(available),
        "status": status,
        "detail": detail,
    }
    if value_source is not None:
        item["value_source"] = value_source
    return item


class FrameSchema:
    """Aggregate ASE frame schema for a group of files with bounded diagnostics."""

    def __init__(self, example_limit: int) -> None:
        self.limit = example_limit
        self.files = 0
        self.frames = 0
        self.frames_per_file: Counter[int] = Counter()
        self.info_keys: set[str] = set()
        self.array_keys: set[str] = set()
        self.result_keys: set[str] = set()
        self.calculators: Counter[str] = Counter()
        self.species: set[str] = set()
        self.units: dict[str, set[str]] = defaultdict(set)
        self.locations: dict[str, Counter[str]] = {q: Counter() for q in QUANTITY_FIELDS}
        self.missing = {q: BoundedExamples(example_limit) for q in QUANTITY_FIELDS}

    def observe(self, atoms: Atoms, label: str) -> dict[str, str | None]:
        self.frames += 1
        self.info_keys.update(map(str, atoms.info))
        self.array_keys.update(atoms.arrays)
        calc = atoms.calc
        self.calculators[type(calc).__name__ if calc is not None else "none"] += 1
        self.result_keys.update(getattr(calc, "results", None) or {})
        self.species.update(atoms.get_chemical_symbols())
        for key, value in atoms.info.items():
            if "unit" in str(key).lower():
                self.units[str(key)].add(str(value))
        found = locate_quantities(atoms)
        for quantity, location in found.items():
            if location:
                self.locations[quantity][location] += 1
            else:
                self.missing[quantity].add(label)
        return found

    def add_file(self, frames: int) -> None:
        self.files += 1
        self.frames_per_file[frames] += 1

    def complete(self, quantity: str) -> bool:
        return self.frames > 0 and self.missing[quantity].count == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "frames": self.frames,
            "frames_per_file": {str(k): v for k, v in sorted(self.frames_per_file.items())},
            "info_keys": sorted(self.info_keys),
            "array_keys": sorted(self.array_keys),
            "calculator_result_keys": sorted(self.result_keys),
            "calculators": dict(sorted(self.calculators.items())),
            "chemical_symbols": sorted(self.species),
            "unit_metadata": {k: sorted(v) for k, v in sorted(self.units.items())},
            "quantities": {
                quantity: {
                    "frames_with_value": sum(self.locations[quantity].values()),
                    "locations": dict(sorted(self.locations[quantity].items())),
                    "missing": self.missing[quantity].to_json("frames without this quantity"),
                }
                for quantity in QUANTITY_FIELDS
            },
        }


    def to_state(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "files": self.files,
            "frames": self.frames,
            "frames_per_file": {str(k): v for k, v in sorted(self.frames_per_file.items())},
            "info_keys": sorted(self.info_keys),
            "array_keys": sorted(self.array_keys),
            "result_keys": sorted(self.result_keys),
            "calculators": dict(sorted(self.calculators.items())),
            "species": sorted(self.species),
            "units": {k: sorted(v) for k, v in sorted(self.units.items())},
            "locations": {q: dict(sorted(c.items())) for q, c in self.locations.items()},
            "missing": {q: examples.to_state() for q, examples in self.missing.items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FrameSchema:
        schema = cls(state["limit"])
        schema.files = state["files"]
        schema.frames = state["frames"]
        schema.frames_per_file = Counter({int(k): v for k, v in state["frames_per_file"].items()})
        schema.info_keys = set(state["info_keys"])
        schema.array_keys = set(state["array_keys"])
        schema.result_keys = set(state["result_keys"])
        schema.calculators = Counter(state["calculators"])
        schema.species = set(state["species"])
        schema.units = defaultdict(set, {k: set(v) for k, v in state["units"].items()})
        schema.locations = {q: Counter(state["locations"][q]) for q in QUANTITY_FIELDS}
        schema.missing = {
            q: BoundedExamples.from_state(state["missing"][q]) for q in QUANTITY_FIELDS
        }
        return schema

    def absorb(self, other: FrameSchema) -> None:
        """Merge a later work unit's frames, as if parsed after this one's."""

        self.files += other.files
        self.frames += other.frames
        self.frames_per_file.update(other.frames_per_file)
        self.info_keys |= other.info_keys
        self.array_keys |= other.array_keys
        self.result_keys |= other.result_keys
        self.calculators.update(other.calculators)
        self.species |= other.species
        for key, values in other.units.items():
            self.units[key] |= values
        for quantity in QUANTITY_FIELDS:
            self.locations[quantity].update(other.locations[quantity])
            self.missing[quantity].absorb(other.missing[quantity])


def _metadata_keywords(text: str) -> list[str]:
    terms = {
        "material": r"\bmaterial\w*",
        "structure": r"\bstructure\w*",
        "trajectory": r"\btrajector\w*",
        "temperature": r"\btemperature\w*",
        "timestep": r"\btime\s*step\w*",
        "diffusion": r"\bdiffusion\w*|\bdiffusivit\w*",
        "msd": r"\bmsd\b|mean[- ]square[d]?[- ]displacement",
        "energy": r"\benerg\w*",
        "force": r"\bforce\w*",
        "stress": r"\bstress\w*",
        "provenance": r"\bprovenance\w*",
    }
    lowered = text.lower()
    return [term for term, pattern in terms.items() if re.search(pattern, lowered)]


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


def _describe_file(path: Path, example_limit: int) -> dict[str, Any]:
    """Schema of an auxiliary file (documentation, YAML, CSV, or unknown)."""

    suffix = path.suffix.lower()
    if suffix in DOC_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "format": "text",
            "path": str(path),
            "lines": len(text.splitlines()),
            "metadata_keywords": _metadata_keywords(text),
        }
    if suffix in YAML_SUFFIXES:
        with path.open("r", encoding="utf-8") as handle:
            keys = sorted(set(_flatten_keys(yaml.safe_load(handle))))
        return {
            "format": "yaml",
            "path": str(path),
            "key_count": len(keys),
            "keys": keys[:200],
            "metadata_keywords": _metadata_keywords(" ".join(keys)),
        }
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        return {"format": "csv", "path": str(path), "columns": header, "rows": rows}
    return {"format": suffix.lstrip(".") or "unknown", "path": str(path)}


def _read_csv(
    path: Path,
    example_limit: int,
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        columns = list(reader.fieldnames)
        rows = list(reader)
    column_summary = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        present = [value for value in values if value not in (None, "")]
        numeric = 0
        for value in present:
            try:
                float(value)
            except ValueError:
                continue
            numeric += 1
        column_summary[column] = {"null": len(values) - len(present), "numeric": numeric}
    return columns, rows, {
        "format": "csv",
        "path": str(path),
        "columns": columns,
        "rows": len(rows),
        "column_summary": column_summary,
    }


def _is_finite_number(value: str | None) -> bool:
    """NaN and inf parse as floats but are not usable reference values."""

    try:
        return math.isfinite(float(value or ""))
    except ValueError:
        return False


def _span_diagnostic(
    groups: dict[str, set[str]], example_limit: int, provenance: str
) -> dict[str, Any]:
    spanning = BoundedExamples(example_limit)
    for value, splits in sorted(groups.items()):
        if len(splits) > 1:
            spanning.add({"value": value, "splits": sorted(splits)})
    return spanning.to_json(provenance)


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


# --------------------------------------------------------------------------- nebDFT2k


def _audit_nebdft2k(files: list[Path], settings: dict[str, Any]) -> dict[str, Any]:
    limit = settings["example_limit"]
    errors: list[str] = []
    warnings: list[str] = []
    schemas: list[dict[str, Any]] = []
    trackers = _trackers("nebDFT2k", limit, LINKAGE_DIMENSIONS + ("published_split",))

    index_path: Path | None = None
    rows: dict[str, dict[str, str]] = {}
    duplicate_rows = BoundedExamples(limit)
    em_dft_invalid = BoundedExamples(limit)
    em_bvse_numeric = 0
    split_counts: Counter[str] = Counter()
    chemsys_splits: dict[str, set[str]] = defaultdict(set)
    material_splits: dict[str, set[str]] = defaultdict(set)
    for path in (p for p in files if p.suffix.lower() == ".csv"):
        try:
            columns, csv_rows, schema = _read_csv(path, limit)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        schemas.append(schema)
        if "edge_id" not in columns:
            continue
        if index_path is not None:
            warnings.append(f"Additional edge_id index ignored for pairing: {path.name}")
            continue
        index_path = path
        for number, row in enumerate(csv_rows, start=1):
            label = f"{path.name}:row{number}"
            for tracker in trackers.values():
                if tracker.dimension != "frame":
                    _observe_source(tracker, row, label)
            edge_id = row.get("edge_id") or ""
            if not edge_id:
                continue
            if edge_id in rows:
                duplicate_rows.add(edge_id)
                continue
            rows[edge_id] = row
            split = row.get("_split") or ""
            split_counts[split or "<missing>"] += 1
            if split:
                if row.get("chemsys"):
                    chemsys_splits[row["chemsys"]].add(split)
                if row.get("material_id"):
                    material_splits[row["material_id"]].add(split)
            if not _is_finite_number(row.get("em_dft")):
                em_dft_invalid.add(edge_id)
            em_bvse_numeric += _is_finite_number(row.get("em_bvse"))

    files_by_key: dict[tuple[str, str], list[Path]] = defaultdict(list)
    unexpected = BoundedExamples(limit)
    for path in (p for p in files if p.suffix.lower() in XYZ_SUFFIXES):
        match = NEB_FILE_PATTERN.match(path.name)
        if match is None or match.group("edge_id") not in rows:
            unexpected.add(path.name)
            continue
        files_by_key[(match.group("edge_id"), match.group("kind"))].append(path)

    duplicate_files = BoundedExamples(limit)
    missing = {"init": BoundedExamples(limit), "relaxed": BoundedExamples(limit)}
    paired = 0
    for edge_id in sorted(rows):
        present = [kind for kind in ("init", "relaxed") if files_by_key.get((edge_id, kind))]
        for kind in ("init", "relaxed"):
            if kind not in present:
                missing[kind].add(f"{edge_id}_{kind}.xyz")
            elif len(files_by_key[(edge_id, kind)]) > 1:
                duplicate_files.add([str(p) for p in files_by_key[(edge_id, kind)]])
        paired += len(present) == 2

    frame_schemas = {"init": FrameSchema(limit), "relaxed": FrameSchema(limit)}
    barrier_differences: list[float] = []
    for (edge_id, kind), paths in sorted(files_by_key.items()):
        path = paths[0]
        count = 0
        energies: list[float | None] = []
        try:
            for index, atoms in enumerate(_iter_frames(path)):
                count += 1
                frame_id = f"{edge_id}:{kind}:{index}"
                trackers["frame"].observe(frame_id, "derived:<edge_id>:<kind>:<frame-index>", DERIVED, frame_id)
                found = frame_schemas[kind].observe(atoms, frame_id)
                energy = None
                if found["energy"] and found["energy"].startswith("calc.results"):
                    energy = atoms.calc.results.get("energy")
                energies.append(None if energy is None else float(energy))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
        finally:
            frame_schemas[kind].add_file(count)
        if kind == "relaxed" and energies and all(value is not None for value in energies):
            if not _is_finite_number(rows[edge_id].get("em_dft")):
                continue
            reference = float(rows[edge_id]["em_dft"])
            barrier_differences.append(abs(max(energies) - energies[0] - reference))

    total_frames = sum(schema.frames for schema in frame_schemas.values())
    capabilities = {
        "index_present": _capability(
            REQUIRED, index_path is not None, "CSV index with an edge_id column",
            value_source=None if index_path is None else index_path.name,
        ),
        "index_edge_ids_unique": _capability(
            REQUIRED, index_path is not None and duplicate_rows.count == 0,
            f"{duplicate_rows.count} duplicated edge_id rows",
        ),
        "all_indexed_hops_have_init": _capability(
            REQUIRED, index_path is not None and missing["init"].count == 0,
            f"{missing['init'].count} indexed hops lack <edge_id>_init.xyz",
        ),
        "all_indexed_hops_have_relaxed": _capability(
            REQUIRED, index_path is not None and missing["relaxed"].count == 0,
            f"{missing['relaxed'].count} indexed hops lack <edge_id>_relaxed.xyz",
        ),
        "no_duplicate_hop_files": _capability(
            REQUIRED, duplicate_files.count == 0,
            f"{duplicate_files.count} hop files present at more than one path",
        ),
        "dft_energy_readable": _capability(
            REQUIRED, all(s.complete("energy") for s in frame_schemas.values()),
            "energy in ASE calculator results for every init and relaxed frame",
            value_source="atoms.calc.results",
        ),
        "dft_forces_readable": _capability(
            REQUIRED, all(s.complete("forces") for s in frame_schemas.values()),
            "forces in ASE calculator results for every init and relaxed frame",
            value_source="atoms.calc.results",
        ),
        "dft_stress_readable": _capability(
            OPTIONAL, all(s.complete("stress") for s in frame_schemas.values()),
            "stress for every init and relaxed frame",
            value_source="atoms.calc.results",
        ),
        "barrier_reference_available": _capability(
            REQUIRED, bool(rows) and em_dft_invalid.count == 0,
            f"{em_dft_invalid.count} indexed hops lack a finite numeric em_dft",
            value_source="index:em_dft",
        ),
        "bvse_reference_available": _capability(
            OPTIONAL, bool(rows) and em_bvse_numeric == len(rows),
            f"{em_bvse_numeric} of {len(rows)} indexed hops have finite numeric em_bvse",
            value_source="index:em_bvse",
        ),
        "no_unindexed_files": _capability(
            OPTIONAL, unexpected.count == 0,
            f"{unexpected.count} XYZ files do not match <indexed edge_id>_<init|relaxed>.xyz",
        ),
    }
    summary = {
        "index_file": None if index_path is None else index_path.name,
        "indexed_hops": len(rows),
        "paired_hops": paired,
        "missing_init_files": missing["init"].to_json("index edge_id without <edge_id>_init.xyz"),
        "missing_relaxed_files": missing["relaxed"].to_json(
            "index edge_id without <edge_id>_relaxed.xyz"
        ),
        "unexpected_unindexed_files": unexpected.to_json("XYZ filename not paired to the index"),
        "duplicate_index_rows": duplicate_rows.to_json("repeated edge_id in index"),
        "duplicate_hop_files": duplicate_files.to_json("same <edge_id>_<kind>.xyz at >1 path"),
        "frame_count_distribution": {
            kind: schema.to_json()["frames_per_file"] for kind, schema in frame_schemas.items()
        },
        "published_split_counts": dict(sorted(split_counts.items())),
        "chemical_systems_spanning_splits": _span_diagnostic(
            chemsys_splits, limit, "index chemsys observed under more than one _split"
        ),
        "materials_spanning_splits": _span_diagnostic(
            material_splits, limit, "index material_id observed under more than one _split"
        ),
        "barrier_reference_check": {
            "statistic": "|max(relaxed E) - relaxed E[0] - em_dft| in eV",
            "hops_compared": len(barrier_differences),
            "absolute_difference": _quantiles(barrier_differences),
            "semantics_verified": False,
            "note": "informational only; the em_dft definition is not verified here",
        },
    }
    schemas.extend(
        {"format": "extxyz", "group": f"*_{kind}.xyz", **schema.to_json()}
        for kind, schema in frame_schemas.items()
    )
    return {
        "trackers": trackers,
        "capabilities": capabilities,
        "schemas": schemas,
        "summary": summary,
        "frames": total_frames,
        "csv_rows": sum(item.get("rows", 0) for item in schemas if item.get("format") == "csv"),
        "splits": dict(sorted(split_counts.items())),
        "errors": errors,
        "warnings": warnings,
        "linkage_values": {
            dimension: sorted(trackers[dimension].values or {}) for dimension in LINKAGE_DIMENSIONS
        },
    }


# --------------------------------------------------------------------------- MPLiTrj


def _investigate_raw_archive(
    archive_path: Path | None, settings: dict[str, Any]
) -> tuple[dict[str, Any], litraj_provenance.ArchiveIndex | None, list[str]]:
    limit = settings["example_limit"]
    if archive_path is None or not archive_path.is_file():
        return {"present": False, "path": None if archive_path is None else str(archive_path)}, None, []
    record: dict[str, Any] = {"present": True, "path": str(archive_path), "access": "read-only zipfile"}
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            index = litraj_provenance.index_archive(archive)
            record["structure"] = litraj_provenance.summarize_archive(index, example_limit=limit)
            inspected = list(index.hops)[: settings["provenance_inspect_hops"]]
            record["sample_hop_contents"] = litraj_provenance.inspect_hops(
                archive, index, inspected, example_limit=limit
            )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record, None, []
    sample, skipped = litraj_provenance.select_sample_materials(
        index,
        settings["provenance_sample_materials"],
        max_members_per_material=settings["provenance_max_members_per_material"],
    )
    record["sample_materials_skipped_as_too_large"] = skipped[:limit]
    return record, index, sample


# MPLiTrj is audited as independent work units so that its source files can be
# parsed concurrently: a cheap parent-side preparation (auxiliary files, raw
# archive central directory, sample material choice), one streaming unit per
# immutable source file, and one provenance-sample unit that needs every file
# unit's capped sample.  Each unit returns compact JSON state whose size does not
# grow with the number of frames, and the parent merges states in canonical
# file order, which reproduces a single serial pass exactly.


def _mplitrj_prepare(
    files: list[Path], settings: dict[str, Any], raw_archive: Path | None
) -> dict[str, Any]:
    limit = settings["example_limit"]
    schemas: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in (p for p in files if p.suffix.lower() not in XYZ_SUFFIXES):
        try:
            schemas.append(_describe_file(path, limit))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    archive_record, index, sample_materials = _investigate_raw_archive(raw_archive, settings)
    return {
        "aux_schemas": schemas,
        "aux_errors": errors,
        "archive_record": archive_record,
        "archive_indexed": index is not None,
        "archive_hops": [] if index is None else list(index.hops),
        "sample_materials": sample_materials,
    }


def mplitrj_source_files(files: list[Path]) -> list[Path]:
    """MPLiTrj extxyz sources in canonical (sorted path) order."""

    return sorted(
        (p for p in files if p.suffix.lower() in XYZ_SUFFIXES), key=lambda p: p.as_posix()
    )


def _mplitrj_file_unit(
    path: str, settings: dict[str, Any], sample_materials: list[str]
) -> dict[str, Any]:
    """Stream one MPLiTrj source file into compact, mergeable JSON state."""

    source = Path(path)
    limit = settings["example_limit"]
    cap = settings["provenance_max_frames_per_material"]
    trackers = _trackers("MPLiTrj", limit, LINKAGE_DIMENSIONS + ("published_split",))
    frame_schema = FrameSchema(limit)
    split_counts: Counter[str] = Counter()
    material_splits: dict[str, set[str]] = defaultdict(set)
    sample_set = set(sample_materials)
    sampled: list[list[Any]] = []
    seen_per_material: Counter[str] = Counter()
    errors: list[str] = []
    filename_split = split_from_filename(source.name)
    count = 0
    try:
        for index_in_file, atoms in enumerate(_iter_frames(source)):
            count += 1
            frame_id = f"{source.name}:{index_in_file}"
            frame_schema.observe(atoms, frame_id)
            info = atoms.info
            for dimension in ("material", "structure", "hop", "neb_path"):
                _observe_source(trackers[dimension], info, frame_id)
            split, field, conflict = _source_value(info, IDENTIFIER_FIELDS["published_split"])
            if split is not None:
                conflict = conflict or (filename_split is not None and split.lower() != filename_split)
                trackers["published_split"].observe(
                    split, field, SOURCE_PROVIDED, frame_id, conflict=conflict
                )
            else:
                split = filename_split
                trackers["published_split"].observe(split, "source_filename", DERIVED, frame_id)
            trackers["chemical_system"].observe(
                chemical_system(atoms.get_chemical_symbols()), "derived:frame_species", DERIVED, frame_id
            )
            trackers["frame"].observe(
                frame_id, "derived:<source-file>:<zero-based-frame-index>", DERIVED, frame_id
            )
            material = _source_value(info, IDENTIFIER_FIELDS["material"])[0]
            split_counts[split or "<missing>"] += 1
            if material is not None and split is not None:
                material_splits[material].add(split)
            if material in sample_set:
                # Keep at most ``cap`` frames per material per file; the merge
                # applies the same cap across files in canonical order.
                if seen_per_material[material] < cap:
                    sampled.append(
                        [frame_id, material, FrameGeometry.from_atoms(atoms).to_state()]
                    )
                seen_per_material[material] += 1
    except Exception as exc:
        errors.append(f"{source}: {type(exc).__name__}: {exc}")
    finally:
        frame_schema.add_file(count)
    return {
        "file": source.name,
        "frames": count,
        "published_split": filename_split,
        "errors": errors,
        "trackers": {name: tracker.to_state() for name, tracker in trackers.items()},
        "frame_schema": frame_schema.to_state(),
        "split_counts": dict(sorted(split_counts.items())),
        "material_splits": {m: sorted(v) for m, v in sorted(material_splits.items())},
        "sample_frames": sampled,
        "sample_seen_per_material": dict(sorted(seen_per_material.items())),
    }


def _merge_mplitrj_units(states: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    """Merge file-unit states given in canonical file order."""

    limit = settings["example_limit"]
    cap = settings["provenance_max_frames_per_material"]
    trackers = _trackers("MPLiTrj", limit, LINKAGE_DIMENSIONS + ("published_split",))
    frame_schema = FrameSchema(limit)
    split_counts: Counter[str] = Counter()
    material_splits: dict[str, set[str]] = defaultdict(set)
    kept: Counter[str] = Counter()
    seen = 0
    sample: list[list[Any]] = []
    per_file: list[dict[str, Any]] = []
    errors: list[str] = []
    for state in states:
        for name, tracker in trackers.items():
            tracker.absorb(IdentifierTracker.from_state(state["trackers"][name]))
        frame_schema.absorb(FrameSchema.from_state(state["frame_schema"]))
        split_counts.update(state["split_counts"])
        for material, splits in state["material_splits"].items():
            material_splits[material].update(splits)
        for frame_id, material, geometry in state["sample_frames"]:
            if kept[material] < cap:
                kept[material] += 1
                sample.append([frame_id, material, geometry])
        seen += sum(state["sample_seen_per_material"].values())
        errors.extend(state["errors"])
        split = state["published_split"]
        per_file.append({
            "file": state["file"],
            "frames": state["frames"],
            "published_split": split,
            "published_split_value_source": "source_filename" if split else None,
        })
    return {
        "trackers": trackers,
        "frame_schema": frame_schema,
        "split_counts": split_counts,
        "material_splits": material_splits,
        "sample_frames": sample,
        "frames_beyond_cap": seen - len(sample),
        "per_file": per_file,
        "errors": errors,
    }


def _mplitrj_provenance_unit(
    raw_archive: str | None,
    settings: dict[str, Any],
    prepared: dict[str, Any],
    sample_frames: list[list[Any]],
    frames_beyond_cap: int,
) -> dict[str, Any]:
    """Structural frame-to-hop mapping check on the bounded provenance sample."""

    limit = settings["example_limit"]
    archive_record = prepared["archive_record"]
    mapping: dict[str, Any] = {
        "status": "not_evaluated",
        "reason": archive_record.get("error", "raw archive unavailable"),
    }
    if not prepared["archive_indexed"] or raw_archive is None:
        return mapping
    flattened = [
        (frame_id, material, FrameGeometry.from_state(geometry))
        for frame_id, material, geometry in sample_frames
    ]
    try:
        with zipfile.ZipFile(raw_archive, mode="r") as archive:
            index = litraj_provenance.index_archive(archive)
            raw_frames, raw_failures = litraj_provenance.load_raw_frames(
                archive, index, prepared["sample_materials"], example_limit=limit
            )
    except Exception as exc:
        mapping["reason"] = f"raw archive frames unreadable: {type(exc).__name__}: {exc}"
        return mapping
    mapping = litraj_provenance.validate_mapping(
        flattened,
        raw_frames,
        example_limit=limit,
        raw_read_failures=raw_failures.count,
        unparseable_edge_ids=archive_record["structure"]["unparseable_edge_ids"]["count"],
    )
    mapping["raw_read_failure_details"] = raw_failures.to_json("archive member read failure")
    mapping["sample_selection"] = (
        f"first {settings['provenance_sample_materials']} sorted archive material IDs "
        f"with <= {settings['provenance_max_members_per_material']} hop members; at most "
        f"{settings['provenance_max_frames_per_material']} flattened frames per material "
        "in file order"
    )
    mapping["flattened_frames_beyond_cap"] = frames_beyond_cap
    if not flattened:
        mapping["reason"] = "no flattened frames belong to the sampled materials"
    return mapping


def _mplitrj_finalize(
    files: list[Path],
    settings: dict[str, Any],
    prepared: dict[str, Any],
    merged: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    limit = settings["example_limit"]
    trackers = merged["trackers"]
    frame_schema = merged["frame_schema"]
    archive_record = json.loads(json.dumps(prepared["archive_record"]))
    warnings: list[str] = []
    if archive_record.get("error"):
        warnings.append(f"Raw archive could not be indexed: {archive_record['error']}")
    archive_record["mapping_validation"] = mapping

    hop_tracker = trackers["hop"]
    if hop_tracker.status == "satisfied":
        same_hop, reason = True, "source-provided hop identifier on every frame"
    elif mapping["status"] == "demonstrated_on_sample":
        same_hop, reason = False, (
            "frame-to-hop mapping demonstrated on a bounded sample only; a full "
            "frame-to-hop provenance index has not been built"
        )
    elif mapping["status"] == "inconclusive":
        same_hop, reason = False, (
            f"sample mapping inconclusive: {mapping['raw_read_failures']} unreadable archive "
            f"members, {mapping['unparseable_edge_ids']} unparseable hop directories"
        )
    elif mapping["status"] == "not_demonstrated":
        same_hop, reason = False, (
            f"{mapping['unmatched']['count']} sampled flattened frames have no structural "
            "match in the raw archive; frame-to-hop mapping not demonstrated"
        )
    else:
        same_hop, reason = False, (
            "flattened frames carry no hop identifier and archive mapping was not evaluated "
            f"({mapping.get('reason', 'no sampled frames')})"
        )

    material_ok = trackers["material"].status == "satisfied"
    split_ok = trackers["published_split"].status == "satisfied"
    capabilities = {
        "material_exclusion_supported": _capability(
            REQUIRED, material_ok, "source-provided material_id on every frame",
            value_source="atoms.info:material_id",
        ),
        "same_hop_exclusion_supported": _capability(REQUIRED, same_hop, reason),
        "published_split_supported": _capability(
            REQUIRED, split_ok, "train/val/test on every frame",
            value_source=",".join(sorted(trackers["published_split"].sources)) or None,
        ),
        "chemical_system_grouping_supported": _capability(
            REQUIRED, trackers["chemical_system"].status == "satisfied",
            "chemical system derived from frame species", value_source="derived:frame_species",
        ),
        "dft_energy_readable": _capability(
            REQUIRED, frame_schema.complete("energy"), "energy for every frame",
            value_source="atoms.calc.results",
        ),
        "dft_forces_readable": _capability(
            REQUIRED, frame_schema.complete("forces"), "forces for every frame",
            value_source="atoms.calc.results",
        ),
        "dft_stress_readable": _capability(
            OPTIONAL, frame_schema.complete("stress"), "stress for every frame",
            value_source="atoms.calc.results",
        ),
        "raw_archive_present": _capability(
            OPTIONAL, bool(archive_record.get("present")), "MPLiTrj_raw.zip for hop provenance",
        ),
    }
    variant = "MPLiTrj_subsample" if files and "MPLiTrj_subsample" in files[0].parts else "MPLiTrj"
    if variant == "MPLiTrj_subsample":
        warnings.append("Full MPLiTrj was absent; audited MPLiTrj_subsample without merging variants.")
    linkage = {
        dimension: sorted(trackers[dimension].values or {}) for dimension in LINKAGE_DIMENSIONS
    }
    linkage["raw_archive_hop"] = list(prepared["archive_hops"])
    schemas = list(prepared["aux_schemas"])
    schemas.append({"format": "extxyz", "group": "*.xyz", **frame_schema.to_json()})
    split_counts = merged["split_counts"]
    return {
        "trackers": trackers,
        "capabilities": capabilities,
        "schemas": schemas,
        "summary": {
            "selected_variant": variant,
            "files": merged["per_file"],
            "published_split_counts": dict(sorted(split_counts.items())),
            "materials_spanning_splits": _span_diagnostic(
                merged["material_splits"], limit,
                "material_id observed in more than one published split",
            ),
            "raw_archive": archive_record,
        },
        "frames": frame_schema.frames,
        "csv_rows": 0,
        "splits": dict(sorted(split_counts.items())),
        "errors": list(prepared["aux_errors"]) + merged["errors"],
        "warnings": warnings,
        "linkage_values": linkage,
    }


def _audit_mplitrj(
    files: list[Path], settings: dict[str, Any], raw_archive: Path | None
) -> dict[str, Any]:
    """Serial composition of the MPLiTrj work units (same code as the parallel path)."""

    prepared = _mplitrj_prepare(files, settings, raw_archive)
    states = [
        _mplitrj_file_unit(str(path), settings, prepared["sample_materials"])
        for path in mplitrj_source_files(files)
    ]
    merged = _merge_mplitrj_units(states, settings)
    mapping = _mplitrj_provenance_unit(
        None if raw_archive is None else str(raw_archive),
        settings,
        prepared,
        merged["sample_frames"],
        merged["frames_beyond_cap"],
    )
    return _mplitrj_finalize(files, settings, prepared, merged, mapping)


# --------------------------------------------------------------------------- FPMD


def inspect_optimade_jsonl(path: Path, *, line_limit: int, example_limit: int) -> dict[str, Any]:
    """Stream at most ``line_limit`` lines of an OPTIMADE JSONL export.

    OPTIMADE JSONL begins with an ``x-optimade`` header line whose shape differs
    from structure entries, so record types are counted rather than assumed.
    """

    record_types: Counter[str] = Counter()
    header_keys: list[str] = []
    ids = BoundedExamples(example_limit)
    without_id = BoundedExamples(example_limit)
    failures = BoundedExamples(example_limit)
    attribute_keys: set[str] = set()
    lines = 0
    truncated = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            if lines >= line_limit:
                truncated = True
                break
            lines += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.add(f"line {lines}: {exc.msg}")
                continue
            if not isinstance(value, dict):
                record_types["non_object"] += 1
                continue
            if "x-optimade" in value:
                record_types["x-optimade header"] += 1
                meta = value["x-optimade"]
                header_keys = sorted(meta) if isinstance(meta, dict) else []
                continue
            record_type = str(value.get("type", "untyped"))
            record_types[record_type] += 1
            if record_type == "structures":
                if value.get("id") not in (None, ""):
                    ids.add(str(value["id"]))
                else:
                    without_id.add(f"line {lines}")
                attributes = value.get("attributes")
                if isinstance(attributes, dict):
                    attribute_keys.update(map(str, attributes))
    keys = sorted(attribute_keys)
    return {
        "format": "optimade.jsonl.gz",
        "path": str(path),
        "lines_inspected": lines,
        "line_limit": line_limit,
        "truncated": truncated,
        "header_present": record_types["x-optimade header"] > 0,
        "header_keys": header_keys,
        "record_types": dict(sorted(record_types.items())),
        "structure_ids": ids.to_json("OPTIMADE entry id"),
        "structure_entries_without_id": without_id.to_json("structures entry lacking id"),
        "structure_attribute_key_count": len(keys),
        "structure_attribute_keys": keys[:200],
        "identifier_like_attribute_keys": [
            key for key in keys if re.search(r"(^|_)(id|ids|material|formula)", key.lower())
        ][:200],
        "parse_failures": failures.to_json("JSON decode failure"),
    }


def inspect_aiida_archive(path: Path, *, example_limit: int) -> dict[str, Any]:
    """List an AiiDA archive's ZIP container without AiiDA and without extraction."""

    if not zipfile.is_zipfile(path):
        return {"path": str(path), "container": "not_zip", "readable": False}
    with zipfile.ZipFile(path, mode="r") as archive:
        infos = archive.infolist()
        names = {info.filename: info for info in infos}
        record: dict[str, Any] = {
            "path": str(path),
            "container": "zip",
            "readable": True,
            "members": len(infos),
            "member_examples": sorted(names)[:example_limit],
            "markers": {
                marker: marker in names for marker in ("metadata.json", "db.sqlite3", "data.json")
            },
        }
        info = names.get("metadata.json")
        if info is not None and info.file_size <= 1024 * 1024:
            metadata = json.loads(archive.read(info).decode("utf-8"))
            if isinstance(metadata, dict):
                record["metadata_keys"] = sorted(metadata)[:100]
                for key in ("export_version", "aiida_version"):
                    if key in metadata:
                        record[key] = metadata[key]
    return record


def _audit_fpmd(files: list[Path], settings: dict[str, Any]) -> dict[str, Any]:
    limit = settings["example_limit"]
    errors: list[str] = []
    schemas: list[dict[str, Any]] = []
    doc_keywords: set[str] = set()
    archives: list[dict[str, Any]] = []
    optimade: list[dict[str, Any]] = []
    for path in files:
        name = path.name.lower()
        try:
            if name.endswith(".aiida"):
                record = inspect_aiida_archive(path, example_limit=limit)
                archives.append(record)
                schemas.append({"format": "aiida", **record})
            elif name.endswith(".jsonl.gz"):
                record = inspect_optimade_jsonl(
                    path, line_limit=settings["optimade_line_limit"], example_limit=limit
                )
                optimade.append(record)
                schemas.append(record)
            else:
                record = _describe_file(path, limit)
                schemas.append(record)
                if record["format"] in {"text", "yaml"}:
                    doc_keywords.update(record.get("metadata_keywords", []))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    archive_names = [Path(item["path"]).name.lower() for item in archives]
    docs_present = any(
        item.get("format") == "text" for item in schemas
    )
    aiida_available = aiida_query_available()
    capabilities = {
        "archive_present": _capability(REQUIRED, bool(archives), f"{len(archives)} .aiida archives"),
        "archive_container_readable": _capability(
            REQUIRED, bool(archives) and all(item.get("readable") for item in archives),
            "every .aiida file opens as a ZIP container",
        ),
        "documentation_present": _capability(REQUIRED, docs_present, "README/description text"),
        "trajectory_archive_present": _capability(
            REQUIRED,
            any("trajector" in name or "screening" in name for name in archive_names),
            "archive named for trajectories or screening",
        ),
        "structure_archive_present": _capability(
            OPTIONAL, any("structure" in name for name in archive_names),
            "archive named for starting structures",
        ),
        "diffusion_reference_documented": _capability(
            REQUIRED, "diffusion" in doc_keywords, "documentation mentions diffusion coefficients",
        ),
        "msd_reference_documented": _capability(
            OPTIONAL, "msd" in doc_keywords, "documentation mentions mean-square displacements",
        ),
        "optimade_export_readable": _capability(
            OPTIONAL, bool(optimade) and all(item["lines_inspected"] for item in optimade),
            "bounded streaming read of OPTIMADE JSONL",
        ),
        "aiida_query_available": _capability(
            REQUIRED, aiida_available,
            "AiiDA query capability available" if aiida_available
            else "AiiDA query capability unavailable in current environment",
        ),
    }
    return {
        "trackers": _trackers("FPMD", limit, ()),
        "capabilities": capabilities,
        "schemas": schemas,
        "summary": {"documentation_keywords": sorted(doc_keywords)},
        "frames": 0,
        "csv_rows": 0,
        "splits": {},
        "errors": errors,
        "warnings": [],
        "linkage_values": {},
    }


# --------------------------------------------------------------------------- BVEL/nebBVSE


def _audit_indexed_pool(
    dataset: str, files: list[Path], settings: dict[str, Any]
) -> dict[str, Any]:
    """Prospective datasets: identifiers come from the published index when present."""

    limit = settings["example_limit"]
    errors: list[str] = []
    schemas: list[dict[str, Any]] = []
    trackers = _trackers(dataset, limit, ("published_split",))
    csv_files = [p for p in files if p.suffix.lower() == ".csv"]
    xyz_files = [p for p in files if p.suffix.lower() in XYZ_SUFFIXES]
    record_source = "index_rows" if csv_files else "frames"
    split_counts: Counter[str] = Counter()
    rows_total = 0
    for path in csv_files:
        try:
            _columns, rows, schema = _read_csv(path, limit)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        schemas.append(schema)
        rows_total += len(rows)
        for number, row in enumerate(rows, start=1):
            label = f"{path.name}:row{number}"
            for tracker in trackers.values():
                _observe_source(tracker, row, label)
            split = _source_value(row, IDENTIFIER_FIELDS["published_split"])[0]
            split_counts[split or "<missing>"] += 1

    frame_schema = FrameSchema(limit)
    for path in xyz_files:
        count = 0
        filename_split = split_from_filename(path.name)
        try:
            for index_in_file, atoms in enumerate(_iter_frames(path)):
                count += 1
                label = f"{path.name}:{index_in_file}"
                frame_schema.observe(atoms, label)
                if record_source != "frames":
                    continue
                for tracker in trackers.values():
                    if tracker.dimension != "published_split":
                        _observe_source(tracker, atoms.info, label)
                split, field, conflict = _source_value(atoms.info, IDENTIFIER_FIELDS["published_split"])
                if split is not None:
                    trackers["published_split"].observe(split, field, SOURCE_PROVIDED, label, conflict=conflict)
                else:
                    trackers["published_split"].observe(filename_split, "source_filename", DERIVED, label)
                split_counts[(split or filename_split) or "<missing>"] += 1
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
        finally:
            frame_schema.add_file(count)
    for path in files:
        if path.suffix.lower() not in XYZ_SUFFIXES | {".csv"}:
            try:
                schemas.append(_describe_file(path, limit))
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    if xyz_files:
        schemas.append({"format": "extxyz", "group": "*.xyz", **frame_schema.to_json()})
    capabilities = {
        "records_present": _capability(
            REQUIRED, bool(rows_total or frame_schema.frames),
            f"identifiers evaluated over {record_source}",
            value_source=record_source,
        ),
    }
    return {
        "trackers": trackers,
        "capabilities": capabilities,
        "schemas": schemas,
        "summary": {"identifier_record_source": record_source},
        "frames": frame_schema.frames,
        "csv_rows": rows_total,
        "splits": dict(sorted(split_counts.items())),
        "errors": errors,
        "warnings": [],
        "linkage_values": {},
    }


# --------------------------------------------------------------------------- orchestration


def _settings(config: dict[str, Any] | None) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    settings.update((config or {}).get("audit", {}))
    return settings


def _blockers(result: dict[str, Any], dataset: str) -> list[str]:
    blockers = [
        f"{name}: {item['detail']}"
        for name, item in result["capabilities"].items()
        if item["requirement"] == REQUIRED and not item["available"]
    ]
    for dimension, tracker in result["trackers"].items():
        spec = ROLES[dataset]["identifiers"][dimension]
        covered_by = spec[2] if len(spec) > 2 else None
        if tracker.requirement == REQUIRED and tracker.status != "satisfied" and covered_by is None:
            blockers.append(
                f"required identifier '{dimension}' is {tracker.status} "
                f"({tracker.missing.count} records missing, "
                f"{tracker.conflicts.count} alias conflicts)"
            )
    return blockers


def audit_dataset(
    dataset: str,
    files: list[Path],
    identities: list[dict[str, Any]],
    *,
    settings: dict[str, Any] | None = None,
    raw_archive: Path | None = None,
) -> dict[str, Any]:
    """Audit one dataset against its role requirements without modifying sources."""

    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    started = utc_now()
    if dataset == "nebDFT2k":
        result = _audit_nebdft2k(files, settings)
    elif dataset == "MPLiTrj":
        result = _audit_mplitrj(files, settings, raw_archive)
    elif dataset == "FPMD":
        result = _audit_fpmd(files, settings)
    else:
        result = _audit_indexed_pool(dataset, files, settings)
    return _dataset_report(dataset, files, identities, settings, result, started)


def _dataset_report(
    dataset: str,
    files: list[Path],
    identities: list[dict[str, Any]],
    settings: dict[str, Any],
    result: dict[str, Any],
    started: str,
) -> dict[str, Any]:
    errors = result["errors"]
    blockers = _blockers(result, dataset)
    if not files:
        errors = [f"{dataset} is not available under the configured raw-data root."] + errors
    status = "fail" if errors else ("partial" if blockers else "pass")
    units_observed: dict[str, list[str]] = defaultdict(list)
    for schema in result["schemas"]:
        for key, values in schema.get("unit_metadata", {}).items():
            units_observed[key] = sorted(set(units_observed[key]) | set(values))
    warnings = list(result["warnings"])
    if result["frames"]:
        warnings.append(
            "Energy/force/stress units are not recorded in the files; ASE/VASP conventions "
            "remain unverified against dataset documentation."
        )
    trackers = result["trackers"]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "role": ROLES[dataset]["role"],
        "started_at": started,
        "completed_at": utc_now(),
        "status": status,
        "inputs": _inputs_summary(identities, settings["example_limit"]),
        "record_counts": {
            "files": len(files),
            "csv_rows": result["csv_rows"],
            "frames": result["frames"],
            **{
                f"distinct_{dimension}": tracker.to_json()["distinct_count"]
                for dimension, tracker in trackers.items()
                if tracker.requirement != NOT_APPLICABLE
            },
        },
        "identifiers": {dimension: tracker.to_json() for dimension, tracker in trackers.items()},
        "capabilities": result["capabilities"],
        "schema_summary": result["schemas"],
        "dataset_summary": result["summary"],
        "units": {"observed_unit_metadata": dict(sorted(units_observed.items())), "verified": False},
        "split_summary": {"counts": result["splits"]},
        "linkage_values": result["linkage_values"],
        "validation_checks": {
            "read_only_source": True,
            "all_files_parseable": not result["errors"],
        },
        "warnings": warnings,
        "errors": errors,
        "blockers": blockers,
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
        "litraj_provenance": Path(litraj_provenance.__file__).resolve(),
        "config_loader": Path(load_config.__code__.co_filename).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _cross_dataset_linkage(
    linkage_values: dict[str, dict[str, list[str]]], example_limit: int
) -> dict[str, Any]:
    neb = linkage_values.get("nebDFT2k", {})
    mpli = linkage_values.get("MPLiTrj", {})
    comparisons = {
        "chemical_system": ("chemical_system", "chemical_system", "nebDFT2k chemsys (source) vs MPLiTrj chemsys derived from frame species"),
        "material": ("material", "material", "material_id in both releases"),
        "hop": ("hop", "hop", "flattened MPLiTrj frames carry no edge_id"),
        "hop_vs_raw_archive": ("hop", "raw_archive_hop", "nebDFT2k edge_id vs MPLiTrj_raw <edge_id>.neb directories"),
    }
    result: dict[str, Any] = {}
    for name, (left_key, right_key, note) in comparisons.items():
        left = set(neb.get(left_key, []))
        right = set(mpli.get(right_key, []))
        shared = sorted(left & right)
        result[name] = {
            "nebDFT2k_count": len(left),
            "MPLiTrj_count": len(right),
            "intersection_count": len(shared),
            "example_limit": example_limit,
            "intersection_examples": shared[:example_limit],
            "note": note,
        }
    return result


class G0ExecutionError(RuntimeError):
    """One or more work units crashed; completed checkpoints were kept."""

    def __init__(self, failures: dict[str, str], failures_path: Path) -> None:
        self.failures = failures
        self.failures_path = failures_path
        detail = "; ".join(f"{key}: {message}" for key, message in failures.items())
        super().__init__(f"Gate G0 work units failed ({detail}); see {failures_path}")


# Spawned workers start clean: no inherited locks, file handles or monkeypatched
# state, and identical behaviour on every platform.
_START_METHOD = "spawn"
_THREAD_LIMIT_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def resolve_workers(requested: Any, settings: dict[str, Any]) -> int:
    """CLI value if given, else the JSON config default; must be a positive integer."""

    value = settings.get("workers", 1) if requested is None else requested
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"workers must be a positive integer, got {value!r}")
    return value


def _scientific_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in settings.items() if k not in OPERATIONAL_SETTINGS}


def _scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    snapshot = json.loads(json.dumps({k: v for k, v in config.items() if k != "_config_path"}))
    for key in OPERATIONAL_SETTINGS:
        snapshot.get("audit", {}).pop(key, None)
    return snapshot


@dataclass(frozen=True)
class WorkUnit:
    key: str
    kind: str
    payload: dict[str, Any]
    checkpoint: Path
    fingerprint: str


def _execute_unit(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level (picklable) worker entry point; returns compact JSON-able results."""

    started = time.perf_counter()
    if kind == "dataset":
        result = audit_dataset(
            payload["dataset"],
            [Path(path) for path in payload["files"]],
            payload["identities"],
            settings=payload["settings"],
        )
    elif kind == "mplitrj_file":
        result = _mplitrj_file_unit(payload["path"], payload["settings"], payload["sample_materials"])
    elif kind == "mplitrj_provenance":
        result = _mplitrj_provenance_unit(
            payload["raw_archive"],
            payload["settings"],
            payload["prepared"],
            payload["sample_frames"],
            payload["frames_beyond_cap"],
        )
    else:
        raise ValueError(f"unknown work unit kind {kind!r}")
    return {"result": result, "elapsed_seconds": round(time.perf_counter() - started, 3)}


def _externalize_linkage(run_root: Path, run_id: str, dataset: str, report: dict[str, Any]) -> dict[str, Any]:
    """Move full identifier sets to a sidecar; they scale with the data.

    The sidecar is used only for cross-dataset linkage and is written before the
    checkpoint that references it, so a reusable checkpoint implies its sidecar.
    """

    values = report["linkage_values"]
    _atomic_json(
        run_root / "linkage" / f"{dataset}.json",
        {"dataset": dataset, "run_id": run_id, "values": values},
    )
    compact = dict(report)
    compact["linkage_values"] = {
        "file": f"linkage/{dataset}.json",
        "counts": {key: len(items) for key, items in values.items()},
        "sha256": _fingerprint(values),
    }
    return compact


def _linkage_sidecar_matches(unit: WorkUnit, result: dict[str, Any]) -> bool:
    summary = result.get("linkage_values", {})
    try:
        stored = json.loads(
            (unit.checkpoint.parent.parent / summary["file"]).read_text(encoding="utf-8")
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return _fingerprint(stored.get("values")) == summary.get("sha256")


def _load_checkpoint(unit: WorkUnit) -> dict[str, Any] | None:
    """Reuse a checkpoint only if it matches this unit's fingerprint exactly."""

    try:
        stored = json.loads(unit.checkpoint.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if stored.get("_unit_fingerprint") != unit.fingerprint:
        return None
    result = stored.get("result")
    # Errored units are recomputed: an error may have been transient.
    if not isinstance(result, dict) or result.get("errors"):
        return None
    if unit.kind == "dataset" and not _linkage_sidecar_matches(unit, result):
        return None
    return stored


@contextmanager
def _worker_pool(workers: int):
    """Bounded process pool; ``None`` for serial in-process execution."""

    if workers == 1:
        yield None
        return
    # One BLAS/OpenMP thread per worker unless the caller chose otherwise;
    # spawned children read these at start-up.
    previous = {name: os.environ.get(name) for name in _THREAD_LIMIT_VARIABLES}
    for name in _THREAD_LIMIT_VARIABLES:
        os.environ.setdefault(name, "1")
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context(_START_METHOD)
        ) as pool:
            yield pool
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_units(
    units: list[WorkUnit],
    pool: ProcessPoolExecutor | None,
    *,
    run_id: str,
    resume: bool,
    progress: Callable[[str], None],
    timings: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Run units, checkpointing each success atomically as it completes.

    Results are keyed by unit and consumed by the caller in canonical order, so
    completion order never reaches the report.
    """

    results: dict[str, dict[str, Any]] = {}
    pending: list[WorkUnit] = []
    for unit in units:
        stored = _load_checkpoint(unit) if resume else None
        if stored is not None:
            results[unit.key] = stored["result"]
            timings[unit.key] = {"elapsed_seconds": stored["elapsed_seconds"], "reused": True}
            progress(f"G0 {unit.key}: reused checkpoint")
        else:
            pending.append(unit)

    def record(unit: WorkUnit, outcome: dict[str, Any]) -> None:
        if unit.kind == "dataset":
            outcome = {
                **outcome,
                "result": _externalize_linkage(
                    unit.checkpoint.parent.parent, run_id, unit.key, outcome["result"]
                ),
            }
        _atomic_json(
            unit.checkpoint,
            {
                "unit": unit.key,
                "run_id": run_id,
                "_unit_fingerprint": unit.fingerprint,
                "elapsed_seconds": outcome["elapsed_seconds"],
                "result": outcome["result"],
            },
        )
        results[unit.key] = outcome["result"]
        timings[unit.key] = {"elapsed_seconds": outcome["elapsed_seconds"], "reused": False}
        result = outcome["result"]
        frames = result.get("frames", result.get("record_counts", {}).get("frames"))
        detail = "" if frames is None else f", frames={frames}"
        progress(f"G0 {unit.key}: complete{detail}, {outcome['elapsed_seconds']:.1f}s")

    def fail(unit: WorkUnit, exc: BaseException) -> None:
        if isinstance(exc, BrokenProcessPool):
            # A dead worker (e.g. killed for memory) breaks every pending future;
            # the culprit cannot be identified, so do not blame each unit.
            failures["worker_pool"] = (
                "a worker process terminated abruptly (e.g. out of memory); "
                "the pool is unusable"
            )
            failures[unit.key] = "not completed: worker pool broken"
        else:
            failures[unit.key] = f"{type(exc).__name__}: {exc}"
        progress(f"G0 {unit.key}: FAILED {failures[unit.key]}")

    if pool is None:
        for unit in pending:
            progress(f"G0 {unit.key}: starting")
            try:
                outcome = _execute_unit(unit.kind, unit.payload)
            except Exception as exc:
                fail(unit, exc)
            else:
                record(unit, outcome)
        return results

    futures: dict[Future, WorkUnit] = {}
    for unit in pending:
        if "worker_pool" in failures:
            failures[unit.key] = "not run: worker pool broken"
            continue
        try:
            futures[pool.submit(_execute_unit, unit.kind, unit.payload)] = unit
        except Exception as exc:  # e.g. BrokenProcessPool after an earlier crash
            fail(unit, exc)
            continue
        progress(f"G0 {unit.key}: starting")
    for future in as_completed(futures):
        unit = futures[future]
        try:
            outcome = future.result()
        except Exception as exc:
            fail(unit, exc)
        else:
            record(unit, outcome)
    return results


def run_audit(
    config_path: str | Path,
    *,
    resume: bool = False,
    workers: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run or resume the Gate G0 audit and return the report and its path.

    Independent work units (each non-MPLiTrj dataset, each MPLiTrj source file,
    then the MPLiTrj provenance sample) run on a bounded process pool; with
    ``workers=1`` the same units run in-process in canonical order.  Only this
    parent process writes checkpoints and the canonical report.
    """

    progress = progress or (lambda message: None)
    wall_start = time.perf_counter()
    config = load_config(config_path)
    settings = _settings(config)
    workers = resolve_workers(workers, settings)
    science = _scientific_settings(settings)
    data_root = Path(config["paths"]["data_root"])
    raw_root = data_root / "raw"
    output_root = _safe_output_root(config)
    hash_limit = int(settings["hash_max_bytes"])
    discovered = discover_datasets(raw_root)
    raw_archive = data_root / settings["mplitrj_raw_archive"] if settings["mplitrj_raw_archive"] else None
    identities = {
        name: [_file_identity(path, data_root, hash_limit) for path in paths]
        for name, paths in discovered.items()
    }
    archive_identity = (
        [_file_identity(raw_archive, data_root, hash_limit)]
        if raw_archive is not None and raw_archive.is_file()
        else []
    )
    configuration = _configuration_record(config)
    implementation = _implementation_record()
    environment = _environment_capabilities()
    git_sha = _git("rev-parse", "HEAD")
    # The worker count is operational, so it is excluded: serial and parallel
    # runs share run IDs and checkpoints.
    run_inputs = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha,
        "configuration": _fingerprint(_scientific_config(config)),
        "implementation": implementation,
        "environment": environment,
        "inputs": identities,
        "raw_archive": archive_identity,
    }
    run_id = _fingerprint(run_inputs)[:16]
    run_root = output_root / "g0" / run_id
    final_path = run_root / "audit.json"
    if resume and final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        datasets_reusable = all(
            item.get("status") in {"pass", "partial"} and not item.get("errors")
            for item in existing.get("datasets", [])
        )
        if (
            existing.get("run_id") == run_id
            and existing.get("status") in {"pass", "partial"}
            and not existing.get("errors")
            and datasets_reusable
            # The run ID ignores operational settings, so only reuse a report
            # whose recorded configuration is the file actually used now.
            and existing.get("configuration", {}).get("sha256") == configuration["sha256"]
            and existing.get("configuration", {}).get("path") == configuration["path"]
        ):
            progress(f"G0: reused complete report {final_path}")
            return existing, final_path

    started = utc_now()
    if final_path.exists():
        # This run supersedes the previous canonical report; set it aside so a
        # failed rerun never leaves an audit.json that looks complete.
        final_path.replace(run_root / "audit.previous.json")
    _atomic_json(
        run_root / "inputs.json",
        {"run_id": run_id, "inputs": identities, "mplitrj_raw_archive": archive_identity},
    )
    base_fingerprint = {
        "run_id": run_id,
        "git_sha": git_sha,
        "implementation": implementation,
        "environment": environment,
        "settings": science,
    }

    def fingerprint(key: str, **extra: Any) -> str:
        return _fingerprint({**base_fingerprint, "unit": key, **extra})

    timings: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    dataset_units = [
        WorkUnit(
            key=name,
            kind="dataset",
            payload={
                "dataset": name,
                "files": [str(path) for path in discovered[name]],
                "identities": identities[name],
                "settings": science,
            },
            checkpoint=run_root / "datasets" / f"{name}.json",
            fingerprint=fingerprint(name, inputs=identities[name]),
        )
        for name in DATASETS
        if name != "MPLiTrj"
    ]
    mpl_files = discovered["MPLiTrj"]
    mpl_sources = mplitrj_source_files(mpl_files)
    mpl_final = WorkUnit(
        key="MPLiTrj",
        kind="dataset",
        payload={},
        checkpoint=run_root / "datasets" / "MPLiTrj.json",
        fingerprint=fingerprint("MPLiTrj", inputs=identities["MPLiTrj"] + archive_identity),
    )
    mpl_report = _load_checkpoint(mpl_final) if resume else None

    with _worker_pool(workers) as pool:
        file_units: list[WorkUnit] = []
        prepared: dict[str, Any] | None = None
        if mpl_report is None:
            progress("G0 MPLiTrj: indexing raw archive and auxiliary files")
            prepare_start = time.perf_counter()
            prepared = _mplitrj_prepare(mpl_files, science, raw_archive)
            timings["MPLiTrj/prepare"] = {
                "elapsed_seconds": round(time.perf_counter() - prepare_start, 3),
                "reused": False,
            }
            by_identifier = {item["path"]: item for item in identities["MPLiTrj"]}
            for path in mpl_sources:
                key = f"MPLiTrj/{path.stem}"
                file_units.append(
                    WorkUnit(
                        key=key,
                        kind="mplitrj_file",
                        payload={
                            "path": str(path),
                            "settings": science,
                            "sample_materials": prepared["sample_materials"],
                        },
                        checkpoint=run_root / "checkpoints" / "MPLiTrj" / f"{path.stem}.json",
                        fingerprint=fingerprint(
                            key,
                            inputs=[by_identifier[str(path)]],
                            sample_materials=prepared["sample_materials"],
                        ),
                    )
                )
            if len({unit.key for unit in file_units}) != len(file_units):
                raise ValueError("MPLiTrj source files must have unique stems for checkpointing")
        else:
            progress("G0 MPLiTrj: reused checkpoint")
            timings["MPLiTrj"] = {"elapsed_seconds": mpl_report["elapsed_seconds"], "reused": True}

        # Longest units first keeps the pool busy; results are re-ordered later.
        first_batch = file_units + dataset_units
        results = _run_units(
            first_batch, pool, run_id=run_id, resume=resume,
            progress=progress, timings=timings, failures=failures,
        )

        if (
            prepared is not None
            and "worker_pool" not in failures
            and not any(unit.key in failures for unit in file_units)
        ):
            merged = _merge_mplitrj_units([results[unit.key] for unit in file_units], science)
            provenance = WorkUnit(
                key="MPLiTrj/provenance_sample",
                kind="mplitrj_provenance",
                payload={
                    "raw_archive": None if raw_archive is None else str(raw_archive),
                    "settings": science,
                    "prepared": prepared,
                    "sample_frames": merged["sample_frames"],
                    "frames_beyond_cap": merged["frames_beyond_cap"],
                },
                checkpoint=run_root / "checkpoints" / "MPLiTrj" / "provenance_sample.json",
                fingerprint=fingerprint(
                    "MPLiTrj/provenance_sample",
                    raw_archive=archive_identity,
                    prepared=_fingerprint(prepared),
                    sample=_fingerprint(merged["sample_frames"]),
                    frames_beyond_cap=merged["frames_beyond_cap"],
                ),
            )
            results.update(
                _run_units(
                    [provenance], pool, run_id=run_id, resume=resume,
                    progress=progress, timings=timings, failures=failures,
                )
            )
            if provenance.key not in failures:
                finalize_start = time.perf_counter()
                result = _mplitrj_finalize(
                    mpl_files, science, prepared, merged, results[provenance.key]
                )
                report = _externalize_linkage(
                    run_root,
                    run_id,
                    "MPLiTrj",
                    _dataset_report(
                        "MPLiTrj", mpl_files, identities["MPLiTrj"], science, result, started
                    ),
                )
                elapsed = round(
                    time.perf_counter() - finalize_start
                    + timings["MPLiTrj/prepare"]["elapsed_seconds"],
                    3,
                )
                _atomic_json(
                    mpl_final.checkpoint,
                    {
                        "unit": "MPLiTrj",
                        "run_id": run_id,
                        "_unit_fingerprint": mpl_final.fingerprint,
                        "elapsed_seconds": elapsed,
                        "result": report,
                    },
                )
                mpl_report = {"result": report}
                timings["MPLiTrj/finalize"] = {
                    "elapsed_seconds": round(time.perf_counter() - finalize_start, 3),
                    "reused": False,
                }
        elif prepared is not None:
            failures.setdefault(
                "MPLiTrj/provenance_sample",
                "not run: an MPLiTrj source-file unit failed or the worker pool broke",
            )

    if failures:
        failures_path = run_root / "failures.json"
        _atomic_json(failures_path, {"run_id": run_id, "failures": dict(sorted(failures.items()))})
        raise G0ExecutionError(dict(sorted(failures.items())), failures_path)
    stale_failures = run_root / "failures.json"
    if stale_failures.exists():
        stale_failures.unlink()

    # Canonical dataset order, independent of completion order.
    dataset_reports: list[dict[str, Any]] = []
    linkage_values: dict[str, dict[str, list[str]]] = {}
    for name in DATASETS:
        report = mpl_report["result"] if name == "MPLiTrj" else results[name]
        linkage_path = run_root / report["linkage_values"]["file"]
        linkage_values[name] = json.loads(linkage_path.read_text(encoding="utf-8"))["values"]
        dataset_reports.append(report)

    statuses = {item["status"] for item in dataset_reports}
    status = "fail" if "fail" in statuses else ("partial" if "partial" in statuses else "pass")
    unit_seconds = {key: value["elapsed_seconds"] for key, value in sorted(timings.items())}
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
        "input_manifest": str(run_root / "inputs.json"),
        "gate_summary": {
            item["dataset"]: {
                "role": item["role"],
                "status": item["status"],
                "blockers": item["blockers"],
                "errors": len(item["errors"]),
            }
            for item in dataset_reports
        },
        "capabilities": {item["dataset"]: item["capabilities"] for item in dataset_reports},
        "linkage_summary": {
            "nebDFT2k_to_MPLiTrj": _cross_dataset_linkage(linkage_values, settings["example_limit"])
        },
        "warnings": [
            f"{item['dataset']}: {warning}" for item in dataset_reports for warning in item["warnings"]
        ],
        "errors": [
            f"{item['dataset']}: {error}" for item in dataset_reports for error in item["errors"]
        ],
        "blockers": [
            f"{item['dataset']}: {blocker}" for item in dataset_reports for blocker in item["blockers"]
        ],
        "datasets": dataset_reports,
        "environment": {"python": sys.version, "platform": platform.platform(), **environment},
        # Operational metadata only; excluded from scientific comparisons.
        "execution": {
            "workers": workers,
            "mode": "in-process" if workers == 1 else f"process pool ({_START_METHOD})",
            "wall_seconds": round(time.perf_counter() - wall_start, 3),
            "unit_elapsed_seconds": unit_seconds,
            "units_reused": sorted(key for key, value in timings.items() if value["reused"]),
            "longest_unit": max(unit_seconds, key=unit_seconds.get) if unit_seconds else None,
        },
        "command": " ".join(sys.argv),
        "seed": config.get("runtime", {}).get("seed"),
    }
    _atomic_json(final_path, report)
    progress(f"G0: report written {final_path}")
    return report, final_path


def concise_summary(report: dict[str, Any], path: Path) -> str:
    """Render the human-facing summary from the canonical JSON report."""

    lines = [f"Gate G0: {report['status']} ({report['run_id']})"]
    for item in report["datasets"]:
        counts = item["record_counts"]
        lines.append(
            f"- {item['dataset']} [{item['role']}]: {item['status']}; files={counts['files']}, "
            f"rows={counts['csv_rows']}, frames={counts['frames']}"
        )
        lines.extend(f"    blocker: {blocker}" for blocker in item["blockers"])
        lines.extend(f"    error: {error}" for error in item["errors"][:3])
    lines.append(f"Blockers: {len(report['blockers'])}; errors: {len(report['errors'])}")
    lines.append(f"JSON report: {path}")
    return "\n".join(lines)
