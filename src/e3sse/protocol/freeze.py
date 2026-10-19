"""Materialize a deterministic E3-SSE protocol from validated read-only inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ase.io import iread

from e3sse.data.dataset_audit import DATASETS
from e3sse.data.dataset_audit import SCHEMA_VERSION as G0_SCHEMA
from e3sse.data.litraj_provenance import parse_edge_id
from e3sse.data.mplitrj_provenance import SCHEMA_VERSION as PROVENANCE_SCHEMA
from e3sse.data.mplitrj_provenance import material_bucket

from .configuration import (
    ProtocolConfigurationError,
    ProtocolConfigurationValidation,
    assess_production_configuration,
    freeze_blockers,
    validate_protocol_configuration,
)
from .contracts import (
    ELIGIBILITY_RULE,
    ENDPOINT_RELAXATION_FILE_KINDS,
    INFERENCE_REQUIRED_FIELDS,
    INFERENCE_RUN_REQUIRED_FIELDS,
    MODEL_PROVENANCE_REQUIRED_FIELDS,
    NEB_PATH_FILE_KINDS,
    REPEAT_FITTED_OBJECTS,
    UNIT_CONVERSIONS,
    canonical_analysis_units,
)
from .decisions import DECISION_REGISTER, DECISION_REGISTER_SCHEMA, validate_decision_register
from .linkage import (
    EvaluationLinkageError,
    combine_partition_universe,
    derive_nebdft2k_linkage,
    validate_nebdft2k_linkage,
)
from .partitions import (
    CHEMICAL_SYSTEM_ALGORITHM,
    analysis_seed,
    build_repeated_partitions,
    classify_primary_population,
    derive_study5_exclusions,
    normalize_chemical_system,
    published_split_summary,
)

CONFIG_SCHEMA = "e3sse.protocol.freeze.config.v1"
DATASET_MANIFEST_SCHEMA = "e3sse.dataset.manifest.v1"
PROTOCOL_SCHEMA = "e3sse.protocol.freeze.v1"
RUN_SCHEMA = "e3sse.protocol.freeze.run.v1"


_PROVENANCE_FIELDS = (
    "derived_frame_id",
    "flattened_file",
    "flattened_frame_index",
    "material_id",
    "edge_id",
    "published_split",
    "mapping_status",
    "raw_file_kind",
    "candidate_count",
    "raw_archive_path",
)


class ProtocolFreezeError(RuntimeError):
    """The protocol cannot be materialized without weakening a frozen invariant."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolFreezeError(f"cannot read JSON {path}: {exc}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Publishing a new inode must never replace an existing scientific artifact.
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ProtocolFreezeError(f"refusing to replace existing artifact: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolFreezeError(f"cannot identify repository Git state: {exc}") from exc


def _validate_repository_state(
    repository: Path, *, expected_sha: str | None, require_clean: bool
) -> tuple[str, str, bool]:
    current_sha = _git(repository, "rev-parse", "HEAD")
    if expected_sha not in {None, "CURRENT"} and current_sha != expected_sha:
        raise ProtocolFreezeError("repository Git SHA mismatch")
    branch = _git(repository, "branch", "--show-current")
    dirty = bool(_git(repository, "status", "--porcelain"))
    if require_clean and dirty:
        raise ProtocolFreezeError("production protocol materialization requires a clean worktree")
    return current_sha, branch, dirty


def _expect_hash(path: Path, expected: str | None, label: str) -> str:
    if not path.is_file():
        raise ProtocolFreezeError(f"missing {label}: {path}")
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise ProtocolFreezeError(f"{label} SHA-256 mismatch")
    return actual


def _validate_g0_input_manifest(
    path: Path, expected: Mapping[str, Any], audit: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    digest = _expect_hash(path, expected.get("sha256"), "G0 input manifest")
    manifest = _load_json(path)
    if manifest.get("run_id") != audit.get("run_id") or not isinstance(
        manifest.get("inputs"), dict
    ):
        raise ProtocolFreezeError("G0 input manifest does not match the audit")
    if set(manifest["inputs"]) != set(DATASETS):
        raise ProtocolFreezeError("G0 input inventory must include every canonical dataset")
    for dataset in audit["datasets"]:
        identities = manifest["inputs"][dataset["dataset"]]
        if not isinstance(identities, list):
            raise ProtocolFreezeError("G0 input inventory entries must be lists")
        digest_payload = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(digest_payload).hexdigest() != dataset["inputs"].get("identity_digest"):
            raise ProtocolFreezeError(f"G0 input inventory digest mismatch: {dataset['dataset']}")
    return manifest, digest


def _safe_member(root: Path, relative: str) -> Path:
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ProtocolFreezeError(f"unsafe provenance member path: {relative}")
    resolved = (root / member).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolFreezeError(f"provenance member escapes index root: {relative}") from exc
    return resolved


def _validate_g0(audit_path: Path, expected: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    digest = _expect_hash(audit_path, expected.get("sha256"), "G0 audit")
    audit = _load_json(audit_path)
    if audit.get("schema_version") != G0_SCHEMA:
        raise ProtocolFreezeError("incompatible G0 schema")
    if expected.get("run_id") and audit.get("run_id") != expected["run_id"]:
        raise ProtocolFreezeError("G0 run ID mismatch")
    datasets = audit.get("datasets")
    if not isinstance(datasets, list):
        raise ProtocolFreezeError("G0 datasets must be a list")
    names = [item.get("dataset") for item in datasets]
    if len(names) != len(set(names)) or set(names) != set(DATASETS):
        raise ProtocolFreezeError("G0 must contain each canonical dataset exactly once")
    # G0 reports "fail" when any dataset fails; a failed transport (FPMD) dataset must
    # not block the static branch, which is gated on its own datasets' statuses.
    statuses = {item.get("dataset"): item.get("status") for item in datasets}
    static_failed = [
        name for name, status in statuses.items() if name != "FPMD" and status == "fail"
    ]
    if audit.get("status") not in {"partial", "pass", "fail"} or static_failed:
        raise ProtocolFreezeError(f"G0 is not usable for the static branch: {static_failed}")
    return audit, digest


def _validate_provenance(
    root: Path, expected: Mapping[str, Any], g0: Mapping[str, Any]
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    digest = _expect_hash(manifest_path, expected.get("manifest_sha256"), "provenance manifest")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != PROVENANCE_SCHEMA or manifest.get("status") != "complete":
        raise ProtocolFreezeError("incompatible or incomplete MPLiTrj provenance manifest")
    if expected.get("index_id") and manifest.get("index_id") != expected["index_id"]:
        raise ProtocolFreezeError("provenance index ID mismatch")
    state = g0.get("mplitrj_provenance_index", {})
    if state.get("status") != "valid":
        raise ProtocolFreezeError("G0 did not validate the provenance index")
    if state.get("index_id") != manifest.get("index_id") or state.get("manifest_sha256") != digest:
        raise ProtocolFreezeError("G0/provenance identity mismatch")
    if not manifest.get("complete_coverage") or manifest.get("coverage_blockers"):
        raise ProtocolFreezeError("provenance coverage is incomplete")

    descriptors = [manifest.get("edges", {}), *manifest.get("shards", [])]
    paths = [item.get("path") for item in descriptors]
    if any(not isinstance(value, str) for value in paths) or len(paths) != len(set(paths)):
        raise ProtocolFreezeError("duplicate or invalid provenance member paths")
    for descriptor in descriptors:
        path = _safe_member(root, descriptor["path"])
        if (
            _expect_hash(path, descriptor.get("sha256"), "provenance member")
            != descriptor["sha256"]
        ):
            raise AssertionError("unreachable")

    source_splits = {
        Path(item["identifier"]).name: item["published_split"]
        for item in manifest.get("flattened_sources", [])
    }
    if len(source_splits) != len(manifest.get("flattened_sources", [])):
        raise ProtocolFreezeError("duplicate flattened provenance source")
    records: list[dict[str, Any]] = []
    seen_frames: set[str] = set()
    counts: Counter[str] = Counter()
    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    indices_by_file: dict[str, set[int]] = defaultdict(set)
    for descriptor in sorted(manifest["shards"], key=lambda item: item["path"]):
        shard = _load_json(_safe_member(root, descriptor["path"]))
        if (
            shard.get("schema_version") != PROVENANCE_SCHEMA
            or shard.get("index_id") != manifest["index_id"]
        ):
            raise ProtocolFreezeError("shard schema or index ID mismatch")
        shard_split = shard.get("split")
        if shard_split not in {"train", "val", "test"}:
            raise ProtocolFreezeError(f"shard has no valid published split: {shard_split!r}")
        shard_records = shard.get("records")
        if not isinstance(shard_records, list) or len(shard_records) != descriptor.get(
            "record_count"
        ):
            raise ProtocolFreezeError("shard record-count mismatch")
        for record in shard_records:
            frame_id = record.get("derived_frame_id")
            expected_frame = f"{record.get('flattened_file')}:{record.get('flattened_frame_index')}"
            if frame_id != expected_frame or frame_id in seen_frames:
                raise ProtocolFreezeError("duplicate or inconsistent provenance frame ID")
            seen_frames.add(frame_id)
            split = record.get("published_split")
            if split not in {"train", "val", "test"} or split != shard_split:
                raise ProtocolFreezeError(f"invalid or shard-inconsistent split: {frame_id}")
            if split != source_splits.get(record.get("flattened_file")):
                raise ProtocolFreezeError("provenance split differs from source split")
            status = record.get("mapping_status")
            counts[status] += 1
            per_split[split][status] += 1
            if status not in {"mapped_unique", "mapped_hop_unique"}:
                raise ProtocolFreezeError("unresolved provenance rows block materialization")
            if not record.get("material_id") or not record.get("edge_id"):
                raise ProtocolFreezeError("mapped provenance row lacks material/hop identity")
            if parse_edge_id(record["edge_id"]) != record["material_id"]:
                raise ProtocolFreezeError("hop/material provenance mismatch")
            # The producer writes material_bucket in the shard body, not the descriptor.
            if shard.get("material_bucket") != material_bucket(record["material_id"]):
                raise ProtocolFreezeError("record is in the wrong material bucket")
            indices_by_file[record["flattened_file"]].add(record["flattened_frame_index"])
            # Retain only the fields the protocol consumes (~929k rows in production).
            records.append({key: record.get(key) for key in _PROVENANCE_FIELDS})
    if len(records) != manifest["counts"]["record_count"]:
        raise ProtocolFreezeError("provenance global record-count mismatch")
    if len(records) != manifest["counts"]["mapped_count"]:
        raise ProtocolFreezeError("provenance is not fully mapped")
    for status in ("mapped_unique", "mapped_hop_unique", "ambiguous", "unmapped", "error"):
        if counts[status] != manifest["counts"].get(f"{status}_count"):
            raise ProtocolFreezeError("provenance status-count mismatch")
    expected_by_split = manifest.get("counts_by_split")
    if not isinstance(expected_by_split, Mapping) or set(expected_by_split) != set(per_split):
        raise ProtocolFreezeError("provenance per-split summary is missing or inconsistent")
    for split, split_counts in per_split.items():
        expected = expected_by_split[split]
        record_count = sum(split_counts.values())
        mapped_count = sum(split_counts[name] for name in ("mapped_unique", "mapped_hop_unique"))
        if (
            expected.get("record_count") != record_count
            or expected.get("mapped_count") != mapped_count
        ):
            raise ProtocolFreezeError(f"provenance per-split counts differ for {split}")
        for status in ("mapped_unique", "mapped_hop_unique", "ambiguous", "unmapped", "error"):
            if expected.get(f"{status}_count") != split_counts[status]:
                raise ProtocolFreezeError(f"provenance per-split status counts differ for {split}")
    for source in manifest["flattened_sources"]:
        name = Path(source["identifier"]).name
        if indices_by_file[name] != set(range(source["frames"])):
            raise ProtocolFreezeError(f"provenance frame indices are incomplete for {name}")
    g0_mpl = next(item for item in g0["datasets"] if item["dataset"] == "MPLiTrj")
    if g0_mpl["record_counts"].get("frames") != len(records):
        raise ProtocolFreezeError("G0/MPLiTrj provenance frame-count mismatch")
    if not g0_mpl.get("capabilities", {}).get("same_hop_exclusion_supported", {}).get("available"):
        raise ProtocolFreezeError("G0 does not support same-hop exclusion")
    edges = _load_json(_safe_member(root, manifest["edges"]["path"]))
    edge_counts = Counter(record["edge_id"] for record in records)
    if manifest["edges"].get("count") != len(edge_counts):
        raise ProtocolFreezeError("provenance edge count differs from mapped hops")
    if edges.get("index_id") != manifest["index_id"] or edges.get("edge_frame_counts") != dict(
        sorted(edge_counts.items())
    ):
        raise ProtocolFreezeError("edge-count table mismatch")
    return manifest, digest, records


def _material_systems_from_index(
    path: Path,
    expected_sha256: str | None,
    provenance: Mapping[str, Any],
    provenance_sha: str,
) -> tuple[dict[str, str], str]:
    digest = _expect_hash(path, expected_sha256, "material/chemical-system linkage")
    payload = _load_json(path)
    if payload.get("schema_version") != "e3sse.material-chemical-system.v1":
        raise ProtocolFreezeError("incompatible material/chemical-system linkage schema")
    if (
        payload.get("provenance_index_id") != provenance["index_id"]
        or payload.get("provenance_manifest_sha256") != provenance_sha
        or payload.get("chemical_system_algorithm") != CHEMICAL_SYSTEM_ALGORITHM
    ):
        raise ProtocolFreezeError("chemical-system linkage is not bound to the provenance index")
    expected_sources = {
        item["identifier"]: item["sha256"] for item in provenance["flattened_sources"]
    }
    if payload.get("flattened_source_sha256") != expected_sources:
        raise ProtocolFreezeError("chemical-system linkage source hashes differ")
    result: dict[str, str] = {}
    for record in payload.get("materials", []):
        material = record.get("material_id")
        system = normalize_chemical_system(record.get("chemical_system_id", ""))
        if not material or (material in result and result[material] != system):
            raise ProtocolFreezeError("missing or conflicting material chemical system")
        result[material] = system
    return result, digest


def _derive_material_systems(
    server_config: Mapping[str, Any], manifest: Mapping[str, Any], records: list[dict[str, Any]]
) -> tuple[dict[str, str], str]:
    """Read only flattened frame species because the v1 provenance index omits chemsys."""

    data_root = Path(server_config["paths"]["data_root"])
    expected_by_frame = {record["derived_frame_id"]: record for record in records}
    systems: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for source in manifest["flattened_sources"]:
        path = (data_root / source["identifier"]).resolve()
        try:
            path.relative_to(data_root.resolve())
        except ValueError as exc:
            raise ProtocolFreezeError("flattened MPLiTrj source escapes data_root") from exc
        if _expect_hash(path, source.get("sha256"), "flattened MPLiTrj source") != source["sha256"]:
            raise AssertionError("unreachable")
        seen = 0
        for index, atoms in enumerate(iread(str(path), index=":")):
            frame_id = f"{path.name}:{index}"
            provenance = expected_by_frame.get(frame_id)
            if provenance is None:
                raise ProtocolFreezeError(f"frame absent from provenance index: {frame_id}")
            material = atoms.info.get("material_id")
            if material != provenance["material_id"]:
                raise ProtocolFreezeError(f"material mismatch for {frame_id}")
            system = normalize_chemical_system(atoms.get_chemical_symbols())
            if material in systems and systems[material] != system:
                raise ProtocolFreezeError(f"conflicting chemical systems for {material}")
            systems[material] = system
            counts[material] += 1
            seen += 1
        if seen != source["frames"]:
            raise ProtocolFreezeError(f"frame count mismatch for {source['identifier']}")
    if set(systems) != {record["material_id"] for record in records}:
        raise ProtocolFreezeError("material/chemical-system linkage is incomplete")
    identity = canonical_sha256(
        {"algorithm": CHEMICAL_SYSTEM_ALGORITHM, "systems": systems, "frame_counts": counts}
    )
    return systems, identity


def _link_records(
    records: list[dict[str, Any]],
    systems: Mapping[str, str],
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_hashes = {
        Path(item["identifier"]).name: item["sha256"]
        for item in provenance["flattened_sources"]
    }
    linked = []
    for record in records:
        material = record["material_id"]
        if material not in systems:
            raise ProtocolFreezeError(f"no chemical system for material {material}")
        linked.append(
            {
                "frame_id": record["derived_frame_id"],
                "provenance_index_id": provenance["index_id"],
                "source_file_sha256": source_hashes[record["flattened_file"]],
                "source_frame_index": record["flattened_frame_index"],
                "input_locator_sha256": canonical_sha256(
                    {
                        "provenance_index_id": provenance["index_id"],
                        "source_file_sha256": source_hashes[record["flattened_file"]],
                        "source_frame_index": record["flattened_frame_index"],
                    }
                ),
                "material_id": material,
                "hop_id": record["edge_id"],
                "chemical_system_id": systems[material],
                "published_split": record["published_split"],
                "mapping_status": record["mapping_status"],
                "raw_file_kind": record.get("raw_file_kind"),
                "origin_evidence_status": (
                    "unequivocal"
                    if record.get("mapping_status") == "mapped_unique"
                    and record.get("candidate_count") == 1
                    and record.get("raw_file_kind")
                    in {"source", "target", "band_optim_step", "traj_init"}
                    else "unresolved"
                ),
                "origin_evidence": (
                    f"MPLiTrj provenance raw_archive_path={record.get('raw_archive_path')}"
                    if record.get("mapping_status") == "mapped_unique"
                    and record.get("candidate_count") == 1
                    and record.get("raw_archive_path")
                    else None
                ),
                "candidate_count": record.get("candidate_count"),
                "raw_archive_path": record.get("raw_archive_path"),
            }
        )
    return linked


def _dataset_manifest(
    audit: Mapping[str, Any],
    audit_sha: str,
    g0_inputs: Mapping[str, Any],
    g0_inputs_sha: str,
    provenance: Mapping[str, Any],
    provenance_sha: str,
) -> dict[str, Any]:
    datasets = []
    for item in audit["datasets"]:
        entry = {
            "dataset": item["dataset"],
            "scientific_role": item["role"],
            "immutable_source": True,
            "source_identifiers": item["inputs"].get("examples", []),
            "input_identity_digest": item["inputs"].get("identity_digest"),
            "content_hash_coverage": {
                "files": len(g0_inputs["inputs"][item["dataset"]]),
                "sha256_skipped": sum(
                    entry.get("sha256") is None for entry in g0_inputs["inputs"][item["dataset"]]
                ),
                "note": "files above the G0 hash_max_bytes are identified by size/mtime only",
            },
            "g0_errors": item.get("errors", []),
            "g0_warnings": item.get("warnings", []),
            "complete_source_inventory": g0_inputs["inputs"][item["dataset"]],
            "record_counts": item["record_counts"],
            "split_information": item["split_summary"],
            "identifier_schema": item["identifiers"],
            "units": item["units"],
            "g0_status": item["status"],
            "g0_blockers": item["blockers"],
            "capabilities": item["capabilities"],
        }
        if item["dataset"] == "MPLiTrj":
            entry["provenance"] = {
                "schema_version": provenance["schema_version"],
                "index_id": provenance["index_id"],
                "manifest_sha256": provenance_sha,
                "counts": provenance["counts"],
                "counts_by_split": provenance["counts_by_split"],
                "complete_coverage": provenance["complete_coverage"],
                "consumed_not_regenerated": True,
                "source_archive": provenance.get("source_archive"),
                "flattened_sources": [
                    {
                        key: source.get(key)
                        for key in ("identifier", "sha256", "frames", "published_split")
                    }
                    for source in provenance["flattened_sources"]
                ],
                "members": [
                    {key: member.get(key) for key in ("path", "sha256", "record_count", "count")}
                    for member in [provenance["edges"], *provenance["shards"]]
                ],
            }
        entry["provenance_artifacts"] = {
            "g0_audit": {"run_id": audit["run_id"], "sha256": audit_sha},
            "g0_input_manifest": {"run_id": g0_inputs["run_id"], "sha256": g0_inputs_sha},
        }
        datasets.append(entry)
    static_names = {"nebDFT2k", "MPLiTrj", "BVEL13k", "nebBVSE122k"}
    by_name = {item["dataset"]: item for item in audit["datasets"]}
    static_ready = all(by_name[name]["status"] == "pass" for name in static_names)
    fpmd = by_name["FPMD"]
    transport_status = {"pass": "ready", "partial": "partial"}.get(fpmd["status"], "blocked")
    result = {
        "manifest_schema_version": DATASET_MANIFEST_SCHEMA,
        "source_g0": {"run_id": audit["run_id"], "sha256": audit_sha},
        "source_g0_input_manifest": {
            "run_id": g0_inputs["run_id"],
            "sha256": g0_inputs_sha,
            "dataset_names": sorted(g0_inputs["inputs"]),
        },
        "datasets": datasets,
        "g0_status": audit["status"],
        "static_branch_status": "ready" if static_ready else "blocked",
        "transport_branch_status": transport_status,
        "transport_blockers": fpmd["blockers"],
    }
    result["manifest_id"] = canonical_sha256(result)
    return result


def _register_resolution(config: Mapping[str, Any], identifier: str) -> dict[str, Any]:
    """The register item (and its resolution, if any) that a protocol field depends on."""

    for item in config["protocol"]["unresolved_items"]:
        if item.get("id") == identifier:
            return {"register_item": identifier, **dict(item)}
    return {"register_item": identifier, "status": "missing_from_configuration"}


def _protocol_content(
    *,
    git_sha: str,
    source_documents: Mapping[str, str],
    dataset_manifest: Mapping[str, Any],
    partition_manifest: Mapping[str, Any],
    exclusions: Mapping[str, Any],
    units: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_schema_version": PROTOCOL_SCHEMA,
        "git_sha": git_sha,
        "source_document_hashes": dict(sorted(source_documents.items())),
        "dataset_manifest": dataset_manifest,
        "partition_definition": partition_manifest,
        "study5_exclusions": exclusions,
        "chemical_system_definition": {
            "algorithm": CHEMICAL_SYSTEM_ALGORITHM,
            "classification": "repository_existing",
            "definition": "sorted unique valid element symbols joined by '-'",
        },
        "inclusion_exclusion_rules": {
            "primary_tm_hubbard_u": config["protocol"]["scientific_resolutions"][
                "primary_tm_hubbard_u"
            ],
            "mixed_anion_family": config["protocol"]["scientific_resolutions"][
                "mixed_anion_family"
            ],
            "failed_or_nonmonotone_neb": "exclude_and_count_when_flag_provided",
            "convergence_flag_evidence": "partition_definition.nebdft2k_source_columns",
            "barrier_definition": "maximum_minus_minimum_reference_image_energy",
            "fpmd_host_melting": "Study_3_exclusion_not_applied_here",
        },
        "leakage_rules": {
            "primary": "exclude_frames_generated_during_target_hop_NEB_optimisation",
            "permitted": ["unequivocal_endpoint_relaxation", "other_hops_same_structure"],
            "target_hop_path_file_kinds": list(NEB_PATH_FILE_KINDS),
            "oracle": "own_path_only_explicit_oracle_never_primary",
            "oracle_label": "oracle_upper_bound",
            "uncertain_provenance": "fail_closed",
            "scope_partition_rule": {
                "global": "training_or_calibration_side_only",
                "family": "training_or_calibration_side_only",
                "element": "training_or_calibration_side_only",
                "material": "target material frames in the target hop's partition: training "
                "side admits Research Design 3.6 frames; test side non-path frames only",
            },
            "eligibility_rule": ELIGIBILITY_RULE,
            "classification": "paper_specified (Research Design 3.6, 5.4)",
            "target_hop_endpoint_relaxation_file_kinds": list(ENDPOINT_RELAXATION_FILE_KINDS),
            "target_hop": {
                "neb_path_frames": "prohibited_for_primary_oracle_only",
                "endpoint_relaxation_frames": "admissible",
                "unidentified_origin": "fail_closed",
            },
            "other_hops_same_structure": "admissible; test-side material scope non-path only",
            "estimator_use": "eligibility is frozen; estimators may use any eligible subset",
            "implementation": "e3sse.protocol.contracts.audit_softening_frame_eligibility",
            "primary_guard": "e3sse.protocol.contracts.require_primary_softening_selection",
        },
        "unit_contract": units,
        "canonical_analysis_units": canonical_analysis_units(),
        "unit_conversion_registry": {
            f"{source}->{target}": factor
            for (source, target), factor in sorted(UNIT_CONVERSIONS.items())
        },
        "decision_register": {
            "schema_version": DECISION_REGISTER_SCHEMA,
            "decisions": validate_decision_register(DECISION_REGISTER),
        },
        "model_groups": config["protocol"]["model_groups"],
        "future_mlip_inference_contract": {
            "record_schema_version": "e3sse.mlip.inference.record.v1",
            "run_schema_version": "e3sse.mlip.inference.run.v1",
            "record_required_fields": list(INFERENCE_REQUIRED_FIELDS),
            "run_required_fields": list(INFERENCE_RUN_REQUIRED_FIELDS),
            "model_provenance_required_fields": list(MODEL_PROVENANCE_REQUIRED_FIELDS),
            "study_1_published_split": "train",
            "prediction_semantics": {
                "energy_kind": "total_configuration_energy",
                "force_sign_convention": "force_equals_negative_energy_gradient",
                "energy_unit": "eV",
                "force_unit": "eV/angstrom",
            },
            "reference_compatibility_policy": (
                "only explicit compatible assessments are primary-analysis eligible"
            ),
            "referential_integrity_policy": (
                "cross-check every result against frozen frame provenance, repeat assignments, "
                "model identity, and protocol_id"
            ),
            "failed_attempt_policy": "retain every attempt with null predictions and failure metadata",
            "idempotency": "record_key = SHA-256(protocol_id, frame_id, weights_sha256, "
            "config_hash); at most one success per key and run",
            "energy_policy": "uncorrected model-native total energies; per-atom outputs are "
            "reconciled as raw_energy * n_atoms",
            "pilot_gate_before_production": [
                "finite-difference force sign check on a few training frames",
                "positive force-parity slope on a small training batch",
                "requested equals effective device/dtype or an explicit fallback_reason",
            ],
        },
        "study_1_contract": {
            "name": "Softening Atlas",
            "published_split": "train",
            "estimator": "zero_intercept_force_parity_projection",
            "model_axis": True,
            "pretraining_group_axis": {
                "A": "MPtrj (MPF for M3GNet)",
                "B1": "MatPES-type",
                "B2": "OMat24-type",
                "purpose": "RQ3 ordered medians and Spearman trend; architecture/dataset "
                "confounding reported as a limitation",
            },
            "inputs": "DFT and ML Cartesian force components only; DFT energies enter only "
            "through the registered configurational-energy decile rule",
            "s_host_definition": "unresolved: s_host_definition",
            "minimum_support": "unresolved: estimator_minimum_support",
            "softening_resolution_levels": ["global", "family", "material", "element"],
            "secondary_groupings": ["configurational_energy_decile"],
            "element_resolved_component_rule": (
                "for target element Z, Eq_17 numerator and denominator use only Cartesian "
                "force components of atoms whose atomic symbol is Z"
            ),
            "anion_families": ["oxide", "sulfide", "halide", "nitride", "other"],
            "anion_family_assignment": config["protocol"]["scientific_resolutions"][
                "mixed_anion_family"
            ],
            "primary_population": "partition_definition.primary_population",
            "leakage_rule": "leakage_rules applied per target hop to be corrected",
            "label_budget_k": [1, 3, 5, 10, 30],
            "label_budget_subsampling": {
                "status": "contract_only_not_executed",
                "sampling_unit": "whole_frame",
                "population": "leakage-admissible frames of the estimator scope and target",
                "draw": "K frames uniformly at random without replacement per repeat",
                "insufficient_population": "record and skip; never sample with replacement",
                "seed_key": [
                    "base_seed",
                    "study1_label_budget",
                    "model_id",
                    "estimator_scope",
                    "scope_target",
                    "target_hop_id",
                    "K",
                    "repeat_index",
                ],
                "repeat_count_and_stability": "label_budget_stability",
                "propagation": "Var(s_K) propagated to barrier error in Study 2 (RQ4)",
            },
            "configurational_energy_deciles": config["protocol"]["scientific_resolutions"][
                "configurational_energy_deciles"
            ],
            "label_budget_stability": config["protocol"]["scientific_resolutions"][
                "label_budget_stability"
            ],
            "repeat_scope": _register_resolution(config, "study1_repeat_population_scope"),
            "bootstrap_unit": "whole_frame",
            "minimum_bootstrap_resamples": 2000,
            "bootstrap_resamples": 10000,
            "production_inference_executed": False,
        },
        "study_2a_contract": {
            "source": "Research Design 5.7, 3.5 Eq. 16",
            "analysis": "fixed_geometry_single_points_on_DFT_NEB_images",
            "not_ml_neb": True,
            "raw_barrier": "max_ML_image_energy_minus_min_ML_image_energy",
            "srt_barrier": "raw_barrier_divided_by_leakage_free_s_hat",
            "comparators": [
                "raw_ML",
                "SRT_global_s",
                "SRT_family_s",
                "SRT_material_s",
                "SRT_oracle_s_reported_separately",
                "linear_recalibration_aEb_plus_b_fitted_on_training_partition_barriers",
                "optional_group_C_fine_tuned_model",
            ],
            "metrics": [
                "MAE_eV",
                "RMSE_eV",
                "signed_mean_error_eV",
                "Spearman_rho",
                "precision_recall_F1_at_c",
                "rho_SRT_Eq_16",
            ],
            "thresholds_eV": [0.3, 0.5],
            "noise_floor": "spread of energies of symmetry-equivalent images where available, "
            "reported alongside all MAEs (Research Design 5.2)",
            "paired_identical_units": True,
        },
        "study_4_contract": {
            "source": "Research Design 5.9, 3.8",
            "predictors": ["raw", "SRT_global", "SRT_family"],
            "uncontrolled_comparator": "naive rule: select if predicted Y < c",
            "draws_per_partition": 200,
            "partitions": "the ten repeated chemical-system partitions",
            "alpha": [0.1, 0.2],
            "threshold_eV": [0.3, 0.5],
            "recorded": ["false_discovery_proportion", "true_discoveries", "selection_size"],
            "realised_fdr": "mean FDP with Monte Carlo standard error (definition unresolved)",
            "predictor_calibration_independence": "unresolved",
        },
        "study_5_contract": {
            "source": "Research Design 5.10 Table 8",
            "status": "contract_only_candidate_pool_not_inspected",
            "selection": "weighted conformal selection",
            "alpha": 0.2,
            "threshold_eV": 0.5,
            "gate_0": "BVEL13k structures excluding all evaluation chemical systems, "
            "transition-metal-free, same element and hop-length criteria as nebDFT2k",
            "exclusion_artifact": "study5_exclusions",
        },
        "inference_roles": {
            "classification": "paper_specified (Research Design 3.8, 4, 5.4)",
            "primary_partition": "published LiTraj split: primary partition for Studies 1-2",
            "study4_partitions": "Study 4 runs on the ten repeated partitions (5.9)",
            "repeated_partitions": "ten chemical-system partitions quantify split variability",
            "global_methods_and_thresholds": (
                "established only from permitted training/development data before any "
                "held-out outcome is inspected; bound by the pilot amendment"
            ),
            "repeat_specific_fits": (
                "softening estimates, recalibration and conformal objects for a repeat use "
                "only that repeat's training/calibration side; thresholds are fixed once in "
                "the pilot (repeated_test_pilot_overlap_rule) and predictor construction "
                "follows study4_predictor_calibration_independence"
            ),
            "repeat_test_side_outcomes": "never used to fit any object of the same repeat",
            "repeat_fit_guard": "e3sse.protocol.contracts.require_repeat_training_side",
            "repeat_fitted_objects": list(REPEAT_FITTED_OBJECTS),
            "study4_predictor": (
                "fixed without calibration or test outcomes (Research Design 3.8); how pooled "
                "s satisfies this is study4_predictor_calibration_independence"
            ),
            "published_validation_split": "unresolved: published_validation_split_role",
        },
        "hypotheses": {
            "source": "Research Design 4 Table 4",
            "RQ1": {
                "hypothesis": "SRT with leakage-free material-level s reduces fixed-geometry "
                "barrier MAE, and the reduction increases with 1-s across models",
                "estimand": "rho_SRT (Eq. 16); slope of rho_SRT on 1-s_bar",
                "support": "rho_SRT >= 0.30 with 95% CI > 0 for models with median s <= 0.9, "
                "and positive dose-response",
                "refute": "rho_SRT < 0.10",
            },
            "RQ2": {
                "hypothesis": "SRT reduces error in log10 D and Ea relative to FPMD",
                "estimand": "paired differences in absolute error",
                "support": "both CIs exclude 0",
            },
            "RQ3": {
                "hypothesis": "s_bar is ordered MPtrj < MatPES-type < OMat24-type, and SRT "
                "gains follow the reverse order",
                "estimand": "ordered medians; Spearman trend",
                "refute": "gains are equal or reversed",
            },
            "RQ4": {
                "hypothesis": "s_K stabilises within +/-0.03 for K <= 10",
                "estimand": "SD(s_K); barrier MAE vs K",
                "report": "plateau K",
            },
            "RQ5": {
                "hypothesis": "H5a realised FDR <= alpha; H5b SRT increases power",
                "estimand": "mean FDP; mean true discoveries",
                "decision": "H5a is a validity check; H5b supported if paired CI > 0",
            },
        },
        "statistical_plan": {
            "randomness": {
                "classification": "engineering_implementation_choice",
                "base_seed": config["protocol"]["partition_base_seed"],
                "derivation": "e3sse.analysis.seed.v1; SHA256 compact JSON array; first 8 bytes big-endian",
                "bootstrap_seed_key": [
                    "base_seed",
                    "cluster_bootstrap",
                    "study",
                    "analysis_id",
                    "partition_id",
                ],
                "bootstrap": {
                    "resamples": 10000,
                    "cluster_order": "ascending UTF-8 identifier order",
                    "sampling": "sample n clusters with replacement for each resample",
                    "generator": "numpy.random.Generator(numpy.random.PCG64DXSM(seed))",
                    "interval": "percentile_2.5_97.5",
                },
                "study4_draw_seeds": {
                    str(repeat): [
                        analysis_seed(
                            config["protocol"]["partition_base_seed"],
                            "study4_hop_draw",
                            repeat,
                            draw,
                        )
                        for draw in range(200)
                    ]
                    for repeat in range(10)
                },
                "study4_hop_selection": (
                    "within each chemical system, sort eligible hop IDs by "
                    "SHA256('e3sse.study4.hop-rank.v1' NUL seed NUL hop_id), then select first"
                ),
                "canonical_hop_order": "ascending UTF-8 hop identifier before SHA256 ranking",
                "tie_policy_classification": "paper_permitted_deterministic_option_U_equals_1",
            },
            "confidence_level": 0.95,
            "interval": "cluster_bootstrap_percentile",
            "paper_minimum_resamples": 2000,
            "frozen_resamples": 10000,
            "study_2_4_cluster": "chemical_system",
            "study_3_cluster": "material",
            "study_4_draws_per_partition": 200,
            "study_4_alpha": [0.1, 0.2],
            "study_4_threshold_eV": [0.3, 0.5],
            "study_4_hop_sampling": "one_hop_per_chemical_system",
            "conformal_tie_policy": "deterministic_conservative_U_equals_1",
            "sensitivity": ["Benjamini-Yekutieli", "threshold_c", "alpha"],
            "primary_endpoints": {
                "P1": "Stage_2a_rho_SRT_for_Group_A_models",
                "P2": "Study_3_paired_reduction_in_absolute_delta_log10_D",
                "P3": "Study_4_realized_FDR",
            },
            "holm_family": config["protocol"]["scientific_resolutions"]["holm_family"],
            "P3_role": "validity_not_superiority",
            "study_2_sample": "all_eligible_fixed_before_outcome_analysis",
            "mde_pilot": "training_only",
            "study_3": "paired_sign_test_and_effect_size_with_material_cluster_uncertainty",
            "realized_fdr": "mean_FDP_over_200_draws_with_Monte_Carlo_standard_error",
            # Jin-Candes validity needs a predictor fixed independently of calibration and
            # test outcomes; pooled s fitted on the calibration side is not yet registered
            # as satisfying this (unresolved: study4_predictor_calibration_independence).
            "fixed_predictor_before_calibration": "unresolved",
            "paired_identical_units": True,
            "primary_selection": "Benjamini_Hochberg_with_monotone_conformal_score",
            "exchangeability_assumption": "jointly_exchangeable_calibration_and_test_chemical_system_units",
            "weighted_conformal": {
                "study5": "Jin_Candes_weighted_selection_modifying_BH_not_BH_on_weighted_pvalues",
                "support_requirement": "prospective_covariate_support_within_calibration_support",
                "density_ratio": "probabilistic_classifier_on_structural_descriptors",
                "unweighted_diagnostic": "classifier_AUC_near_0.5",
            },
        },
        "study_5_evaluation_scope": _register_resolution(
            config, "study5_evaluation_system_scope"
        ),
        "decision_gates": config["protocol"]["decision_gates"],
        "evaluation_seal": {
            "state": "sealed",
            "denied": [
                "held_out_test_outcome_analysis",
                "test_model_performance_summary",
                "test_derived_tuning",
                "fpmd_transport_reference_performance",
            ],
        },
        "g0_status": dataset_manifest["g0_status"],
        "static_branch_status": dataset_manifest["static_branch_status"],
        "transport_branch_status": dataset_manifest["transport_branch_status"],
        "transport_branch_note": (
            "status at materialization only; Study 3 unseals through a separate hashed "
            "transport addendum, so this protocol_id is unaffected"
        ),
        "unresolved_items": config["protocol"]["unresolved_items"],
        "known_deviations": config["protocol"]["known_deviations"],
    }


def materialize_protocol(
    config_path: str | Path, *, resume: bool = False, git_sha: str | None = None
) -> tuple[dict[str, Any], Path]:
    """Validate inputs and atomically materialize the canonical Phase B artifacts.

    A commit cannot contain its own SHA, so the reviewed commit is pinned with
    ``git_sha`` (CLI ``--git-sha``) when the tracked config says ``CURRENT``; it must
    equal HEAD of a clean worktree and is recorded in ``resolved_config.json``.
    """

    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    if git_sha is not None:
        if not isinstance(git_sha, str) or len(git_sha) != 40 or set(git_sha) - set(
            "0123456789abcdef"
        ):
            raise ProtocolFreezeError("--git-sha must be a lowercase 40-character commit SHA")
        pinned = config.get("inputs", {}).get("repository_git_sha")
        if pinned not in {None, "CURRENT", git_sha}:
            raise ProtocolFreezeError("--git-sha contradicts the configured repository_git_sha")
        config["inputs"]["repository_git_sha"] = git_sha
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ProtocolFreezeError("incompatible protocol-freeze config schema")
    try:
        if config.get("materialization_mode") == "production":
            configuration_state = assess_production_configuration(config)
        else:
            configuration_state = validate_protocol_configuration(config, allow_synthetic_test=True)
            configuration_state = ProtocolConfigurationValidation(
                configuration_state.materialization_mode,
                production_eligible=False,
                freeze_blockers=freeze_blockers(config),
            )
    except ProtocolConfigurationError as exc:
        raise ProtocolFreezeError(str(exc)) from exc
    production_mode = configuration_state.materialization_mode == "production"
    base = config_path.parent
    paths = config["paths"]
    repository = _resolve(base, paths["repository_root"])
    expected_git = config["inputs"].get("repository_git_sha")
    current_sha, git_branch, git_dirty = _validate_repository_state(
        repository,
        expected_sha=expected_git,
        require_clean=production_mode,
    )

    source_documents: dict[str, str] = {}
    for name, record in sorted(config["source_documents"].items()):
        path = _resolve(base, record["path"])
        source_documents[name] = _expect_hash(path, record.get("sha256"), f"source document {name}")

    g0_spec = config["inputs"]["g0_audit"]
    g0_path = _resolve(base, g0_spec["path"])
    audit, audit_sha = _validate_g0(g0_path, g0_spec)
    g0_inputs_spec = config["inputs"]["g0_input_manifest"]
    g0_inputs, g0_inputs_sha = _validate_g0_input_manifest(
        _resolve(base, g0_inputs_spec["path"]), g0_inputs_spec, audit
    )
    server_spec = config["inputs"]["server_config"]
    server_path = _resolve(base, server_spec["path"])
    server_sha = _expect_hash(server_path, server_spec.get("sha256"), "server config")
    if audit.get("configuration", {}).get("sha256") != server_sha:
        raise ProtocolFreezeError("G0 was not produced from the supplied server config")
    server_config = _load_json(server_path)

    provenance_spec = config["inputs"]["mplitrj_provenance"]
    provenance_root = _resolve(base, provenance_spec["root"])
    provenance, provenance_sha, provenance_records = _validate_provenance(
        provenance_root, provenance_spec, audit
    )

    linkage_spec = config["inputs"].get("material_chemical_system_linkage")
    if linkage_spec:
        systems, linkage_sha = _material_systems_from_index(
            _resolve(base, linkage_spec["path"]),
            linkage_spec.get("sha256"),
            provenance,
            provenance_sha,
        )
    else:
        systems, linkage_sha = _derive_material_systems(
            server_config, provenance, provenance_records
        )
    provenance_materials = {record["material_id"] for record in provenance_records}
    if set(systems) != provenance_materials:
        raise ProtocolFreezeError("material/chemical-system linkage coverage is not exact")
    linked = _link_records(provenance_records, systems, provenance)

    neb_dataset = next(item for item in audit["datasets"] if item["dataset"] == "nebDFT2k")
    neb_linkage_spec = config["inputs"].get("nebdft2k_partition_linkage")
    if neb_linkage_spec:
        neb_linkage_path = _resolve(base, neb_linkage_spec["path"])
        neb_linkage_sha = _expect_hash(
            neb_linkage_path, neb_linkage_spec.get("sha256"), "nebDFT2k partition linkage"
        )
        neb_linkage_payload = _load_json(neb_linkage_path)
    else:
        try:
            neb_linkage_payload = derive_nebdft2k_linkage(
                g0_inputs["inputs"]["nebDFT2k"],
                data_root=Path(server_config["paths"]["data_root"]),
                g0_run_id=audit["run_id"],
                g0_input_identity_digest=neb_dataset["inputs"]["identity_digest"],
            )
        except EvaluationLinkageError as exc:
            raise ProtocolFreezeError(str(exc)) from exc
        neb_linkage_sha = canonical_sha256(neb_linkage_payload)
    try:
        neb_records = validate_nebdft2k_linkage(
            neb_linkage_payload,
            g0_run_id=audit["run_id"],
            g0_input_identity_digest=neb_dataset["inputs"]["identity_digest"],
        )
        partition_records, partition_universe = combine_partition_universe(linked, neb_records)
    except EvaluationLinkageError as exc:
        raise ProtocolFreezeError(str(exc)) from exc

    published = published_split_summary(partition_records)
    published["test_chemical_systems"] = sorted(
        {row["chemical_system_id"] for row in partition_records if row["published_split"] == "test"}
    )
    repeated = build_repeated_partitions(
        partition_records, base_seed=int(config["protocol"]["partition_base_seed"])
    )
    universe_systems = sorted({row["chemical_system_id"] for row in partition_records})
    tm_resolution = config["protocol"]["scientific_resolutions"]["primary_tm_hubbard_u"]
    if tm_resolution.get("status") == "resolved":
        population = classify_primary_population(
            universe_systems, tm_resolution["excluded_elements"]
        )
        primary_population = {
            "status": "resolved",
            "predicate_id": tm_resolution["predicate_id"],
            "excluded_elements": sorted(tm_resolution["excluded_elements"]),
            "assignment_by_chemical_system": population,
            "counts": dict(sorted(Counter(population.values()).items())),
        }
    else:
        primary_population = {
            "status": "unresolved",
            "reason": "TM/Hubbard-U element set is not registered; no system is classified",
        }
    partition_manifest = {
        "schema_version": "e3sse.partition.manifest.v1",
        "linkage_sha256": linkage_sha,
        "nebdft2k_linkage_sha256": neb_linkage_sha,
        "partition_universe": partition_universe,
        "published_split": published,
        "repeated_partitions": repeated,
        "primary_population": primary_population,
        "nebdft2k_source_columns": neb_linkage_payload.get("source_columns"),
        "linkage_validation": {
            "frames": len(linked),
            "materials": len({row["material_id"] for row in linked}),
            "hops": len({row["hop_id"] for row in linked}),
            "chemical_systems": len({row["chemical_system_id"] for row in linked}),
            "nebdft2k_hops": len(neb_records),
            "all_records_group_assignable": True,
            "zero_repeat_overlap": all(
                repeat["overlap_count"] == 0 for repeat in repeated["partitions"]
            ),
        },
    }
    partition_manifest["manifest_id"] = canonical_sha256(partition_manifest)
    exclusions = derive_study5_exclusions(published["test_chemical_systems"], repeated)
    exclusions["source_partition_manifest_id"] = partition_manifest["manifest_id"]
    exclusions["included_partitions"] = [
        "published_test",
        *[f"repeat_{index}_test" for index in range(10)],
    ]
    exclusions["scope"] = _register_resolution(config, "study5_evaluation_system_scope")
    exclusions["manifest_id"] = canonical_sha256(exclusions)
    dataset_manifest = _dataset_manifest(
        audit, audit_sha, g0_inputs, g0_inputs_sha, provenance, provenance_sha
    )
    if dataset_manifest["static_branch_status"] != "ready":
        raise ProtocolFreezeError("static branch is not ready")

    protocol = _protocol_content(
        git_sha=current_sha,
        source_documents=source_documents,
        dataset_manifest=dataset_manifest,
        partition_manifest=partition_manifest,
        exclusions=exclusions,
        units=config["protocol"]["unit_contract"],
        config=config,
    )
    blockers = list(configuration_state.freeze_blockers)
    production_eligible = production_mode and not blockers
    protocol["materialization_mode"] = configuration_state.materialization_mode
    protocol["production_eligible"] = production_eligible
    protocol["freeze_blockers"] = sorted(set(blockers))
    if not production_mode:
        protocol["freeze_status"] = "synthetic_fixture_only"
    else:
        protocol["freeze_status"] = "frozen" if production_eligible else "candidate_unfrozen"
    protocol_id = canonical_sha256(protocol)
    protocol["protocol_id"] = protocol_id
    config_sha = file_sha256(config_path)
    implementation_paths = {
        path.relative_to(repository).as_posix(): path
        for path in sorted((repository / "src" / "e3sse").rglob("*.py"))
    }
    implementation_paths["scripts/build_protocol_freeze.py"] = (
        repository / "scripts" / "build_protocol_freeze.py"
    )
    code_hashes = {name: file_sha256(path) for name, path in implementation_paths.items()}
    input_hashes = {
        "config": config_sha,
        "g0_audit": audit_sha,
        "g0_input_manifest": g0_inputs_sha,
        "mplitrj_provenance_manifest": provenance_sha,
        "nebdft2k_partition_linkage": neb_linkage_sha,
        "server_config": server_sha,
        "material_chemical_system_linkage": linkage_sha,
        **{f"source_document:{name}": digest for name, digest in source_documents.items()},
    }
    fingerprint = canonical_sha256(
        {
            "run_schema": RUN_SCHEMA,
            "protocol_schema": PROTOCOL_SCHEMA,
            "git_sha": current_sha,
            "code_sha256": code_hashes,
            "input_hashes": input_hashes,
        }
    )
    output_root = _resolve(base, paths["output_root"])
    data_root = Path(server_config["paths"]["data_root"]).resolve()
    configured_outputs = Path(server_config["paths"]["outputs_root"]).resolve()
    if output_root == configured_outputs or not output_root.is_relative_to(configured_outputs):
        raise ProtocolFreezeError(
            "protocol output_root must be a dedicated directory under outputs_root"
        )
    try:
        output_root.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ProtocolFreezeError("protocol outputs cannot be written below the data root")
    run_root = output_root / protocol_id
    run_manifest_path = run_root / "run_manifest.json"
    artifacts = {
        "resolved_config.json": config,
        "g0_input_manifest.json": g0_inputs,
        "nebdft2k_partition_linkage.json": neb_linkage_payload,
        "partition_membership.json": {
            "schema_version": "e3sse.partition-membership.v1",
            "records": partition_records,
            "records_sha256": canonical_sha256(partition_records),
        },
        "dataset_manifest.json": dataset_manifest,
        "partition_manifest.json": partition_manifest,
        "study5_exclusions.json": exclusions,
        "protocol.json": protocol,
    }
    expected_artifact_hashes = {
        name: hashlib.sha256(
            (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
        ).hexdigest()
        for name, payload in artifacts.items()
    }
    if resume and run_manifest_path.is_file():
        run_manifest = _load_json(run_manifest_path)
        if run_manifest.get("fingerprint") != fingerprint:
            raise ProtocolFreezeError("stale protocol-freeze checkpoint")
        recorded_hashes = run_manifest.get("artifact_sha256")
        if not isinstance(recorded_hashes, Mapping) or set(recorded_hashes) != set(artifacts):
            raise ProtocolFreezeError("resumed run has an incomplete artifact inventory")
        if recorded_hashes != expected_artifact_hashes:
            raise ProtocolFreezeError("resumed artifact hashes differ from reconstructed content")
        for name, digest in recorded_hashes.items():
            artifact_path = run_root / name
            if not artifact_path.is_file() or file_sha256(artifact_path) != digest:
                raise ProtocolFreezeError(f"modified resumed artifact: {name}")
        return protocol, run_root / "protocol.json"
    checkpoint_path = run_root / "checkpoint.json"
    checkpoint = {
        "schema_version": RUN_SCHEMA,
        "fingerprint": fingerprint,
        "protocol_id": protocol_id,
        "artifact_sha256": expected_artifact_hashes,
    }
    # An empty run_root (crash between mkdir and the checkpoint link) is treated as absent.
    if run_root.exists() and any(run_root.iterdir()):
        if not resume:
            raise ProtocolFreezeError(
                "existing protocol artifacts require --resume; refusing overwrite"
            )
        if not checkpoint_path.is_file() or _load_json(checkpoint_path) != checkpoint:
            raise ProtocolFreezeError("stale or missing interrupted-run checkpoint")
    else:
        _atomic_json(checkpoint_path, checkpoint)
    for name, payload in artifacts.items():
        artifact_path = run_root / name
        if artifact_path.exists():
            if file_sha256(artifact_path) != expected_artifact_hashes[name]:
                raise ProtocolFreezeError(f"modified interrupted artifact: {name}")
        else:
            _atomic_json(artifact_path, payload)
    run_manifest = {
        "schema_version": RUN_SCHEMA,
        "fingerprint": fingerprint,
        "protocol_id": protocol_id,
        "git_sha": current_sha,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "code_sha256": code_hashes,
        "input_hashes": input_hashes,
        "partition_base_seed": config["protocol"]["partition_base_seed"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "dependency_versions": {
            name: importlib.metadata.version(name) for name in ("ase", "numpy", "scipy")
        },
        "platform": platform.platform(),
        "command": shlex.join(
            [
                sys.executable,
                "scripts/build_protocol_freeze.py",
                "--config",
                str(config_path),
                *(["--resume"] if resume else []),
                *(["--git-sha", git_sha] if git_sha else []),
            ]
        ),
        "artifact_sha256": {name: file_sha256(run_root / name) for name in sorted(artifacts)},
    }
    _atomic_json(run_manifest_path, run_manifest)
    return protocol, run_root / "protocol.json"
