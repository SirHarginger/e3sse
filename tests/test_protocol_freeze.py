from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from protocol_fixtures import (
    canonical_units,
    complete_inference_record,
)

from e3sse.data.mplitrj_provenance import material_bucket
from e3sse.protocol.contracts import (
    OutcomeAccessDenied,
    require_outcome_access,
    validate_inference_record,
    validate_inference_run_manifest,
    validate_study1_inference_record,
    validate_unit_contract,
)
from e3sse.protocol.freeze import (
    ProtocolFreezeError,
    _validate_repository_state,
    canonical_sha256,
    materialize_protocol,
)
from e3sse.protocol.partitions import (
    build_repeated_partitions,
    derive_study5_exclusions,
    normalize_chemical_system,
    published_split_summary,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _units(*, verified: bool = True) -> dict:
    return canonical_units(verified=verified)


def _records() -> list[dict]:
    systems = ["Li-O", "Li-S", "Cl-Li", "Li-N", "F-Li"]
    splits = ["train", "train", "val", "test", "test"]
    split_indices = {"train": 0, "val": 0, "test": 0}
    records = []
    for index, (system, split) in enumerate(zip(systems, splits, strict=True)):
        frame_index = split_indices[split]
        split_indices[split] += 1
        records.append(
            {
                "frame_id": f"MPLiTrj_{split}.xyz:{frame_index}",
                "material_id": f"mp-{index + 1}",
                "hop_id": f"mp-{index + 1}_0_1_0_0_0",
                "chemical_system_id": system,
                "published_split": split,
            }
        )
    return records


def test_chemical_system_normalization_and_distinction():
    assert normalize_chemical_system("O-Li") == "Li-O"
    assert normalize_chemical_system("Li2O") == "Li-O"
    assert normalize_chemical_system(["O", "Li", "O"]) == "Li-O"
    assert normalize_chemical_system("Li-S") != normalize_chemical_system("Li-O")
    with pytest.raises(ValueError):
        normalize_chemical_system("not chemistry")


def test_partitions_are_deterministic_grouped_and_exactly_ten():
    records = _records()
    first = build_repeated_partitions(records, base_seed=2026)
    second = build_repeated_partitions(reversed(records), base_seed=2026)
    assert first == second
    assert first["repeats"] == len(first["partitions"]) == 10
    for repeat in first["partitions"]:
        train = set(repeat["training_or_calibration_chemical_systems"])
        test = set(repeat["test_chemical_systems"])
        assert len(train) == 3
        assert len(test) == 2
        assert not train & test
        assert repeat["overlap_count"] == 0
        assignment = repeat["assignment_by_chemical_system"]
        for row in records:
            assert assignment[normalize_chemical_system(row["chemical_system_id"])] in {
                "training_or_calibration",
                "test",
            }


def test_published_split_and_study5_exclusions_preserve_source_labels():
    rows = _records()
    published = published_split_summary(rows)
    assert published["record_counts"] == {"train": 2, "val": 1, "test": 2}
    published["test_chemical_systems"] = sorted(
        row["chemical_system_id"] for row in rows if row["published_split"] == "test"
    )
    repeated = build_repeated_partitions(rows, base_seed=7)
    exclusions = derive_study5_exclusions(published["test_chemical_systems"], repeated)
    assert set(exclusions["chemical_system_ids"]) == set(published["test_chemical_systems"]).union(
        *(set(partition["test_chemical_systems"]) for partition in repeated["partitions"])
    )


def test_partition_rounding_and_unit_identity_fail_closed():
    rows = _records()[:3]
    repeated = build_repeated_partitions(rows, base_seed=11)
    for repeat in repeated["partitions"]:
        assert len(repeat["training_or_calibration_chemical_systems"]) == 2
        assert len(repeat["test_chemical_systems"]) == 1
    conflicting = [dict(rows[0]), {**rows[1], "material_id": rows[0]["material_id"]}]
    with pytest.raises(ValueError, match="span chemical systems"):
        build_repeated_partitions(conflicting, base_seed=11)


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_small_group_universes_keep_calibration_at_least_as_large_as_test(size):
    for repeat in build_repeated_partitions(_records()[:size], base_seed=3)["partitions"]:
        assert len(repeat["training_or_calibration_chemical_systems"]) >= len(
            repeat["test_chemical_systems"]
        )
        assert sum(repeat["achieved_fractions"].values()) == pytest.approx(1)


def test_units_outcome_seal_and_future_inference_schema():
    validate_unit_contract(_units())
    with pytest.raises(ValueError, match="unresolved"):
        validate_unit_contract(_units(verified=False))
    wrong = _units()
    wrong["dft_force"]["canonical_analysis_unit"] = "K"
    with pytest.raises(ValueError, match="dft_force"):
        validate_unit_contract(wrong)
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(role="test", outcome="barrier_mae")
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(role="unknown", outcome="barrier_mae")
    # Training outcomes follow registration too (Research Design 4 and 8).
    with pytest.raises(OutcomeAccessDenied, match="sealed before protocol freeze"):
        require_outcome_access(role="train", outcome="pilot_summary")
    record = {
        "schema_version": "e3sse.mlip.inference.record.v1",
        "frame_id": "train.xyz:0",
        "material_id": "mp-1",
        "chemical_system_id": "Li-O",
        "hop_id": None,
        "published_split": "train",
        "repeat_partition_membership": {
            str(index): "training_or_calibration" for index in range(10)
        },
        "provenance_index_id": "index-1",
        "source_file_sha256": "d" * 64,
        "source_frame_index": 0,
        "model_provenance": {
            "canonical_name": "synthetic-model",
            "group": "A",
            "architecture_family": "synthetic",
            "pretraining_dataset": "synthetic-only",
            "upstream": "local-test-fixture",
            "release_revision": "v1",
            "checkpoint_id": "toy-v1",
            "weights_sha256": "e" * 64,
            "package_name": "synthetic",
            "package_version": "0.0.0",
            "adapter_name": "synthetic-adapter",
            "adapter_version": "1",
            "load_options": {},
            "license_identifier": "test-only",
            "license_verified": True,
            "fine_tuned": False,
        },
        "reference_compatibility": {
            "target_functional": "PBE",
            "target_hubbard_u_policy": "no_hubbard_u",
            "target_spin_policy": "non_spin_polarized",
            "target_evidence": "synthetic fixture",
            "model_training_settings": "same synthetic settings",
            "status": "compatible",
            "evidence": "synthetic fixture",
            "primary_analysis_eligible": True,
            "exclusion_reason": None,
        },
        "n_atoms": 1,
        "energy_kind": "total_configuration_energy",
        "force_sign_convention": "force_equals_negative_energy_gradient",
        "energy": -1.0,
        "energy_unit": "eV",
        "forces": [[0.0, 0.0, 0.0]],
        "force_unit": "eV/angstrom",
        "requested_device": "cpu",
        "effective_device": "cpu",
        "requested_dtype": "float64",
        "effective_dtype": "float64",
        "run_id": "run",
        "protocol_id": "protocol",
        "git_sha": "a" * 40,
        "config_hash": "b" * 64,
        "input_hash": "c" * 64,
        "status": "success",
    }
    record = complete_inference_record(record)
    validate_inference_record(record)
    record.pop("force_unit")
    with pytest.raises(ValueError, match="missing"):
        validate_inference_record(record)


def test_inference_contract_rejects_semantic_and_failed_record_errors():
    _, record = _valid_inference_record()
    record["forces"] = [[True, 0.0, 0.0]]
    with pytest.raises(ValueError, match="finite shape"):
        validate_inference_record(record)
    _, record = _valid_inference_record()
    record["published_split"] = "test"
    with pytest.raises(ValueError, match="training split"):
        validate_inference_record(record)
    _, record = _valid_inference_record()
    record["reference_compatibility"]["status"] = "unknown"
    record["reference_compatibility"]["primary_analysis_eligible"] = False
    with pytest.raises(ValueError, match="not eligible"):
        validate_inference_record(record)
    _, record = _valid_inference_record()
    record.update({"status": "failed", "energy": None, "raw_energy": None, "forces": None})
    with pytest.raises(ValueError, match="failure metadata"):
        validate_inference_record(record)


def _valid_inference_record() -> tuple[dict, dict]:
    """Reuse the fully explicit fixture assembled by the primary contract test."""

    membership = {str(index): "training_or_calibration" for index in range(10)}
    record = {
        "schema_version": "e3sse.mlip.inference.record.v1",
        "frame_id": "train.xyz:0",
        "material_id": "mp-1",
        "chemical_system_id": "Li-O",
        "hop_id": None,
        "published_split": "train",
        "repeat_partition_membership": membership,
        "provenance_index_id": "index-1",
        "source_file_sha256": "d" * 64,
        "source_frame_index": 0,
        "model_provenance": {
            "canonical_name": "synthetic-model",
            "group": "A",
            "architecture_family": "synthetic",
            "pretraining_dataset": "synthetic-only",
            "upstream": "local-test-fixture",
            "release_revision": "v1",
            "checkpoint_id": "toy-v1",
            "weights_sha256": "e" * 64,
            "package_name": "synthetic",
            "package_version": "0.0.0",
            "adapter_name": "synthetic-adapter",
            "adapter_version": "1",
            "load_options": {},
            "license_identifier": "test-only",
            "license_verified": True,
            "fine_tuned": False,
        },
        "reference_compatibility": {
            "target_functional": "PBE",
            "target_hubbard_u_policy": "no_hubbard_u",
            "target_spin_policy": "non_spin_polarized",
            "target_evidence": "synthetic fixture",
            "model_training_settings": "same synthetic settings",
            "status": "compatible",
            "evidence": "synthetic fixture",
            "primary_analysis_eligible": True,
            "exclusion_reason": None,
        },
        "n_atoms": 1,
        "energy_kind": "total_configuration_energy",
        "force_sign_convention": "force_equals_negative_energy_gradient",
        "energy": -1.0,
        "energy_unit": "eV",
        "forces": [[0.0, 0.0, 0.0]],
        "force_unit": "eV/angstrom",
        "requested_device": "cpu",
        "effective_device": "cpu",
        "requested_dtype": "float64",
        "effective_dtype": "float64",
        "run_id": "run",
        "protocol_id": "protocol",
        "git_sha": "a" * 40,
        "config_hash": "b" * 64,
        "input_hash": "c" * 64,
        "status": "success",
    }
    return membership, complete_inference_record(record)


def test_inference_run_manifest_contract():
    _, inference = _valid_inference_record()
    manifest = {
        "schema_version": "e3sse.mlip.inference.run.v1",
        "run_id": "run",
        "protocol_id": "protocol",
        "input_manifest_sha256": "a" * 64,
        "inference_config_sha256": "b" * 64,
        "git_sha": "c" * 40,
        "git_branch": "feature",
        "git_dirty": False,
        "started_at_utc": "2026-01-01T00:00:00Z",
        "finished_at_utc": "2026-01-01T00:01:00Z",
        "command": "synthetic",
        "python_version": "3",
        "dependency_versions": {},
        "environment_lock_sha256": "d" * 64,
        "random_seeds": {},
        "hardware": {},
        "determinism": {},
        "model_artifacts": [inference["model_provenance"]],
        "output_sha256": "e" * 64,
        "record_counts_by_status": {"success": 1, "failed": 0},
    }
    validate_inference_run_manifest(manifest)
    manifest["output_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_inference_run_manifest(manifest)


def test_inference_record_cross_checks_frozen_scientific_linkage():
    membership, record = _valid_inference_record()
    frozen_fields = (
        "frame_id",
        "material_id",
        "chemical_system_id",
        "hop_id",
        "published_split",
        "provenance_index_id",
        "source_file_sha256",
        "source_frame_index",
        "n_atoms",
        "atomic_numbers_sha256",
        "analysis_population",
        "input_hash",
    )
    frozen_frame = {field: record[field] for field in frozen_fields}
    validate_study1_inference_record(
        record,
        frozen_frame=frozen_frame,
        frozen_repeat_membership=membership,
        frozen_model=record["model_provenance"],
        protocol_id="protocol",
    )
    record["material_id"] = "mp-wrong"
    with pytest.raises(ValueError, match="frozen frame"):
        validate_study1_inference_record(
            record,
            frozen_frame=frozen_frame,
            frozen_repeat_membership=membership,
            frozen_model=record["model_provenance"],
            protocol_id="protocol",
        )


def _phase_b_fixture(tmp_path: Path) -> tuple[Path, Path]:
    server_config = tmp_path / "server.json"
    _write(
        server_config,
        {
            "paths": {
                "data_root": str(tmp_path / "data"),
                "outputs_root": str(tmp_path / "outputs"),
            }
        },
    )
    records = _records()
    provenance_records = [
        {
            "flattened_file": row["frame_id"].split(":")[0],
            "flattened_frame_index": int(row["frame_id"].split(":")[1]),
            "derived_frame_id": row["frame_id"],
            "published_split": row["published_split"],
            "material_id": row["material_id"],
            "edge_id": row["hop_id"],
            "raw_file_kind": "band_optim_step",
            "mapping_status": "mapped_unique",
        }
        for row in records
    ]
    provenance_root = tmp_path / "provenance" / "index-1"
    # Producer layout: one shard per (published split, material bucket), with the
    # split and bucket recorded in the shard body (mplitrj_provenance.write_shards).
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in provenance_records:
        key = (record["published_split"], material_bucket(record["material_id"]))
        grouped.setdefault(key, []).append(record)
    shard_descriptors = []
    for (split, bucket), rows in sorted(grouped.items()):
        relative = f"{split}/material-bucket-{bucket}.json"
        _write(
            provenance_root / relative,
            {
                "schema_version": "e3sse.mplitrj.provenance.v1",
                "index_id": "index-1",
                "split": split,
                "material_bucket": bucket,
                "records": rows,
            },
        )
        shard_descriptors.append(
            {
                "path": relative,
                "sha256": _sha(provenance_root / relative),
                "record_count": len(rows),
            }
        )
    edges = provenance_root / "edges.json"
    _write(
        edges,
        {
            "index_id": "index-1",
            "edge_frame_counts": {row["hop_id"]: 1 for row in records},
        },
    )
    counts = {
        "record_count": 5,
        "mapped_count": 5,
        "mapped_unique_count": 5,
        "mapped_hop_unique_count": 0,
        "ambiguous_count": 0,
        "unmapped_count": 0,
        "error_count": 0,
        "coverage_fraction": 1.0,
    }
    split_sizes = {"train": 2, "val": 1, "test": 2}
    counts_by_split = {
        split: {
            "record_count": size,
            "mapped_count": size,
            "mapped_unique_count": size,
            "mapped_hop_unique_count": 0,
            "ambiguous_count": 0,
            "unmapped_count": 0,
            "error_count": 0,
            "coverage_fraction": 1.0,
        }
        for split, size in split_sizes.items()
    }
    sources = {}
    for row in records:
        name, index = row["frame_id"].split(":")
        sources.setdefault(name, {"published_split": row["published_split"], "frames": 0})
        sources[name]["frames"] = max(sources[name]["frames"], int(index) + 1)
    manifest = {
        "schema_version": "e3sse.mplitrj.provenance.v1",
        "status": "complete",
        "index_id": "index-1",
        "complete_coverage": True,
        "coverage_blockers": [],
        "counts": counts,
        "counts_by_split": counts_by_split,
        "flattened_sources": [
            {
                "identifier": name,
                "published_split": source["published_split"],
                "frames": source["frames"],
                "sha256": "unused-with-explicit-linkage",
            }
            for name, source in sorted(sources.items())
        ],
        "edges": {"path": "edges.json", "sha256": _sha(edges), "count": 5},
        "shards": shard_descriptors,
    }
    manifest_path = provenance_root / "manifest.json"
    _write(manifest_path, manifest)

    linkage = tmp_path / "linkage.json"
    _write(
        linkage,
        {
            "schema_version": "e3sse.material-chemical-system.v1",
            "provenance_index_id": "index-1",
            "provenance_manifest_sha256": _sha(manifest_path),
            "chemical_system_algorithm": "sorted-unique-elements-v1",
            "flattened_source_sha256": {
                name: "unused-with-explicit-linkage" for name in sorted(sources)
            },
            "materials": [
                {"material_id": row["material_id"], "chemical_system_id": row["chemical_system_id"]}
                for row in records
            ],
        },
    )
    datasets = []
    input_inventory = {
        name: [{"identifier": name, "sha256": "a" * 64}]
        for name in ("nebDFT2k", "MPLiTrj", "FPMD", "BVEL13k", "nebBVSE122k")
    }
    for name in ("nebDFT2k", "MPLiTrj", "FPMD", "BVEL13k", "nebBVSE122k"):
        status = "partial" if name == "FPMD" else "pass"
        blockers = (
            ["diffusion_reference_recoverable", "msd_reference_recoverable"]
            if name == "FPMD"
            else []
        )
        datasets.append(
            {
                "dataset": name,
                "role": f"synthetic {name}",
                "status": status,
                "inputs": {
                    "examples": [name],
                    "identity_digest": canonical_sha256(input_inventory[name]),
                },
                "record_counts": {"files": 1, "frames": 5 if name == "MPLiTrj" else 0},
                "split_summary": {"counts": {"train": 2, "val": 1, "test": 2}},
                "identifiers": {},
                "units": {"verified": False, "observed_unit_metadata": {}},
                "blockers": blockers,
                "capabilities": {"same_hop_exclusion_supported": {"available": name == "MPLiTrj"}},
                "errors": [],
            }
        )
    audit = {
        "schema_version": "e3sse.g0.audit.v2",
        "run_id": "g0-toy",
        "status": "partial",
        "configuration": {"sha256": _sha(server_config)},
        "mplitrj_provenance_index": {
            "status": "valid",
            "index_id": "index-1",
            "manifest_sha256": _sha(manifest_path),
        },
        "datasets": datasets,
    }
    audit_path = tmp_path / "g0" / "audit.json"
    _write(audit_path, audit)
    g0_inputs = tmp_path / "g0" / "inputs.json"
    _write(
        g0_inputs,
        {
            "run_id": "g0-toy",
            "inputs": input_inventory,
            "mplitrj_raw_archive": None,
        },
    )
    neb_linkage = tmp_path / "nebdft2k_partition_linkage.json"
    neb_records = [
        {
            "edge_id": row["hop_id"],
            "material_id": row["material_id"],
            "chemical_system_id": row["chemical_system_id"],
            "published_split": row["published_split"],
            "eligibility_status": "included",
            "exclusion_reason": None,
            "source_identifier": f"nebDFT2k/{row['hop_id']}",
            "source_sha256": "f" * 64,
        }
        for row in records
    ]
    normalized_neb_records = [
        {
            "record_kind": "nebdft2k_hop",
            "frame_id": None,
            "hop_id": row["edge_id"],
            "material_id": row["material_id"],
            "chemical_system_id": normalize_chemical_system(row["chemical_system_id"]),
            "published_split": row["published_split"],
            "eligibility_status": row["eligibility_status"],
            "exclusion_reason": row["exclusion_reason"],
            "source_identifier": row["source_identifier"],
            "source_sha256": row["source_sha256"],
        }
        for row in neb_records
    ]
    _write(
        neb_linkage,
        {
            "schema_version": "e3sse.nebdft2k.partition-linkage.v1",
            "contains_outcomes": False,
            "source_g0_run_id": "g0-toy",
            "source_input_identity_digest": canonical_sha256(input_inventory["nebDFT2k"]),
            "chemical_system_algorithm": "sorted-unique-elements-v1",
            "record_count": len(neb_records),
            "records_sha256": canonical_sha256(normalized_neb_records),
            "records": neb_records,
        },
    )
    docs = {}
    for name in ("guide", "design", "narrative"):
        path = tmp_path / f"{name}.docx"
        path.write_bytes(name.encode())
        docs[name] = {"path": str(path), "sha256": _sha(path)}
    config = {
        "schema_version": "e3sse.protocol.freeze.config.v1",
        "materialization_mode": "synthetic_test",
        "paths": {
            "repository_root": str(Path.cwd()),
            "output_root": str(tmp_path / "outputs" / "freeze"),
        },
        "inputs": {
            "repository_git_sha": "CURRENT",
            "g0_audit": {"path": str(audit_path), "run_id": "g0-toy", "sha256": _sha(audit_path)},
            "g0_input_manifest": {"path": str(g0_inputs), "sha256": _sha(g0_inputs)},
            "server_config": {"path": str(server_config), "sha256": _sha(server_config)},
            "mplitrj_provenance": {
                "root": str(provenance_root),
                "index_id": "index-1",
                "manifest_sha256": _sha(manifest_path),
            },
            "nebdft2k_partition_linkage": {
                "path": str(neb_linkage),
                "sha256": _sha(neb_linkage),
            },
            "material_chemical_system_linkage": {"path": str(linkage), "sha256": _sha(linkage)},
        },
        "source_documents": docs,
        "protocol": {
            "partition_base_seed": 2026,
            "unit_contract": _units(),
            "model_groups": {"A": [], "B1": [], "B2": [], "C": []},
            "decision_gates": {
                name: {
                    "paper_pass_condition": f"synthetic {name}",
                    "paper_contingency": "synthetic",
                    "source": "synthetic fixture",
                }
                for name in ("G0", "G1", "G2", "G3", "G4")
            },
            "unresolved_items": [],
            "known_deviations": ["synthetic fixture only"],
            "scientific_resolutions": {
                "primary_tm_hubbard_u": {
                    "status": "resolved",
                    "predicate_id": "toy",
                    "excluded_elements": ["N"],
                    "evidence": "synthetic",
                },
                "mixed_anion_family": {
                    "status": "resolved",
                    "precedence": ["toy"],
                    "evidence": "synthetic",
                },
                "configurational_energy_deciles": {
                    "status": "resolved",
                    "energy_reference": "toy",
                    "fit_population": "toy",
                    "binning_rule": "toy",
                    "evidence": "synthetic",
                },
                "label_budget_stability": {
                    "status": "resolved",
                    "repeat_count": 1,
                    "stability_metric": "toy",
                    "selection_rule": "toy",
                    "evidence": "synthetic",
                },
                "holm_family": {"status": "resolved", "family": "toy", "evidence": "synthetic"},
            },
        },
    }
    config_path = tmp_path / "protocol.json"
    _write(config_path, config)
    return config_path, audit_path


def test_end_to_end_materialization_resume_hashing_and_raw_immutability(tmp_path):
    config, _audit = _phase_b_fixture(tmp_path)
    inputs = [path for path in tmp_path.rglob("*") if path.is_file()]
    before = {path: _sha(path) for path in inputs}
    protocol, path = materialize_protocol(config)
    first_mtime = path.stat().st_mtime_ns
    resumed, resumed_path = materialize_protocol(config, resume=True)
    assert resumed == protocol
    assert resumed_path == path
    assert path.stat().st_mtime_ns == first_mtime
    assert protocol["protocol_id"] == canonical_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_id"}
    )
    assert protocol["g0_status"] == "partial"
    assert protocol["static_branch_status"] == "ready"
    assert protocol["transport_branch_status"] == "partial"
    assert before == {path: _sha(path) for path in inputs}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd() / "src")
    cli = subprocess.run(
        [
            sys.executable,
            "scripts/build_protocol_freeze.py",
            "--config",
            str(config),
            "--resume",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert cli.returncode == 0, cli.stderr
    assert f"Protocol ID: {protocol['protocol_id']}" in cli.stdout


def test_missing_invalid_inputs_stale_resume_and_blocking_unresolved(tmp_path):
    config_path, audit = _phase_b_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["inputs"]["g0_audit"]["sha256"] = "0" * 64
    _write(config_path, config)
    with pytest.raises(ProtocolFreezeError, match="G0 audit SHA"):
        materialize_protocol(config_path)
    config["inputs"]["g0_audit"]["sha256"] = _sha(audit)
    config["protocol"]["unresolved_items"] = [{"id": "scientific-choice", "blocking": True}]
    _write(config_path, config)
    with pytest.raises(ProtocolFreezeError, match="unknown protocol register item"):
        materialize_protocol(config_path)
    config["protocol"]["unresolved_items"] = [
        {
            "id": "s_bar_definition",
            "blocking": True,
            "resolution_category": "requires_preanalysis_scientific_decision",
        }
    ]
    _write(config_path, config)
    protocol, path = materialize_protocol(config_path)
    assert protocol["freeze_status"] == "synthetic_fixture_only"
    assert "blocking protocol item unresolved: s_bar_definition" in protocol["freeze_blockers"]
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(role="test", outcome="barrier_mae", frozen_protocol=protocol)
    with pytest.raises(ProtocolFreezeError, match="require --resume"):
        materialize_protocol(config_path)
    run_manifest = path.parent / "run_manifest.json"
    payload = json.loads(run_manifest.read_text())
    payload["fingerprint"] = "stale"
    _write(run_manifest, payload)
    with pytest.raises(ProtocolFreezeError, match="stale"):
        materialize_protocol(config_path, resume=True)


def test_resume_completes_interrupted_write_without_rewriting_verified_artifacts(
    tmp_path, monkeypatch
):
    from e3sse.protocol import freeze

    config, _ = _phase_b_fixture(tmp_path)
    original_write = freeze._atomic_json

    def interrupted_write(path, value):
        if path.name == "partition_manifest.json":
            raise OSError("synthetic interruption")
        original_write(path, value)

    monkeypatch.setattr(freeze, "_atomic_json", interrupted_write)
    with pytest.raises(OSError, match="synthetic interruption"):
        materialize_protocol(config)
    existing = list((tmp_path / "outputs").rglob("dataset_manifest.json"))
    assert len(existing) == 1
    timestamp = existing[0].stat().st_mtime_ns
    monkeypatch.setattr(freeze, "_atomic_json", original_write)
    protocol, path = materialize_protocol(config, resume=True)
    assert path.is_file()
    assert protocol["production_eligible"] is False
    assert existing[0].stat().st_mtime_ns == timestamp


def test_repository_safety_hook_blocks_raw_deletion():
    hook = Path(".agents/hooks/safety_guard.py")
    blocked = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "rm data/raw/x"}}),
        text=True,
        capture_output=True,
        check=False,
    )
    allowed = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": "python scripts/build_protocol_freeze.py --config config.json --resume"
                },
            }
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked.returncode == 2
    assert "research data are immutable" in blocked.stderr
    assert allowed.returncode == 0


def test_production_repository_state_requires_exact_clean_commit(monkeypatch, tmp_path):
    responses = {
        ("rev-parse", "HEAD"): "a" * 40,
        ("branch", "--show-current"): "feature/protocol-freeze",
        ("status", "--porcelain"): " M scientific.py",
    }
    monkeypatch.setattr(
        "e3sse.protocol.freeze._git", lambda _root, *args: responses[args]
    )
    with pytest.raises(ProtocolFreezeError, match="clean worktree"):
        _validate_repository_state(tmp_path, expected_sha="a" * 40, require_clean=True)
    sha, branch, dirty = _validate_repository_state(
        tmp_path, expected_sha="a" * 40, require_clean=False
    )
    assert (sha, branch, dirty) == ("a" * 40, "feature/protocol-freeze", True)
