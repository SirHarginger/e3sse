from __future__ import annotations

import hashlib
import json

import pytest
from protocol_fixtures import (
    canonical_units,
    complete_inference_record,
)

from e3sse.protocol.contracts import (
    OutcomeAccessDenied,
    build_pilot_amendment,
    require_outcome_access,
    validate_inference_record,
    validate_inference_run_manifest,
    validate_model_provenance,
)


def _model() -> dict:
    return {
        "canonical_name": "fixture-model",
        "group": "A",
        "architecture_family": "fixture",
        "pretraining_dataset": "fixture-data",
        "upstream": "fixture-registry",
        "release_revision": "v1.0.0",
        "checkpoint_id": "fixture-v1",
        "weights_sha256": "a" * 64,
        "package_name": "fixture-package",
        "package_version": "1.0.0",
        "adapter_name": "fixture-adapter",
        "adapter_version": "1.0.0",
        "load_options": {},
        "license_identifier": "MIT",
        "license_verified": True,
        "fine_tuned": False,
        "reference_compatibility": {
            "target_functional": "PBE",
            "target_hubbard_u_policy": "no_hubbard_u",
            "target_spin_policy": "non_spin_polarized",
            "target_evidence": "fixture protocol",
            "model_training_settings": "compatible fixture settings",
            "status": "compatible",
            "evidence": "fixture assessment",
            "primary_analysis_eligible": True,
            "exclusion_reason": None,
        },
    }


def _record() -> dict:
    return complete_inference_record({
        "schema_version": "e3sse.mlip.inference.record.v1",
        "frame_id": "MPLiTrj_train.xyz:0",
        "material_id": "mp-1",
        "chemical_system_id": "Li-O",
        "hop_id": "mp-1_0_1_0_0_0",
        "published_split": "train",
        "repeat_partition_membership": {
            "0": "training_or_calibration",
            "1": "test",
        },
        "provenance_index_id": "fixture-index",
        "source_file_sha256": "b" * 64,
        "source_frame_index": 0,
        "model_provenance": _model(),
        "reference_compatibility": {
            "target_functional": "PBE",
            "target_hubbard_u_policy": "no_hubbard_u",
            "target_spin_policy": "non_spin_polarized",
            "target_evidence": "fixture protocol",
            "model_training_settings": "compatible fixture settings",
            "status": "compatible",
            "evidence": "fixture assessment",
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
        "run_id": "fixture-run",
        "protocol_id": "fixture-protocol",
        "git_sha": "c" * 40,
        "config_hash": "d" * 64,
        "input_hash": "e" * 64,
        "status": "success",
    })


def _run_manifest(record: dict) -> dict:
    return {
        "schema_version": "e3sse.mlip.inference.run.v1",
        "run_id": record["run_id"],
        "protocol_id": record["protocol_id"],
        "input_manifest_sha256": "f" * 64,
        "inference_config_sha256": record["config_hash"],
        "git_sha": record["git_sha"],
        "git_branch": "feature/protocol-freeze",
        "git_dirty": False,
        "started_at_utc": "2026-10-12T08:00:00Z",
        "finished_at_utc": "2026-10-12T08:01:00Z",
        "command": "python -m fixture",
        "python_version": "3.12.0",
        "dependency_versions": {"fixture-package": "1.0.0"},
        "environment_lock_sha256": "1" * 64,
        "random_seeds": {"inference": 2026},
        "hardware": {"device": "cpu"},
        "determinism": {"enabled": True},
        "model_artifacts": [record["model_provenance"]],
        "output_sha256": "2" * 64,
        "record_counts_by_status": {"success": 1, "failed": 0},
    }


_MEMBERSHIP = [
    {"frame_id": None, "hop_id": "hop-train", "published_split": "train"},
    {"frame_id": None, "hop_id": "hop-test", "published_split": "test"},
]


def _frozen_protocol() -> dict:
    units = canonical_units()
    protocol = {
        "protocol_schema_version": "e3sse.protocol.freeze.v1",
        "git_sha": "3" * 40,
        "source_document_hashes": {"research_design": "4" * 64},
        "dataset_manifest": {"manifest_schema_version": "fixture"},
        "partition_definition": {
            "schema_version": "fixture",
            "partition_universe": {
                "membership_sha256": hashlib.sha256(
                    json.dumps(_MEMBERSHIP, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            },
        },
        "leakage_rules": {"uncertain_provenance": "fail_closed"},
        "unit_contract": units,
        "study_1_contract": {"published_split": "train"},
        "evaluation_seal": {"state": "sealed"},
        "static_branch_status": "ready",
        "materialization_mode": "production",
        "production_eligible": True,
        "freeze_status": "frozen",
        "freeze_blockers": [],
        "transport_branch_status": "partial",
        "unresolved_items": [],
    }
    payload = json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    protocol["protocol_id"] = hashlib.sha256(payload).hexdigest()
    return protocol


def test_model_provenance_validator_is_shared_and_fail_closed():
    model = _model()
    validate_model_provenance(model)
    model["license_verified"] = False
    with pytest.raises(ValueError, match="license"):
        validate_model_provenance(model)
    record = _record()
    record["model_provenance"] = model
    with pytest.raises(ValueError, match="license"):
        validate_inference_record(record)
    manifest = _run_manifest(_record())
    manifest["model_artifacts"] = [model]
    with pytest.raises(ValueError, match="license"):
        validate_inference_run_manifest(manifest)


def test_primary_and_repeat_analysis_contexts_are_distinct():
    record = _record()
    validate_inference_record(record, analysis_context="primary")
    record["published_split"] = "test"
    with pytest.raises(ValueError, match="published training split"):
        validate_inference_record(record, analysis_context="primary")
    validate_inference_record(record, analysis_context="repeat", repeat_id="0")
    with pytest.raises(ValueError, match="calibration side"):
        validate_inference_record(record, analysis_context="repeat", repeat_id="1")
    with pytest.raises(ValueError, match="requires a frozen repeat_id"):
        validate_inference_record(record, analysis_context="repeat")


def test_run_manifest_checks_time_order_counts_and_record_identity():
    record = _record()
    manifest = _run_manifest(record)
    validate_inference_run_manifest(manifest, records=[record])
    manifest["finished_at_utc"] = "2026-10-12T07:59:59Z"
    with pytest.raises(ValueError, match="cannot precede"):
        validate_inference_run_manifest(manifest, records=[record])
    manifest = _run_manifest(record)
    manifest["record_counts_by_status"] = {"success": 0, "failed": 1}
    with pytest.raises(ValueError, match="differ from supplied records"):
        validate_inference_run_manifest(manifest, records=[record])
    manifest = _run_manifest(record)
    record["run_id"] = "other-run"
    with pytest.raises(ValueError, match="run_id"):
        validate_inference_run_manifest(manifest, records=[record])


def test_nontraining_outcome_access_requires_a_complete_validated_freeze():
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(
            role="test",
            outcome="barrier_mae",
            frozen_protocol={"protocol_id": "not-a-freeze"},
        )
    protocol = _frozen_protocol()
    with pytest.raises(OutcomeAccessDenied, match="pilot amendment"):
        require_outcome_access(role="test", outcome="barrier_mae", frozen_protocol=protocol)
    amendment = build_pilot_amendment(
        protocol,
        partition_records=_MEMBERSHIP,
        training_unit_ids=["hop-train"],
        frozen_thresholds={"c_eV": [0.3, 0.5]},
        mde_analysis={"status": "reported"},
    )
    with pytest.raises(ValueError, match="published training split"):
        build_pilot_amendment(
            protocol,
            partition_records=_MEMBERSHIP,
            training_unit_ids=["hop-test"],
            frozen_thresholds={"c_eV": [0.3]},
            mde_analysis={"status": "reported"},
        )
    require_outcome_access(
        role="test", outcome="barrier_mae", frozen_protocol=protocol, pilot_amendment=amendment
    )
    forged = {**amendment, "data_roles": ["train", "test"]}
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(
            role="test", outcome="barrier_mae", frozen_protocol=protocol, pilot_amendment=forged
        )
    protocol["static_branch_status"] = "blocked"
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(
            role="test", outcome="barrier_mae", frozen_protocol=protocol, pilot_amendment=amendment
        )
    synthetic = _frozen_protocol()
    synthetic["materialization_mode"] = "synthetic_test"
    synthetic["production_eligible"] = False
    synthetic["freeze_status"] = "synthetic_fixture_only"
    synthetic_without_id = {key: value for key, value in synthetic.items() if key != "protocol_id"}
    synthetic["protocol_id"] = hashlib.sha256(
        json.dumps(
            synthetic_without_id,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(role="test", outcome="barrier_mae", frozen_protocol=synthetic)


