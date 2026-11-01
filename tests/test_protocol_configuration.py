from __future__ import annotations

import copy

import pytest
from protocol_fixtures import canonical_units, production_evidence, resolved_register

from e3sse.protocol.chemistry import (
    ANION_FAMILY_RULE_ID,
    TM_RULE_ID,
    primary_exclusion_elements,
)
from e3sse.protocol.configuration import (
    ProtocolConfigurationError,
    validate_protocol_configuration,
    validate_protocol_template,
)


def _units() -> dict:
    return canonical_units()


def _compatibility() -> dict:
    return {
        "target_functional": "PBE",
        "target_hubbard_u_policy": "no_hubbard_u",
        "target_spin_policy": "non_spin_polarized",
        "target_evidence": "source documentation",
        "model_training_settings": "documented training settings",
        "status": "compatible",
        "evidence": "compatibility assessment",
        "primary_analysis_eligible": True,
        "exclusion_reason": None,
    }


def _model(group: str, index: int = 0) -> dict:
    return {
        "canonical_name": f"model-{group}-{index}",
        "group": group,
        "frozen_group": group,
        "resolution_status": "resolved",
        "architecture_family": "test architecture",
        "pretraining_dataset": "test dataset",
        "upstream": "https://example.invalid/model",
        "release_revision": "v1",
        "checkpoint_id": f"checkpoint-{group}-{index}",
        "weights_sha256": "a" * 64,
        "package_name": "test-package",
        "package_version": "1.0",
        "adapter_name": "test-adapter",
        "adapter_version": "1",
        "load_options": {},
        "license_identifier": "test-license",
        "license_verified": True,
        "fine_tuned": False,
        "reference_compatibility": _compatibility(),
    }


def _resolved_items() -> list[dict]:
    return resolved_register()


def _production_config() -> dict:
    digest = "b" * 64
    return {
        "schema_version": "e3sse.protocol.freeze.config.v1",
        "materialization_mode": "production",
        "paths": {"repository_root": "..", "output_root": "/safe/outputs"},
        "inputs": {
            "repository_git_sha": "c" * 40,
            "g0_audit": {"path": "/input/audit.json", "sha256": digest},
            "g0_input_manifest": {"path": "/input/inputs.json", "sha256": digest},
            "server_config": {"path": "/input/server.json", "sha256": digest},
            "mplitrj_provenance": {"root": "/input/index", "manifest_sha256": digest},
            "nebdft2k_partition_linkage": {
                "path": "/input/nebdft2k-linkage.json",
                "sha256": digest,
            },
            "material_chemical_system_linkage": None,
            "production_evidence": production_evidence(
                [_model("A", index) for index in range(4)] + [_model("B1"), _model("B2")]
            ),
        },
        "source_documents": {"design": {"path": "design.docx", "sha256": digest}},
        "protocol": {
            "partition_base_seed": 2026,
            "unit_contract": _units(),
            "model_groups": {
                "A": [_model("A", index) for index in range(4)],
                "B1": [_model("B1")],
                "B2": [_model("B2")],
                "C": [],
            },
            "scientific_resolutions": {
                "primary_tm_hubbard_u": {
                    "status": "resolved",
                    "predicate_id": TM_RULE_ID,
                    "excluded_elements": primary_exclusion_elements(),
                    "evidence": "registration",
                },
                "mixed_anion_family": {
                    "status": "resolved",
                    "rule_id": ANION_FAMILY_RULE_ID,
                    "evidence": "registration",
                },
                "configurational_energy_deciles": {
                    "status": "resolved",
                    "energy_reference": "per-material minimum",
                    "fit_population": "published training split",
                    "binning_rule": "registered quantile rule",
                    "evidence": "registration",
                },
                "label_budget_stability": {
                    "status": "resolved",
                    "repeat_count": 20,
                    "stability_metric": "registered interval width",
                    "selection_rule": "registered threshold",
                    "evidence": "registration",
                },
                "holm_family": {
                    "status": "resolved",
                    "family": "registered hypothesis family",
                    "evidence": "registration",
                },
            },
            "unresolved_items": _resolved_items(),
            "decision_gates": {
                name: {
                    "paper_pass_condition": f"registered {name}",
                    "paper_contingency": "registered contingency",
                    "source": "Research Design 6 Table 9",
                }
                for name in ("G0", "G1", "G2", "G3", "G4")
            },
            "known_deviations": [],
        },
    }


def test_complete_production_configuration_is_eligible():
    result = validate_protocol_configuration(_production_config())
    assert result.materialization_mode == "production"
    assert result.production_eligible is True


def test_removing_unresolved_checklist_does_not_resolve_scientific_field():
    config = _production_config()
    config["protocol"]["scientific_resolutions"]["primary_tm_hubbard_u"].update(
        {"status": "unresolved", "predicate_id": None, "predicate": None, "evidence": None}
    )
    config["protocol"]["unresolved_items"] = []
    with pytest.raises(ProtocolConfigurationError, match="not frozen"):
        validate_protocol_configuration(config)


def test_production_requires_pinned_git_and_artifact_hashes():
    config = _production_config()
    config["inputs"]["repository_git_sha"] = "CURRENT"
    with pytest.raises(ProtocolConfigurationError, match="repository_git_sha"):
        validate_protocol_configuration(config)
    config = _production_config()
    config["inputs"]["g0_audit"]["sha256"] = None
    with pytest.raises(ProtocolConfigurationError, match="g0_audit.sha256"):
        validate_protocol_configuration(config)


def test_wrong_dimensional_unit_is_rejected():
    config = _production_config()
    config["protocol"]["unit_contract"]["dft_force"]["canonical_analysis_unit"] = "K"
    with pytest.raises(ProtocolConfigurationError, match="dft_force"):
        validate_protocol_configuration(config)


def test_model_provenance_and_compatibility_are_enforced():
    config = _production_config()
    config["protocol"]["model_groups"]["B1"][0]["weights_sha256"] = "not-a-hash"
    with pytest.raises(ProtocolConfigurationError, match="weights_sha256"):
        validate_protocol_configuration(config)
    config = _production_config()
    config["protocol"]["model_groups"]["B2"][0]["reference_compatibility"][
        "primary_analysis_eligible"
    ] = False
    with pytest.raises(ProtocolConfigurationError, match="not eligible for primary analysis"):
        validate_protocol_configuration(config)


def test_anion_and_tm_resolutions_must_register_the_frozen_rules():
    config = _production_config()
    config["protocol"]["scientific_resolutions"]["mixed_anion_family"]["rule_id"] = "custom"
    with pytest.raises(ProtocolConfigurationError, match="frozen rule dominant-anion"):
        validate_protocol_configuration(config)
    config = _production_config()
    config["protocol"]["scientific_resolutions"]["primary_tm_hubbard_u"]["excluded_elements"] = [
        "Fe"
    ]
    with pytest.raises(ProtocolConfigurationError, match="frozen rule pymatgen"):
        validate_protocol_configuration(config)

def test_synthetic_mode_is_explicitly_nonproduction():
    config = _production_config()
    config["materialization_mode"] = "synthetic_test"
    config["inputs"]["repository_git_sha"] = "CURRENT"
    config["protocol"]["model_groups"] = {"A": [], "B1": [], "B2": [], "C": []}
    with pytest.raises(ProtocolConfigurationError, match="not production"):
        validate_protocol_configuration(config)
    result = validate_protocol_configuration(config, allow_synthetic_test=True)
    assert result.production_eligible is False


def test_checked_in_unresolved_configuration_is_a_valid_template():
    import json
    from pathlib import Path

    config = json.loads(Path("configs/protocol_freeze.json").read_text(encoding="utf-8"))
    result = validate_protocol_template(config)
    assert result.production_eligible is False
    with pytest.raises(ProtocolConfigurationError):
        validate_protocol_configuration(config)


def test_template_validation_does_not_mutate_input():
    config = _production_config()
    before = copy.deepcopy(config)
    validate_protocol_template(config)
    assert config == before


def test_deleting_or_unblocking_a_required_item_does_not_clear_it():
    config = _production_config()
    config["protocol"]["unresolved_items"] = [
        item for item in config["protocol"]["unresolved_items"] if item["id"] != "s_bar_definition"
    ]
    with pytest.raises(ProtocolConfigurationError, match="missing from configuration: s_bar"):
        validate_protocol_configuration(config)
    config = _production_config()
    for item in config["protocol"]["unresolved_items"]:
        if item["id"] == "s_host_definition":
            item.pop("resolution")
    with pytest.raises(ProtocolConfigurationError, match="unresolved: s_host_definition"):
        validate_protocol_configuration(config)
