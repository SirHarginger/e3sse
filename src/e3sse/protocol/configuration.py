"""Fail-closed validation for protocol-freeze configuration.

Template validation deliberately differs from materialization validation.  A
checked-in template may document unresolved fields, but those fields can never
be made materialization-ready by deleting a checklist entry or selecting the
synthetic-test mode.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ase.data import atomic_numbers

from .chemistry import (
    ANION_FAMILY_RULE_ID,
    TM_RULE_ID,
    primary_exclusion_elements,
)
from .contracts import (
    CANONICAL_UNITS,
    MODEL_PROVENANCE_REQUIRED_FIELDS,
    REQUIRED_UNITS,
    validate_model_provenance,
    validate_unit_contract,
)
from .decisions import required_item_blockers, validate_unresolved_items

CONFIG_SCHEMA = "e3sse.protocol.freeze.config.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_REQUIRED_GROUPS = ("A", "B1", "B2", "C")
_REQUIRED_GATES = ("G0", "G1", "G2", "G3", "G4")
_GATE_FIELDS = ("paper_pass_condition", "paper_contingency", "source")
_ANION_FAMILIES = {"oxide", "sulfide", "halide", "nitride", "other"}
_COMPATIBILITY_FIELDS = (
    "target_functional",
    "target_hubbard_u_policy",
    "target_spin_policy",
    "target_evidence",
    "model_training_settings",
    "status",
    "evidence",
    "primary_analysis_eligible",
    "exclusion_reason",
)
_RESOLUTION_FIELDS: dict[str, tuple[str, ...]] = {
    "primary_tm_hubbard_u": ("predicate_id", "excluded_elements"),
    "mixed_anion_family": ("rule_id",),
    "configurational_energy_deciles": (
        "energy_reference",
        "fit_population",
        "binning_rule",
    ),
    "label_budget_stability": ("repeat_count", "stability_metric", "selection_rule"),
    "holm_family": ("family",),
}


class ProtocolConfigurationError(ValueError):
    """Raised when a template or materialization configuration is invalid."""


@dataclass(frozen=True)
class ProtocolConfigurationValidation:
    """Result whose eligibility flag cannot be confused with a production freeze."""

    materialization_mode: str
    production_eligible: bool
    freeze_blockers: tuple[str, ...] = ()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolConfigurationError(f"{label} must be an object")
    return value


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_nonempty(record: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    missing = [field for field in fields if not _nonempty(record.get(field))]
    if missing:
        raise ProtocolConfigurationError(f"{label} has missing/empty fields: {missing}")


def _require_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProtocolConfigurationError(f"{label} must be a pinned lowercase SHA-256")


def _validate_shape(config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ProtocolConfigurationError("unsupported protocol-freeze configuration schema")
    mode = config.get("materialization_mode")
    if mode not in {"production", "synthetic_test"}:
        raise ProtocolConfigurationError(
            "materialization_mode must be 'production' or 'synthetic_test'"
        )
    paths = _mapping(config.get("paths"), "paths")
    _require_nonempty(paths, ("repository_root", "output_root"), "paths")
    inputs = _mapping(config.get("inputs"), "inputs")
    protocol = _mapping(config.get("protocol"), "protocol")
    if not isinstance(protocol.get("partition_base_seed"), int) or isinstance(
        protocol.get("partition_base_seed"), bool
    ):
        raise ProtocolConfigurationError("partition_base_seed must be an integer")
    return mode, inputs, protocol


def _validate_template_units(protocol: Mapping[str, Any]) -> None:
    units = _mapping(protocol.get("unit_contract"), "protocol.unit_contract")
    missing = [name for name in REQUIRED_UNITS if name not in units]
    if missing:
        raise ProtocolConfigurationError(f"unit contract missing: {missing}")
    for name in REQUIRED_UNITS:
        record = _mapping(units[name], f"unit_contract.{name}")
        if record.get("canonical_analysis_unit") != CANONICAL_UNITS[name]:
            raise ProtocolConfigurationError(
                f"unit_contract.{name} canonical analysis unit differs from the frozen contract"
            )
        evidence = _mapping(
            record.get("production_source_unit_evidence"),
            f"unit_contract.{name}.production_source_unit_evidence",
        )
        if evidence.get("status") not in {
            "unresolved",
            "verified",
            "not_retained",
            "deferred_study3",
        }:
            raise ProtocolConfigurationError(f"unit_contract.{name} has invalid status")


def _validate_template_resolutions(protocol: Mapping[str, Any]) -> None:
    resolutions = _mapping(protocol.get("scientific_resolutions"), "scientific_resolutions")
    missing = sorted(set(_RESOLUTION_FIELDS) - set(resolutions))
    if missing:
        raise ProtocolConfigurationError(f"scientific resolution records missing: {missing}")
    for name, fields in _RESOLUTION_FIELDS.items():
        record = _mapping(resolutions[name], f"scientific_resolutions.{name}")
        if record.get("status") not in {"resolved", "unresolved"}:
            raise ProtocolConfigurationError(f"scientific resolution has invalid status: {name}")
        for field in ("status", "evidence", *fields):
            if field not in record:
                raise ProtocolConfigurationError(
                    f"scientific resolution field is absent: {name}.{field}"
                )


def _validate_template_models(protocol: Mapping[str, Any]) -> None:
    groups = _mapping(protocol.get("model_groups"), "protocol.model_groups")
    if set(groups) != set(_REQUIRED_GROUPS):
        raise ProtocolConfigurationError("model_groups must contain exactly A, B1, B2, and C")
    for group in _REQUIRED_GROUPS:
        if not isinstance(groups[group], list):
            raise ProtocolConfigurationError(f"model group {group} must be a list")
        for index, model in enumerate(groups[group]):
            _mapping(model, f"model_groups.{group}[{index}]")


def _validate_template_gates(protocol: Mapping[str, Any]) -> None:
    gates = _mapping(protocol.get("decision_gates"), "decision_gates")
    if set(gates) != set(_REQUIRED_GATES):
        raise ProtocolConfigurationError("decision_gates must define exactly G0 through G4")
    for name in _REQUIRED_GATES:
        gate = _mapping(gates[name], f"decision_gates.{name}")
        _require_nonempty(gate, _GATE_FIELDS, f"decision_gates.{name}")


def validate_protocol_template(config: Mapping[str, Any]) -> ProtocolConfigurationValidation:
    """Validate the checked-in shape while permitting explicit unresolved values."""

    mode, _inputs, protocol = _validate_shape(config)
    _validate_template_units(protocol)
    _validate_template_resolutions(protocol)
    _validate_template_models(protocol)
    _validate_template_gates(protocol)
    try:
        validate_unresolved_items(protocol.get("unresolved_items"))
    except (TypeError, ValueError) as exc:
        raise ProtocolConfigurationError(str(exc)) from exc
    deviations = protocol.get("known_deviations")
    if not isinstance(deviations, list) or any(not _nonempty(item) for item in deviations):
        raise ProtocolConfigurationError("known_deviations must be a list of non-empty strings")
    return ProtocolConfigurationValidation(mode, production_eligible=False)


def _validate_trust_anchors(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    """Report every missing pin at once; any one keeps the protocol unfrozen."""

    errors: list[str] = []

    def collect(check: Callable[[], None]) -> None:
        try:
            check()
        except ProtocolConfigurationError as exc:
            errors.append(str(exc))

    def git_pin() -> None:
        git_sha = inputs.get("repository_git_sha")
        if not isinstance(git_sha, str) or _GIT_SHA.fullmatch(git_sha) is None:
            raise ProtocolConfigurationError("production repository_git_sha must be pinned")

    collect(git_pin)
    specs = {
        "g0_audit": "sha256",
        "g0_input_manifest": "sha256",
        "server_config": "sha256",
        "mplitrj_provenance": "manifest_sha256",
    }
    for name, digest_field in specs.items():

        def pinned(name: str = name, digest_field: str = digest_field) -> None:
            spec = _mapping(inputs.get(name), f"inputs.{name}")
            location = "root" if name == "mplitrj_provenance" else "path"
            _require_nonempty(spec, (location,), f"inputs.{name}")
            _require_sha(spec.get(digest_field), f"inputs.{name}.{digest_field}")

        collect(pinned)
    for name in ("material_chemical_system_linkage", "nebdft2k_partition_linkage"):

        def optional_pin(name: str = name) -> None:
            spec = inputs.get(name)
            if spec is not None:
                spec = _mapping(spec, f"inputs.{name}")
                _require_nonempty(spec, ("path",), f"inputs.{name}")
                _require_sha(spec.get("sha256"), f"inputs.{name}.sha256")

        collect(optional_pin)

    def documents() -> None:
        records = _mapping(config.get("source_documents"), "source_documents")
        if not records:
            raise ProtocolConfigurationError("source_documents must not be empty")
        for name, value in records.items():
            document = _mapping(value, f"source_documents.{name}")
            _require_nonempty(document, ("path",), f"source_documents.{name}")
            _require_sha(document.get("sha256"), f"source_documents.{name}.sha256")

    collect(documents)
    if errors:
        raise ProtocolConfigurationError("; ".join(errors))


def _validate_resolution(name: str, record: Mapping[str, Any]) -> None:
    fields = _RESOLUTION_FIELDS[name]
    if record.get("status") != "resolved" or not _nonempty(record.get("evidence")):
        raise ProtocolConfigurationError(f"scientific resolution is not frozen: {name}")
    for field in fields:
        value = record.get(field)
        if field == "excluded_elements":
            if (
                not isinstance(value, list)
                or not value
                or len(value) != len(set(value))
                or any(atomic_numbers.get(item, 0) <= 0 for item in value)
            ):
                raise ProtocolConfigurationError(
                    "primary_tm_hubbard_u.excluded_elements must list unique element symbols"
                )
            if sorted(value) != primary_exclusion_elements() or record.get(
                "predicate_id"
            ) != TM_RULE_ID:
                raise ProtocolConfigurationError(
                    f"primary_tm_hubbard_u must register the frozen rule {TM_RULE_ID}"
                )
        elif field == "repeat_count":
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ProtocolConfigurationError(
                    "label_budget_stability.repeat_count must be a positive integer"
                )
        elif field == "rule_id" and name == "mixed_anion_family":
            if value != ANION_FAMILY_RULE_ID:
                raise ProtocolConfigurationError(
                    f"mixed_anion_family.rule_id must be the frozen rule {ANION_FAMILY_RULE_ID}"
                )
        elif not _nonempty(value):
            raise ProtocolConfigurationError(
                f"scientific resolution is incomplete: {name}.{field}"
            )


def _validate_compatibility(value: Any, label: str) -> None:
    compatibility = _mapping(value, label)
    missing = [field for field in _COMPATIBILITY_FIELDS if field not in compatibility]
    if missing:
        raise ProtocolConfigurationError(f"{label} missing fields: {missing}")
    _require_nonempty(compatibility, _COMPATIBILITY_FIELDS[:7], label)
    if (
        compatibility["status"] != "compatible"
        or compatibility["primary_analysis_eligible"] is not True
        or compatibility["exclusion_reason"] is not None
    ):
        raise ProtocolConfigurationError(f"{label} is not primary-analysis compatible")


def _validate_model(model: Mapping[str, Any], group: str, index: int) -> None:
    label = f"model_groups.{group}[{index}]"
    if model.get("resolution_status") != "resolved" or model.get("frozen_group") != group:
        raise ProtocolConfigurationError(f"{label} identity is unresolved or mis-grouped")
    try:
        validate_model_provenance(model)
    except (TypeError, ValueError) as exc:
        raise ProtocolConfigurationError(f"{label}: {exc}") from exc
    missing = [field for field in MODEL_PROVENANCE_REQUIRED_FIELDS if field not in model]
    if missing:
        raise ProtocolConfigurationError(f"{label} missing provenance fields: {missing}")
    text_fields = set(MODEL_PROVENANCE_REQUIRED_FIELDS) - {
        "load_options",
        "license_verified",
        "fine_tuned",
        "reference_compatibility",
    }
    _require_nonempty(model, sorted(text_fields), label)
    _require_sha(model.get("weights_sha256"), f"{label}.weights_sha256")
    if not isinstance(model.get("load_options"), Mapping):
        raise ProtocolConfigurationError(f"{label}.load_options must be an object")
    if model.get("license_verified") is not True:
        raise ProtocolConfigurationError(f"{label} license must be verified")
    if not isinstance(model.get("fine_tuned"), bool):
        raise ProtocolConfigurationError(f"{label}.fine_tuned must be boolean")
    if model["fine_tuned"]:
        _require_nonempty(model, ("parent_checkpoint_id", "training_data_provenance"), label)
    _validate_compatibility(
        model.get("reference_compatibility"), f"{label}.reference_compatibility"
    )


def _model_checks(protocol: Mapping[str, Any]) -> list[tuple[str, Callable[[], None]]]:
    groups = _mapping(protocol["model_groups"], "protocol.model_groups")
    expected_counts = {"A": 4, "B1": 1, "B2": 1}
    checks: list[tuple[str, Callable[[], None]]] = []

    def count_check(group: str, count: int) -> Callable[[], None]:
        def check() -> None:
            if len(groups[group]) != count:
                raise ProtocolConfigurationError(
                    f"model group {group} must contain exactly {count} resolved model(s)"
                )

        return check

    for group, count in expected_counts.items():
        checks.append((f"model_group_{group}_count", count_check(group, count)))

    def optional_c() -> None:
        if len(groups["C"]) > 1:
            raise ProtocolConfigurationError(
                "optional model group C may contain at most one model"
            )

    checks.append(("model_group_C_count", optional_c))
    for group in _REQUIRED_GROUPS:
        for index, model in enumerate(groups[group]):
            # Group C is optional (Research Design 5.5): an unenabled placeholder is recorded
            # in the protocol but neither blocks the freeze nor enters any analysis.
            if group == "C" and model.get("resolution_status") == "optional_unresolved":
                continue
            checks.append(
                (
                    f"model_{group}_{index}",
                    lambda model=model, group=group, index=index: _validate_model(
                        _mapping(model, f"model_groups.{group}[{index}]"), group, index
                    ),
                )
            )
    return checks


def freeze_blockers(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Every reason a production configuration cannot yet be frozen, without short-circuit.

    The template must already be valid.  Each entry is a human-readable message;
    an empty result means every pin, unit, scientific resolution, model identity,
    and checklist item required for a production freeze is present.
    """

    _mode, inputs, protocol = _validate_shape(config)
    resolutions = _mapping(protocol["scientific_resolutions"], "scientific_resolutions")
    checks: list[tuple[str, Callable[[], None]]] = [
        ("trust_anchors", lambda: _validate_trust_anchors(config, inputs)),
        (
            "unit_contract",
            lambda: validate_unit_contract(_mapping(protocol["unit_contract"], "unit_contract")),
        ),
    ]
    for name in _RESOLUTION_FIELDS:
        checks.append(
            (
                f"resolution_{name}",
                lambda name=name: _validate_resolution(
                    name, _mapping(resolutions[name], f"scientific_resolutions.{name}")
                ),
            )
        )
    checks.extend(_model_checks(protocol))
    blockers: list[str] = []
    for _label, check in checks:
        try:
            check()
        except (ProtocolConfigurationError, TypeError, ValueError) as exc:
            blockers.append(str(exc))
    items = validate_unresolved_items(protocol.get("unresolved_items"))
    blockers.extend(required_item_blockers(items))
    return tuple(blockers)


def validate_protocol_configuration(
    config: Mapping[str, Any], *, allow_synthetic_test: bool = False
) -> ProtocolConfigurationValidation:
    """Validate configuration for a production freeze, defaulting to production-only.

    ``allow_synthetic_test`` exists solely for local fixture execution.  Its
    result always has ``production_eligible=False`` and therefore cannot be
    interpreted as authorization to create a production protocol freeze.
    """

    template = validate_protocol_template(config)
    if template.materialization_mode == "synthetic_test":
        if not allow_synthetic_test:
            raise ProtocolConfigurationError("synthetic_test is not production materialization")
        return template
    blockers = freeze_blockers(config)
    if blockers:
        raise ProtocolConfigurationError("; ".join(blockers))
    return ProtocolConfigurationValidation("production", production_eligible=True)


def assess_production_configuration(
    config: Mapping[str, Any],
) -> ProtocolConfigurationValidation:
    """Validate shape and report blockers so a sealed, unfrozen candidate can be built."""

    template = validate_protocol_template(config)
    if template.materialization_mode != "production":
        raise ProtocolConfigurationError("assessment applies to production configurations")
    blockers = freeze_blockers(config)
    return ProtocolConfigurationValidation(
        "production", production_eligible=not blockers, freeze_blockers=blockers
    )
