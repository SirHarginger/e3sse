"""Fail-closed scientific contracts used before production inference."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

INFERENCE_RECORD_SCHEMA = "e3sse.mlip.inference.record.v1"
INFERENCE_RUN_SCHEMA = "e3sse.mlip.inference.run.v1"
PROTOCOL_FREEZE_SCHEMA = "e3sse.protocol.freeze.v1"

REQUIRED_UNITS = (
    "position",
    "cell",
    "dft_energy",
    "ml_energy",
    "dft_force",
    "ml_force",
    "stress",
    "barrier",
    "temperature",
)

CANONICAL_UNITS = {
    "position": "angstrom",
    "cell": "angstrom",
    "dft_energy": "eV",
    "ml_energy": "eV",
    "dft_force": "eV/angstrom",
    "ml_force": "eV/angstrom",
    "stress": "eV/angstrom^3",
    "barrier": "eV",
    "temperature": "K",
}

# Canonical analysis units are frozen by the protocol; only the production source
# units remain to be evidenced in Phase B.
CANONICAL_UNIT_CLASSIFICATION = {
    "position": ("engineering_choice", "angstrom implied by eV/angstrom forces"),
    "cell": ("engineering_choice", "angstrom, consistent with positions"),
    "dft_energy": ("paper_specified", "Research Design Nomenclature (U, eV)"),
    "ml_energy": ("paper_specified", "Research Design Nomenclature (U, eV)"),
    "dft_force": ("paper_specified", "Research Design Nomenclature (F, eV/angstrom)"),
    "ml_force": ("paper_specified", "Research Design Nomenclature (F, eV/angstrom)"),
    "stress": ("engineering_choice", "eV/angstrom^3, consistent with energy and length"),
    "barrier": ("paper_specified", "Research Design Nomenclature (E_b, eV)"),
    "temperature": ("paper_specified", "Research Design Nomenclature (T, K)"),
}


def canonical_analysis_units() -> dict[str, dict[str, str]]:
    """Frozen canonical analysis units with the authority for each."""

    return {
        name: {
            "unit": CANONICAL_UNITS[name],
            "classification": CANONICAL_UNIT_CLASSIFICATION[name][0],
            "source": CANONICAL_UNIT_CLASSIFICATION[name][1],
        }
        for name in REQUIRED_UNITS
    }


INFERENCE_REQUIRED_FIELDS = (
    "schema_version",
    "frame_id",
    "material_id",
    "chemical_system_id",
    "hop_id",
    "published_split",
    "repeat_partition_membership",
    "analysis_population",
    "provenance_index_id",
    "source_file_sha256",
    "source_frame_index",
    "model_provenance",
    "reference_compatibility",
    "n_atoms",
    "atomic_numbers_sha256",
    "energy_kind",
    "raw_energy_kind",
    "raw_energy",
    "energy_reference_scheme",
    "energy_corrections_applied",
    "force_sign_convention",
    "energy",
    "energy_unit",
    "forces",
    "force_unit",
    "requested_device",
    "effective_device",
    "requested_dtype",
    "effective_dtype",
    "run_id",
    "protocol_id",
    "git_sha",
    "config_hash",
    "input_hash",
    "record_key",
    "attempt_id",
    "attempt_number",
    "status",
)

INFERENCE_RUN_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "protocol_id",
    "input_manifest_sha256",
    "inference_config_sha256",
    "git_sha",
    "git_branch",
    "git_dirty",
    "started_at_utc",
    "finished_at_utc",
    "command",
    "python_version",
    "dependency_versions",
    "environment_lock_sha256",
    "random_seeds",
    "hardware",
    "determinism",
    "model_artifacts",
    "output_sha256",
    "record_counts_by_status",
)

MODEL_PROVENANCE_REQUIRED_FIELDS = (
    "canonical_name",
    "group",
    "architecture_family",
    "pretraining_dataset",
    "upstream",
    "release_revision",
    "checkpoint_id",
    "weights_sha256",
    "package_name",
    "package_version",
    "adapter_name",
    "adapter_version",
    "load_options",
    "license_identifier",
    "license_verified",
    "fine_tuned",
    "reference_compatibility",
)
_MODEL_NONTEXT_FIELDS = {"load_options", "license_verified", "fine_tuned", "reference_compatibility"}
COMPATIBILITY_FIELDS = (
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

# Reference settings of the LiTraj DFT data (Research Design Table 6).
REFERENCE_TARGET = {
    "target_functional": "PBE",
    "target_hubbard_u_policy": "no_hubbard_u",
    "target_spin_policy": "non_spin_polarized",
}
_ANALYSIS_POPULATIONS = {"primary", "tm_hubbard_u_secondary"}
PILOT_AMENDMENT_SCHEMA = "e3sse.protocol.pilot-amendment.v1"
TRANSPORT_ADDENDUM_SCHEMA = "e3sse.protocol.transport-addendum.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_DTYPES = {"float32", "float64", "bfloat16", "mixed-float16", "mixed-bfloat16"}
_ANALYSIS_CONTEXTS = {"primary", "repeat"}

# Exact SI (2019) elementary charge and CODATA 2018 recommended values.  Only the
# pairs below are recognized; any other source unit leaves the quantity unresolved.
_EV_PER_HARTREE = 27.211386245988
_ANGSTROM_PER_BOHR = 0.529177210903
_GPA_PER_EV_PER_ANGSTROM3 = 160.2176634
UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("angstrom", "angstrom"): 1.0,
    ("nm", "angstrom"): 10.0,
    ("bohr", "angstrom"): _ANGSTROM_PER_BOHR,
    ("eV", "eV"): 1.0,
    ("meV", "eV"): 1.0e-3,
    ("hartree", "eV"): _EV_PER_HARTREE,
    ("rydberg", "eV"): _EV_PER_HARTREE / 2.0,
    ("eV/angstrom", "eV/angstrom"): 1.0,
    ("meV/angstrom", "eV/angstrom"): 1.0e-3,
    ("hartree/bohr", "eV/angstrom"): _EV_PER_HARTREE / _ANGSTROM_PER_BOHR,
    ("rydberg/bohr", "eV/angstrom"): _EV_PER_HARTREE / 2.0 / _ANGSTROM_PER_BOHR,
    ("eV/angstrom^3", "eV/angstrom^3"): 1.0,
    ("GPa", "eV/angstrom^3"): 1.0 / _GPA_PER_EV_PER_ANGSTROM3,
    ("kbar", "eV/angstrom^3"): 0.1 / _GPA_PER_EV_PER_ANGSTROM3,
    ("K", "K"): 1.0,
}

# Roles whose outcomes are part of the frozen method itself (pilot/training data).
_TUNING_ROLES = {"train"}
_TUNING_PURPOSES = {"threshold_selection", "method_tuning", "hyperparameter_selection"}
_OUTCOME_PURPOSES = {"preregistered_analysis", *_TUNING_PURPOSES}


class OutcomeAccessDenied(PermissionError):
    """Raised when a sealed scientific outcome is requested before freeze."""


def _content_id(value: Mapping[str, Any], id_field: str) -> str:
    content = {key: item for key, item in value.items() if key != id_field}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _sorted_ids_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted({str(value) for value in values}), separators=(",", ":")).encode()
    ).hexdigest()


def build_pilot_amendment(
    protocol: Mapping[str, Any],
    *,
    partition_records: Sequence[Mapping[str, Any]],
    training_unit_ids: Iterable[str],
    frozen_thresholds: Mapping[str, Any],
    mde_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind training-only pilot decisions to a frozen protocol and its verified units."""

    validate_protocol_freeze_for_outcome_access(protocol)
    units = list(training_unit_ids)
    _verify_training_units(protocol, partition_records, units)
    amendment = {
        "schema_version": PILOT_AMENDMENT_SCHEMA,
        "parent_protocol_id": protocol["protocol_id"],
        "data_roles": ["train"],
        "training_membership_sha256": protocol["partition_definition"]["partition_universe"][
            "membership_sha256"
        ],
        "training_units_sha256": _sorted_ids_sha256(units),
        "training_unit_count": len(set(units)),
        "frozen_thresholds": dict(frozen_thresholds),
        "mde_analysis": dict(mde_analysis),
    }
    amendment["amendment_id"] = _content_id(amendment, "amendment_id")
    validate_pilot_amendment(amendment, protocol)
    return amendment


def build_transport_addendum(protocol: Mapping[str, Any], *, g0_audit_path: Any) -> dict[str, Any]:
    """Record later G0 transport readiness from the audit file itself, never a caller claim."""

    from pathlib import Path

    path = Path(g0_audit_path)
    payload = path.read_bytes()
    audit = json.loads(payload)
    fpmd = [item for item in audit.get("datasets", []) if item.get("dataset") == "FPMD"]
    if len(fpmd) != 1:
        raise ValueError("G0 audit must contain exactly one FPMD dataset")
    status = fpmd[0].get("status")
    addendum = {
        "schema_version": TRANSPORT_ADDENDUM_SCHEMA,
        "parent_protocol_id": protocol["protocol_id"],
        "g0_audit_sha256": hashlib.sha256(payload).hexdigest(),
        "g0_run_id": audit.get("run_id"),
        "fpmd_g0_status": status,
        "transport_branch_status": {"pass": "ready", "partial": "partial"}.get(status, "blocked"),
        "blockers": list(fpmd[0].get("blockers", [])),
    }
    addendum["addendum_id"] = _content_id(addendum, "addendum_id")
    return addendum


def validate_pilot_amendment(amendment: Any, protocol: Mapping[str, Any]) -> None:
    """Second seal: training-only pilot thresholds/MDE bound to the frozen protocol.

    Research Design section 4 fixes thresholds during the training-split pilot and
    section 5.1 keeps test outcomes sealed until methods and thresholds are frozen,
    so test-side access requires this hashed amendment in addition to the freeze.
    """

    if not isinstance(amendment, Mapping):
        raise TypeError("pilot amendment must be an object")
    if amendment.get("schema_version") != PILOT_AMENDMENT_SCHEMA:
        raise ValueError("unsupported pilot amendment schema")
    if amendment.get("parent_protocol_id") != protocol.get("protocol_id"):
        raise ValueError("pilot amendment is not bound to this protocol")
    if amendment.get("data_roles") != ["train"]:
        raise ValueError("pilot amendment may use training-split outcomes only")
    membership = protocol["partition_definition"]["partition_universe"]["membership_sha256"]
    if amendment.get("training_membership_sha256") != membership:
        raise ValueError("pilot amendment was not verified against the frozen membership")
    _require_sha256(amendment.get("training_units_sha256"), "pilot training units")
    for field in ("frozen_thresholds", "mde_analysis"):
        if not isinstance(amendment.get(field), Mapping) or not amendment[field]:
            raise ValueError(f"pilot amendment lacks {field}")
    if amendment.get("amendment_id") != _content_id(amendment, "amendment_id"):
        raise ValueError("pilot amendment content hash mismatch")


def validate_transport_addendum(addendum: Any, protocol: Mapping[str, Any]) -> None:
    """Study 3 unseal: a later, hashed G0 transport readiness bound to the static protocol.

    Keeping transport readiness outside the protocol hash means resolving the FPMD
    blockers never changes the identity of Studies 1, 2a and 4.
    """

    if not isinstance(addendum, Mapping):
        raise TypeError("transport addendum must be an object")
    if addendum.get("schema_version") != TRANSPORT_ADDENDUM_SCHEMA:
        raise ValueError("unsupported transport addendum schema")
    if addendum.get("parent_protocol_id") != protocol.get("protocol_id"):
        raise ValueError("transport addendum is not bound to this protocol")
    _require_sha256(addendum.get("g0_audit_sha256"), "transport addendum G0 audit")
    if (
        addendum.get("fpmd_g0_status") != "pass"
        or addendum.get("transport_branch_status") != "ready"
        or addendum.get("blockers") != []
    ):
        raise ValueError("transport branch is not ready")
    if addendum.get("addendum_id") != _content_id(addendum, "addendum_id"):
        raise ValueError("transport addendum content hash mismatch")


def _verify_membership(
    protocol: Mapping[str, Any], partition_records: Sequence[Mapping[str, Any]]
) -> str:
    """Return the frozen membership digest after checking the supplied records match it."""

    expected = protocol["partition_definition"]["partition_universe"]["membership_sha256"]
    actual = hashlib.sha256(
        json.dumps(list(partition_records), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual != expected:
        raise ValueError("partition records differ from the frozen membership")
    return actual


def _verify_training_units(
    protocol: Mapping[str, Any],
    partition_records: Sequence[Mapping[str, Any]] | None,
    unit_ids: Iterable[str] | None,
) -> None:
    """Resolve the caller's units against the frozen membership instead of trusting a label."""

    if partition_records is None or unit_ids is None:
        raise ValueError("training outcome access requires partition records and unit IDs")
    _verify_membership(protocol, partition_records)
    split_by_unit: dict[str, set[str]] = {}
    for record in partition_records:
        for key in ("frame_id", "hop_id"):
            if record.get(key) is not None:
                split_by_unit.setdefault(str(record[key]), set()).add(record["published_split"])
    units = list(unit_ids)
    if not units:
        raise ValueError("unit IDs must be non-empty")
    wrong = [unit for unit in units if split_by_unit.get(str(unit)) != {"train"}]
    if wrong:
        raise ValueError(f"units are not in the published training split: {wrong[:5]}")


def require_outcome_access(
    *,
    role: str,
    outcome: str,
    frozen_protocol: Mapping[str, Any] | None = None,
    purpose: str = "preregistered_analysis",
    pilot_amendment: Mapping[str, Any] | None = None,
    transport_addendum: Mapping[str, Any] | None = None,
    partition_records: Sequence[Mapping[str, Any]] | None = None,
    unit_ids: Iterable[str] | None = None,
) -> None:
    """Default-deny scientific outcomes; identifiers, counts, hashes, schemas never pass here.

    Seal 1 (registration): every role, training included, requires a validated
    frozen protocol, because Study 1 and the pilot follow registration
    (Research Design 4 and 8).  Training access is verified against the frozen
    membership, never a caller-declared role.  Tuning and threshold selection
    are training-only.  Seal 2: non-training outcomes also require the hashed
    training-only pilot amendment.  FPMD outcomes also require a transport
    addendum declaring the transport branch ready.
    """

    roles = {
        "train",
        "calibration",
        "validation",
        "test",
        "repeated_test",
        "fpmd_reference",
        "prospective",
    }
    if role not in roles:
        raise OutcomeAccessDenied(f"unknown outcome-bearing role {role!r}")
    if not _nonempty_string(outcome):
        raise OutcomeAccessDenied("outcome must be a non-empty identifier")
    if purpose not in _OUTCOME_PURPOSES:
        raise OutcomeAccessDenied(f"unknown outcome-access purpose {purpose!r}")
    if purpose in _TUNING_PURPOSES and role not in _TUNING_ROLES:
        raise OutcomeAccessDenied(f"{purpose} may use training-split outcomes only, not {role}")
    try:
        validate_protocol_freeze_for_outcome_access(frozen_protocol)
    except (KeyError, TypeError, ValueError):
        raise OutcomeAccessDenied(f"{role} outcome {outcome!r} is sealed before protocol freeze")
    try:
        if role in _TUNING_ROLES:
            _verify_training_units(frozen_protocol, partition_records, unit_ids)
            return
        validate_pilot_amendment(pilot_amendment, frozen_protocol)
        if role == "fpmd_reference":
            validate_transport_addendum(transport_addendum, frozen_protocol)
    except (KeyError, TypeError, ValueError) as exc:
        raise OutcomeAccessDenied(f"{role} outcome {outcome!r} remains sealed: {exc}") from exc


def validate_protocol_freeze_for_outcome_access(
    protocol: Mapping[str, Any] | None,
) -> None:
    """Validate a complete scientific freeze, rather than trusting a caller boolean."""

    if not isinstance(protocol, Mapping):
        raise TypeError("frozen protocol must be an object")
    required = (
        "protocol_schema_version",
        "protocol_id",
        "git_sha",
        "source_document_hashes",
        "dataset_manifest",
        "partition_definition",
        "leakage_rules",
        "unit_contract",
        "study_1_contract",
        "evaluation_seal",
        "static_branch_status",
        "materialization_mode",
        "production_eligible",
        "freeze_status",
        "freeze_blockers",
        "unresolved_items",
    )
    missing = [field for field in required if field not in protocol]
    if missing:
        raise ValueError(f"frozen protocol missing fields: {missing}")
    if protocol["protocol_schema_version"] != PROTOCOL_FREEZE_SCHEMA:
        raise ValueError("unsupported frozen protocol schema")
    _require_sha256(protocol["protocol_id"], "protocol_id")
    if _GIT_SHA.fullmatch(str(protocol["git_sha"])) is None:
        raise ValueError("invalid frozen protocol Git SHA")
    documents = protocol["source_document_hashes"]
    if not isinstance(documents, Mapping) or not documents:
        raise ValueError("source document hashes must be non-empty")
    for name, digest in documents.items():
        if not _nonempty_string(name):
            raise ValueError("source document names must be non-empty")
        _require_sha256(digest, f"source document {name}")
    if protocol["static_branch_status"] != "ready":
        raise ValueError("static protocol branch is not ready")
    if (
        protocol["materialization_mode"] != "production"
        or protocol["production_eligible"] is not True
        or protocol["freeze_status"] != "frozen"
    ):
        raise ValueError("protocol is not a production-eligible frozen materialization")
    if protocol["freeze_blockers"] != []:
        raise ValueError("frozen protocol retains freeze blockers")
    unresolved = protocol["unresolved_items"]
    if not isinstance(unresolved, list) or any(
        not isinstance(item, Mapping) for item in unresolved
    ):
        raise TypeError("unresolved_items must be a list of objects")
    if any(item.get("blocking", True) for item in unresolved):
        raise ValueError("frozen protocol retains blocking unresolved items")
    seal = protocol["evaluation_seal"]
    if not isinstance(seal, Mapping) or seal.get("state") != "sealed":
        raise ValueError("evaluation seal is not present")
    validate_unit_contract(protocol["unit_contract"])
    for field in (
        "dataset_manifest",
        "partition_definition",
        "leakage_rules",
        "study_1_contract",
    ):
        if not isinstance(protocol[field], Mapping) or not protocol[field]:
            raise ValueError(f"{field} must be a non-empty object")
    content = {key: value for key, value in protocol.items() if key != "protocol_id"}
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if digest != protocol["protocol_id"]:
        raise ValueError("frozen protocol content hash mismatch")


def unit_conversion_id(source_unit: str, canonical_unit: str) -> str:
    """Return the only accepted ``conversion`` label for a recognized unit pair."""

    if (source_unit, canonical_unit) not in UNIT_CONVERSIONS:
        raise ValueError(f"unrecognized unit conversion: {source_unit!r} -> {canonical_unit!r}")
    return "identity" if source_unit == canonical_unit else f"{source_unit}->{canonical_unit}"


def unit_conversion_factor(quantity: str, record: Mapping[str, Any]) -> float:
    """Multiplicative factor from a proven production source unit to the canonical unit."""

    validate_unit_contract({**_placeholder_units(), quantity: record})
    evidence = record["production_source_unit_evidence"]
    if quantity == "stress" and evidence.get("status") == "not_retained":
        raise ValueError("stress is not retained; no conversion exists")
    return UNIT_CONVERSIONS[(evidence["source_unit"], record["canonical_analysis_unit"])]


def _placeholder_units() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "canonical_analysis_unit": unit,
            "production_source_unit_evidence": {
                "source_unit": unit,
                "conversion": "identity",
                "status": "verified",
                "evidence": "placeholder used only to validate one quantity",
            },
        }
        for name, unit in CANONICAL_UNITS.items()
    }


def unresolved_source_units(units: Mapping[str, Any]) -> list[str]:
    """Quantities whose production source-unit evidence is not yet proven."""

    unresolved = []
    for name in REQUIRED_UNITS:
        record = units.get(name)
        evidence = record.get("production_source_unit_evidence") if isinstance(
            record, Mapping
        ) else None
        if not isinstance(evidence, Mapping):
            unresolved.append(name)
            continue
        if name == "stress" and evidence.get("status") == "not_retained":
            if not _nonempty_string(evidence.get("evidence")):
                unresolved.append(name)
            continue
        source = evidence.get("source_unit")
        if (
            evidence.get("status") != "verified"
            or not _nonempty_string(source)
            or not _nonempty_string(evidence.get("evidence"))
            or (source, CANONICAL_UNITS[name]) not in UNIT_CONVERSIONS
            or evidence.get("conversion") != unit_conversion_id(source, CANONICAL_UNITS[name])
        ):
            unresolved.append(name)
    return unresolved


def validate_unit_contract(units: Mapping[str, Mapping[str, Any]]) -> None:
    """Frozen canonical analysis units plus proven production source units.

    Canonical units must equal :data:`CANONICAL_UNITS`; they never depend on data.
    A production source unit is resolved only when evidenced from production
    schemas and its declared conversion is exactly the registered label for a pair
    in :data:`UNIT_CONVERSIONS`.  Unrecognized source units stay unresolved rather
    than being converted, and ASE conventions are never assumed.
    """

    missing = [name for name in REQUIRED_UNITS if name not in units]
    if missing:
        raise ValueError(f"unit contract missing: {missing}")
    wrong = [
        name
        for name in REQUIRED_UNITS
        if not isinstance(units[name], Mapping)
        or units[name].get("canonical_analysis_unit") != CANONICAL_UNITS[name]
    ]
    if wrong:
        raise ValueError(f"canonical analysis units differ from the frozen contract: {wrong}")
    unresolved = unresolved_source_units(units)
    if unresolved:
        raise ValueError(f"production source-unit evidence unresolved: {unresolved}")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_utc_timestamp(value: Any, label: str) -> datetime:
    if not _nonempty_string(value) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be an RFC 3339 UTC timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{label} must be UTC")
    return parsed


def validate_model_provenance(model: Mapping[str, Any]) -> None:
    """Validate the immutable model identity shared by records and run manifests."""

    if not isinstance(model, Mapping):
        raise TypeError("model provenance must be an object")
    missing = [field for field in MODEL_PROVENANCE_REQUIRED_FIELDS if field not in model]
    if missing:
        raise ValueError(f"model provenance missing fields: {missing}")
    text_fields = set(MODEL_PROVENANCE_REQUIRED_FIELDS) - _MODEL_NONTEXT_FIELDS
    if any(not _nonempty_string(model[field]) for field in text_fields):
        raise ValueError("model provenance identifiers must be non-empty")
    validate_reference_compatibility(model["reference_compatibility"])
    if model["group"] not in {"A", "B1", "B2", "C"}:
        raise ValueError("invalid frozen model group")
    _require_sha256(model["weights_sha256"], "weights_sha256")
    if not isinstance(model["load_options"], Mapping):
        raise TypeError("model load_options must be an object")
    if model["license_verified"] is not True:
        raise ValueError("model license must be verified")
    if not isinstance(model["fine_tuned"], bool):
        raise TypeError("fine_tuned must be boolean")
    if model["fine_tuned"] != (model["group"] == "C"):
        raise ValueError("only frozen Group C may contain a fine-tuned model")
    fine_tuning_fields = ("parent_checkpoint_id", "training_data_provenance")
    if model["fine_tuned"] and not all(
        _nonempty_string(model.get(field)) for field in fine_tuning_fields
    ):
        raise ValueError("fine-tuned models require parent checkpoint and training provenance")
    if not model["fine_tuned"] and any(
        model.get(field) is not None for field in fine_tuning_fields
    ):
        raise ValueError("pretrained models cannot carry fine-tuning provenance")


def validate_reference_compatibility(compatibility: Any) -> None:
    """Primary eligibility requires the pinned LiTraj reference settings, not a self-claim."""

    if not isinstance(compatibility, Mapping):
        raise TypeError("reference_compatibility must be an object")
    if any(field not in compatibility for field in COMPATIBILITY_FIELDS):
        raise ValueError("reference compatibility assessment is incomplete")
    if any(not _nonempty_string(compatibility[field]) for field in COMPATIBILITY_FIELDS[:7]):
        raise ValueError("reference compatibility evidence must be explicit")
    mismatched = [
        field for field, value in REFERENCE_TARGET.items() if compatibility[field] != value
    ]
    if mismatched:
        raise ValueError(f"reference compatibility targets differ from LiTraj: {mismatched}")
    if (
        compatibility["status"] != "compatible"
        or compatibility["primary_analysis_eligible"] is not True
    ):
        raise ValueError("model/reference settings are not eligible for primary analysis")
    if compatibility["exclusion_reason"] is not None:
        raise ValueError("eligible primary records cannot have an exclusion reason")


def inference_record_key(
    *, protocol_id: str, frame_id: str, weights_sha256: str, config_hash: str
) -> str:
    """Idempotent key: one final prediction per protocol, frame, checkpoint and config."""

    payload = json.dumps(
        ["e3sse.mlip.record-key.v1", protocol_id, frame_id, weights_sha256, config_hash],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_inference_record(
    record: Mapping[str, Any],
    *,
    analysis_context: str = "primary",
    repeat_id: str | None = None,
    expected_protocol_id: str | None = None,
    expected_repeat_ids: Iterable[str] | None = None,
) -> None:
    """Validate the future Study 1 result envelope without invoking a model."""

    missing = [field for field in INFERENCE_REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"inference record missing fields: {missing}")
    if record["schema_version"] != INFERENCE_RECORD_SCHEMA:
        raise ValueError("unsupported inference record schema")
    if analysis_context not in _ANALYSIS_CONTEXTS:
        raise ValueError("analysis_context must be primary or repeat")
    if record["published_split"] not in {"train", "val", "test"}:
        raise ValueError("invalid published split")
    for field in (
        "frame_id",
        "material_id",
        "chemical_system_id",
        "provenance_index_id",
        "run_id",
        "protocol_id",
    ):
        if not _nonempty_string(record[field]):
            raise ValueError(f"{field} must be a non-empty identifier")
    if record["hop_id"] is not None and not _nonempty_string(record["hop_id"]):
        raise ValueError("hop_id must be null or a non-empty identifier")
    _require_sha256(record["source_file_sha256"], "source_file_sha256")
    _require_sha256(record["config_hash"], "config_hash")
    _require_sha256(record["input_hash"], "input_hash")
    if (
        not isinstance(record["source_frame_index"], int)
        or isinstance(record["source_frame_index"], bool)
        or record["source_frame_index"] < 0
    ):
        raise ValueError("source_frame_index must be a non-negative integer")
    if (
        not isinstance(record["n_atoms"], int)
        or isinstance(record["n_atoms"], bool)
        or record["n_atoms"] <= 0
    ):
        raise ValueError("n_atoms must be a positive integer")
    if _GIT_SHA.fullmatch(str(record["git_sha"])) is None:
        raise ValueError("git_sha must be a lowercase 40-character commit SHA")
    if expected_protocol_id is not None and record["protocol_id"] != expected_protocol_id:
        raise ValueError("record protocol_id differs from the frozen protocol")
    membership = record["repeat_partition_membership"]
    if not isinstance(membership, Mapping) or not membership:
        raise ValueError("repeat_partition_membership must be non-empty")
    if expected_repeat_ids is not None and set(membership) != set(expected_repeat_ids):
        raise ValueError("repeat partition IDs differ from the frozen protocol")
    if set(membership.values()) - {"training_or_calibration", "test"}:
        raise ValueError("invalid repeat partition membership")
    if analysis_context == "primary":
        if repeat_id is not None:
            raise ValueError("repeat_id is not valid for the primary analysis")
        if record["published_split"] != "train":
            raise ValueError("primary Study 1 inference requires the published training split")
    else:
        if not _nonempty_string(repeat_id) or repeat_id not in membership:
            raise ValueError("repeat analysis requires a frozen repeat_id")
        if membership[repeat_id] != "training_or_calibration":
            raise ValueError("repeat Study 1 inference requires the calibration side")

    model = record["model_provenance"]
    validate_model_provenance(model)

    compatibility = record["reference_compatibility"]
    validate_reference_compatibility(compatibility)
    if dict(compatibility) != dict(model["reference_compatibility"]):
        raise ValueError("record reference compatibility differs from frozen model assessment")
    if record["analysis_population"] not in _ANALYSIS_POPULATIONS:
        raise ValueError("analysis_population must be primary or tm_hubbard_u_secondary")
    _require_sha256(record["atomic_numbers_sha256"], "atomic_numbers_sha256")
    expected_key = inference_record_key(
        protocol_id=record["protocol_id"],
        frame_id=record["frame_id"],
        weights_sha256=model["weights_sha256"],
        config_hash=record["config_hash"],
    )
    if record["record_key"] != expected_key:
        raise ValueError("record_key is not the idempotent key of this prediction")
    if not _nonempty_string(record["attempt_id"]) or (
        not isinstance(record["attempt_number"], int)
        or isinstance(record["attempt_number"], bool)
        or record["attempt_number"] < 1
    ):
        raise ValueError("every record requires attempt_id and a positive attempt_number")
    for requested, effective in (
        ("requested_device", "effective_device"),
        ("requested_dtype", "effective_dtype"),
    ):
        if record[requested] != record[effective] and not _nonempty_string(
            record.get("fallback_reason")
        ):
            raise ValueError(f"{effective} differs from {requested} without fallback_reason")
    if record["raw_energy_kind"] not in {"total", "per_atom"}:
        raise ValueError("raw_energy_kind must be total or per_atom")
    # Raw uMLIP predictions stay distinct from corrected values (AGENTS.md).
    if record["energy_reference_scheme"] != "model_native" or record[
        "energy_corrections_applied"
    ] != []:
        raise ValueError("records must hold uncorrected model-native energies")
    if record["requested_dtype"] not in _DTYPES or record["effective_dtype"] not in _DTYPES:
        raise ValueError("unsupported requested/effective dtype")
    if not _nonempty_string(record["requested_device"]) or not _nonempty_string(
        record["effective_device"]
    ):
        raise ValueError("requested/effective device identifiers are required")
    if record["energy_kind"] != "total_configuration_energy":
        raise ValueError("energy must be total configuration energy")
    if record["force_sign_convention"] != "force_equals_negative_energy_gradient":
        raise ValueError("force sign convention is invalid")
    if record["status"] not in {"success", "failed"}:
        raise ValueError("invalid inference status")
    if record["status"] == "success":
        if not _finite_number(record["energy"]) or not _finite_number(record["raw_energy"]):
            raise ValueError("successful inference requires finite energy")
        scale = record["n_atoms"] if record["raw_energy_kind"] == "per_atom" else 1
        if not math.isclose(
            record["energy"], record["raw_energy"] * scale, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise ValueError("energy is not the total configuration energy of raw_energy")
        forces = record["forces"]
        if (
            not isinstance(forces, list)
            or len(forces) != record["n_atoms"]
            or any(
                not isinstance(row, list)
                or len(row) != 3
                or any(not _finite_number(value) for value in row)
                for row in forces
            )
        ):
            raise ValueError("forces must have finite shape (n_atoms, 3)")
        if any(
            record.get(field) is not None
            for field in ("failure_stage", "error_code", "error_type", "error_message")
        ):
            raise ValueError("successful records cannot carry failure metadata")
    else:
        if any(record[field] is not None for field in ("energy", "raw_energy", "forces")):
            raise ValueError("failed inference must have null predictions")
        failure_fields = (
            "failure_stage",
            "error_code",
            "error_type",
            "error_message",
            "retryable",
            "attempt_at_utc",
            "log_reference",
        )
        if any(field not in record for field in failure_fields):
            raise ValueError("failed inference requires complete failure metadata")
        if any(
            not _nonempty_string(record[field])
            for field in failure_fields
            if field != "retryable"
        ):
            raise ValueError("failure identifiers and message must be non-empty")
        if not isinstance(record["retryable"], bool):
            raise ValueError("invalid retry metadata")
    if record["energy_unit"] != "eV" or record["force_unit"] != "eV/angstrom":
        raise ValueError("MLIP adapters must normalize energy/forces to eV and eV/angstrom")


def validate_study1_inference_record(
    record: Mapping[str, Any],
    *,
    analysis_context: str = "primary",
    repeat_id: str | None = None,
    frozen_frame: Mapping[str, Any],
    frozen_repeat_membership: Mapping[str, str],
    frozen_model: Mapping[str, Any],
    protocol_id: str,
) -> None:
    """Cross-check a future result against frozen frame, partition, and model facts."""

    validate_inference_record(
        record,
        analysis_context=analysis_context,
        repeat_id=repeat_id,
        expected_protocol_id=protocol_id,
        expected_repeat_ids=frozen_repeat_membership,
    )
    linkage_fields = (
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
    if any(record[field] != frozen_frame.get(field) for field in linkage_fields):
        raise ValueError("inference record differs from frozen frame provenance")
    if dict(record["repeat_partition_membership"]) != dict(frozen_repeat_membership):
        raise ValueError("inference record differs from frozen repeat assignments")
    if dict(record["model_provenance"]) != dict(frozen_model):
        raise ValueError("inference record differs from frozen model identity")


def validate_inference_run_manifest(
    manifest: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
    expected_protocol_id: str | None = None,
    allow_dirty: bool = False,
) -> None:
    """Validate the immutable, versioned envelope for a future inference run.

    Production runs must come from a clean worktree and the frozen protocol_id.
    Within a run each record_key has at most one success and attempt IDs are
    unique, so a resumed, chunked run can never double-count a frame.
    """

    missing = [field for field in INFERENCE_RUN_REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"inference run manifest missing fields: {missing}")
    if manifest["schema_version"] != INFERENCE_RUN_SCHEMA:
        raise ValueError("unsupported inference run manifest schema")
    for field in (
        "input_manifest_sha256",
        "inference_config_sha256",
        "environment_lock_sha256",
        "output_sha256",
    ):
        _require_sha256(manifest[field], field)
    if _GIT_SHA.fullmatch(str(manifest["git_sha"])) is None:
        raise ValueError("invalid inference run Git SHA")
    for field in (
        "run_id",
        "protocol_id",
        "git_branch",
        "command",
        "python_version",
    ):
        if not _nonempty_string(manifest[field]):
            raise ValueError(f"{field} must be non-empty")
    if not isinstance(manifest["git_dirty"], bool):
        raise TypeError("git_dirty must be boolean")
    if manifest["git_dirty"] and not allow_dirty:
        raise ValueError("production inference requires a clean Git worktree")
    if expected_protocol_id is not None and manifest["protocol_id"] != expected_protocol_id:
        raise ValueError("inference run belongs to a different protocol_id")
    for field in (
        "dependency_versions",
        "random_seeds",
        "hardware",
        "determinism",
        "record_counts_by_status",
    ):
        if not isinstance(manifest[field], Mapping):
            raise TypeError(f"{field} must be an object")
    if not isinstance(manifest["model_artifacts"], list) or not manifest["model_artifacts"]:
        raise ValueError("model_artifacts must be non-empty")
    model_identities = []
    for model in manifest["model_artifacts"]:
        validate_model_provenance(model)
        identity = (model["canonical_name"], model["checkpoint_id"], model["weights_sha256"])
        if identity in model_identities:
            raise ValueError("duplicate model artifact identity")
        model_identities.append(identity)
    started = _require_utc_timestamp(manifest["started_at_utc"], "started_at_utc")
    finished = _require_utc_timestamp(manifest["finished_at_utc"], "finished_at_utc")
    if finished < started:
        raise ValueError("finished_at_utc cannot precede started_at_utc")
    counts = manifest["record_counts_by_status"]
    if set(counts) != {"success", "failed"} or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
        raise ValueError("record status counts must be non-negative success/failed integers")
    if sum(counts.values()) == 0:
        raise ValueError("inference run must account for at least one record")
    if records is not None:
        actual_counts = Counter(record.get("status") for record in records)
        if set(actual_counts) - {"success", "failed"}:
            raise ValueError("run records contain an invalid status")
        if dict(counts) != {status: actual_counts[status] for status in ("success", "failed")}:
            raise ValueError("record status counts differ from supplied records")
        artifact_models = [dict(model) for model in manifest["model_artifacts"]]
        successes = Counter(r.get("record_key") for r in records if r.get("status") == "success")
        if any(count > 1 for count in successes.values()):
            raise ValueError("a record_key has more than one successful prediction")
        attempts = Counter(record.get("attempt_id") for record in records)
        if any(count > 1 for count in attempts.values()):
            raise ValueError("duplicate attempt_id in run records")
        for record in records:
            if record.get("run_id") != manifest["run_id"]:
                raise ValueError("record run_id differs from run manifest")
            if record.get("protocol_id") != manifest["protocol_id"]:
                raise ValueError("record protocol_id differs from run manifest")
            if record.get("git_sha") != manifest["git_sha"]:
                raise ValueError("record Git SHA differs from run manifest")
            if record.get("config_hash") != manifest["inference_config_sha256"]:
                raise ValueError("record config hash differs from run manifest")
            if (
                not isinstance(record.get("model_provenance"), Mapping)
                or dict(record["model_provenance"]) not in artifact_models
            ):
                raise ValueError("record model identity is absent from run manifest")


# Research Design 3.6 eligibility rule, frozen as a single rule (no policy switch).
ELIGIBILITY_RULE = "research_design_3.6_v1"
# LiTraj raw hop files (src/e3sse/data/litraj_provenance.py): ``band_optim_step`` frames
# and the initial interpolated ``traj_init`` path are generated by the hop's NEB
# optimisation; ``source``/``target`` hold the relaxed endpoints.
NEB_PATH_FILE_KINDS = ("band_optim_step", "traj_init")
ENDPOINT_RELAXATION_FILE_KINDS = ("source", "target")
_FROZEN_FRAME_FIELDS = (
    "hop_id",
    "material_id",
    "chemical_system_id",
    "published_split",
    "mapping_status",
    "raw_file_kind",
    "origin_evidence_status",
    "origin_evidence",
)
# Published-split roles fixed by the paper (5.4); validation stays unresolved.
PUBLISHED_SPLIT_ROLES = {"train": "training_or_calibration", "test": "test"}


def _partition_roles(protocol: Mapping[str, Any], partition: str | int) -> tuple[Any, str]:
    """Role function over frozen membership rows, plus a digest of its definition."""

    if partition == "published":
        roles = dict(PUBLISHED_SPLIT_ROLES)
        digest = hashlib.sha256(json.dumps(roles, sort_keys=True).encode()).hexdigest()
        return (lambda row: roles.get(row.get("published_split"))), digest
    if not isinstance(partition, int) or isinstance(partition, bool):
        raise TypeError("partition must be 'published' or a repeat index")
    repeats = protocol["partition_definition"]["repeated_partitions"]["partitions"]
    matches = [item for item in repeats if item.get("repeat_index") == partition]
    if len(matches) != 1:
        raise ValueError(f"repeat {partition!r} is not in the frozen partition manifest")
    assignment = dict(matches[0]["assignment_by_chemical_system"])
    digest = hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()
    return (lambda row: assignment.get(row.get("chemical_system_id"))), digest
_PATH_FILE_KINDS = set(NEB_PATH_FILE_KINDS)
_ENDPOINT_FILE_KINDS = set(ENDPOINT_RELAXATION_FILE_KINDS)


def audit_softening_frame_eligibility(
    records: Iterable[Mapping[str, Any]],
    *,
    target_hop_id: str,
    estimator_scope: str,
    scope_evidence: Mapping[str, Any],
    allowed_partition_role: str,
    target_material_id: str | None = None,
    target_family: str | None = None,
    target_element: str | None = None,
    protocol: Mapping[str, Any],
    partition_records: Sequence[Mapping[str, Any]],
    partition: str | int,
    estimator_kind: str = "primary",
) -> dict[str, Any]:
    """Select leakage-free frames and count every eligibility decision reason.

    Research Design 3.6 (paper-specified): for the primary estimate correcting a
    target hop, frames generated during that hop's NEB optimisation are
    prohibited; endpoint-relaxation frames and frames of other hops of the same
    structure are admissible; the hop's own path frames are oracle-only.  A frame
    whose origin cannot be identified fails closed.  Research Design 5.4: pooled
    (global/family/element) scopes use the training/calibration side, and test-side
    material-level s uses only non-path frames of the target material.

    Partition roles and every eligibility-critical frame fact (hop, material,
    chemical system, published split, mapping status, file kind, origin evidence)
    are read from the frozen membership (``partition_records``, verified against
    the protocol's membership digest), never from the caller.  ``partition`` is
    ``"published"`` (per-frame published split; validation frames fail closed while
    their role is unresolved) or a repeat index (frozen chemical-system assignment).

    This freezes eligibility, not use: an estimator may use any subset of the
    eligible frames.  ``estimator_kind="oracle"`` selects only the target hop's own
    path frames; its result is labelled and rejected by
    :func:`require_primary_softening_selection`.
    """

    if not _nonempty_string(target_hop_id):
        raise ValueError("target_hop_id must be a non-empty identifier")
    if estimator_scope not in {"global", "family", "material", "element"}:
        raise ValueError("invalid softening estimator scope")
    membership_sha256 = _verify_membership(protocol, partition_records)
    frozen_frames = {
        str(row["frame_id"]): row for row in partition_records if row.get("frame_id") is not None
    }
    role_of, assignment_sha256 = _partition_roles(protocol, partition)
    if estimator_kind not in {"primary", "oracle"}:
        raise ValueError("estimator_kind must be primary or oracle")
    if estimator_scope == "material":
        if allowed_partition_role not in {"training_or_calibration", "test"}:
            raise ValueError("material scope requires the target hop's partition role")
    elif allowed_partition_role != "training_or_calibration":
        raise ValueError("pooled softening estimation requires the calibration side")
    targets = {
        "global": None,
        "family": target_family,
        "material": target_material_id,
        "element": target_element,
    }
    scope_target = targets[estimator_scope]
    if estimator_scope != "global" and not _nonempty_string(scope_target):
        raise ValueError(f"{estimator_scope} scope requires its explicit target")
    if not isinstance(scope_evidence, Mapping):
        raise TypeError("scope_evidence must be an object")
    if (
        scope_evidence.get("status") != "verified"
        or scope_evidence.get("scope") != estimator_scope
        or scope_evidence.get("target") != scope_target
        or not _nonempty_string(scope_evidence.get("source"))
    ):
        raise ValueError("scope evidence must verify the requested scope and target")

    eligible: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for item in records:
        record = dict(item)
        frozen = frozen_frames.get(str(record.get("frame_id")))
        if frozen is None:
            reasons["excluded_not_in_frozen_membership"] += 1
            continue
        conflicting = [
            field
            for field in _FROZEN_FRAME_FIELDS
            if field in record and record[field] != frozen.get(field)
        ]
        if conflicting:
            raise ValueError(f"record contradicts the frozen membership: {conflicting}")
        record.update({field: frozen.get(field) for field in _FROZEN_FRAME_FIELDS})
        if record.get("mapping_status") not in {"mapped_unique", "mapped_hop_unique"}:
            reasons["excluded_unresolved_mapping"] += 1
            continue
        role = role_of(frozen)
        if role is None:
            reasons["excluded_unassigned_partition_role"] += 1
            continue
        if record.get("partition_role") not in {None, role}:
            raise ValueError("record partition_role contradicts the frozen assignment")
        record["partition_role"] = role
        if role != allowed_partition_role:
            reasons["excluded_partition_role"] += 1
            continue
        if not _nonempty_string(record.get("hop_id")):
            reasons["excluded_missing_hop_identity"] += 1
            continue
        if estimator_scope == "material" and record.get("material_id") != target_material_id:
            reasons["excluded_material_scope"] += 1
            continue
        if estimator_scope == "family":
            if record.get("anion_family_evidence_status") != "verified":
                reasons["excluded_unverified_family"] += 1
                continue
            if record.get("anion_family") != target_family:
                reasons["excluded_family_scope"] += 1
                continue
        if estimator_scope == "element":
            elements = record.get("elements")
            if record.get("element_evidence_status") != "verified" or not isinstance(
                elements, (list, tuple)
            ):
                reasons["excluded_unverified_elements"] += 1
                continue
            if target_element not in elements:
                reasons["excluded_element_scope"] += 1
                continue
        is_target_hop = record["hop_id"] == target_hop_id
        if estimator_kind == "oracle":
            if is_target_hop and record.get("raw_file_kind") in _PATH_FILE_KINDS:
                eligible.append(record)
                reasons["oracle_own_path"] += 1
            else:
                reasons["excluded_not_own_path"] += 1
            continue
        kind = record.get("raw_file_kind")
        proven_endpoint = (
            kind in _ENDPOINT_FILE_KINDS
            and record.get("origin_evidence_status") == "unequivocal"
            and _nonempty_string(record.get("origin_evidence"))
        )
        if is_target_hop:
            if kind in _PATH_FILE_KINDS:
                reasons["excluded_target_hop_neb_path"] += 1
            elif proven_endpoint:
                eligible.append(record)
                reasons["eligible_target_hop_endpoint_relaxation"] += 1
            else:
                reasons["excluded_target_hop_unidentified_origin"] += 1
            continue
        if allowed_partition_role == "test":
            # Section 5.4: test-side fitting uses non-path frames only, so other hops'
            # NEB path frames of a test material (e.g. reverse hops) are excluded.
            if kind in _PATH_FILE_KINDS:
                reasons["excluded_test_side_path_frame"] += 1
                continue
            if not proven_endpoint:
                reasons["excluded_test_side_unidentified_origin"] += 1
                continue
        eligible.append(record)
        reasons["eligible_other_hop"] += 1
    selection = {
        "schema_version": "e3sse.softening.frame_eligibility.v3",
        "estimator_kind": estimator_kind,
        "analysis_label": "primary" if estimator_kind == "primary" else "oracle_upper_bound",
        "primary_eligible": estimator_kind == "primary",
        "eligibility_rule": ELIGIBILITY_RULE,
        "target_hop_id": target_hop_id,
        "estimator_scope": estimator_scope,
        "scope_target": scope_target,
        "scope_evidence": dict(scope_evidence),
        "allowed_partition_role": allowed_partition_role,
        "protocol_id": protocol.get("protocol_id"),
        "partition": partition,
        "membership_sha256": membership_sha256,
        "partition_assignment_sha256": assignment_sha256,
        "eligible_records": eligible,
        "eligible_count": len(eligible),
        "reason_counts": dict(sorted(reasons.items())),
        "input_count": sum(reasons.values()),
    }
    selection["selection_sha256"] = _content_id(selection, "selection_sha256")
    return selection


# Objects refit per repeat.  Thresholds are fixed once in the pilot (Research Design 4)
# and predictor construction is the open study4_predictor_calibration_independence item.
REPEAT_FITTED_OBJECTS = (
    "softening_estimate",
    "recalibration",
    "conformal_object",
)


def require_repeat_training_side(
    protocol: Mapping[str, Any],
    *,
    repeat_index: int,
    fitted_object: str,
    chemical_system_ids: Iterable[str],
) -> None:
    """A repeat-specific fit may use only that repeat's frozen training/calibration side.

    Research Design 5.4: no repeated-test-side outcome may fit that repeat's
    predictor, softening estimate, recalibration, threshold, or conformal object.
    Sides are read from the frozen partition manifest, never from the caller.
    """

    if fitted_object not in REPEAT_FITTED_OBJECTS:
        raise ValueError(f"unknown repeat-fitted object {fitted_object!r}")
    repeats = protocol["partition_definition"]["repeated_partitions"]["partitions"]
    matches = [item for item in repeats if item.get("repeat_index") == repeat_index]
    if len(matches) != 1:
        raise ValueError(f"repeat {repeat_index!r} is not in the frozen partition manifest")
    assignment = matches[0]["assignment_by_chemical_system"]
    systems = list(chemical_system_ids)
    if not systems:
        raise ValueError("a repeat-specific fit needs at least one chemical system")
    from .partitions import normalize_chemical_system

    offending = sorted(
        {
            str(system)
            for system in systems
            if assignment.get(normalize_chemical_system(system)) != "training_or_calibration"
        }
    )
    if offending:
        raise ValueError(
            f"repeat {repeat_index} {fitted_object} uses test-side or unassigned "
            f"chemical systems: {offending[:5]}"
        )


def require_primary_softening_selection(
    selection: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    """Reject oracle, tampered, foreign-protocol or wrong-side selections.

    For a repeat, pooled-scope selections must draw only on that repeat's
    training/calibration side (:func:`require_repeat_training_side`).  Test-side
    material-scope selections are admissible under Research Design 5.4 because the
    audit has already restricted them to non-path frames of the target material.
    """

    if (
        selection.get("schema_version") != "e3sse.softening.frame_eligibility.v3"
        or selection.get("estimator_kind") != "primary"
        or selection.get("analysis_label") != "primary"
        or selection.get("primary_eligible") is not True
        or selection.get("eligibility_rule") != ELIGIBILITY_RULE
    ):
        raise ValueError("only a primary leakage-audited selection may feed primary results")
    if selection.get("selection_sha256") != _content_id(selection, "selection_sha256"):
        raise ValueError("softening selection was modified after the leakage audit")
    if selection.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("softening selection belongs to a different protocol")
    if selection.get("membership_sha256") != protocol["partition_definition"][
        "partition_universe"
    ]["membership_sha256"]:
        raise ValueError("softening selection was audited against another membership")
    _, assignment_sha256 = _partition_roles(protocol, selection.get("partition"))
    if selection.get("partition_assignment_sha256") != assignment_sha256:
        raise ValueError("softening selection used a different partition assignment")
    target = selection.get("target_hop_id")
    records = selection.get("eligible_records", [])
    for record in records:
        if record.get("hop_id") == target and record.get("raw_file_kind") not in _ENDPOINT_FILE_KINDS:
            raise ValueError("primary selection contains a target-hop path frame")
    pooled = selection.get("estimator_scope") != "material"
    if pooled and records:
        if selection.get("partition") == "published":
            if any(record.get("published_split") != "train" for record in records):
                raise ValueError("pooled published-split selection includes non-training frames")
        else:
            require_repeat_training_side(
                protocol,
                repeat_index=selection["partition"],
                fitted_object="softening_estimate",
                chemical_system_ids=[record["chemical_system_id"] for record in records],
            )
