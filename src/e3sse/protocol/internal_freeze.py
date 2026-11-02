"""The reliability-gated internal-freeze protocol, layered on an immutable parent protocol.

Lineage is by exact identity, never by a sequence number.  The internal-freeze
protocol records ``parent_protocol_id`` and ``parent_protocol_sha256`` (the SHA-256
of the parent's exact ``protocol.json`` bytes) for the registration-gated historical
protocol.  It inherits every parent rule (datasets, partitions, leakage, units,
Study 1/2a/4/5 contracts, statistical plan) by reference, adds amendments, and
records ``supersedes_protocol_id`` for governance.  Its own ``protocol_id`` is
content-derived.  The parent protocol, its ID, its files and its OSF-registration
semantics are never modified.

Governance: ``internal_version_controlled_freeze``.  No external registration is
required at runtime, and pre-registration is not claimed.  The freeze is the content
hash plus a reviewed, clean Git commit.  A production materialization is ``frozen``
only when no blocking open item remains.  Until then it is ``candidate_unfrozen`` and
grants no outcome access.

Second seal: non-training outcomes need the parent pilot amendment plus a hashed
internal pilot record binding that amendment to this protocol.  The record replaces
the parent's "registered OSF update".  A pilot amendment may never override a value
frozen by an amendment here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from .contracts import (
    validate_pilot_amendment,
    validate_protocol_freeze_for_outcome_access,
)
from .registration import protocol_file_sha256

INTERNAL_FREEZE_SCHEMA = "e3sse.internal-freeze-protocol"
INTERNAL_FREEZE_CONFIG_SCHEMA = "e3sse.internal-freeze-protocol.config"
INTERNAL_PILOT_RECORD_SCHEMA = "e3sse.internal-freeze-pilot-record"
PROTOCOL_NAME = "E3-SSE reliability-gated internal-freeze protocol"
PARENT_PROTOCOL_NAME = "E3-SSE registration-gated historical protocol"
# Pilot amendments may fix Study 1/2 settings (parent rule) but never a value that an
# internal-freeze amendment already freezes.
FROZEN_AMENDMENT_PILOT_KEY_PREFIXES = (
    "study2_applicability_", "study2_deployment_gate_", "study2_selective_",
    "study2_representation_", "study2_label_budget_downstream_", "study2_prospective_",
)
GOVERNANCE = {
    "governance_mode": "internal_version_controlled_freeze",
    "external_registration_required": False,
    "preregistration_claimed": False,
}
REQUIRED_AMENDMENTS = (
    "transfer_benchmark",
    "applicability",
    "selective_prediction",
    "deployment_gate",
    "endpoints",
    "label_budget_downstream",
    "representation",
    "selection",
    "prospective",
)
MATERIALIZATION_MODES = ("production", "synthetic")
BLOCKER_CLASSES = ("local_engineering", "server_evidence", "owner_decision", "documentation")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class InternalFreezeError(ValueError):
    """An internal-freeze protocol or its configuration would weaken a frozen invariant."""


def _content_id(value: Mapping[str, Any], id_field: str) -> str:
    content = {key: item for key, item in value.items() if key != id_field}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _exact(actual: Any, expected: Any) -> bool:
    """Equal and of the same type, so ``0`` never passes for ``False``."""

    return type(actual) is type(expected) and actual == expected


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InternalFreezeError(message)


def _invariant_violations(amendments: Mapping[str, Any]) -> list[str]:
    """Scientific invariants that no internal-freeze configuration may relax."""

    problems = []
    gate = amendments["deployment_gate"]
    if gate.get("barrier_labels_used") is not False:
        problems.append("deployment_gate.barrier_labels_used must be false")
    level = gate.get("quantile_level")
    if not isinstance(level, (int, float)) or isinstance(level, bool) or not 0 < level < 1:
        problems.append("deployment_gate.quantile_level must be in (0, 1)")
    combined = amendments["applicability"].get("diagnostics", {}).get("combined", {})
    if combined.get("learned_weights") is not False:
        problems.append("applicability combined score may not use learned weights")
    projections = amendments["representation"].get("projections", {})
    if projections.get("quantitative_use") != "forbidden":
        problems.append("representation projections must be forbidden for quantitative use")
    methods = amendments["transfer_benchmark"].get("methods", {})
    oracle = methods.get("B6_oracle_own_path", {})
    if oracle.get("role") != "oracle_upper_bound_never_primary":
        problems.append("B6 own-path calibration must stay an oracle, never primary")
    if "transition_state_image_rule" not in amendments["transfer_benchmark"]:
        problems.append("transfer_benchmark must state the transition-state image rule")
    framings = amendments["selective_prediction"].get("framings", {})
    if "selective_primary" not in framings:
        problems.append("selective_prediction must define the selective_primary framing")
    return problems


def validate_internal_freeze_config(config: Any) -> None:
    """Reject a internal-freeze configuration that is incomplete or weakens an invariant."""

    _require(isinstance(config, Mapping), "internal-freeze config must be an object")
    _require(config.get("schema_version") == INTERNAL_FREEZE_CONFIG_SCHEMA, "unsupported internal-freeze config")
    parent = config.get("parent_protocol")
    _require(isinstance(parent, Mapping), "internal-freeze config needs parent_protocol")
    for field in ("protocol_id", "protocol_sha256"):
        _require(_SHA256.fullmatch(str(parent.get(field))) is not None, f"parent {field} invalid")
    _require(config.get("protocol_name") == PROTOCOL_NAME, f"protocol_name must be {PROTOCOL_NAME!r}")
    governance = config.get("governance")
    _require(isinstance(governance, Mapping), "internal-freeze config needs governance")
    for key, value in GOVERNANCE.items():
        _require(_exact(governance.get(key), value), f"governance.{key} must be {value!r}")
    _require(bool(governance.get("deviation_statement")), "governance needs a deviation_statement")
    amendments = config.get("amendments")
    _require(isinstance(amendments, Mapping), "internal-freeze config needs amendments")
    missing = [name for name in REQUIRED_AMENDMENTS if not isinstance(amendments.get(name), Mapping)]
    _require(not missing, f"internal-freeze config lacks amendments {missing}")
    problems = _invariant_violations(amendments)
    _require(not problems, "; ".join(problems))
    items = config.get("open_items")
    _require(isinstance(items, list), "open_items must be a list")
    ids = []
    for item in items:
        _require(isinstance(item, Mapping) and bool(item.get("id")), "open item needs an id")
        _require(isinstance(item.get("blocking"), bool), f"open item {item.get('id')} needs blocking")
        _require(item.get("blocker_class") in BLOCKER_CLASSES,
                 f"open item {item.get('id')} needs blocker_class in {BLOCKER_CLASSES}")
        if not item["blocking"]:
            # Clearing a blocker must leave a trace: a resolution record or a stated reason.
            resolution = item.get("resolution")
            resolved = (isinstance(resolution, Mapping) and resolution.get("status") == "resolved"
                        and bool(resolution.get("rule")) and bool(resolution.get("evidence")))
            _require(resolved or bool(item.get("non_blocking_reason")),
                     f"non-blocking open item {item['id']} needs a resolution "
                     "{status, rule, evidence} or a non_blocking_reason")
        ids.append(item["id"])
    _require(len(ids) == len(set(ids)), "duplicate open item ids")


def local_engineering_blockers(config: Mapping[str, Any]) -> list[str]:
    """Blocking local items; the engineering checkpoint requires none."""

    return sorted(i["id"] for i in config["open_items"]
                  if i["blocking"] and i.get("blocker_class") == "local_engineering")


def internal_freeze_blockers(config: Mapping[str, Any]) -> list[str]:
    """Blocking open items; an internal-freeze protocol with any of them is never frozen."""

    return sorted(item["id"] for item in config["open_items"] if item["blocking"])


def build_internal_freeze_protocol(
    parent: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    git_sha: str,
    materialization_mode: str = "production",
) -> dict[str, Any]:
    """Internal-freeze protocol content bound to exactly this frozen parent protocol."""

    validate_protocol_freeze_for_outcome_access(parent)
    validate_internal_freeze_config(config)
    _require(materialization_mode in MATERIALIZATION_MODES, "unknown materialization mode")
    _require(_GIT_SHA.fullmatch(str(git_sha)) is not None, "internal-freeze protocol needs a 40-hex Git SHA")
    expected = config["parent_protocol"]
    _require(parent["protocol_id"] == expected["protocol_id"], "parent protocol_id mismatch")
    _require(
        protocol_file_sha256(parent) == expected["protocol_sha256"],
        "parent protocol.json SHA-256 mismatch",
    )
    blockers = internal_freeze_blockers(config)
    frozen = materialization_mode == "production" and not blockers
    internal_protocol = {
        "protocol_schema": INTERNAL_FREEZE_SCHEMA,
        "protocol_name": config["protocol_name"],
        "parent_protocol_id": parent["protocol_id"],
        "parent_protocol_sha256": expected["protocol_sha256"],
        "parent_protocol_name": PARENT_PROTOCOL_NAME,
        "parent_protocol_git_sha": parent["git_sha"],
        "supersedes_protocol_id": parent["protocol_id"],
        "supersedes_protocol_sha256": expected["protocol_sha256"],
        "inheritance": "every parent rule applies unless an amendment explicitly extends it",
        "git_sha": git_sha,
        "governance": dict(config["governance"]),
        "outcome_access_declaration": config.get("outcome_access_declaration"),
        "public_stage_names": config.get("public_stage_names"),
        "amendments": dict(config["amendments"]),
        "open_items": list(config["open_items"]),
        "declared_scope_revisions": list(config.get("declared_scope_revisions", [])),
        "freeze_blockers": blockers,
        "materialization_mode": materialization_mode,
        "freeze_status": "frozen" if frozen else "candidate_unfrozen",
        "evaluation_seal": {"state": "sealed"},
    }
    internal_protocol["protocol_id"] = _content_id(internal_protocol, "protocol_id")
    return internal_protocol


def validate_internal_freeze_protocol(internal_protocol: Any, parent: Mapping[str, Any]) -> None:
    """Outcome-access validation: a frozen, production internal-freeze protocol of exactly this parent."""

    validate_protocol_freeze_for_outcome_access(parent)
    _require(isinstance(internal_protocol, Mapping), "internal-freeze protocol must be an object")
    _require(internal_protocol.get("protocol_schema") == INTERNAL_FREEZE_SCHEMA, "unsupported internal-freeze protocol")
    _require(_SHA256.fullmatch(str(internal_protocol.get("protocol_id"))) is not None, "invalid protocol_id")
    _require(internal_protocol.get("protocol_id") == _content_id(internal_protocol, "protocol_id"),
             "internal-freeze protocol content hash mismatch")
    _require(internal_protocol.get("parent_protocol_id") == parent["protocol_id"],
             "internal-freeze protocol is bound to a different parent protocol")
    _require(internal_protocol.get("parent_protocol_sha256") == protocol_file_sha256(parent),
             "internal-freeze protocol is bound to a different parent protocol.json")
    governance = internal_protocol.get("governance") or {}
    for key, value in GOVERNANCE.items():
        _require(_exact(governance.get(key), value), f"internal-freeze governance.{key} must be {value!r}")
    _require(_GIT_SHA.fullmatch(str(internal_protocol.get("git_sha"))) is not None, "invalid internal-freeze protocol Git SHA")
    _require(internal_protocol.get("materialization_mode") == "production"
             and internal_protocol.get("freeze_status") == "frozen"
             and internal_protocol.get("freeze_blockers") == [],
             "internal-freeze protocol is not a frozen production materialization")
    _require((internal_protocol.get("evaluation_seal") or {}).get("state") == "sealed",
             "internal-freeze evaluation seal is not present")
    # Defence in depth: a hand-built dict with a matching hash must still be a valid freeze.
    amendments = internal_protocol.get("amendments")
    _require(isinstance(amendments, Mapping)
             and all(isinstance(amendments.get(n), Mapping) for n in REQUIRED_AMENDMENTS),
             "internal-freeze protocol lacks required amendments")
    problems = _invariant_violations(amendments)
    _require(not problems, "; ".join(problems))
    items = internal_protocol.get("open_items")
    _require(isinstance(items, list), "internal-freeze protocol lacks open_items")
    _require(sorted(i.get("id") for i in items if i.get("blocking", True)) == [],
             "internal-freeze protocol retains blocking open items")


def build_internal_pilot_record(
    internal_protocol: Mapping[str, Any], parent: Mapping[str, Any], pilot_amendment: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the training-only pilot amendment to this internal-freeze protocol (replaces the OSF update)."""

    validate_internal_freeze_protocol(internal_protocol, parent)
    validate_pilot_amendment(pilot_amendment, parent)
    _reject_frozen_amendment_overrides(pilot_amendment)
    record = {
        "schema_version": INTERNAL_PILOT_RECORD_SCHEMA,
        "internal_freeze_protocol_id": internal_protocol["protocol_id"],
        "parent_protocol_id": parent["protocol_id"],
        "pilot_amendment_id": pilot_amendment["amendment_id"],
    }
    record["record_id"] = _content_id(record, "record_id")
    return record


def _reject_frozen_amendment_overrides(pilot_amendment: Mapping[str, Any]) -> None:
    keys = [*pilot_amendment.get("frozen_thresholds", {}), *pilot_amendment.get("mde_analysis", {})]
    clashing = [key for key in keys if str(key).startswith(FROZEN_AMENDMENT_PILOT_KEY_PREFIXES)]
    _require(not clashing, f"pilot amendment may not override internal-freeze settings: {clashing}")


def validate_internal_pilot_record(
    record: Any,
    internal_protocol: Mapping[str, Any],
    parent: Mapping[str, Any],
    pilot_amendment: Mapping[str, Any],
) -> None:
    validate_pilot_amendment(pilot_amendment, parent)
    _reject_frozen_amendment_overrides(pilot_amendment)
    _require(isinstance(record, Mapping), "internal pilot record must be an object")
    _require(record.get("schema_version") == INTERNAL_PILOT_RECORD_SCHEMA,
             "unsupported internal pilot record")
    _require(record.get("internal_freeze_protocol_id") == internal_protocol.get("protocol_id"),
             "internal pilot record is bound to a different internal-freeze protocol")
    _require(record.get("parent_protocol_id") == parent["protocol_id"],
             "internal pilot record is bound to a different parent protocol")
    _require(record.get("pilot_amendment_id") == pilot_amendment.get("amendment_id"),
             "internal pilot record names a different pilot amendment")
    _require(record.get("record_id") == _content_id(record, "record_id"),
             "internal pilot record content hash mismatch")


PRESPECIFIED_PURPOSE = "prespecified_analysis"


def internal_freeze_provenance(internal_protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Fields every outcome-access record writes under internal-freeze governance."""

    return {
        "governance": f"internal_freeze:{internal_protocol['protocol_id']}",
        "governance_mode": GOVERNANCE["governance_mode"],
        "analysis_purpose_label": PRESPECIFIED_PURPOSE,
        "preregistration_claimed": False,
    }
