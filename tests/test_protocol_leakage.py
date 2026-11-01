"""Research Design 3.6 / 5.4 frame-eligibility rule, case by case (synthetic only)."""

from __future__ import annotations

import pytest
from protocol_fixtures import frozen_universe, with_systems

from e3sse.protocol.contracts import (
    audit_softening_frame_eligibility,
    require_primary_softening_selection,
)
from e3sse.protocol.freeze import canonical_sha256

TARGET = "mp-1_0_1_0_0_0"


_COUNTER = iter(range(10**6))


def _frame(identifier, *, hop=TARGET, kind, side="training_or_calibration", **extra):
    record = {
        "id": identifier,
        "frame_id": f"MPLiTrj_{'train' if side != 'test' else 'test'}.xyz:{next(_COUNTER)}",
        "mapping_status": "mapped_unique",
        "hop_id": hop,
        "raw_file_kind": kind,
        "partition_role": side,
        "material_id": "mp-1",
        "published_split": "train" if side != "test" else "test",
        "origin_evidence_status": "unresolved",
        "origin_evidence": None,
    }
    if kind in {"source", "target"}:
        record.update(origin_evidence_status="unequivocal", origin_evidence="raw_archive_path")
    record.update(extra)
    return record


def _audit(records, *, scope="global", side="training_or_calibration", kind="primary",
           partition=0):
    material = "mp-1" if scope == "material" else None
    rows = with_systems(records)
    protocol, membership = frozen_universe(rows)
    selection = audit_softening_frame_eligibility(
        rows,
        protocol=protocol,
        partition_records=membership,
        partition=partition,
        target_hop_id=TARGET,
        estimator_scope=scope,
        scope_evidence={"status": "verified", "scope": scope, "target": material, "source": "t"},
        allowed_partition_role=side,
        target_material_id=material,
        estimator_kind=kind,
    )
    selection["_protocol"] = protocol
    return selection


def _guard(selection):
    protocol = selection.pop("_protocol")
    require_primary_softening_selection(selection, protocol)
    selection["_protocol"] = protocol


def _ids(selection):
    return [record["id"] for record in selection["eligible_records"]]


@pytest.mark.parametrize("kind", ["band_optim_step", "traj_init"])
def test_target_hop_neb_path_frame_is_prohibited(kind):
    selection = _audit([_frame("path", kind=kind)])
    assert _ids(selection) == []
    assert selection["reason_counts"] == {"excluded_target_hop_neb_path": 1}


@pytest.mark.parametrize("kind", ["source", "target"])
def test_target_hop_endpoint_relaxation_frame_is_admissible(kind):
    selection = _audit([_frame("endpoint", kind=kind)])
    assert _ids(selection) == ["endpoint"]
    assert selection["reason_counts"] == {"eligible_target_hop_endpoint_relaxation": 1}
    assert selection["eligibility_rule"] == "research_design_3.6_v1"
    _guard(selection)


@pytest.mark.parametrize("kind", ["band_optim_step", "traj_init", "source", None])
def test_other_hop_frame_of_same_structure_is_admissible(kind):
    selection = _audit([_frame("other", hop="mp-1_1_2_0_0_0", kind=kind)])
    assert _ids(selection) == ["other"]
    _guard(selection)


@pytest.mark.parametrize(
    "record",
    [
        _frame("ambiguous", hop=None, kind=None, mapping_status="ambiguous"),
        _frame("unmapped", hop=None, kind=None, mapping_status="unmapped"),
        _frame("repeated-endpoint", kind=None, mapping_status="mapped_hop_unique"),
        _frame("unproven-endpoint", kind="source", origin_evidence_status="unresolved"),
        _frame("unknown-file", kind="other"),
    ],
)
def test_unidentified_provenance_fails_closed(record):
    selection = _audit([record])
    assert _ids(selection) == []
    assert set(selection["reason_counts"]) <= {
        "excluded_unresolved_mapping",
        "excluded_target_hop_unidentified_origin",
    }


def test_own_path_is_oracle_only_and_rejected_for_primary_output():
    records = [
        _frame("path", kind="band_optim_step"),
        _frame("init", kind="traj_init"),
        _frame("endpoint", kind="source"),
        _frame("other", hop="mp-1_1_2_0_0_0", kind="band_optim_step"),
    ]
    assert _ids(_audit(records)) == ["endpoint", "other"]
    oracle = _audit(records, kind="oracle")
    assert _ids(oracle) == ["path", "init"]
    assert (oracle["analysis_label"], oracle["primary_eligible"]) == ("oracle_upper_bound", False)
    protocol = oracle.pop("_protocol")
    with pytest.raises(ValueError, match="only a primary"):
        require_primary_softening_selection(oracle, protocol)
    forged = {**oracle, "estimator_kind": "primary", "analysis_label": "primary"}
    forged["primary_eligible"] = True
    with pytest.raises(ValueError, match="modified after the leakage audit"):
        require_primary_softening_selection(forged, protocol)
    forged["selection_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "selection_sha256"}
    )
    with pytest.raises(ValueError, match="target-hop path frame"):
        require_primary_softening_selection(forged, protocol)


def test_test_side_material_scope_uses_non_path_frames_only():
    records = [
        _frame("own-endpoint", kind="target", side="test"),
        _frame("other-endpoint", hop="mp-1_1_2_0_0_0", kind="source", side="test"),
        _frame("other-path", hop="mp-1_1_2_0_0_0", kind="band_optim_step", side="test"),
        _frame("other-unknown", hop="mp-1_1_2_0_0_0", kind=None, side="test"),
    ]
    selection = _audit(records, scope="material", side="test")
    assert _ids(selection) == ["own-endpoint", "other-endpoint"]
    assert selection["reason_counts"]["excluded_test_side_path_frame"] == 1
    assert selection["reason_counts"]["excluded_test_side_unidentified_origin"] == 1
    _guard(selection)
    for scope in ("global", "family", "element"):
        with pytest.raises(ValueError, match="calibration side"):
            _audit(records, scope=scope, side="test")


def test_eligibility_facts_and_roles_come_from_the_frozen_membership():
    rows = with_systems([_frame("x", hop="mp-1_1_2_0_0_0", kind="source")])
    protocol, membership = frozen_universe(rows)
    kwargs = {
        "protocol": protocol,
        "partition_records": membership,
        "partition": 0,
        "target_hop_id": TARGET,
        "estimator_scope": "global",
        "scope_evidence": {"status": "verified", "scope": "global", "target": None, "source": "t"},
        "allowed_partition_role": "training_or_calibration",
    }
    # A caller cannot relabel a frame's system, kind, or hop.
    for field, value in (("chemical_system_id", "Li-S"), ("raw_file_kind", "target"),
                         ("hop_id", TARGET)):
        with pytest.raises(ValueError, match="contradicts the frozen membership"):
            audit_softening_frame_eligibility([{**rows[0], field: value}], **kwargs)
    with pytest.raises(ValueError, match="frozen membership"):
        audit_softening_frame_eligibility(rows, **{**kwargs, "partition_records": membership[:0]})
    unknown = audit_softening_frame_eligibility([{"frame_id": "absent.xyz:0"}], **kwargs)
    assert unknown["reason_counts"] == {"excluded_not_in_frozen_membership": 1}
    with pytest.raises(ValueError, match="not in the frozen partition manifest"):
        audit_softening_frame_eligibility(rows, **{**kwargs, "partition": 7})
    # The guard re-derives the partition assignment from the protocol.
    selection = audit_softening_frame_eligibility(rows, **kwargs)
    require_primary_softening_selection(selection, protocol)
    other = {**protocol, "protocol_id": "e" * 64}
    with pytest.raises(ValueError, match="different protocol"):
        require_primary_softening_selection(selection, other)


def test_published_partition_uses_per_frame_split_and_seals_validation():
    records = [
        _frame("train-frame", hop="mp-1_1_2_0_0_0", kind="source"),
        _frame("test-frame", hop="mp-1_1_2_0_0_0", kind="source", side="test"),
        _frame("val-frame", hop="mp-1_1_2_0_0_0", kind="source", published_split="val"),
    ]
    selection = _audit(records, partition="published")
    assert _ids(selection) == ["train-frame"]
    assert selection["reason_counts"] == {
        "eligible_other_hop": 1,
        "excluded_partition_role": 1,
        "excluded_unassigned_partition_role": 1,
    }
    _guard(selection)
