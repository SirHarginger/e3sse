"""Label-budget (exactly K frames) vs full-estimate support, with leakage at every K."""

from __future__ import annotations

import numpy as np
import pytest
from protocol_fixtures import frozen_universe, with_systems

from e3sse.protocol.contracts import (
    LABEL_BUDGET_K,
    audit_softening_frame_eligibility,
    validate_softening_support,
)
from e3sse.protocol.freeze import canonical_sha256

TARGET = "mp-1_0_1_0_0_0"


def _record(index, *, hop, kind):
    record = {
        "frame_id": f"MPLiTrj_train.xyz:{index}",
        "mapping_status": "mapped_unique",
        "hop_id": hop,
        "raw_file_kind": kind,
        "partition_role": "training_or_calibration",
        "material_id": "mp-1",
        "published_split": "train",
        "origin_evidence_status": "unresolved",
        "origin_evidence": None,
    }
    if kind in {"source", "target"}:
        record.update(origin_evidence_status="unequivocal", origin_evidence="raw_archive_path")
    return record


def _material_selection():
    records = [_record(i, hop="mp-1_1_2_0_0_0", kind="band_optim_step") for i in range(70)]
    records.append(_record(70, hop=TARGET, kind="band_optim_step"))  # own NEB path
    records.append(_record(71, hop=TARGET, kind="source"))  # own endpoint relaxation
    rows = with_systems(records)
    protocol, membership = frozen_universe(rows)
    selection = audit_softening_frame_eligibility(
        rows,
        protocol=protocol,
        partition_records=membership,
        partition=0,
        target_hop_id=TARGET,
        estimator_scope="material",
        scope_evidence={"status": "verified", "scope": "material", "target": "mp-1", "source": "t"},
        allowed_partition_role="training_or_calibration",
        target_material_id="mp-1",
    )
    return protocol, selection


def _frames(selection, k, *, magnitude=0.5):
    ids = [record["frame_id"] for record in selection["eligible_records"]][:k]
    return [{"frame_id": frame_id, "dft_forces": np.full((4, 3), magnitude)} for frame_id in ids]


def _population(selection):
    return _frames(selection, len(selection["eligible_records"]))


@pytest.mark.parametrize("k", [1, 3, 5, 10, 30])
def test_every_registered_k_is_a_valid_label_budget_estimate(k):
    protocol, selection = _material_selection()
    summary = validate_softening_support(
        selection,
        protocol,
        _frames(selection, k),
        purpose="label_budget_subsample",
        k=k,
        population=_population(selection),
    )
    assert (summary["frames"], summary["k"]) == (k, k)
    assert LABEL_BUDGET_K == (1, 3, 5, 10, 30)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_undersized_sets_stay_rejected_as_full_estimates(k):
    protocol, selection = _material_selection()
    with pytest.raises(ValueError, match=">= 10 frames"):
        validate_softening_support(selection, protocol, _frames(selection, k), purpose="full_estimate")


def test_full_estimate_keeps_the_informative_force_component_rule():
    protocol, selection = _material_selection()
    weak = _frames(selection, 10, magnitude=0.05)  # every |F_DFT| below 0.1 eV/A
    with pytest.raises(ValueError, match="components"):
        validate_softening_support(selection, protocol, weak, purpose="full_estimate")
    validate_softening_support(selection, protocol, _frames(selection, 10), purpose="full_estimate")


def test_label_budget_rejects_unregistered_k_and_count_mismatch():
    protocol, selection = _material_selection()
    population = _population(selection)
    with pytest.raises(ValueError, match="one of"):
        validate_softening_support(
            selection, protocol, _frames(selection, 2), purpose="label_budget_subsample", k=2,
            population=population,
        )
    with pytest.raises(ValueError, match="exactly 5"):
        validate_softening_support(
            selection, protocol, _frames(selection, 4), purpose="label_budget_subsample", k=5,
            population=population,
        )
    with pytest.raises(ValueError, match="finite"):
        bad = _frames(selection, 1)
        bad[0]["dft_forces"] = np.full((4, 3), np.nan)
        validate_softening_support(
            selection, protocol, bad, purpose="label_budget_subsample", k=1, population=population
        )


def test_label_budget_population_is_derived_by_the_contract_not_chosen():
    protocol, selection = _material_selection()
    population = _population(selection)
    subsample = _frames(selection, 1)

    def run(frames, pop):
        return validate_softening_support(
            selection, protocol, frames, purpose="label_budget_subsample", k=1, population=pop
        )

    with pytest.raises(ValueError, match="frozen population"):
        run(subsample, None)
    # A cherry-picked subset of the eligible frames is never a valid population.
    with pytest.raises(ValueError, match="every eligible frame"):
        run(subsample, population[:60])
    with pytest.raises(ValueError, match="every eligible frame"):
        run([population[-1]], population[:-1])
    # Weak frames are dropped by the contract; too few informative frames is an error.
    n_weak = len(population) - 59
    weak = [dict(frame) for frame in population]
    for frame in weak[-n_weak:]:
        frame["dft_forces"] = np.full((4, 3), 0.01)
    with pytest.raises(ValueError, match=">= 60 informative population frames, got 59"):
        run(subsample, weak)
    one_weak = [dict(frame) for frame in population]
    one_weak[-1]["dft_forces"] = np.full((4, 3), 0.01)
    with pytest.raises(ValueError, match="outside the frozen population"):
        run([one_weak[-1]], one_weak)
    summary = run(subsample, one_weak)
    assert summary["population_frames"] == len(population) - 1
    # The population digest is deterministic, order-free, and bound to the forces.
    reference = run(subsample, population)["population_sha256"]
    assert run(subsample, list(reversed(population)))["population_sha256"] == reference
    assert summary["population_sha256"] != reference


def test_label_budget_subsample_forces_must_match_the_population():
    protocol, selection = _material_selection()
    tampered = _frames(selection, 3)
    tampered[1]["dft_forces"] = np.full((4, 3), 0.7)
    with pytest.raises(ValueError, match="differ from the population"):
        validate_softening_support(
            selection, protocol, tampered, purpose="label_budget_subsample", k=3,
            population=_population(selection),
        )


def test_full_estimate_rejects_a_label_budget_population_or_k():
    protocol, selection = _material_selection()
    frames = _frames(selection, 10)
    with pytest.raises(ValueError, match="neither a label-budget K nor a population"):
        validate_softening_support(
            selection, protocol, frames, purpose="full_estimate", population=_population(selection)
        )
    with pytest.raises(ValueError, match="neither a label-budget K nor a population"):
        validate_softening_support(selection, protocol, frames, purpose="full_estimate", k=10)


@pytest.mark.parametrize("k", [1, 3, 5, 10, 30])
def test_same_hop_leakage_is_prohibited_at_every_k(k):
    protocol, selection = _material_selection()
    eligible = {record["frame_id"] for record in selection["eligible_records"]}
    own_path = "MPLiTrj_train.xyz:70"
    assert own_path not in eligible
    assert "MPLiTrj_train.xyz:71" in eligible  # endpoint relaxation stays admissible
    leaky = [*_frames(selection, k - 1), {"frame_id": own_path, "dft_forces": np.ones((4, 3))}]
    with pytest.raises(ValueError, match="not leakage-eligible"):
        validate_softening_support(
            selection, protocol, leaky, purpose="label_budget_subsample", k=k,
            population=_population(selection),
        )
    # Smuggling the path frame into a re-hashed selection is caught by the primary guard.
    forged = {
        **selection,
        "eligible_records": [
            *selection["eligible_records"],
            {"frame_id": own_path, "hop_id": TARGET, "raw_file_kind": "band_optim_step",
             "material_id": "mp-1"},
        ],
    }
    forged["selection_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "selection_sha256"}
    )
    with pytest.raises(ValueError, match="target-hop path frame"):
        validate_softening_support(
            forged, protocol, leaky, purpose="label_budget_subsample", k=k,
            population=_population(selection),
        )


def test_support_thresholds_must_equal_the_protocols_frozen_parameters():
    import copy

    protocol, selection = _material_selection()
    drifted = copy.deepcopy(protocol)
    drifted["unresolved_items"][0]["resolution"]["parameters"]["full_estimate_min_frames"] = 5
    with pytest.raises(ValueError, match="frozen estimator_minimum_support"):
        validate_softening_support(
            selection, drifted, _frames(selection, 10), purpose="full_estimate"
        )
