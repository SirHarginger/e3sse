"""Conformal selection: orientation, edge cases, BH/BY, frozen draws, FDR and MC s.e."""

from __future__ import annotations

import numpy as np
import pytest
from wcs_published_reference import unit_keys

from e3sse.protocol.partitions import analysis_seed
from e3sse.selection.conformal import (
    benjamini_hochberg,
    benjamini_yekutieli,
    conformal_pvalues,
    draw_one_hop_per_system,
    hop_rank,
    orient_low_barrier_scores,
    realized_fdr,
    selection_outcome,
    study4_draw_seed,
    weighted_conformal_pvalues,
    weighted_conformalized_selection,
)

C = 0.5


def test_low_predicted_barrier_gets_small_pvalue_and_ties_are_conservative():
    cal_hat = [0.2, 0.4, 0.6, 0.8]
    cal_y = [0.6, 0.7, 0.9, 0.3]  # nulls (Y >= c) at predictions 0.2, 0.4, 0.6
    p = conformal_pvalues(cal_hat, cal_y, [0.1, 0.4, 0.7], C)
    assert p.tolist() == pytest.approx([1 / 5, 3 / 5, 4 / 5])
    assert np.all(p >= 1 / 5) and np.all(p <= 1)
    # Non-null calibration units never enter.
    assert conformal_pvalues([0.1], [0.2], [0.0], C)[0] == pytest.approx(1 / 2)
    with pytest.raises(ValueError):
        conformal_pvalues([0.1, np.nan], [0.6, 0.6], [0.1], C)


def test_bh_empty_all_and_step_up_selection():
    assert not benjamini_hochberg([0.5, 0.9], 0.1).any()
    assert benjamini_hochberg([0.001, 0.002], 0.1).all()
    # Step-up: p_(3)=0.06 <= 0.1*3/4 admits the smaller ones too.
    assert benjamini_hochberg([0.04, 0.05, 0.06, 0.9], 0.1).tolist() == [True, True, True, False]
    assert benjamini_hochberg([], 0.1).size == 0
    by = benjamini_yekutieli([0.04, 0.05, 0.06, 0.9], 0.1)
    assert by.sum() <= 3


def test_granularity_constraint_blocks_small_calibration_sets():
    # With n = 9 calibration units every p >= 0.1, so at alpha = 0.1 and m = 20
    # nothing can be selected unless m / (alpha (n + 1)) = 20 units reach p = 0.1.
    p = conformal_pvalues(np.linspace(0.6, 1.0, 9), np.full(9, 0.9), np.linspace(0, 1, 20), C)
    assert p.min() == pytest.approx(0.1)
    assert not benjamini_hochberg(p, 0.1).any()


def test_selection_outcome_counts_fdp_and_power():
    record = selection_outcome([True, True, False, False], [0.2, 0.6, 0.1, 0.9], C)
    assert record == {"selection_size": 2, "false_discoveries": 1, "true_discoveries": 1,
                      "fdp": 0.5, "power": 0.5, "n_test": 4, "n_fast_test": 2}
    assert selection_outcome([False, False], [0.2, 0.9], C)["fdp"] == 0.0


def test_frozen_one_hop_per_system_draw_is_deterministic_and_order_free():
    hops = {"Li-O": ["h3", "h1", "h2"], "Li-S": ["h9"]}
    seed = study4_draw_seed(2026, 0, 0)
    assert seed == analysis_seed(2026, "study4_hop_draw", 0, 0)
    drawn = draw_one_hop_per_system(hops, seed)
    assert drawn["Li-S"] == "h9"
    assert drawn["Li-O"] == min(["h1", "h2", "h3"], key=lambda h: hop_rank(seed, h))
    assert draw_one_hop_per_system({"Li-O": ["h2", "h1", "h3"], "Li-S": ["h9"]}, seed) == drawn
    with pytest.raises(ValueError):
        draw_one_hop_per_system({"Li-O": []}, seed)


def test_realized_fdr_and_mc_se_follow_the_frozen_definition():
    fdp = {p: [0.1 * (p % 2), 0.1 * (p % 2)] for p in range(10)}
    summary = realized_fdr(fdp, 0.1)
    means = np.array([0.1 * (p % 2) for p in range(10)])
    assert summary["realized_fdr"] == pytest.approx(means.mean())
    assert summary["mc_se"] == pytest.approx(means.std(ddof=1) / np.sqrt(10))
    assert summary["g4_pass"]
    with pytest.raises(ValueError):
        realized_fdr({0: [0.1]}, 0.1)


def test_fdr_is_controlled_on_exchangeable_synthetic_data_for_any_predictor():
    rng = np.random.default_rng(2026)
    for noise in (0.05, 0.5):  # predictor quality changes power, never validity
        fdps = {}
        for partition in range(10):
            values = []
            for _ in range(200):
                y = rng.uniform(0.0, 1.0, size=300)
                hat = y + rng.normal(scale=noise, size=y.size)
                cal, test = slice(0, 150), slice(150, 300)
                p = conformal_pvalues(hat[cal], y[cal], hat[test], C)
                values.append(selection_outcome(benjamini_hochberg(p, 0.1), y[test], C)["fdp"])
            fdps[partition] = values
        summary = realized_fdr(fdps, 0.1)
        assert summary["realized_fdr"] <= 0.1 + 2 * summary["mc_se"]


def test_weighted_pvalues_reduce_to_unweighted_conformal_without_shift():
    rng = np.random.default_rng(1)
    cal_hat, cal_y = rng.uniform(size=50), rng.uniform(size=50)
    test_hat = rng.uniform(size=20)
    v, vt = orient_low_barrier_scores(cal_hat, cal_y, test_hat, C)
    weighted = weighted_conformal_pvalues(v, np.ones(50), vt, np.ones(20))
    assert weighted == pytest.approx(conformal_pvalues(cal_hat, cal_y, test_hat, C))
    # Fast predicted barriers get small p-values (orientation).
    assert np.corrcoef(test_hat, weighted)[0, 1] > 0.5


def test_wcs_rejects_invalid_inputs_and_handles_empty_selection():
    with pytest.raises(ValueError):
        weighted_conformalized_selection([0.1], [0.0], [0.1], [1.0], 0.1, **unit_keys(1, 1))
    with pytest.raises(ValueError):
        weighted_conformalized_selection([0.1], [1.0], [0.1], [1.0], 1.0, **unit_keys(1, 1))
    with pytest.raises(ValueError, match="tie keys"):
        weighted_conformalized_selection([0.1], [1.0], [0.1], [1.0], 0.1)
    with pytest.raises(ValueError, match="tie_rule"):
        weighted_conformalized_selection([0.1], [1.0], [0.1], [1.0], 0.1, tie_rule="strict")
    result = weighted_conformalized_selection(np.zeros(5), np.ones(5), np.full(3, -1.0),
                                              np.ones(3), 0.1, **unit_keys(5, 3))
    assert not result["selected"].any()


def test_wcs_controls_fdr_under_known_covariate_shift():
    rng = np.random.default_rng(7)
    q, fdps, sizes = 0.2, [], []
    for draw in range(300):
        x_cal = rng.normal(size=200)
        x_test = rng.normal(loc=0.7, size=60)
        y_cal = 0.5 + 0.3 * x_cal + rng.normal(scale=0.2, size=200)
        y_test = 0.5 + 0.3 * x_test + rng.normal(scale=0.2, size=60)
        hat_cal, hat_test = 0.5 + 0.25 * x_cal, 0.5 + 0.25 * x_test  # fixed predictor
        # True density ratio N(0.7,1)/N(0,1) evaluated at x, up to a constant.
        w_cal, w_test = np.exp(0.7 * x_cal), np.exp(0.7 * x_test)
        v, vt = orient_low_barrier_scores(hat_cal, y_cal, hat_test, C)
        selected = weighted_conformalized_selection(v, w_cal, vt, w_test, q,
                                                    **unit_keys(200, 60, str(draw)))["selected"]
        outcome = selection_outcome(selected, y_test, C)
        fdps.append(outcome["fdp"])
        sizes.append(outcome["selection_size"])
    fdps = np.asarray(fdps)
    assert np.mean(sizes) > 1  # control is not vacuous
    assert fdps.mean() <= q + 2 * fdps.std(ddof=1) / np.sqrt(fdps.size)
    assert (fdps >= 0).all()


def test_wcs_stays_valid_with_tied_discrete_predictions():
    # All units null; a 3-valued predictor makes calibration-test ties ubiquitous.
    rng = np.random.default_rng(3)
    fdps = []
    for _ in range(200):
        cal_hat, test_hat = rng.choice([0.2, 0.4, 0.6], 100), rng.choice([0.2, 0.4, 0.6], 40)
        cal_y, test_y = rng.uniform(0.6, 1.0, 100), rng.uniform(0.6, 1.0, 40)
        v, vt = orient_low_barrier_scores(cal_hat, cal_y, test_hat, C)
        selected = weighted_conformalized_selection(v, np.ones(100), vt, np.ones(40), 0.2,
                                                    tie_rule="e3sse_conservative_le")
        fdps.append(selection_outcome(selected["selected"], test_y, C)["fdp"])
    assert np.mean(fdps) <= 0.2
