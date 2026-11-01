"""Covariate-shift weights: estimator, diagnostics, and FDR under known/estimated/
misspecified/no weighting in a controlled synthetic shift."""

from __future__ import annotations

import numpy as np
import pytest
from wcs_published_reference import unit_keys

from e3sse.selection.conformal import (
    orient_low_barrier_scores,
    selection_outcome,
    weighted_conformalized_selection,
)
from e3sse.selection.density_ratio import (
    auc,
    fit_domain_classifier,
    predicted_q_probability,
    shift_weights,
    weight_diagnostics,
)

SHIFT = 1.0  # P: x ~ N(0, 1);  Q: x ~ N(SHIFT, 1);  true w(x) = exp(SHIFT x - SHIFT^2 / 2)


def _true_w(x):
    return np.exp(SHIFT * x - SHIFT ** 2 / 2)


def test_estimated_weights_recover_the_true_ratio_and_are_deterministic():
    rng = np.random.default_rng(0)
    xp, xq = rng.normal(size=(4000, 1)), rng.normal(loc=SHIFT, size=(4000, 1))
    model = fit_domain_classifier(xp, xq, l2=1e-3)
    assert fit_domain_classifier(xp, xq, l2=1e-3) == model
    grid = np.linspace(-1.5, 2.5, 9)[:, None]
    estimated = shift_weights(model, grid)["weights"]
    assert np.allclose(np.log(estimated), np.log(_true_w(grid[:, 0])), atol=0.15)


def test_diagnostics_report_overlap_ess_auc_and_clipping():
    rng = np.random.default_rng(1)
    xp, xq = rng.normal(size=(500, 2)), rng.normal(size=(500, 2))
    model = fit_domain_classifier(xp, xq)
    prob_p, prob_q = predicted_q_probability(model, xp), predicted_q_probability(model, xq)
    calibration = shift_weights(model, xp)["weights"]
    weights = shift_weights(model, xq)
    diagnostics = weight_diagnostics(calibration, weights["weights"], prob_p, prob_q, q=0.2)
    assert abs(diagnostics["classifier_auc"] - 0.5) < 0.06  # no shift
    assert diagnostics["calibration_effective_sample_size"] > 450
    assert diagnostics["unselectable_test_fraction"] == 0.0
    with pytest.raises(ValueError):
        auc(np.array([]), np.array([0.5]))
    with pytest.raises(ValueError, match="positive"):
        shift_weights(model, xq, clip_max=0.0)
    with pytest.raises(ValueError, match="width"):
        predicted_q_probability(model, np.ones((3, 5)))
    clipped = shift_weights(model, xq, clip_max=0.9)
    assert clipped["n_clipped"] > 0 and "guarantee interpretation changes" in clipped["interpretation"]
    assert auc(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0
    with pytest.raises(ValueError):
        fit_domain_classifier([[np.nan]], [[0.0]])


def _realized_fdr(weighting, draws=300, q=0.2):
    rng = np.random.default_rng(7)
    fdps = []
    for draw in range(draws):
        x_cal, x_test = rng.normal(size=200), rng.normal(loc=SHIFT, size=60)
        # Heteroscedastic noise: the predictor is worse where Q has more mass.
        noise = lambda x, k: rng.normal(scale=0.1 + 0.15 * np.abs(x), size=k)
        y_cal, y_test = 0.6 + 0.2 * x_cal + noise(x_cal, 200), 0.6 + 0.2 * x_test + noise(x_test, 60)
        hat_cal, hat_test = 0.6 + 0.2 * x_cal, 0.6 + 0.2 * x_test
        if weighting == "known":
            w_cal, w_test = _true_w(x_cal), _true_w(x_test)
        elif weighting == "estimated":  # fitted on an independent sample (Theorem 3.5)
            model = fit_domain_classifier(rng.normal(size=(300, 1)),
                                          rng.normal(loc=SHIFT, size=(300, 1)))
            w_cal = shift_weights(model, x_cal[:, None])["weights"]
            w_test = shift_weights(model, x_test[:, None])["weights"]
        elif weighting == "misspecified":
            w_cal, w_test = np.exp(-SHIFT * x_cal), np.exp(-SHIFT * x_test)  # wrong direction
        else:
            w_cal, w_test = np.ones(200), np.ones(60)
        v, vt = orient_low_barrier_scores(hat_cal, y_cal, hat_test, 0.5)
        selected = weighted_conformalized_selection(v, w_cal, vt, w_test, q,
                                                    **unit_keys(len(v), len(vt), str(draw)))["selected"]
        fdps.append(selection_outcome(selected, y_test, 0.5)["fdp"])
    fdps = np.asarray(fdps)
    return fdps.mean(), fdps.std(ddof=1) / np.sqrt(fdps.size)


@pytest.mark.parametrize("weighting", ["known", "estimated"])
def test_known_and_independently_estimated_weights_control_fdr(weighting):
    fdr, se = _realized_fdr(weighting)
    assert fdr <= 0.2 + 2 * se


def test_misspecified_and_absent_weights_are_reported_not_assumed_valid():
    # Recorded for the design report; no validity claim is made for these arms.
    for weighting in ("misspecified", "none"):
        fdr, se = _realized_fdr(weighting, draws=100)
        assert 0.0 <= fdr <= 1.0 and se >= 0.0


def _shift_region_bias_fdr(weighting, draws=300, q=0.2):
    """Predeclared DGP family 'unmodelled shift-region bias' (covariate shift + predictor error).

    Y = 0.8 - 0.3 x + 0.5 * 1{x > 1} + noise; the predictor misses the bump, so units with
    x > 1 look fast but are slow.  P: x ~ N(0, 1) sees few of them; Q: x ~ N(1.5, 1) many.
    """

    rng = np.random.default_rng(23)
    shift = 1.5
    fdps, powers = [], []
    for draw in range(draws):
        x_cal, x_test = rng.normal(size=300), rng.normal(loc=shift, size=80)
        truth = lambda x: 0.8 - 0.3 * x + 0.5 * (x > 1)
        y_cal = truth(x_cal) + rng.normal(scale=0.1, size=300)
        y_test = truth(x_test) + rng.normal(scale=0.1, size=80)
        hat_cal, hat_test = 0.8 - 0.3 * x_cal, 0.8 - 0.3 * x_test
        if weighting == "known":
            w_cal, w_test = np.exp(shift * x_cal), np.exp(shift * x_test)
        elif weighting == "estimated":
            model = fit_domain_classifier(rng.normal(size=(400, 1)),
                                          rng.normal(loc=shift, size=(400, 1)))
            w_cal = shift_weights(model, x_cal[:, None])["weights"]
            w_test = shift_weights(model, x_test[:, None])["weights"]
        else:
            w_cal, w_test = np.ones(300), np.ones(80)
        v, vt = orient_low_barrier_scores(hat_cal, y_cal, hat_test, 0.5)
        selected = weighted_conformalized_selection(v, w_cal, vt, w_test, q,
                                                    **unit_keys(len(v), len(vt), str(draw)))["selected"]
        outcome = selection_outcome(selected, y_test, 0.5)
        fdps.append(outcome["fdp"])
        powers.append(outcome["power"] or 0.0)
    fdps = np.asarray(fdps)
    return fdps.mean(), fdps.std(ddof=1) / np.sqrt(fdps.size), float(np.mean(powers))


@pytest.mark.parametrize("weighting", ["known", "estimated"])
def test_weighted_selection_stays_valid_under_shift_region_predictor_bias(weighting):
    fdr, se, _ = _shift_region_bias_fdr(weighting)
    assert fdr <= 0.2 + 2 * se
