"""Production WCS vs an independent transcription of the published algorithm.

The reference (tests/wcs_published_reference.py) follows arXiv:2307.09291v2 eqs. 4-6
and Algorithm 1 with deterministic pruning.  Scenarios cover continuous scores,
binary, 3-valued and heavy ties, all-equal scores, a single calibration null, zero
and all selections, constant and unequal weights, and no or severe shift.
"""

from __future__ import annotations

import numpy as np
import pytest
from wcs_published_reference import unit_keys, wcs_deterministic

from e3sse.selection.conformal import (
    lexicographic_tie_free_scores,
    orient_low_barrier_scores,
    selection_outcome,
    weighted_conformalized_selection,
)

SCENARIOS = ("continuous", "binary", "three_valued", "heavy_ties", "all_equal",
             "single_null", "zero_selection", "all_selection")


def _fixture(rng, scenario):
    n, m = int(rng.integers(3, 30)), int(rng.integers(1, 20))
    if scenario == "continuous":
        v, vt = rng.normal(size=n), rng.normal(size=m)
    elif scenario == "binary":
        v, vt = rng.integers(0, 2, n).astype(float), rng.integers(0, 2, m).astype(float)
    elif scenario == "three_valued":
        v, vt = rng.integers(0, 3, n).astype(float), rng.integers(0, 3, m).astype(float)
    elif scenario == "heavy_ties":
        v, vt = np.round(rng.normal(size=n)), np.round(rng.normal(size=m))
    elif scenario == "all_equal":
        v, vt = np.zeros(n), np.zeros(m)
    elif scenario == "single_null":
        # One calibration null; the clipped score puts the non-nulls at +M.
        v = np.full(n, 1.0e6)
        v[0] = rng.normal()
        vt = rng.normal(size=m)
    elif scenario == "zero_selection":
        v, vt = rng.normal(size=n) - 5.0, rng.normal(size=m) + 5.0
    else:  # all_selection: many calibration units, every test score far below them
        v, vt = rng.normal(size=max(n, 20)) + 5.0, rng.normal(size=m) - 5.0
    shift = rng.choice(["constant", "unequal", "severe"])
    if shift == "constant":
        w, wt = np.ones(len(v)), np.ones(m)
    elif shift == "unequal":
        w, wt = rng.uniform(0.2, 3.0, len(v)), rng.uniform(0.2, 3.0, m)
    else:
        w, wt = np.exp(rng.normal(scale=2.0, size=len(v))), np.exp(rng.normal(scale=2.0, size=m))
    return v, w, vt, wt, float(rng.choice([0.05, 0.1, 0.2, 0.5]))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_conservative_tie_sensitivity_matches_the_reference_with_ties_counted(scenario):
    rng = np.random.default_rng(SCENARIOS.index(scenario))
    for _ in range(60):
        v, w, vt, wt, q = _fixture(rng, scenario)
        got = weighted_conformalized_selection(v, w, vt, wt, q, tie_rule="e3sse_conservative_le")
        expected, p, sizes = wcs_deterministic(v.tolist(), w.tolist(), vt.tolist(), wt.tolist(),
                                               q, count_ties=True)
        assert got["selected"].tolist() == expected
        assert got["pvalues"] == pytest.approx(p)
        assert got["first_step_sizes"].tolist() == sizes
        if scenario == "zero_selection":
            assert not any(expected)
        if scenario == "all_selection" and q >= 0.1 and np.ptp(w) == 0:
            assert all(expected)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_tie_free_rule_is_the_published_algorithm_on_lexicographic_scores(scenario):
    rng = np.random.default_rng(1000 + SCENARIOS.index(scenario))
    for draw in range(60):
        v, w, vt, wt, q = _fixture(rng, scenario)
        got = weighted_conformalized_selection(v, w, vt, wt, q,
                                               tie_rule="tie_free_lexicographic",
                                               **unit_keys(len(v), len(vt), str(draw)))
        keys = unit_keys(len(v), len(vt), str(draw))
        jv, jvt = lexicographic_tie_free_scores(v, vt, keys["calibration_tie_keys"],
                                                keys["test_tie_keys"])
        assert len(set(np.concatenate([jv, jvt]).tolist())) == len(jv) + len(jvt)
        expected, _, _ = wcs_deterministic(jv.tolist(), w.tolist(), jvt.tolist(), wt.tolist(), q,
                                           count_ties=False)
        assert got["selected"].tolist() == expected


def test_lexicographic_tie_keys_preserve_strict_order_exactly_and_are_reproducible():
    rng = np.random.default_rng(5)
    # Includes values a float epsilon apart, which additive jitter could reorder.
    v = np.concatenate([rng.integers(0, 4, 50).astype(float), [1.0, np.nextafter(1.0, 2.0)]])
    vt = rng.integers(0, 4, 30).astype(float)
    keys = unit_keys(len(v), len(vt), "order")
    a, at = lexicographic_tie_free_scores(v, vt, keys["calibration_tie_keys"], keys["test_tie_keys"])
    b, bt = lexicographic_tie_free_scores(v, vt, keys["calibration_tie_keys"], keys["test_tie_keys"])
    assert len(set(np.concatenate([a, at]).tolist())) == len(a) + len(at)
    assert np.array_equal(a, b) and np.array_equal(at, bt)
    pooled, jittered = np.concatenate([v, vt]), np.concatenate([a, at])
    strict = pooled[:, None] < pooled[None, :]
    assert (jittered[:, None] < jittered[None, :])[strict].all()


@pytest.mark.parametrize("tie_rule", ["tie_free_lexicographic", "e3sse_conservative_le"])
@pytest.mark.parametrize("levels", [2, 3, None])
def test_both_tie_rules_control_fdr_with_discrete_and_continuous_predictors(tie_rule, levels):
    rng = np.random.default_rng(17)
    q, fdps = 0.2, []
    for draw in range(300):
        x_cal, x_test = rng.normal(size=120), rng.normal(loc=0.5, size=40)
        y_cal = 0.5 + 0.3 * x_cal + rng.normal(scale=0.2, size=120)
        y_test = 0.5 + 0.3 * x_test + rng.normal(scale=0.2, size=40)
        hat_cal, hat_test = 0.5 + 0.25 * x_cal, 0.5 + 0.25 * x_test
        if levels is not None:  # discretize the predictor to force ties
            edges = np.quantile(hat_cal, np.linspace(0, 1, levels + 1)[1:-1])
            hat_cal, hat_test = np.digitize(hat_cal, edges) * 0.2, np.digitize(hat_test, edges) * 0.2
        v, vt = orient_low_barrier_scores(hat_cal, y_cal, hat_test, 0.5)
        result = weighted_conformalized_selection(v, np.exp(0.5 * x_cal), vt,
                                                  np.exp(0.5 * x_test), q, tie_rule=tie_rule,
                                                  **unit_keys(120, 40, str(draw)))
        fdps.append(selection_outcome(result["selected"], y_test, 0.5)["fdp"])
    fdps = np.asarray(fdps)
    assert fdps.mean() <= q + 2 * fdps.std(ddof=1) / np.sqrt(fdps.size)


def _selection_with_eq6_weight(v, w, vt, wt, q, focal_weight):
    """Reference WCS where the eq. 6 added term uses w_j (published) or w_l (ConfSelect code)."""

    m, total = len(vt), sum(w)
    p = [(sum(wi for vi, wi in zip(v, w) if vi < vt[j]) + wt[j]) / (total + wt[j]) for j in range(m)]
    sizes = []
    for j in range(m):
        aux = []
        for l in range(m):
            if l == j:
                continue
            added = (wt[j] if focal_weight == "w_j" else wt[l]) * (vt[j] < vt[l])
            aux.append((sum(wi for vi, wi in zip(v, w) if vi < vt[l]) + added) / (total + wt[j]))
        k_star = max(k for k in range(1, m + 1) if 1 + sum(a <= q * k / m for a in aux) >= k)
        sizes.append(1 + sum(a <= q * k_star / m for a in aux))
    first = [p[j] <= q * sizes[j] / m for j in range(m)]
    r_star = max([r for r in range(1, m + 1)
                  if sum(first[j] and sizes[j] <= r for j in range(m)) >= r], default=0)
    return [bool(first[j] and sizes[j] <= r_star) for j in range(m)]


def test_published_eq6_focal_weight_is_w_j_not_the_reference_code_w_l():
    """Biometrika eq. 6 adds w(X_{n+j}) 1{Vhat_{n+j} < Vhat_l}; the official ConfSelect code
    (feffee17) uses w_l.  Production must follow the published paper on cases where they differ."""

    rng = np.random.default_rng(2026)
    discriminating = 0
    for draw in range(400):
        n, m = int(rng.integers(5, 30)), int(rng.integers(2, 15))
        w, wt = rng.uniform(0.2, 3.0, n), rng.uniform(0.2, 3.0, m)
        v, vt = rng.normal(size=n), rng.normal(size=m)
        q = float(rng.choice([0.1, 0.2, 0.5]))
        published = _selection_with_eq6_weight(v, w, vt, wt, q, "w_j")
        code_variant = _selection_with_eq6_weight(v, w, vt, wt, q, "w_l")
        got = weighted_conformalized_selection(v, w, vt, wt, q,
                                               tie_rule="tie_free_lexicographic",
                                               **unit_keys(n, m, str(draw)))
        assert got["selected"].tolist() == published
        discriminating += published != code_variant
    assert discriminating > 0  # the fixture set really distinguishes the two indices


def test_tie_keys_bind_to_units_so_input_order_cannot_change_the_selection():
    rng = np.random.default_rng(31)
    v, vt = rng.integers(0, 3, 40).astype(float), rng.integers(0, 3, 15).astype(float)
    w, wt = rng.uniform(0.5, 2.0, 40), rng.uniform(0.5, 2.0, 15)
    keys = unit_keys(40, 15, "order")
    base = weighted_conformalized_selection(v, w, vt, wt, 0.3, **keys)["selected"]
    perm_c, perm_t = rng.permutation(40), rng.permutation(15)
    shuffled = weighted_conformalized_selection(
        v[perm_c], w[perm_c], vt[perm_t], wt[perm_t], 0.3,
        calibration_tie_keys=[keys["calibration_tie_keys"][i] for i in perm_c],
        test_tie_keys=[keys["test_tie_keys"][i] for i in perm_t])["selected"]
    assert shuffled.tolist() == base[perm_t].tolist()
    with pytest.raises(ValueError, match="unique"):
        weighted_conformalized_selection(v, w, vt, wt, 0.3, calibration_tie_keys=[1] * 40,
                                         test_tie_keys=keys["test_tie_keys"])
