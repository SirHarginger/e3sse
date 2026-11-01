"""Synthetic PES checks for SRT, applicability diagnostics and selective prediction.

DFT reference: a host spring lattice plus a migrating Li in a double well
    E_DFT = k/2 sum_host |r|^2 + h (1 - (x/a)^2)^2 + k/2 (y^2 + z^2),
with the barrier h at x = 0.  ML = s E_DFT + C + eps * delta, where
delta = exp(-x^2 / w^2) - (k_h/2) sum_host r_x^2 is a non-uniform residual: it changes
the barrier and tilts the force vectors.  eps = 0 is exact uniform softening.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from e3sse.barriers.fixed_geometry import (
    barrier,
    common_population,
    metrics,
    rho_srt,
    srt_barrier,
)
from e3sse.selection.selective_prediction import (
    aurc,
    correct_or_raw_curve,
    oracle_aurc,
    random_aurc,
    risk_coverage,
    summarize,
    summarize_defined_subset,
)
from e3sse.softening.applicability import (
    combined_component_counts,
    combined_score,
    energy_median_split,
    frame_median_rho_f,
    gate_decision,
    gate_thresholds,
    log_ratio_disagreement,
    percentile_ranks,
    rho_f,
    rho_f_dft_denominator,
    scale_uncertainty,
)
from e3sse.softening.frame_statistics import (
    bootstrap_indices,
    eq17,
    residual_fraction,
)

K, H, A, W, K_H = 1.0, 0.4, 1.0, 0.3, 0.2
N_HOST = 6


def _dft(positions: np.ndarray) -> tuple[float, np.ndarray]:
    li, host = positions[0], positions[1:]
    x = li[0]
    energy = 0.5 * K * (host ** 2).sum() + H * (1 - (x / A) ** 2) ** 2 + 0.5 * K * (li[1:] ** 2).sum()
    forces = np.zeros_like(positions)
    forces[1:] = -K * host
    forces[0, 0] = 4 * H * x * (1 - (x / A) ** 2) / A ** 2
    forces[0, 1:] = -K * li[1:]
    return float(energy), forces


def _delta(positions: np.ndarray) -> tuple[float, np.ndarray]:
    x = positions[0, 0]
    host_x = positions[1:, 0]
    energy = math.exp(-(x ** 2) / W ** 2) - 0.5 * K_H * (host_x ** 2).sum()
    forces = np.zeros_like(positions)
    forces[0, 0] = 2 * x / W ** 2 * math.exp(-(x ** 2) / W ** 2)
    forces[1:, 0] = K_H * host_x
    return energy, forces


def _ml(positions, s, eps, offset=-3.7):
    e, f = _dft(positions)
    d, g = _delta(positions)
    return s * e + offset + eps * d, s * f + eps * g


def _frames(n=40, seed=7):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        positions = rng.normal(scale=0.35, size=(N_HOST + 1, 3))
        positions[0, 0] = rng.uniform(-1.1, 1.1)
        frames.append(positions)
    return frames


def _sums(frames, s, eps):
    rows = []
    for positions in frames:
        _, x = _dft(positions)
        _, y = _ml(positions, s, eps)
        rows.append({"sxy": float((x * y).sum()), "sxx": float((x * x).sum()),
                     "syy": float((y * y).sum())})
    return rows


def _pooled(rows):
    return {k: math.fsum(r[k] for r in rows) for k in ("sxy", "sxx", "syy")}


def _path(n_images=7):
    images = []
    for x in np.linspace(-A, A, n_images):
        positions = np.zeros((N_HOST + 1, 3))
        positions[1:, 0] = np.linspace(-0.2, 0.2, N_HOST)
        positions[0, 0] = x
        images.append(positions)
    return images


def _barriers(s, eps):
    images = _path()
    dft = barrier([_dft(p)[0] for p in images])
    raw = barrier([_ml(p, s, eps)[0] for p in images])
    return dft, raw


def test_perfect_uniform_softening_is_recovered_exactly():
    s = 0.8
    pooled = _pooled(_sums(_frames(), s, 0.0))
    s_hat = eq17(pooled["sxy"], pooled["sxx"])
    assert s_hat == pytest.approx(s, rel=1e-12)
    assert rho_f(**pooled)["value"] == pytest.approx(0.0, abs=1e-7)
    dft, raw = _barriers(s, 0.0)
    assert dft == pytest.approx(H)
    assert raw == pytest.approx(s * H)  # additive ML offset cancels
    assert srt_barrier(raw, s_hat) == pytest.approx(dft, rel=1e-12)


def test_nonuniform_residual_raises_rho_f_monotonically_and_survives_srt():
    s = 0.8
    rhos, residuals = [], []
    for eps in (0.0, 0.02, 0.05, 0.1, 0.2):
        pooled = _pooled(_sums(_frames(), s, eps))
        s_hat = eq17(pooled["sxy"], pooled["sxx"])
        rhos.append(rho_f(**pooled)["value"])
        dft, raw = _barriers(s, eps)
        residuals.append(abs(srt_barrier(raw, s_hat) - dft))
    assert all(b > a for a, b in itertools.pairwise(rhos))
    assert all(b > a for a, b in itertools.pairwise(residuals))
    assert residuals[0] == pytest.approx(0.0, abs=1e-9)


def test_rho_f_identities_and_scale_invariance():
    pooled = _pooled(_sums(_frames(), 0.8, 0.1))
    value = rho_f(**pooled)["value"]
    assert value ** 2 == pytest.approx(residual_fraction(**pooled))
    # Rescaling either force field leaves rho_F unchanged; swapping DFT and ML too.
    scaled = {"sxy": 3 * pooled["sxy"], "sxx": pooled["sxx"], "syy": 9 * pooled["syy"]}
    assert rho_f(**scaled)["value"] == pytest.approx(value)
    swapped = {"sxy": pooled["sxy"], "sxx": pooled["syy"], "syy": pooled["sxx"]}
    assert rho_f(**swapped)["value"] == pytest.approx(value)
    # DFT-denominator variant is tan(theta): same ranking.
    assert rho_f_dft_denominator(**pooled) == pytest.approx(value / math.sqrt(1 - value ** 2))
    assert rho_f(sxy=0.0, sxx=0.0, syy=1.0)["reason"] == "zero_dft_force_norm"
    assert rho_f(sxy=0.0, sxx=1.0, syy=0.0)["value"] is None
    # Anti-aligned forces are not "uniform softening".
    assert rho_f(sxy=-1.0, sxx=1.0, syy=1.0)["reason"] == "nonpositive_force_projection"
    assert rho_f_dft_denominator(sxy=-1.0, sxx=1.0, syy=1.0) is None
    assert rho_f_dft_denominator(sxy=float("nan"), sxx=1.0, syy=1.0) is None


def test_dft_force_noise_inflates_rho_f_most_on_low_force_frames():
    rng = np.random.default_rng(3)
    x = rng.normal(size=3000)
    y = 0.8 * x
    noise = rng.normal(scale=0.05, size=x.size)
    low, high = 0.1 * x, 2.0 * x
    for scale_frames in (low, high):
        assert rho_f(float(((scale_frames + noise) * 0.8 * scale_frames).sum()),
                     float(((scale_frames + noise) ** 2).sum()),
                     float(((0.8 * scale_frames) ** 2).sum()))["value"] > 0
    rho_low = rho_f(float(((low + noise) * 0.8 * low).sum()), float(((low + noise) ** 2).sum()),
                    float(((0.8 * low) ** 2).sum()))["value"]
    rho_high = rho_f(float(((high + noise) * 0.8 * high).sum()), float(((high + noise) ** 2).sum()),
                     float(((0.8 * high) ** 2).sum()))["value"]
    assert rho_low > rho_high
    assert rho_f(float((x * y).sum()), float((x * x).sum()), float((y * y).sum()))["value"] < 1e-7


def test_scale_uncertainty_shrinks_with_frames_and_is_deterministic():
    def widths(n):
        rows = _sums(_frames(n=n, seed=11), 0.8, 0.1)
        pooled = _pooled(rows)
        s_hat = eq17(pooled["sxy"], pooled["sxx"])
        draws = []
        for sample in bootstrap_indices(2026, len(rows), 500):
            sxy = math.fsum(rows[i]["sxy"] for i in sample)
            sxx = math.fsum(rows[i]["sxx"] for i in sample)
            draws.append(sxy / sxx)
        return scale_uncertainty(s_hat, draws)["value"]

    assert widths(80) < widths(10)
    assert widths(20) == widths(20)
    assert scale_uncertainty(None, [0.8, 0.9])["value"] is None
    dropped = scale_uncertainty(0.8, [0.8, None, 0.9])
    assert dropped["value"] is None and dropped["n_undefined_draws"] == 1


def test_frame_median_and_disagreement_diagnostics():
    rows = _sums(_frames(), 0.8, 0.0)
    assert frame_median_rho_f(rows)["value"] == pytest.approx(0.0, abs=1e-7)
    same = log_ratio_disagreement((8.0, 10.0), (4.0, 5.0), numerator_supported=True,
                                  denominator_supported=True, label="li_host")
    assert same["value"] == pytest.approx(0.0)
    differ = log_ratio_disagreement((9.0, 10.0), (4.0, 5.0), numerator_supported=True,
                                    denominator_supported=True, label="li_host")
    assert differ["value"] == pytest.approx(math.log(0.9 / 0.8))
    unsupported = log_ratio_disagreement((9.0, 10.0), (4.0, 5.0), numerator_supported=False,
                                         denominator_supported=True, label="li_host")
    assert unsupported["value"] is None and "unsupported" in unsupported["reason"]
    low, high = energy_median_split([0.3, 0.1, 0.2, 0.2, 0.5])
    assert low == [1, 2, 3] and high == [0, 4]


def test_percentile_ranks_ties_and_combined_score_without_learning():
    assert percentile_ranks([0.2, None, 0.1, 0.2]) == [pytest.approx(5 / 6), None,
                                                       pytest.approx(1 / 3), pytest.approx(5 / 6)]
    combined = combined_score({"rho_F": [0.1, 0.3, None], "scale": [0.2, 0.1, None]})
    assert combined[0] == pytest.approx(0.75) and combined[1] == pytest.approx(0.75)
    assert combined[2] is None
    with pytest.raises(ValueError):
        combined_score({"a": [1.0], "b": [1.0, 2.0]})
    with pytest.raises(ValueError):
        percentile_ranks([0.1, float("nan")])
    assert combined_component_counts({"a": [0.1, None], "b": [None, None]}) == [1, 0]


def test_label_free_gate_uses_supported_units_and_exact_diagnostic_set():
    population = {"rho_F": [0.1, 0.2, 0.3, 0.4, None, 0.9],
                  "scale_uncertainty": [0.01, 0.02, 0.03, 0.04, 0.05, 0.9]}
    supported = [True, True, True, True, True, False]
    result = gate_thresholds(population, supported, 0.8)
    thresholds = result["thresholds"]
    assert thresholds["rho_F"] == pytest.approx(np.quantile([0.1, 0.2, 0.3, 0.4], 0.8))
    assert result["accounting"]["rho_F"] == {"n_population": 6, "n_used": 4,
                                             "n_unsupported": 1, "n_undefined_supported": 1}
    ok = {"rho_F": 0.1, "scale_uncertainty": 0.01}
    assert gate_decision(supported=True, diagnostics=ok, thresholds=thresholds) == (True, [])
    passed, reasons = gate_decision(supported=False,
                                    diagnostics={"rho_F": None, "scale_uncertainty": 0.5},
                                    thresholds=thresholds)
    assert not passed
    assert reasons == ["support_rule_failed", "rho_F_undefined", "scale_uncertainty_above_threshold"]
    # A threshold set missing a gated diagnostic can never silently gate on the rest.
    with pytest.raises(ValueError, match="exactly"):
        gate_decision(supported=True, diagnostics=ok, thresholds={"rho_F": 0.3})
    with pytest.raises(ValueError, match="exactly"):
        gate_thresholds({"rho_F": [0.1]}, [True], 0.8)
    with pytest.raises(ValueError):
        gate_thresholds(population, supported, 1.0)


def test_risk_coverage_aurc_ties_random_and_oracle():
    errors = [0.4, 0.1, 0.3, 0.2]
    perfect = [4, 1, 3, 2]
    assert aurc(perfect, errors) == pytest.approx(oracle_aurc(errors))
    assert random_aurc(errors) == pytest.approx(0.25)
    # All-tied scores equal the random baseline, independent of input order.
    assert aurc([1, 1, 1, 1], errors) == pytest.approx(random_aurc(errors))
    curve = risk_coverage([1, None, 1, 0], errors)
    assert curve["risk"][-1] == pytest.approx(np.mean(errors))
    assert curve["risk"][0] == pytest.approx(0.2)
    # Order invariance under permutation of units.
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 3, size=30).astype(float).tolist()
    errs = rng.uniform(size=30).tolist()
    perm = rng.permutation(30)
    assert aurc(scores, errs) == pytest.approx(
        aurc([scores[i] for i in perm], [errs[i] for i in perm]))
    # Monte Carlo random orderings average to the analytic baseline.
    mc = np.mean([aurc(rng.permutation(30).tolist(), errs) for _ in range(4000)])
    assert mc == pytest.approx(random_aurc(errs), rel=0.02)
    summary = summarize(perfect, errors, target_coverage=0.5)
    assert summary["normalized_gain"] == pytest.approx(1.0)
    assert summary["risk_at_target_coverage"] == pytest.approx(0.15)


def test_rho_f_ranking_beats_random_on_synthetic_heterogeneous_hops():
    """Hops with larger non-uniform residual have larger rho_F and larger SRT error."""

    rng = np.random.default_rng(5)
    scores, srt_errors, raw_errors = [], [], []
    for index, eps in enumerate(rng.uniform(0.0, 0.25, size=24)):
        pooled = _pooled(_sums(_frames(n=20, seed=100 + index), 0.8, eps))
        s_hat = eq17(pooled["sxy"], pooled["sxx"])
        dft, raw = _barriers(0.8, eps)
        scores.append(rho_f(**pooled)["value"])
        srt_errors.append(abs(srt_barrier(raw, s_hat) - dft))
        raw_errors.append(abs(raw - dft))
    assert aurc(scores, srt_errors) < random_aurc(srt_errors)
    hybrid = correct_or_raw_curve(scores, srt_errors, raw_errors)
    assert hybrid["risk"][0] == pytest.approx(np.mean(raw_errors))
    assert hybrid["risk"][-1] == pytest.approx(np.mean(srt_errors))
    # An unsupported hop keeps its raw prediction: never imputed, never "corrected".
    partial = correct_or_raw_curve([*scores, None], [*srt_errors, None], [*raw_errors, 0.3])
    assert partial["n_uncorrectable"] == 1
    assert partial["coverage"][-1] == pytest.approx(24 / 25)
    assert partial["risk"][-1] == pytest.approx((sum(srt_errors) + 0.3) / 25)
    with pytest.raises(ValueError, match="undefined score"):
        correct_or_raw_curve([0.1], [None], [0.2])
    subset = summarize_defined_subset([*scores, None], [*srt_errors, 9.0])
    assert subset["n_undefined_scores_excluded"] == 1
    assert subset["aurc"] == pytest.approx(aurc(scores, srt_errors))


def test_paired_metrics_use_one_hashed_common_population():
    reference = {"h1": 0.4, "h2": 0.6, "h3": 0.2}
    raw = {"h1": 0.32, "h2": 0.48, "h3": 0.16}
    srt = {"h1": 0.4, "h2": 0.6, "h3": None}
    population = common_population(
        {"raw": [h for h, v in raw.items() if v is not None],
         "srt": [h for h, v in srt.items() if v is not None]}, label="test")
    assert population["hop_ids"] == ["h1", "h2"] and population["n_excluded"] == 1
    m_raw = metrics(raw, reference, population, [0.5])
    m_srt = metrics(srt, reference, population, [0.5])
    assert m_raw["population_sha256"] == m_srt["population_sha256"]
    assert m_srt["mae_eV"] == pytest.approx(0.0)
    assert m_raw["signed_mean_error_eV"] < 0
    assert rho_srt(raw, srt, reference, population) == pytest.approx(1.0)
    assert srt_barrier(0.3, None) is None
    with pytest.raises(ValueError):
        metrics(srt, reference, {"hop_ids": ["h3"], "population_sha256": "x"}, [0.5])
    with pytest.raises(ValueError):
        barrier([1.0])
