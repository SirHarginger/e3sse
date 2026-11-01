"""Conformal selection of fast Li hops with FDR control (Research Design 3.8, 5.9).

Frozen parent rules (``statistical_plan``, ``study_4_contract``):

* Null H0j: Y_{n+j} >= c (the hop is not fast).  Jin-Candes clipped monotone score,
  oriented for small outcomes, with the deterministic conservative tie rule U_j = 1
  (Eq. 20):

      p_j = (1 + #{i : Y_i >= c, Yhat_i <= Yhat_{n+j}}) / (n + 1).

  A small predicted barrier gives a small p-value.  Only calibration nulls count.
* Benjamini-Hochberg selection (Eq. 21).  Benjamini-Yekutieli is a sensitivity
  analysis.
* Study 4 draws one hop per chemical system.  Within each system, eligible hop IDs are
  sorted by SHA256('e3sse.study4.hop-rank.v1' NUL seed NUL hop_id) and the first is
  taken.  Seeds are ``analysis_seed(base, 'study4_hop_draw', partition, draw)``.
* FDP_{p,d} = |R ∩ H0| / max(|R|, 1); FDR_p = mean_d FDP_{p,d}; realised FDR =
  mean_p FDR_p; MC s.e. = SD_p(FDR_p) / sqrt(P) with ddof = 1.  G4 passes iff
  realised FDR <= alpha + 2 MC s.e. (``g4_monte_carlo_standard_error_definition``).

The predictor values Yhat must be fixed without calibration or test outcomes.  That
is the caller's contract (``study4_predictor_calibration_independence``).  This
module never fits a predictor.  Weighted conformalized selection for the
prospective pool (below) follows Jin and Candes, Algorithm 1, read in full.  Fitting
the density-ratio weights is not done here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from e3sse.protocol.partitions import analysis_seed

HOP_RANK_DOMAIN = "e3sse.study4.hop-rank.v1"


def conformal_pvalues(
    calibration_predictions: Sequence[float],
    calibration_outcomes: Sequence[float],
    test_predictions: Sequence[float],
    threshold: float,
) -> np.ndarray:
    """Eq. 20 with U_j = 1 (deterministic, conservative)."""

    cal_hat = np.asarray(calibration_predictions, dtype=np.float64)
    cal_y = np.asarray(calibration_outcomes, dtype=np.float64)
    test_hat = np.asarray(test_predictions, dtype=np.float64)
    if cal_hat.shape != cal_y.shape or cal_hat.ndim != 1:
        raise ValueError("calibration predictions and outcomes must be 1-D and aligned")
    for name, array in (("calibration predictions", cal_hat), ("calibration outcomes", cal_y),
                        ("test predictions", test_hat)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
    null_hat = np.sort(cal_hat[cal_y >= threshold])
    counts = np.searchsorted(null_hat, test_hat, side="right")  # #{null Yhat_i <= Yhat_j}
    return (1.0 + counts) / (cal_hat.size + 1.0)


def benjamini_hochberg(pvalues: Sequence[float], alpha: float) -> np.ndarray:
    """Boolean selection mask of the BH step-up rule (Eq. 21)."""

    p = np.asarray(pvalues, dtype=np.float64)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.sort(p)
    passing = np.nonzero(order <= alpha * np.arange(1, m + 1) / m)[0]
    if passing.size == 0:
        return np.zeros(m, dtype=bool)
    k_max = int(passing[-1]) + 1
    return p <= alpha * k_max / m


def benjamini_yekutieli(pvalues: Sequence[float], alpha: float) -> np.ndarray:
    """BY sensitivity: BH at alpha / H_m (arbitrary dependence)."""

    m = len(pvalues)
    harmonic = math.fsum(1.0 / k for k in range(1, m + 1)) if m else 1.0
    return benjamini_hochberg(pvalues, alpha / harmonic) if m else np.zeros(0, dtype=bool)


def selection_outcome(selected: Sequence[bool], test_outcomes: Sequence[float],
                      threshold: float) -> dict[str, Any]:
    """Per-draw record: false/true discoveries, selection size, FDP."""

    mask = np.asarray(selected, dtype=bool)
    y = np.asarray(test_outcomes, dtype=np.float64)
    if mask.shape != y.shape:
        raise ValueError("selection mask and outcomes must be aligned")
    false = int((mask & (y >= threshold)).sum())
    true = int((mask & (y < threshold)).sum())
    size = int(mask.sum())
    fast = int((y < threshold).sum())
    return {
        "selection_size": size,
        "false_discoveries": false,
        "true_discoveries": true,
        "fdp": false / max(size, 1),
        "power": true / fast if fast else None,
        "n_test": int(y.size),
        "n_fast_test": fast,
    }


def hop_rank(seed: int, hop_id: str) -> str:
    return hashlib.sha256(f"{HOP_RANK_DOMAIN}\0{seed}\0{hop_id}".encode()).hexdigest()


def draw_one_hop_per_system(hops_by_system: Mapping[str, Sequence[str]], seed: int) -> dict[str, str]:
    """Frozen Study-4 draw: the lowest SHA rank within each chemical system."""

    drawn = {}
    for system in sorted(hops_by_system):
        hops = sorted({str(h) for h in hops_by_system[system]})
        if not hops:
            raise ValueError(f"chemical system {system!r} has no eligible hop")
        drawn[system] = min(hops, key=lambda hop: (hop_rank(seed, hop), hop))
    return drawn


def study4_draw_seed(base_seed: int, partition: int, draw: int) -> int:
    return analysis_seed(base_seed, "study4_hop_draw", partition, draw)


def realized_fdr(fdp_by_partition: Mapping[Any, Sequence[float]], alpha: float) -> dict[str, Any]:
    """Frozen G4 summary: partition means, their mean, MC s.e. over partitions."""

    if len(fdp_by_partition) < 2:
        raise ValueError("MC s.e. needs at least two partitions")
    per_partition = {}
    for partition, values in sorted(fdp_by_partition.items(), key=lambda kv: str(kv[0])):
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"partition {partition!r} needs finite FDP draws")
        per_partition[partition] = float(array.mean())
    means = np.asarray(list(per_partition.values()))
    fdr = float(means.mean())
    mc_se = float(means.std(ddof=1) / math.sqrt(means.size))
    return {
        "realized_fdr": fdr,
        "mc_se": mc_se,
        "fdr_by_partition": per_partition,
        "alpha": alpha,
        "g4_pass": fdr <= alpha + 2.0 * mc_se,
        "n_partitions": int(means.size),
    }


# --------------------------------------------------------------------------- weighted
#
# Weighted Conformalized Selection (WCS), Jin and Candes, "Model-free selective
# inference under covariate shift via weighted conformal p-values" (arXiv:2307.09291v2,
# Algorithm 1, eqs. 5-6 and 9; Biometrika 113(1), asaf066, 2026).  The paper selects
# units with LARGE responses (null: Y <= c).  E3-SSE selects LOW barriers (null:
# Y >= c), so orient_low_barrier_scores maps Y' = -Y, c' = -c and uses the clipped
# monotone score V(x, y') = M 1{y' > c'} + c' 1{y' <= c'} - mu'(x), mu'(x) = -Yhat(x).
# Weighted p-values are not PRDS, so plain BH on them is not valid (paper, sec. 2.2).
# WCS uses auxiliary p-values (eq. 6) to calibrate each unit's threshold.  Pruning is
# deterministic (eq. 9), so the output is reproducible; Theorem 3.1 covers all three
# pruning options.  The weights must be fitted on data independent of the calibration
# and test units (Theorem 3.5), which is the caller's frozen contract.

CLIP_SCORE_M = 1.0e6  # "M -> infinity": any value above every |c' - mu'(x)| works


def orient_low_barrier_scores(
    calibration_predictions: Sequence[float],
    calibration_outcomes: Sequence[float],
    test_predictions: Sequence[float],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clipped monotone scores for the 'fast hop' alternative Y < c."""

    cal_hat = np.asarray(calibration_predictions, dtype=np.float64)
    cal_y = np.asarray(calibration_outcomes, dtype=np.float64)
    test_hat = np.asarray(test_predictions, dtype=np.float64)
    for name, array in (("calibration predictions", cal_hat), ("calibration outcomes", cal_y),
                        ("test predictions", test_hat)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
    if np.abs(cal_hat - threshold).max(initial=0.0) >= CLIP_SCORE_M / 2 or \
            np.abs(test_hat - threshold).max(initial=0.0) >= CLIP_SCORE_M / 2:
        raise ValueError("predictions too large for the clipping constant")
    c_prime = -threshold
    mu_cal, mu_test = -cal_hat, -test_hat
    fast = -cal_y > c_prime  # Y < c
    calibration_scores = np.where(fast, CLIP_SCORE_M, c_prime) - mu_cal
    test_scores = c_prime - mu_test
    return calibration_scores, test_scores


TIE_RULES = ("tie_free_lexicographic", "e3sse_conservative_le")


TIE_KEY_DOMAIN = "e3sse.wcs-tie-key"


def unit_tie_key(protocol_id: str, analysis_id: str, unit_id: str) -> int:
    """Frozen per-unit tie key: first 8 bytes of SHA-256(domain, protocol, analysis, unit).

    It depends only on frozen identifiers, never on scores, outcomes, ranks or input
    order, so it cannot be redrawn after scores are seen.
    """

    payload = "\0".join((TIE_KEY_DOMAIN, protocol_id, analysis_id, str(unit_id))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def lexicographic_tie_free_scores(calibration_scores, test_scores, calibration_keys,
                                  test_keys) -> tuple[np.ndarray, np.ndarray]:
    """Exact tie-free ordering by (score, unit tie key), returned as pooled integer ranks.

    Unequal scores keep their order exactly; the comparison is lexicographic, not
    ``score + epsilon``.  Equal scores are ordered by the unit's frozen key.  WCS uses
    only order comparisons, so replacing scores by pooled ranks is exact.
    """

    v = np.asarray(calibration_scores, dtype=np.float64)
    vt = np.asarray(test_scores, dtype=np.float64)
    keys = np.asarray(list(calibration_keys) + list(test_keys), dtype=np.uint64)
    if keys.size != v.size + vt.size:
        raise ValueError("one tie key per calibration and test unit is required")
    if np.unique(keys).size != keys.size:
        raise ValueError("tie keys must be unique (duplicate unit identifiers?)")
    pooled = np.concatenate([v, vt])
    order = np.lexsort((keys, pooled))  # primary: score; secondary: unit tie key
    ranks = np.empty(pooled.size, dtype=np.float64)
    ranks[order] = np.arange(pooled.size, dtype=np.float64)
    return ranks[: v.size], ranks[v.size:]


def weighted_conformal_pvalues(calibration_scores, calibration_weights, test_scores,
                               test_weights) -> np.ndarray:
    """(sum_i w_i 1{V_i <= Vhat_j} + w_j) / (sum_i w_i + w_j).

    On tie-free scores this is exactly the published WCS p-value, eq. 5 (strict ``<``),
    because no calibration score equals a test score.  On scores with ties it is the
    E3-SSE conservative modification of eq. 5 (calibration ties counted).  Eq. 4 is
    the randomized marginal p-value, not the WCS p-value, so this function is not
    described as eq. 4.
    """

    v, w = _weighted_inputs(calibration_scores, calibration_weights)
    vt, wt = _weighted_inputs(test_scores, test_weights)
    below = (w[None, :] * (v[None, :] <= vt[:, None])).sum(axis=1)
    return (below + wt) / (w.sum() + wt)


def _weighted_inputs(scores, weights) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.shape != w.shape or v.ndim != 1:
        raise ValueError("scores and weights must be aligned 1-D arrays")
    if not (np.isfinite(v).all() and np.isfinite(w).all()) or (w <= 0).any():
        raise ValueError("scores must be finite and weights finite and positive")
    return v, w


def _bh_count_with_forced(pvalues: np.ndarray, q: float, m: int) -> int:
    """k* = max{k : 1 + #{p_l <= q k / m} >= k} for the m-1 auxiliary p-values (step 5)."""

    ordered = np.sort(pvalues)
    best = 1
    for k in range(1, m + 1):
        if 1 + int(np.searchsorted(ordered, q * k / m, side="right")) >= k:
            best = k
    return best


def weighted_conformalized_selection(calibration_scores, calibration_weights, test_scores,
                                     test_weights, q: float, *,
                                     tie_rule: str = "tie_free_lexicographic",
                                     calibration_tie_keys: Sequence[int] | None = None,
                                     test_tie_keys: Sequence[int] | None = None,
                                     ) -> dict[str, Any]:
    """Algorithm 1 with deterministic pruning (eq. 9); returns the mask and p-values.

    ``tie_rule``:
    * ``tie_free_lexicographic`` (primary): WCS (Algorithm 1, eqs. 5, 6, 9) with
      auxiliary i.i.d. unit tie keys (``unit_tie_key``).  Treating the key as an extra
      covariate that is independent of (X, Y) and identically distributed under P and
      Q, the score (V, key) is tie-free, stays monotone in y, and dQ/dP is unchanged, so
      Theorem 3.1 applies to the augmented problem.  FDR <= q then holds marginally
      over the key draw, not conditionally on the realised keys (statistical-validator
      review, 2026-11-01).
    * ``e3sse_conservative_le`` (sensitivity): calibration ties counted in the main
      p-value only.  This matches the official ConfSelect ``weighted_CS`` with U_j = 1
      except for that code's eq. 6 weight index (it uses w_l; the published eq. 6 uses
      w_j).  Its validity rests on the reviewer's one-line extension of the C.1
      argument, not on a published theorem; it is never the primary method.
    """

    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    if tie_rule not in TIE_RULES:
        raise ValueError(f"tie_rule must be one of {TIE_RULES}")
    v, w = _weighted_inputs(calibration_scores, calibration_weights)
    vt, wt = _weighted_inputs(test_scores, test_weights)
    if tie_rule == "tie_free_lexicographic":
        if calibration_tie_keys is None or test_tie_keys is None:
            raise ValueError("tie_free_lexicographic needs frozen unit tie keys (unit_tie_key)")
        v, vt = lexicographic_tie_free_scores(v, vt, calibration_tie_keys, test_tie_keys)
    m = vt.size
    p = weighted_conformal_pvalues(v, w, vt, wt)
    if m == 0:
        return {"selected": np.zeros(0, dtype=bool), "pvalues": p, "r_star": 0,
                "tie_rule": tie_rule}
    # Eq. 6 auxiliary p-values use the strict "<" in every mode: calibration ties enter
    # only the main p-value (the official ConfSelect weighted_CS with U_j = 1).  The
    # added test-unit term carries w_j, the weight of the unit replaced by 0
    # (arXiv eq. 6); official ConfSelect code uses w_l there (open item).
    cal_below = (w[None, :] * (v[None, :] < vt[:, None])).sum(axis=1)  # per test unit l
    sizes = np.empty(m, dtype=np.int64)
    for j in range(m):
        # Eq. 6: auxiliary p-values of the other units with unit j's score as reference.
        aux = (cal_below + wt[j] * (vt[j] < vt)) / (w.sum() + wt[j])
        aux = np.delete(aux, j)
        sizes[j] = _bh_count_with_forced(aux, q, m)  # |R_{j->0}|
    first_step = p <= q * sizes / m
    r_star = 0
    for r in range(m, 0, -1):
        if int((first_step & (sizes <= r)).sum()) >= r:
            r_star = r
            break
    selected = first_step & (sizes <= r_star) if r_star else np.zeros(m, dtype=bool)
    return {"selected": selected, "pvalues": p, "first_step_sizes": sizes, "r_star": r_star,
            "pruning": "deterministic", "tie_rule": tie_rule}
