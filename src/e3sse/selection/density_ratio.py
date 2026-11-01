"""Covariate-shift weights w(x) = q_X(x) / p_X(x) for weighted conformal selection.

This is a separate quantity from the applicability support score.  It is estimated
with a probabilistic domain classifier: an L2-regularized logistic regression,
fitted deterministically by iteratively reweighted least squares (pure numpy, no
random initialization).  With P the calibration-side population and Q the
prospective-side population (both post-Gate-1 and predictor-eligible),

    w(x) = (n_P / n_Q) * pi(x) / (1 - pi(x)),   pi(x) = Pr(Q | x).

Jin & Candes Theorem 3.5: with weights fitted on data independent of the
calibration and test units, FDR <= q * E[gamma^2 / (1 + q (gamma^2 - 1) / m)], where
gamma = sup_x max(w_hat / w, w / w_hat).  The guarantee is therefore exact only for
known (oracle) weights.  With estimated weights it is an inflation bound that
shrinks as w_hat becomes consistent.  Clipping changes the weights and is recorded as
changing the theoretical interpretation.  Only label-free covariates enter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DENSITY_RATIO_SCHEMA = "e3sse.covariate-shift-weights"


def _design(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1)), (x - mean) / sd])


def fit_domain_classifier(p_features: Sequence[Sequence[float]],
                          q_features: Sequence[Sequence[float]], *,
                          l2: float = 1.0, max_iter: int = 100,
                          tolerance: float = 1e-10) -> dict[str, Any]:
    """Fit Pr(Q | x) by ridge-penalized IRLS; the intercept is not penalized."""

    xp = np.asarray(p_features, dtype=np.float64)
    xq = np.asarray(q_features, dtype=np.float64)
    if xp.ndim != 2 or xq.ndim != 2 or xp.shape[1] != xq.shape[1]:
        raise ValueError("P and Q features must be 2-D with equal width")
    if not (np.isfinite(xp).all() and np.isfinite(xq).all()):
        raise ValueError("features must be finite (missing features fail closed upstream)")
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    x = np.vstack([xp, xq])
    y = np.concatenate([np.zeros(len(xp)), np.ones(len(xq))])
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    design = _design(x, mean, sd)
    penalty = np.full(design.shape[1], l2)
    penalty[0] = 0.0
    beta = np.zeros(design.shape[1])
    converged = False
    for iteration in range(1, max_iter + 1):
        eta = np.clip(design @ beta, -35.0, 35.0)
        prob = 1.0 / (1.0 + np.exp(-eta))
        weight = prob * (1.0 - prob)
        gradient = design.T @ (y - prob) - penalty * beta
        hessian = design.T @ (design * weight[:, None]) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.abs(step).max() < tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError("domain classifier did not converge; the fit is not used")
    return {"schema": DENSITY_RATIO_SCHEMA, "coefficients": beta.tolist(), "mean": mean.tolist(),
            "sd": sd.tolist(), "l2": l2, "n_p": len(xp), "n_q": len(xq),
            "iterations": iteration}


def predicted_q_probability(model: dict[str, Any], features: Sequence[Sequence[float]]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]) or not np.isfinite(x).all():
        raise ValueError("features must be finite with the fitted width")
    design = _design(x, np.asarray(model["mean"]), np.asarray(model["sd"]))
    eta = np.clip(design @ np.asarray(model["coefficients"]), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-eta))


def shift_weights(model: dict[str, Any], features: Sequence[Sequence[float]], *,
                  clip_max: float | None = None) -> dict[str, Any]:
    """w(x) = (n_P/n_Q) pi/(1-pi); optional clipping is flagged, never silent."""

    if clip_max is not None and not clip_max > 0:
        raise ValueError("clip_max must be positive")
    prob = predicted_q_probability(model, features)
    weights = (model["n_p"] / model["n_q"]) * prob / (1.0 - prob)
    clipped = 0
    if clip_max is not None:
        clipped = int((weights > clip_max).sum())
        weights = np.minimum(weights, clip_max)
    return {"weights": weights, "clip_max": clip_max, "n_clipped": clipped,
            "interpretation": "estimated weights (Theorem 3.5 bound)"
            + ("; clipped: weights altered, guarantee interpretation changes"
               if clip_max is not None else "")}


def auc(scores_p: np.ndarray, scores_q: np.ndarray) -> float:
    """Mann-Whitney AUC of Pr(Q|x) separating Q from P (0.5 = no detectable shift)."""

    if len(scores_p) == 0 or len(scores_q) == 0:
        raise ValueError("AUC needs both P and Q scores")
    combined = np.concatenate([scores_p, scores_q])
    order = combined.argsort(kind="stable")
    ranks = np.empty(len(combined))
    sorted_values = combined[order]
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_p, n_q = len(scores_p), len(scores_q)
    return float((ranks[n_p:].sum() - n_q * (n_q + 1) / 2.0) / (n_p * n_q))


def weight_diagnostics(calibration_weights: np.ndarray, test_weights: np.ndarray,
                       prob_p: np.ndarray, prob_q: np.ndarray, *, q: float,
                       overlap_epsilon: float = 0.01) -> dict[str, Any]:
    """Diagnostics reported with every weighted analysis.

    The ESS is computed on the calibration weights, which is where weighting thins the
    evidence.  ``unselectable_test_fraction`` is the share of test units with
    w_j / (sum_i w_i + w_j) > q.  Their eq. 5 p-value is at least that ratio, so they can
    never be selected.  The AUC is reported as given (in-sample or out-of-sample
    according to the caller).
    """

    w = np.asarray(calibration_weights, dtype=np.float64)
    wt = np.asarray(test_weights, dtype=np.float64)
    if w.size == 0 or wt.size == 0:
        raise ValueError("diagnostics need calibration and test weights")
    floor = wt / (w.sum() + wt)
    return {
        "calibration_effective_sample_size": float(w.sum() ** 2 / (w ** 2).sum()),
        "n_calibration": int(w.size),
        "n_test": int(wt.size),
        "max_calibration_weight": float(w.max()),
        "max_test_weight": float(wt.max()),
        "calibration_weight_quantiles": dict(zip(("q05", "q50", "q95"),
                                                 np.quantile(w, [0.05, 0.5, 0.95]).tolist(),
                                                 strict=True)),
        "unselectable_test_fraction": float((floor > q).mean()),
        "classifier_auc": auc(np.asarray(prob_p), np.asarray(prob_q)),
        "q_fraction_outside_overlap": float((np.asarray(prob_q) > 1 - overlap_epsilon).mean()),
        "p_fraction_outside_overlap": float((np.asarray(prob_p) < overlap_epsilon).mean()),
    }
