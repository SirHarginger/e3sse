"""Selective prediction: risk-coverage curves and AURC for correct-or-abstain.

Internal-freeze protocol ``amendments.selective_prediction``.  Units are ordered by an
applicability score, most trustworthy (smallest score) first.  At coverage k/n the
selective risk is the mean error of the first k units, and

    AURC = (1/n) * sum_{k=1..n} risk(k)          (Geifman et al.)

Ties: units with equal scores are exchangeable, so a tie group contributes its
*expected* risk under uniformly random within-tie order.  The partial sum over the
first j members of a tie group then has expectation j * (group mean error), which
makes the curve independent of input order.  Undefined scores (``None``) are ordered
after every defined score and form one tie group.

Random abstention baseline: for a uniformly random ordering, E[risk(k)] equals the
full-coverage mean error for every k, because a uniformly random k-subset has
expected mean equal to the population mean.  So AURC_random = MAE exactly, and it is
used analytically, never estimated by Monte Carlo.  It is a reference, not a method.  The oracle orders by the
realised error and is a diagnostic upper bound only.

The scoring functions take scores and errors as separate arguments.  A score must be
computed upstream from pre-outcome information only, and nothing here derives a
score from the errors except ``oracle_aurc``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _validated(scores: Sequence[float | None], errors: Sequence[float]) -> np.ndarray:
    if len(scores) != len(errors):
        raise ValueError("scores and errors must have the same length")
    if not len(errors):
        raise ValueError("risk-coverage needs at least one unit")
    e = np.asarray(errors, dtype=np.float64)
    if not np.isfinite(e).all() or (e < 0).any():
        raise ValueError("errors must be finite and non-negative")
    return e


def _tie_groups(scores: Sequence[float | None]) -> list[list[int]]:
    defined = sorted({s for s in scores if s is not None})
    for s in defined:
        if not np.isfinite(s):
            raise ValueError("defined scores must be finite")
    groups = [[i for i, s in enumerate(scores) if s == value] for value in defined]
    undefined = [i for i, s in enumerate(scores) if s is None]
    if undefined:
        groups.append(undefined)
    return groups


def risk_coverage(scores: Sequence[float | None], errors: Sequence[float]) -> dict[str, Any]:
    """Expected (tie-averaged) risk at every coverage k/n, k = 1..n."""

    e = _validated(scores, errors)
    n = len(e)
    cumulative = np.empty(n, dtype=np.float64)
    position = 0
    running = 0.0
    for group in _tie_groups(scores):
        mean = float(e[group].mean())
        for j in range(1, len(group) + 1):
            cumulative[position] = running + j * mean
            position += 1
        running += float(e[group].sum())
    k = np.arange(1, n + 1, dtype=np.float64)
    return {"coverage": k / n, "risk": cumulative / k}


def aurc(scores: Sequence[float | None], errors: Sequence[float]) -> float:
    return float(risk_coverage(scores, errors)["risk"].mean())


def random_aurc(errors: Sequence[float]) -> float:
    """Expected AURC of uniformly random abstention (= full-coverage mean error)."""

    return float(_validated([0.0] * len(errors), errors).mean())


def oracle_aurc(errors: Sequence[float]) -> float:
    """AURC when ordering by the realised error itself (upper bound, never deployable)."""

    e = _validated([0.0] * len(errors), errors)
    ordered = np.sort(e)
    return float((np.cumsum(ordered) / np.arange(1, len(e) + 1)).mean())


def risk_at_coverage(scores: Sequence[float | None], errors: Sequence[float],
                     coverage: float) -> float:
    """Risk at the smallest k with k/n >= coverage."""

    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    curve = risk_coverage(scores, errors)
    index = int(np.searchsorted(curve["coverage"], coverage - 1e-12))
    return float(curve["risk"][index])


def summarize(scores: Sequence[float | None], errors: Sequence[float],
              *, target_coverage: float = 0.8) -> dict[str, float | None]:
    """AURC, E-AURC, normalized gain, and risk at full and target coverage."""

    value = aurc(scores, errors)
    rand = random_aurc(errors)
    best = oracle_aurc(errors)
    span = rand - best
    return {
        "aurc": value,
        "aurc_random": rand,
        "aurc_oracle": best,
        "e_aurc": value - best,
        # Descriptive only: a ratio, unstable when the oracle span is small; never tested.
        "normalized_gain": (rand - value) / span if span > 0 else None,
        "risk_full_coverage": rand,
        "risk_at_target_coverage": risk_at_coverage(scores, errors, target_coverage),
        "n_units": len(errors),
        "n_undefined_scores": sum(s is None for s in scores),
    }


def correct_or_raw_curve(
    scores: Sequence[float | None],
    srt_errors: Sequence[float | None],
    raw_errors: Sequence[float],
) -> dict[str, Any]:
    """Secondary framing: correct the first k units, keep raw for the rest; risk over all n.

    Coverage is the fraction of all n units that are corrected.  A unit without an SRT
    prediction (unsupported s, ``srt_errors[i] is None``) can never be corrected.  It
    must carry an undefined score, always keeps its raw prediction, and is counted in
    ``n_uncorrectable``.  The curve therefore ends at coverage n_correctable / n.  No
    SRT error is ever imputed.  Ties are averaged as in ``risk_coverage``.
    """

    raw = _validated(scores, raw_errors)
    if len(srt_errors) != len(raw):
        raise ValueError("srt_errors must align with raw_errors")
    for score, error in zip(scores, srt_errors, strict=True):
        if error is None and score is not None:
            raise ValueError("a unit without an SRT prediction must have an undefined score")
    correctable = [i for i, e in enumerate(srt_errors) if e is not None]
    srt = _validated([0.0] * len(correctable), [srt_errors[i] for i in correctable])
    benefit = raw[correctable] - srt
    local_scores = [scores[i] for i in correctable]
    n = len(raw)
    total = np.empty(len(correctable) + 1, dtype=np.float64)
    total[0] = raw.sum()
    position = 0
    for group in _tie_groups(local_scores):
        mean_benefit = float(benefit[group].mean())
        base = total[position]
        for j in range(1, len(group) + 1):
            total[position + j] = base - j * mean_benefit
        position += len(group)
    return {"coverage": np.arange(len(correctable) + 1) / n, "risk": total / n,
            "n_uncorrectable": n - len(correctable)}


def summarize_defined_subset(scores: Sequence[float | None], errors: Sequence[float],
                             *, target_coverage: float = 0.8) -> dict[str, Any]:
    """``summarize`` restricted to units with a defined score (reported alongside)."""

    keep = [i for i, s in enumerate(scores) if s is not None]
    if not keep:
        return {"n_units": 0, "n_undefined_scores": len(scores)}
    result = summarize([scores[i] for i in keep], [errors[i] for i in keep],
                       target_coverage=target_coverage)
    result["n_undefined_scores_excluded"] = len(scores) - len(keep)
    return result
