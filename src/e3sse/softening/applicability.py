"""Applicability diagnostics: is one scalar s an adequate local model of the error?

Every diagnostic is a pure function of the Eq. 17 sufficient statistics of the frames
behind the s being diagnosed (``frame_statistics``: sxy = sum F_DFT F_ML,
sxx = sum F_DFT^2, syy = sum F_ML^2 per frame and component set).  No barrier, no
target-hop outcome and no own-path frame can enter: the functions take no outcome
argument.  Larger values mean less trustworthy (internal-freeze protocol
``amendments.applicability.orientation``).

Force non-collinearity ``rho_F``.  With s = Sxy/Sxx,

    sum |F_ML - s F_DFT|^2 / sum |F_ML|^2 = 1 - Sxy^2 / (Sxx Syy) = sin^2(theta),

where theta is the angle between the stacked DFT and ML force vectors.  rho_F = |sin
theta| is therefore invariant to rescaling either force field, symmetric in DFT and
ML, bounded in [0, 1] and zero iff the forces are exactly proportional (uniform
softening).  rho_F^2 equals ``frame_statistics.residual_fraction``.  The variant with
the DFT denominator, sum |F_ML - s F_DFT|^2 / (s^2 Sxx), equals tan^2(theta).  It is
monotone in rho_F, so both give identical rankings, risk-coverage curves and AURC.

Undefined values are ``None`` together with a reason, never imputed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .frame_statistics import eq17, residual_fraction


class Diagnostic(dict):
    """``{"value": float | None, "reason": str | None}`` plus diagnostic-specific fields."""


def _undefined(reason: str, **extra: Any) -> Diagnostic:
    return Diagnostic(value=None, reason=reason, **extra)


def rho_f(sxy: float, sxx: float, syy: float) -> Diagnostic:
    """|sin theta| = sqrt(residual_fraction): a display transform of the stored quantity.

    The canonical stored statistic is ``frame_statistics.residual_fraction``
    (= sin^2 theta), computed from the same Eq. 17 sums as s.  rho_F is its square
    root, a strictly increasing transform, so rankings, risk-coverage orderings and
    AURC are identical whichever one is used.  No second statistic is stored.
    Additional rule: undefined when Sxy <= 0, because sin(theta) ignores the sign, so
    anti-aligned forces would otherwise look perfectly uniform.
    """

    if not (math.isfinite(sxy) and math.isfinite(sxx) and math.isfinite(syy)):
        return _undefined("non_finite_sums")
    if sxx <= 0.0:
        return _undefined("zero_dft_force_norm")
    if syy <= 0.0:
        return _undefined("zero_ml_force_norm")
    if sxy <= 0.0:
        return _undefined("nonpositive_force_projection")
    fraction = residual_fraction(sxy, sxx, syy)
    return Diagnostic(value=math.sqrt(min(1.0, fraction)), reason=None,
                      residual_fraction=fraction)


def rho_f_dft_denominator(sxy: float, sxx: float, syy: float) -> float | None:
    """|tan theta|: the DFT-denominator variant (sensitivity only; same ranking as rho_F).

    Defined exactly where ``rho_f`` is (same finiteness and sign rules).
    """

    sine = rho_f(sxy, sxx, syy)["value"]
    if sine is None or sine >= 1.0:
        return None
    return sine / math.sqrt(1.0 - sine * sine)


def frame_median_rho_f(rows: Sequence[Mapping[str, float]]) -> Diagnostic:
    """Pre-specified sensitivity: median over frames of the per-frame |sin theta_f|."""

    values = [
        d["value"]
        for d in (rho_f(row["sxy"], row["sxx"], row["syy"]) for row in rows)
        if d["value"] is not None
    ]
    if not values:
        return _undefined("no_frame_with_defined_rho_F")
    values.sort()
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])
    return Diagnostic(value=median, reason=None, n_frames=len(values))


def scale_uncertainty(s_hat: float | None, bootstrap: Sequence[float]) -> Diagnostic:
    """Relative half-width (q97.5 - q2.5) / (2 s_hat) of the whole-frame bootstrap."""

    import numpy as np

    if s_hat is None or not math.isfinite(s_hat) or s_hat <= 0.0:
        return _undefined("s_hat_undefined_or_nonpositive")
    n_undefined = sum(b is None for b in bootstrap)
    if n_undefined:
        # A resample with an undefined slope is never dropped silently.
        return _undefined("undefined_bootstrap_draws", n_undefined_draws=n_undefined)
    draws = np.asarray(list(bootstrap), dtype=np.float64)
    if draws.size < 2 or not np.isfinite(draws).all():
        return _undefined("insufficient_bootstrap_draws")
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    return Diagnostic(
        value=float((high - low) / (2.0 * s_hat)),
        reason=None,
        bootstrap_sd=float(np.std(draws, ddof=1)),
    )


def log_ratio_disagreement(
    numerator: tuple[float, float] | None,
    denominator: tuple[float, float] | None,
    *,
    numerator_supported: bool,
    denominator_supported: bool,
    label: str,
) -> Diagnostic:
    """|ln(s_a / s_b)| for two support-gated scales given as (sxy, sxx) sums."""

    if not numerator_supported or not denominator_supported or numerator is None or denominator is None:
        return _undefined(f"{label}_component_set_unsupported")
    s_a, s_b = eq17(*numerator), eq17(*denominator)
    if s_a is None or s_b is None or s_a <= 0.0 or s_b <= 0.0:
        return _undefined(f"{label}_scale_undefined_or_nonpositive")
    return Diagnostic(value=abs(math.log(s_a / s_b)), reason=None, s_a=s_a, s_b=s_b)


def energy_median_split(energies: Sequence[float]) -> tuple[list[int], list[int]]:
    """Frozen split: indices at or below the median dE_f are low, the rest high.

    The median frame goes to the low half.  The order is by (energy, original index),
    so the split is deterministic under ties.
    """

    order = sorted(range(len(energies)), key=lambda i: (energies[i], i))
    cut = (len(order) + 1) // 2
    return sorted(order[:cut]), sorted(order[cut:])


def percentile_ranks(values: Sequence[float | None]) -> list[float | None]:
    """Average-rank percentiles in (0, 1]; ``None`` stays ``None``."""

    if any(v is not None and not math.isfinite(v) for v in values):
        raise ValueError("diagnostic values must be finite or None")
    defined = sorted((v, i) for i, v in enumerate(values) if v is not None)
    ranks: list[float | None] = [None] * len(values)
    n = len(defined)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and defined[end + 1][0] == defined[start][0]:
            end += 1
        average = (start + end) / 2.0 + 1.0
        for _, index in defined[start:end + 1]:
            ranks[index] = average / n
        start = end + 1
    return ranks


def combined_score(columns: Mapping[str, Sequence[float | None]]) -> list[float | None]:
    """Equal-weight mean of within-population percentile ranks; no learned weights.

    A unit whose component scores are all undefined has an undefined combined score.
    Ranks depend on the population, so callers recompute this inside every bootstrap
    replicate and record the population hash and per-unit component count
    (``combined_component_counts``).
    """

    ranked = {name: percentile_ranks(values) for name, values in columns.items()}
    lengths = {len(values) for values in ranked.values()}
    if len(lengths) != 1:
        raise ValueError("all diagnostic columns must cover the same units")
    (n,) = lengths
    combined: list[float | None] = []
    for i in range(n):
        parts = [ranked[name][i] for name in ranked if ranked[name][i] is not None]
        combined.append(sum(parts) / len(parts) if parts else None)
    return combined


def combined_component_counts(columns: Mapping[str, Sequence[float | None]]) -> list[int]:
    """Number of defined component diagnostics behind each unit's combined score."""

    rows = list(zip(*columns.values(), strict=True))
    return [sum(v is not None for v in row) for row in rows]


GATED_DIAGNOSTICS = ("rho_F", "scale_uncertainty")


def gate_thresholds(
    population: Mapping[str, Sequence[float | None]],
    supported: Sequence[bool],
    quantile_level: float,
) -> dict[str, Any]:
    """Label-free thresholds over the *supported* units of the frozen threshold population.

    Returns the thresholds and an accounting of units used, unsupported and undefined.
    The gated diagnostics are exactly ``GATED_DIAGNOSTICS``.
    """

    import numpy as np

    if not 0.0 < quantile_level < 1.0:
        raise ValueError("quantile_level must lie in (0, 1)")
    if set(population) != set(GATED_DIAGNOSTICS):
        raise ValueError(f"gate thresholds need exactly {GATED_DIAGNOSTICS}")
    thresholds: dict[str, float] = {}
    accounting: dict[str, dict[str, int]] = {}
    for name in GATED_DIAGNOSTICS:
        values = population[name]
        if len(values) != len(supported):
            raise ValueError("support flags must align with the diagnostic values")
        kept = [v for v, ok in zip(values, supported, strict=True) if ok and v is not None]
        defined = np.asarray(kept, dtype=np.float64)
        if defined.size == 0:
            raise ValueError(f"no supported, defined {name} value in the threshold population")
        thresholds[name] = float(np.quantile(defined, quantile_level, method="linear"))
        accounting[name] = {
            "n_population": len(values),
            "n_used": int(defined.size),
            "n_unsupported": sum(not ok for ok in supported),
            "n_undefined_supported": sum(ok and v is None
                                         for v, ok in zip(values, supported, strict=True)),
        }
    return {"thresholds": thresholds, "quantile_level": quantile_level, "accounting": accounting}


def gate_decision(
    *, supported: bool, diagnostics: Mapping[str, float | None], thresholds: Mapping[str, float]
) -> tuple[bool, list[str]]:
    """Apply SRT iff supported and every gated diagnostic is defined and within threshold."""

    if set(thresholds) != set(GATED_DIAGNOSTICS):
        raise ValueError(f"gate decision needs thresholds for exactly {GATED_DIAGNOSTICS}")
    reasons = [] if supported else ["support_rule_failed"]
    for name, threshold in sorted(thresholds.items()):
        value = diagnostics.get(name)
        if value is None:
            reasons.append(f"{name}_undefined")
        elif value > threshold:
            reasons.append(f"{name}_above_threshold")
    return not reasons, reasons
