"""Chemical-system cluster bootstrap and Holm adjustment (frozen statistical plan).

``statistical_plan.randomness.bootstrap``: clusters in ascending UTF-8 identifier
order, n clusters sampled with replacement per resample, generator
``numpy.random.Generator(numpy.random.PCG64DXSM(seed))`` (via
``frame_statistics.iter_bootstrap_indices``), percentile 2.5/97.5 interval.  Seeds
come from ``protocol.partitions.analysis_seed`` over canonical identifiers only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def cluster_bootstrap(clusters: Sequence[str], statistic: Callable[[Any], float | None], *,
                      seed: int, resamples: int) -> list[float | None]:
    """``statistic(unit_indices)`` over resamples of whole clusters.

    ``clusters[i]`` is unit i's cluster; the statistic receives an index array of the
    resampled units (duplicates kept).
    """

    import numpy as np

    from e3sse.softening.frame_statistics import iter_bootstrap_indices

    names = sorted(set(clusters))
    members = {name: [] for name in names}
    for index, name in enumerate(clusters):
        members[name].append(index)
    member_arrays = [np.asarray(members[name], dtype=np.int64) for name in names]
    values: list[float | None] = []
    for block in iter_bootstrap_indices(seed, len(names), resamples):
        for draw in np.atleast_2d(block):
            units = np.concatenate([member_arrays[i] for i in draw])
            values.append(statistic(units))
    return values


def percentile_interval(values: Sequence[float | None], level: float = 0.95) -> dict[str, Any]:
    import numpy as np

    defined = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    undefined = len(values) - int(defined.size)
    if defined.size < 2:
        return {"low": None, "high": None, "undefined_resamples": undefined}
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(defined, [tail, 1.0 - tail], method="linear")
    return {"low": float(low), "high": float(high), "undefined_resamples": undefined}


def one_sided_p(values: Sequence[float | None], *, null_at_or_below: float = 0.0) -> float | None:
    """Fraction of bootstrap replicates <= the null value (percentile-bootstrap p)."""

    defined = [v for v in values if v is not None and math.isfinite(v)]
    if not defined:
        return None
    return sum(v <= null_at_or_below for v in defined) / len(defined)


def p_resolution(values: Sequence[float | None]) -> dict[str, Any]:
    """Bootstrap resolution of a percentile p: a p of 0 is reported as < 1/B."""

    defined = sum(v is not None and math.isfinite(v) for v in values)
    p = one_sided_p(values)
    return {"bootstrap_replicates_defined": defined,
            "p_below_resolution": p == 0.0,
            "p_reported_as": None if p is None else (f"< {1 / defined:.6g}" if p == 0.0 else p)}


def holm(pvalues: Mapping[str, float | None]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone) over the whole declared family.

    The family size m is every declared member: an undefined p enters as 1 (it can
    never be rejected) instead of leaving the family, so m is never reduced after the
    data are seen.  Callers record which members were undefined.
    """

    defined = sorted(((1.0 if p is None else p, name) for name, p in pvalues.items()),
                     key=lambda x: (x[0], x[1]))
    m = len(defined)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (p, name) in enumerate(defined):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted
