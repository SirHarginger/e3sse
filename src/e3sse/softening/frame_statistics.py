"""Per-frame sufficient statistics for the Eq. 17 force-parity softening scale.

Research Design Eq. 17 (frozen ``study_1_contract.estimator``,
``zero_intercept_force_parity_projection``): with DFT force components x_a and ML
force components y_a of a frame set S,

    s_hat = sum_a y_a x_a / sum_a x_a^2,

so ``F_ML ~ s F_DFT``; s < 1 is softening.  Because the numerator and denominator
are sums over Cartesian components, the per-frame, per-element sums

    sxy = sum x*y,  sxx = sum x^2,  syy = sum y^2,  n_informative = #{|x| >= 0.1}

are sufficient: summing them over any frame subset (and any element subset) gives
exactly the same slope as concatenating the raw components.  This also holds for
bootstrap resamples, where frame i drawn c_i times contributes c_i times its sums.
The residual fraction of the zero-intercept projection,
    min_s sum (y - s x)^2 / sum y^2 = 1 - sxy^2 / (sxx * syy),
follows from the same sums.  Element-resolved scales use the components of atoms of
one species (``element_resolved_component_rule``); s_Li uses Li atoms and s_host all
non-Li atoms of the same frames (``s_host_definition``).

Per-frame sums use ``math.fsum`` (exactly rounded), so they do not depend on numpy
version, BLAS or thread count in the isolated model environments.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

STATISTICS_SCHEMA = "e3sse.study1.frame-statistics.v1"
INFORMATIVE_FORCE_EV_PER_A = 0.1  # frozen estimator_minimum_support.informative_force_eV_per_A
LITHIUM_Z = 3
# Component sets derived from the per-element sums; element sets are "Z<number>".
COMPONENT_SETS = ("all", "Li", "host")


def element_statistics(
    numbers: Sequence[int], dft_forces: Any, ml_forces: Any | None = None
) -> list[dict[str, Any]]:
    """Per-element sums for one frame, ascending atomic number.

    ``ml_forces=None`` gives the DFT-only part (the model-independent reference).
    """

    import numpy as np

    numbers = np.asarray(numbers, dtype=np.int64)
    x = np.asarray(dft_forces, dtype=np.float64)
    if x.shape != (len(numbers), 3) or not np.isfinite(x).all():
        raise ValueError("DFT forces must be finite with shape (n_atoms, 3)")
    y = None
    if ml_forces is not None:
        y = np.asarray(ml_forces, dtype=np.float64)
        if y.shape != x.shape:
            raise ValueError("ML forces must have the DFT force shape")
    rows = []
    for z in sorted(set(numbers.tolist())):
        mask = numbers == z
        xs = x[mask].ravel()
        row = {
            "Z": int(z),
            "n_atoms": int(mask.sum()),
            "sxx": math.fsum((xs * xs).tolist()),
            "n_informative": int((np.abs(xs) >= INFORMATIVE_FORCE_EV_PER_A).sum()),
        }
        if y is not None:
            ys = y[mask].ravel()
            row["sxy"] = math.fsum((xs * ys).tolist())
            row["syy"] = math.fsum((ys * ys).tolist())
        rows.append(row)
    return rows


def atomic_numbers_sha256(numbers: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(z)) for z in numbers).encode()).hexdigest()


def component_set_members(name: str, z: int) -> bool:
    """Whether element ``z`` contributes to component set ``name``."""

    if name == "all":
        return True
    if name == "Li":
        return z == LITHIUM_Z
    if name == "host":
        return z != LITHIUM_Z
    if name.startswith("Z") and name[1:].isdigit():
        return z == int(name[1:])
    raise ValueError(f"unknown component set {name!r}")


def combine(rows: Iterable[Mapping[str, Any]], component_set: str) -> dict[str, Any]:
    """Exactly rounded sums of one frame's element rows for one component set."""

    chosen = [row for row in rows if component_set_members(component_set, int(row["Z"]))]
    result = {
        "n_atoms": sum(int(row["n_atoms"]) for row in chosen),
        "sxx": math.fsum(row["sxx"] for row in chosen),
        "n_informative": sum(int(row["n_informative"]) for row in chosen),
    }
    if all("sxy" in row for row in chosen):
        result["sxy"] = math.fsum(row["sxy"] for row in chosen)
        result["syy"] = math.fsum(row["syy"] for row in chosen)
    return result


def eq17(sxy: float, sxx: float) -> float | None:
    """Eq. 17 slope, or ``None`` when the denominator is zero (undefined)."""

    return None if sxx <= 0.0 else sxy / sxx


def residual_fraction(sxy: float, sxx: float, syy: float) -> float | None:
    """Unexplained fraction of sum y^2 after the zero-intercept projection."""

    if sxx <= 0.0 or syy <= 0.0:
        return None
    return max(0.0, 1.0 - (sxy * sxy) / (sxx * syy))


def support_status(
    *,
    n_frames: int,
    n_informative: int,
    n_materials: int,
    pooled: bool,
    parameters: Mapping[str, Any],
) -> list[str]:
    """Unmet frozen ``estimator_minimum_support`` conditions (empty list = supported).

    ``n_informative`` is counted over all atoms of the estimate's frames, as the
    frozen contract (``validate_softening_support``) does, for every component set.
    There is no fallback.
    """

    reasons = []
    if n_frames < parameters["full_estimate_min_frames"]:
        reasons.append(f"fewer_than_{parameters['full_estimate_min_frames']}_frames")
    if n_informative < parameters["full_estimate_min_informative_components"]:
        reasons.append(
            f"fewer_than_{parameters['full_estimate_min_informative_components']}"
            "_informative_components"
        )
    if pooled and n_materials < parameters["pooled_min_materials"]:
        reasons.append(f"fewer_than_{parameters['pooled_min_materials']}_materials")
    return reasons


# --------------------------------------------------------------------------- draws
#
# numpy's Generator methods (integers, choice, permutation) may change their streams
# between numpy feature releases; the PCG64DXSM bit generator's raw 64-bit stream does
# not.  Draws are therefore mapped from the raw stream of the frozen bit generator
# (statistical_plan.randomness.bootstrap.generator) by fixed, version-independent rules,
# recorded in every analysis manifest as DRAW_MAPPING.

DRAW_MAPPING = {
    "bit_generator": "numpy.random.PCG64DXSM(seed).random_raw (uint64 stream)",
    "with_replacement": "index = raw % n; raw >= 2**64 - (2**64 % n) rejected, stream "
                        "continues; resample b takes the next n accepted values",
    "without_replacement": "first k of a stable argsort of n raw keys, sorted ascending",
    "reason": "numpy Generator method streams may change between numpy feature releases",
}

_TWO_64 = 1 << 64


def _bit_generator(seed: int) -> Any:
    import numpy as np

    return np.random.PCG64DXSM(seed)


def iter_bootstrap_indices(
    seed: int, n_clusters: int, n_resamples: int, *, max_values: int = 4_000_000
) -> Iterator[Any]:
    """Cluster indices, uniform with replacement, as (resamples, n_clusters) blocks.

    Resample b uses the next ``n_clusters`` accepted values of one stream, so the
    blocks concatenate to the same matrix whatever the block size.  Unbiased: raw
    values at or above the largest multiple of n are rejected and the stream
    continues, so every index has probability exactly 1/n.
    """

    import numpy as np

    if n_clusters < 1 or n_resamples < 1:
        raise ValueError("bootstrap needs at least one cluster and one resample")
    bit = _bit_generator(seed)
    remainder = _TWO_64 % n_clusters  # zero for powers of two: no rejection needed
    limit = None if remainder == 0 else np.uint64(_TWO_64 - remainder)
    per_block = max(1, max_values // n_clusters)
    done = 0
    while done < n_resamples:
        block = min(per_block, n_resamples - done)
        needed = block * n_clusters
        kept, total = [], 0
        while total < needed:
            raw = bit.random_raw(needed - total)
            if limit is not None:
                raw = raw[raw < limit]
            kept.append(raw)
            total += len(raw)
        values = np.concatenate(kept) % np.uint64(n_clusters)
        yield values.astype(np.int64).reshape(block, n_clusters)
        done += block


def bootstrap_indices(seed: int, n_clusters: int, n_resamples: int) -> Any:
    """All resamples at once (small problems and tests)."""

    import numpy as np

    return np.concatenate(list(iter_bootstrap_indices(seed, n_clusters, n_resamples)))


def draw_without_replacement(seed: int, population: int, k: int) -> Any:
    """``k`` distinct indices of ``range(population)``, uniformly at random.

    Rank of one raw 64-bit key per population member (stable sort; a key tie,
    probability ~ N^2 / 2^65, keeps population order).
    """

    import numpy as np

    if not 1 <= k <= population:
        raise ValueError("need 1 <= k <= population")
    keys = _bit_generator(seed).random_raw(population)
    return np.sort(np.argsort(keys, kind="stable")[:k])


def percentile_interval(values: Any, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval (statistical_plan ``percentile_2.5_97.5``)."""

    import numpy as np

    tail = (1.0 - level) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [tail, 1.0 - tail],
                            method="linear")
    return float(low), float(high)
