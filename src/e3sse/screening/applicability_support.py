"""Label-free applicability support for the prospective reliability gate.

Internal-freeze protocol ``amendments.prospective.support_gate``.  A unit (a hop
represented in a feature space) is *supported* when its mean distance to the k
nearest members of the frozen development reference population is at or below
tau_d.  tau_d is the frozen quantile of the leave-one-out distances inside that
population.  Everything is fitted on the development population only: the
standardization, the reference set and tau_d.  The gate is therefore a fixed function
of X for conformal-calibration and prospective units alike, and no outcome enters.

The feature space is chosen by a deterministic hierarchy, never by a weighted mix:

1. a stable frozen-model representation (verified in ``embedding_capability_manifest``);
2. otherwise the compact label-free descriptor space below;
3. otherwise reliability-gated prospective SRT is unavailable, and ungated
   SRT-family is reported only as a labelled secondary baseline.

Distances are always computed in the original (standardized) feature space.
Projections (PCA/UMAP) are never used here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from e3sse.protocol.chemistry import anion_family, composition_counts

SUPPORT_SCHEMA = "e3sse.prospective-applicability-support"
SOURCES = ("frozen_model_representation", "descriptor_space", "unavailable")
DESCRIPTOR_NAMES = (
    "li_atomic_fraction",
    "mean_atomic_number",
    "sd_atomic_number",
    "volume_per_atom_A3",
    "li_number_density_per_A3",
    "hop_length_A",
    "em_bvse_eV",
    "family_oxide",
    "family_sulfide",
    "family_halide",
    "family_nitride",
    "family_other",
)
_FAMILIES = ("oxide", "sulfide", "halide", "nitride", "other")
FAMILY_COLUMNS = tuple(DESCRIPTOR_NAMES.index(f"family_{f}") for f in _FAMILIES)


def applicability_source(*, representation_verified: bool, descriptors_available: bool) -> str:
    """Deterministic fallback hierarchy (no combination of sources)."""

    if representation_verified:
        return "frozen_model_representation"
    if descriptors_available:
        return "descriptor_space"
    return "unavailable"


def descriptor_vector(
    symbols: Sequence[str],
    *,
    cell_volume_A3: float,
    hop_length_A: float | None,
    em_bvse_eV: float | None,
) -> list[float | None]:
    """Compact label-free descriptors: composition, density, hop geometry, BVSE.

    ``symbols`` and ``cell_volume_A3`` come from the pristine host structure of the
    material (never a vacancy NEB supercell).  ``hop_length_A`` is |r_t + o.L - r_s|
    using the hop's own cell offset o, not the minimum image.  ``em_bvse_eV`` is the
    nebBVSE122k barrier of h*_3.

    Uses only structure and BVSE covariates; no DFT or ML-derived outcome.  A
    missing covariate stays ``None``, and units with any ``None`` are unsupported
    (fail closed) rather than imputed.
    """

    from ase.data import atomic_numbers

    if not symbols:
        raise ValueError("a descriptor needs at least one atom")
    if not math.isfinite(cell_volume_A3) or cell_volume_A3 <= 0:
        raise ValueError("cell volume must be positive and finite")
    z = np.asarray([atomic_numbers[s] for s in symbols], dtype=np.float64)
    n = len(symbols)
    n_li = sum(s == "Li" for s in symbols)
    family = anion_family(composition_counts(symbols))
    return [
        n_li / n,
        float(z.mean()),
        float(z.std(ddof=0)),
        cell_volume_A3 / n,
        n_li / cell_volume_A3,
        None if hop_length_A is None else float(hop_length_A),
        None if em_bvse_eV is None else float(em_bvse_eV),
        *[1.0 if family == name else 0.0 for name in _FAMILIES],
    ]


def _matrix(rows: Sequence[Sequence[float | None]]) -> tuple[np.ndarray, np.ndarray]:
    complete = np.asarray([all(v is not None for v in row) for row in rows], dtype=bool)
    width = len(rows[0]) if len(rows) else 0
    values = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        if complete[i]:
            values[i] = np.asarray(row, dtype=np.float64)
    if complete.any() and not np.isfinite(values[complete]).all():
        raise ValueError("feature values must be finite")
    return values, complete


def fit_reference(
    development_rows: Sequence[Sequence[float | None]],
    *,
    chemical_systems: Sequence[str],
    k: int,
    quantile_level: float,
    unscaled_columns: Sequence[int] = (),
) -> dict[str, Any]:
    """Freeze standardization, reference set and tau_d on the development population.

    tau_d uses leave-one-chemical-system-out distances: a unit's neighbours never come
    from its own chemical system.  Sibling hops would otherwise make tau_d too tight
    for genuinely new systems.
    """

    if k < 1:
        raise ValueError("k must be positive")
    if not 0.0 < quantile_level < 1.0:
        raise ValueError("quantile_level must lie in (0, 1)")
    if len(chemical_systems) != len(development_rows):
        raise ValueError("one chemical system per development unit is required")
    values, complete = _matrix(development_rows)
    reference = values[complete]
    groups = np.asarray([str(g) for g, ok in zip(chemical_systems, complete, strict=True) if ok])
    if reference.shape[0] <= k:
        raise ValueError("the reference population needs more than k complete units")
    mean = reference.mean(axis=0)
    sd = reference.std(axis=0, ddof=0)
    sd[sd == 0.0] = 1.0  # a constant feature carries no distance information
    for column in unscaled_columns:  # e.g. family one-hots: fixed unit weight, never z-scored
        mean[column], sd[column] = 0.0, 1.0
    z = (reference - mean) / sd
    distances = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
    distances[groups[:, None] == groups[None, :]] = np.inf  # leave one chemical system out
    nearest = np.sort(distances, axis=1)[:, :k]
    if not np.isfinite(nearest).all():
        raise ValueError("fewer than k reference units outside some chemical system")
    loo = nearest.mean(axis=1)
    return {
        "schema": SUPPORT_SCHEMA,
        "k": k,
        "quantile_level": quantile_level,
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "reference_z": z.tolist(),
        "tau_d": float(np.quantile(loo, quantile_level, method="linear")),
        "n_reference": int(z.shape[0]),
        "n_incomplete_excluded": int((~complete).sum()),
        "n_chemical_systems": len(set(groups.tolist())),
        "threshold_rule": "leave-one-chemical-system-out k-NN distance quantile",
        "unscaled_columns": sorted(int(c) for c in unscaled_columns),
    }


def support_distances(reference: Mapping[str, Any],
                      rows: Sequence[Sequence[float | None]]) -> list[float | None]:
    """Mean distance to the k nearest reference units; ``None`` for incomplete units."""

    values, complete = _matrix(rows)
    z_ref = np.asarray(reference["reference_z"], dtype=np.float64)
    mean = np.asarray(reference["mean"], dtype=np.float64)
    sd = np.asarray(reference["sd"], dtype=np.float64)
    result: list[float | None] = [None] * len(rows)
    for i in np.nonzero(complete)[0]:
        z = (values[i] - mean) / sd
        d = np.sort(np.sqrt(((z_ref - z) ** 2).sum(axis=1)))[: reference["k"]]
        result[int(i)] = float(d.mean())
    return result


def support_decisions(reference: Mapping[str, Any],
                      distances: Sequence[float | None]) -> list[dict[str, Any]]:
    """Supported iff the distance is defined and <= tau_d; incomplete units fail closed."""

    tau = reference["tau_d"]
    return [
        {"supported": d is not None and d <= tau, "distance": d,
         "reason": None if d is not None and d <= tau
         else ("missing_feature" if d is None else "outside_support")}
        for d in distances
    ]
