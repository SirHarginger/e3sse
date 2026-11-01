"""Fixed-geometry barriers on DFT-NEB images: raw, SRT and paired comparisons.

Parent ``study_2a_contract``: ML single points on the exact DFT-NEB images (never
ML-NEB), barrier = max image energy - min image energy (frozen
``inclusion_exclusion_rules.barrier_definition``, LiTraj ``get_barrier``), SRT =
raw / s_hat.  The barrier is invariant to an additive energy offset, so per-model
energy references cancel.

Paired comparisons use one frozen common hop set on which every compared method is
defined.  Its canonical SHA-256 travels with every metric.  Methods are never
compared on different populations.  Keep DFT, raw, SRT and fitted values in
separately named fields.  Nothing here converts one into another.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

COMMON_POPULATION_SCHEMA = "e3sse.fixed-geometry-transfer.common-population"


def barrier(image_energies_eV: Sequence[float]) -> float:
    """max - min over the path images (eV); needs >= 2 finite energies."""

    energies = np.asarray(image_energies_eV, dtype=np.float64)
    if energies.ndim != 1 or energies.size < 2 or not np.isfinite(energies).all():
        raise ValueError("a barrier needs at least two finite image energies")
    return float(energies.max() - energies.min())


def srt_barrier(raw_barrier_eV: float, s_hat: float | None) -> float | None:
    """Raw / s_hat; ``None`` (unsupported, no fallback) when s_hat is undefined."""

    if s_hat is None or not np.isfinite(s_hat) or s_hat <= 0.0:
        return None
    return float(raw_barrier_eV / s_hat)


def common_population(
    hop_ids_by_method: Mapping[str, Sequence[str]], *, label: str
) -> dict[str, Any]:
    """Hops defined for every method, sorted, with a canonical SHA-256 and exclusions."""

    if not hop_ids_by_method:
        raise ValueError("common population needs at least one method")
    sets = {method: set(map(str, hops)) for method, hops in hop_ids_by_method.items()}
    union = set().union(*sets.values())
    common = sorted(set.intersection(*sets.values()))
    artifact = {
        "schema_version": COMMON_POPULATION_SCHEMA,
        "label": label,
        "methods": sorted(sets),
        "hop_ids": common,
        "n_common": len(common),
        "n_excluded": len(union) - len(common),
        "excluded_by_method": {m: len(union - s) for m, s in sorted(sets.items())},
    }
    artifact["population_sha256"] = hashlib.sha256(
        json.dumps(common, separators=(",", ":")).encode()
    ).hexdigest()
    return artifact


def _paired(predicted: Mapping[str, float], reference: Mapping[str, float],
            hop_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    missing = [h for h in hop_ids if h not in predicted or h not in reference
               or predicted[h] is None or reference[h] is None]
    if missing:
        raise ValueError(f"{len(missing)} hops of the common population lack a value")
    return (np.asarray([predicted[h] for h in hop_ids], dtype=np.float64),
            np.asarray([reference[h] for h in hop_ids], dtype=np.float64))


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Average-rank Spearman correlation (the project's single implementation)."""

    from e3sse.softening.study1_analysis import spearman_rho

    return spearman_rho(x.tolist(), y.tolist())


def classification(predicted: np.ndarray, reference: np.ndarray, threshold_eV: float) -> dict:
    """A hop is "fast" when its barrier is below c (Research Design 3.8)."""

    pred = predicted < threshold_eV
    true = reference < threshold_eV
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall > 0 else None)
    return {"threshold_eV": threshold_eV, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def metrics(predicted: Mapping[str, float], reference: Mapping[str, float],
            population: Mapping[str, Any], thresholds_eV: Sequence[float]) -> dict[str, Any]:
    """Point metrics of one predictor on the frozen common population."""

    hops = population["hop_ids"]
    if not hops:
        raise ValueError("metrics need a non-empty common population")
    p, r = _paired(predicted, reference, hops)
    error = p - r
    return {
        "population_sha256": population["population_sha256"],
        "n": len(hops),
        "mae_eV": float(np.abs(error).mean()),
        "rmse_eV": float(np.sqrt((error ** 2).mean())),
        "signed_mean_error_eV": float(error.mean()),
        "median_absolute_error_eV": float(np.median(np.abs(error))),
        "spearman": spearman(p, r),
        "classification": [classification(p, r, c) for c in thresholds_eV],
    }


def rho_srt(raw: Mapping[str, float], srt: Mapping[str, float], reference: Mapping[str, float],
            population: Mapping[str, Any]) -> float | None:
    """Research Design Eq. 16: 1 - mean|SRT - DFT| / mean|raw - DFT| on paired hops."""

    hops = population["hop_ids"]
    raw_p, ref = _paired(raw, reference, hops)
    srt_p, _ = _paired(srt, reference, hops)
    denominator = np.abs(raw_p - ref).mean()
    if denominator == 0:
        return None
    return float(1.0 - np.abs(srt_p - ref).mean() / denominator)
