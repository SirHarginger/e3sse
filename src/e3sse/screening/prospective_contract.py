"""Frozen prospective contract: population funnel, family-scale status, model selection.

These functions encode decisions only; they read no data and no outcome.

Population funnel (``amendments.prospective.populations``), each a subset of the
previous one:

    gate1 -> ml_neb_attempted -> ml_neb_converged -> conformal_predictor_eligible
          -> selected
    applicability_supported is a subset of ml_neb_converged

Every converged unit gets exactly one frozen prediction: family SRT when it is
supported and the family scale exists, otherwise the raw ML-NEB barrier, labelled
with its reason.  So predictor-eligible equals converged.  ML-NEB *failure* (no
converged band) is a different event from applicability *abstention* (converged
but outside support).  Formal FDR applies only to the predictor-eligible units that
WCS was run on.  Attrition is reported against the Gate-1 population.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

FAILURE_CATEGORIES = ("optimizer_non_convergence", "calculator_exception", "non_finite_value",
                      "geometry_validity", "timeout", "other_deterministic")
PREDICTION_SOURCES = ("srt_family", "raw_outside_support", "raw_family_scale_unavailable")


def funnel(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and rates with explicit denominators from per-unit status records.

    Each unit has ``attempted`` (bool), ``failure`` (None or a FAILURE_CATEGORIES
    member), ``supported`` (bool, meaningful only when converged) and ``selected``
    (bool).
    """

    gate1 = len(units)
    for unit in units:
        if not unit["attempted"] and unit["failure"] is not None:
            raise ValueError("a unit that was not attempted cannot carry a failure")
    attempted = [u for u in units if u["attempted"]]
    failures = [u for u in attempted if u["failure"] is not None]
    for unit in failures:
        if unit["failure"] not in FAILURE_CATEGORIES:
            raise ValueError(f"unknown failure category {unit['failure']!r}")
    converged = [u for u in attempted if u["failure"] is None]
    supported = [u for u in converged if u["supported"]]
    # Units in the frozen weight-fitting hold-out slice are converged but never enter WCS.
    holdout = [u for u in converged if u.get("weight_fit_holdout", False)]
    wcs_run = [u for u in converged if not u.get("weight_fit_holdout", False)]
    selected = [u for u in wcs_run if u["selected"]]
    if any(u["selected"] for u in units if u not in wcs_run):
        raise ValueError("only units in the WCS-run population can be selected")

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "counts": {"gate1": gate1, "ml_neb_attempted": len(attempted),
                   "ml_neb_converged": len(converged),
                   "applicability_supported": len(supported),
                   "conformal_predictor_eligible": len(converged),
                   "weight_fit_holdout": len(holdout), "wcs_run": len(wcs_run),
                   "selected": len(selected),
                   "failures_by_category": {c: sum(u["failure"] == c for u in failures)
                                            for c in FAILURE_CATEGORIES}},
        "rates": {
            "ml_neb_failure_rate": rate(len(failures), len(attempted)),  # / attempted
            "applicability_coverage": rate(len(supported), len(converged)),  # / converged
            "predictor_coverage": rate(len(converged), gate1),  # / Gate-1 population
            "selection_rate": rate(len(selected), len(wcs_run)),  # / WCS-run units
        },
        "fdr_population": "wcs_run",
    }


def prediction_source(*, supported: bool, family_scale_available: bool) -> str:
    """Hybrid predictor branch.  The raw fallback is labelled and never called family SRT."""

    if not supported:
        return "raw_outside_support"
    if not family_scale_available:
        return "raw_family_scale_unavailable"
    return "srt_family"


def anion_family_record(composition: Mapping[str, float]) -> dict[str, Any]:
    """Frozen dominant-anion-count family plus a descriptive mixed-anion flag.

    The family is the parent rule (strictly largest family by atom count; no family
    element or a tie between the two largest -> other).  ``mixed_anion`` (two or more
    family element groups present) is reported descriptively and never changes the
    family.
    """

    from e3sse.protocol.chemistry import anion_family

    groups = {"oxide": {"O"}, "sulfide": {"S"}, "halide": {"F", "Cl", "Br", "I"},
              "nitride": {"N"}}
    present = sorted(g for g, elements in groups.items()
                     if any(composition.get(e, 0) > 0 for e in elements))
    return {"family": anion_family(composition), "mixed_anion": len(present) >= 2,
            "anion_groups_present": present}


MODEL_SELECTION_RELATIVE_FLOOR = 0.5


def select_prospective_model(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Deterministic lexicographic rule over development summaries of the deployed predictor.

    Order of operations (fixed): feasibility first, then the common hop set.  Each
    record has ``model``, ``convergence_rate`` (on the per-system rate-limiting
    development hops) and ``independent_support_coverage``.  The latter is measured on
    an independent label-free set, never on the development set that defines tau_d,
    where coverage is about the quantile level by construction.  Records also carry
    ``mae_common_eV`` and ``rmse_common_eV`` of the complete deployed predictor on
    the per-system rate-limiting development hops converged for every feasible model.

    1. Feasibility (relative): convergence_rate and independent_support_coverage each
       >= 0.5 x the best value among the candidates.
    2. Primary: lowest MAE.  3. Tie-breakers: lower RMSE, higher convergence, name.
    The selected model's development MAE is winner's-curse biased and is never
    reported as a performance estimate.
    """

    if not records:
        raise ValueError("no candidate models")
    names = [r["model"] for r in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate model names")
    for r in records:
        for key in ("convergence_rate", "independent_support_coverage", "mae_common_eV",
                    "rmse_common_eV"):
            value = r[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{r['model']}: {key} must be finite")
    best_convergence = max(r["convergence_rate"] for r in records)
    best_coverage = max(r["independent_support_coverage"] for r in records)
    feasible_names = {r["model"] for r in records
                      if r["convergence_rate"] >= MODEL_SELECTION_RELATIVE_FLOOR * best_convergence
                      and r["independent_support_coverage"]
                      >= MODEL_SELECTION_RELATIVE_FLOOR * best_coverage}
    if not feasible_names:
        raise ValueError("no feasible model")
    ranked = sorted((r for r in records if r["model"] in feasible_names),
                    key=lambda r: (r["mae_common_eV"], r["rmse_common_eV"],
                                   -r["convergence_rate"], r["model"]))
    return {"selected_model": ranked[0]["model"], "feasible": sorted(feasible_names),
            "excluded_infeasible": sorted(set(names) - feasible_names),
            "ranking": [r["model"] for r in ranked],
            "note": "development MAE of the selected model is not a performance estimate"}
