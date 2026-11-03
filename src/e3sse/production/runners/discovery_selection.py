"""discovery_selection: weighted conformalized selection and the candidate evidence package.

``amendments.prospective.weighted_conformal`` (published Biometrika eq. 5/6 with the
focal test weight w_j; ``selection.conformal.weighted_conformalized_selection``):

* calibration units: converged deployment-calibration units (P, weight-fit slice
  excluded) with a weight; their DFT barriers are read here, after the pilot seal;
* test units: converged, weighted, selectable pool units (Q, slice excluded);
* scores: ``orient_low_barrier_scores`` for the alternative Y < c (c = 0.5 eV);
* primary tie handling: exact lexicographic (score, tie key), tie key =
  ``unit_tie_key(internal protocol ID, "study5_wcs", unit ID)`` (SHA-256 bound);
  sensitivity: the conservative ``e3sse_conservative_le`` rule, and the unweighted
  selection, both labelled;
* q = alpha = 0.2.

The FDR statement covers only the predictor-eligible population WCS ran on; the full
funnel (Gate 1 -> attempted -> converged -> supported -> eligible -> selected), with
ML-NEB failures and support abstentions as separate causes, is reported against the
Gate-1 population.  Candidates are prospective screening results with approximate FDR
control, not confirmed superionic conductors.
"""

from __future__ import annotations

from typing import Any

from e3sse.production.runners.context import RunnerContext

CALIBRATION_BARRIERS = "nebdft2k_conformal_calibration_barriers"
ANALYSIS_ID = "study5_wcs"


def funnel_units(population, pool_units, slice_records, bvse_index) -> list[dict[str, Any]]:
    """Funnel records for every Gate-1 pool unit (Gate 0 passed)."""

    by_key = {u["unit_key"]: u for u in pool_units}
    slice_by_key = {r["unit_key"]: r for r in slice_records}
    units = []
    for unit in population["units"]:
        if unit["role"] not in ("pool", "pool_weight_slice"):
            continue
        key = unit["unit_key"]
        holdout = unit["role"] == "pool_weight_slice"
        if holdout:
            record = slice_by_key.get(key)
            status = record["status"] if record else None
            failure = record["failure_category"] if record else None
            supported = False
        else:
            record = by_key.get(key)
            status = record["ml_neb_status"] if record else None
            failure = record["failure_category"] if record else None
            supported = bool(record and record["supported"])
        units.append({"unit_key": key, "attempted": status is not None,
                      "failure": failure if status is not None and status != "converged" else None,
                      "supported": supported, "selected": False, "weight_fit_holdout": holdout,
                      "not_attempted_reason": None if status is not None else
                      (bvse_index.get(key) or {}).get("status", "no_bvse_path")})
    return units


def run(ctx: RunnerContext) -> dict[str, Any]:
    import numpy as np

    from e3sse.production.runners.ml_neb_units import selected_model
    from e3sse.production.runners.transfer import dft_barriers
    from e3sse.screening.prospective_contract import funnel
    from e3sse.selection.conformal import (
        orient_low_barrier_scores,
        unit_tie_key,
        weighted_conformalized_selection,
    )

    prospective = ctx.frozen_amendments()["prospective"]
    alpha, threshold = float(prospective["alpha"]), float(prospective["threshold_eV"])
    protocol_id = ctx.internal_freeze()["protocol_id"]
    model = selected_model(ctx)
    weights = ctx.upstream("density_ratio", "density_ratio")
    calibration = [u for u in ctx.upstream("deployment_calibration", "deployed_units")["units"]
                   if u["prediction"] is not None and u["unit_key"] in weights["calibration_weights"]]
    pool = ctx.upstream("discovery", "deployed_units")["units"]
    tests = [u for u in pool if u["prediction"] is not None and u["unit_key"] in weights["test_weights"]]
    population = ctx.upstream("development_population", "population_manifest")
    slices = ctx.upstream("weight_slice_ml_neb", "ml_neb_records", model=model)["records"]
    units = funnel_units(population, pool, slices, ctx.upstream("bvse_reproduction", "bvse_path_index"))
    result: dict[str, Any] = {"model": model, "alpha": alpha, "threshold_eV": threshold}
    if weights.get("classifier_status") != "converged" or not calibration or not tests:
        result["status"] = "not_run: no converged weights or empty calibration/test population"
    else:
        reference = dft_barriers(ctx, [u["unit_id"] for u in calibration], role="train",
                                 kind=CALIBRATION_BARRIERS)
        cal_pred = np.asarray([u["prediction"]["prediction_eV"] for u in calibration])
        cal_y = np.asarray([reference[u["unit_id"]] for u in calibration])
        test_pred = np.asarray([u["prediction"]["prediction_eV"] for u in tests])
        cal_scores, test_scores = orient_low_barrier_scores(cal_pred, cal_y, test_pred, threshold)
        cal_w = np.asarray([weights["calibration_weights"][u["unit_key"]] for u in calibration])
        test_w = np.asarray([weights["test_weights"][u["unit_key"]] for u in tests])
        cal_keys = [unit_tie_key(protocol_id, ANALYSIS_ID, u["unit_id"]) for u in calibration]
        test_keys = [unit_tie_key(protocol_id, ANALYSIS_ID, u["unit_id"]) for u in tests]
        primary = weighted_conformalized_selection(
            cal_scores, cal_w, test_scores, test_w, alpha, tie_rule="tie_free_lexicographic",
            calibration_tie_keys=cal_keys, test_tie_keys=test_keys)
        conservative = weighted_conformalized_selection(cal_scores, cal_w, test_scores, test_w, alpha,
                                                        tie_rule="e3sse_conservative_le")
        unweighted = weighted_conformalized_selection(
            cal_scores, np.ones_like(cal_w), test_scores, np.ones_like(test_w), alpha,
            tie_rule="tie_free_lexicographic", calibration_tie_keys=cal_keys,
            test_tie_keys=test_keys)
        selected = {u["unit_key"] for u, s in zip(tests, primary["selected"]) if s}
        for unit in units:
            unit["selected"] = unit["unit_key"] in selected
        candidates = [{"unit_key": u["unit_key"], "unit_id": u["unit_id"],
                       "anion_family": u["anion_family"], "raw_ml_neb_eV": u["prediction"]["raw_eV"],
                       "deployed_prediction_eV": u["prediction"]["prediction_eV"],
                       "prediction_source": u["prediction"]["source"],
                       "support_distance": u["support_distance"],
                       "weight": float(w), "conformal_p": float(p),
                       "selected": bool(s), "selected_conservative_tie_rule": bool(sc),
                       "selected_unweighted": bool(su),
                       "validation_level": "prospective screening, approximate FDR control"}
                      for u, w, p, s, sc, su in zip(tests, test_w, primary["pvalues"],
                                                    primary["selected"], conservative["selected"],
                                                    unweighted["selected"])]
        result.update(status="run", n_calibration=len(calibration), n_test=len(tests),
                      n_selected=int(primary["selected"].sum()),
                      n_selected_conservative=int(conservative["selected"].sum()),
                      n_selected_unweighted=int(unweighted["selected"].sum()),
                      candidates=sorted(candidates, key=lambda c: (not c["selected"], c["conformal_p"],
                                                                   c["unit_id"])),
                      fdr_scope="the predictor-eligible WCS population only",
                      guarantee="approximate FDR control under the stated assumptions "
                                "(Theorem 3.5); not a confirmed superionic conductor list")
    result["funnel"] = funnel(units)
    outputs = {"discovery_selection": ctx.publish("discovery_selection.json", result)}
    return {"outputs": outputs, "summary": {"status": result["status"],
                                            "selected": result.get("n_selected")}}
