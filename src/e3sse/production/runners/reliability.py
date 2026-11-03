"""reliability: correct-or-abstain behaviour of the pre-outcome diagnostics (Study 2a).

``amendments.selective_prediction`` and ``endpoints``, on the primary-population
evaluation hops of the transfer stage, per model:

* scores: rho_F, scale_uncertainty, li_host_disagreement, energy_dependence (from the
  B3 frames), latent_support (the leave-one-chemical-system-out distance of the hop's
  converged development band when it is a development unit and the model's branch is
  latent; otherwise undefined), combined (equal-weight percentile ranks; descriptive),
  and the label-free |raw barrier| baseline (descriptive);
* selective_primary: abstained hops receive no SRT prediction (B3 SRT errors, units
  with B3 defined); correct_or_raw_secondary: abstained hops keep raw B0;
* AURC, E-AURC, normalized gain, risk at coverage 0.8, both on all units (undefined
  scores ordered last) and on the defined subset; the analytic random baseline;
* Spearman of each score with the correction benefit |raw error| - |SRT error|,
  with a chemical-system cluster-bootstrap interval;
* P4/H6: Delta-AURC = AURC(rho_F) - AURC_random under selective_primary, one-sided
  cluster-bootstrap p (fraction of replicates >= 0), Holm across eligible Group-A
  models;
* realised coverage of the deployment gate.

Support thresholds never use a target barrier: they were fixed on the training side
(transfer) or the disjoint development population (path_representation).
"""

from __future__ import annotations

import math
from typing import Any

from e3sse.production.runners.context import RunnerContext

SCORES = ("rho_F", "scale_uncertainty", "li_host_disagreement", "energy_dependence")


def _clean(summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in summary.items():
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        out[key] = value
    return out


def model_reliability(rows: list[dict[str, Any]], *, latent_distance: dict[str, float],
                      seed: int, resamples: int) -> dict[str, Any]:
    import numpy as np

    from e3sse.production.runners.inference_stats import (
        cluster_bootstrap,
        one_sided_p,
        percentile_interval,
    )
    from e3sse.selection.selective_prediction import (
        aurc,
        correct_or_raw_curve,
        random_aurc,
        risk_coverage,
        summarize,
        summarize_defined_subset,
    )
    from e3sse.softening.applicability import combined_score
    from e3sse.softening.study1_analysis import spearman_rho

    raw_errors = [abs(r["raw_barrier_eV"] - r["dft_barrier_eV"]) for r in rows]
    srt_errors = [None if r["srt_material_eV"] is None else abs(r["srt_material_eV"] - r["dft_barrier_eV"])
                  for r in rows]
    columns = {name: [r["scores"][name] for r in rows] for name in SCORES}
    columns["latent_support"] = [latent_distance.get(r["hop_id"]) for r in rows]
    columns["combined"] = combined_score({n: columns[n] for n in SCORES})
    columns["abs_raw_barrier_baseline"] = [abs(r["raw_barrier_eV"]) for r in rows]
    corrected = [i for i, e in enumerate(srt_errors) if e is not None]
    clusters = [rows[i]["chemical_system"] for i in corrected]
    selective_errors = [srt_errors[i] for i in corrected]
    benefit = [raw_errors[i] - srt_errors[i] for i in corrected]
    result: dict[str, Any] = {"n_hops": len(rows), "n_with_srt": len(corrected),
                              "random_aurc_selective": random_aurc(selective_errors)
                              if selective_errors else None, "scores": {}}
    for name, values in columns.items():
        selective_scores = [values[i] for i in corrected]
        entry: dict[str, Any] = {}
        if selective_errors:
            entry["selective_primary"] = _clean(summarize(selective_scores, selective_errors))
            curve = risk_coverage(selective_scores, selective_errors)
            entry["selective_primary_curve"] = {"coverage": np.asarray(curve["coverage"]).tolist(),
                                                "risk": np.asarray(curve["risk"]).tolist()}
            entry["selective_primary_defined_subset"] = _clean(
                summarize_defined_subset(selective_scores, selective_errors))
            pairs = [(s, b) for s, b in zip(selective_scores, benefit) if s is not None]
            entry["spearman_with_benefit"] = spearman_rho([p[0] for p in pairs], [p[1] for p in pairs]) \
                if len(pairs) > 2 else None
            scores_array = np.asarray([np.nan if s is None else s for s in selective_scores])
            benefit_array = np.asarray(benefit)

            def association(idx, a=scores_array, b=benefit_array):
                keep = idx[~np.isnan(a[idx])]
                return spearman_rho(a[keep].tolist(), b[keep].tolist()) if keep.size > 2 else None

            entry["spearman_ci95"] = percentile_interval(
                cluster_bootstrap(clusters, association, seed=seed, resamples=resamples))
        secondary_scores = [values[i] if srt_errors[i] is not None else None
                            for i in range(len(rows))]
        curve = correct_or_raw_curve(secondary_scores, srt_errors, raw_errors)
        entry["correct_or_raw_secondary"] = {
            "risk_full_coverage": float(curve["risk"][-1]), "risk_no_correction": float(curve["risk"][0]),
            "n_uncorrectable": curve["n_uncorrectable"]}
        result["scores"][name] = entry
    if selective_errors:
        rho = [columns["rho_F"][i] for i in corrected]
        errors_array = np.asarray(selective_errors)

        def delta(idx):
            return aurc([rho[i] for i in idx], errors_array[idx].tolist()) - \
                random_aurc(errors_array[idx].tolist())

        point = aurc(rho, selective_errors) - random_aurc(selective_errors)
        replicates = cluster_bootstrap(clusters, delta, seed=seed, resamples=resamples)
        result["P4"] = {"delta_aurc": float(point), "ci95": percentile_interval(replicates),
                        "p_one_sided": one_sided_p([-v if v is not None else None
                                                    for v in replicates], null_at_or_below=0.0),
                        "p_resolution": {
                            "bootstrap_replicates_defined": sum(v is not None for v in replicates),
                            "p_below_resolution": one_sided_p(
                                [-v if v is not None else None for v in replicates]) == 0.0},
                        "p_definition": "fraction of cluster-bootstrap replicates with "
                                        "Delta-AURC* >= 0 (approximate percentile-bootstrap p)"}
    result["gate_realised_coverage"] = (sum(bool(r["gate_passes"]) for r in rows) / len(rows)
                                        if rows else None)
    return result


def run(ctx: RunnerContext) -> dict[str, Any]:
    from e3sse.production.runners.inference_stats import holm
    from e3sse.production.runners.pilot import pilot_eligible_models
    from e3sse.production.runners.representation import frozen_path_representation
    from e3sse.protocol.partitions import analysis_seed
    from e3sse.softening.study1_inputs import statistical_constants

    constants = statistical_constants(ctx.parent_protocol)
    records = ctx.upstream("transfer", "transfer_hop_records")
    representation = frozen_path_representation(ctx)["models"]
    population = ctx.upstream("development_population", "population_manifest")
    dev_edge = {u["unit_key"]: u["unit_id"] for u in population["units"] if u["role"] == "development"}
    ctx.require_outcome(role="test", outcome="transfer_errors",
                        outcome_kind="nebdft2k_fixed_geometry_evaluation_barriers",
                        unit_ids=sorted({r["hop_id"] for r in records}))
    eligible = pilot_eligible_models(ctx)
    results = {}
    for model in sorted({r["model"] for r in records}):
        rows = [r for r in records if r["model"] == model and r["population"] == "primary"]
        branch = representation.get(model, {})
        latent = {}
        if branch.get("branch") == "latent_support":
            reference = branch["reference"]
            latent = {dev_edge[key]: distance for key, distance in
                      zip(reference["unit_keys"], reference["loo_distances"]) if key in dev_edge}
        results[model] = model_reliability(
            rows, latent_distance=latent,
            seed=analysis_seed(constants["base_seed"], "cluster_bootstrap", "study2a",
                               "reliability", "published"),
            resamples=constants["bootstrap_resamples"])
    # The family is every eligible model; a model without P4 enters Holm with p = 1.
    p4 = {m: results.get(m, {}).get("P4", {}).get("p_one_sided") for m in eligible}
    adjusted = holm(p4)
    deltas = {m: results.get(m, {}).get("P4", {}).get("delta_aurc") for m in eligible}
    h6 = {"supported": any(p < 0.05 for p in adjusted.values()),
          # refuted needs every model's Delta-AURC defined and >= 0; undefined -> inconclusive
          "refuted": bool(deltas) and all(d is not None and d >= 0 for d in deltas.values()),
          "holm_adjusted_p": adjusted,
          "undefined_p_entered_as_1": sorted(m for m, p in p4.items() if p is None)}
    h6["verdict"] = "supported" if h6["supported"] else "refuted" if h6["refuted"] else "inconclusive"
    outputs = {"reliability_results": ctx.publish("reliability_results.json", {
        "models": results, "P4_H6": h6,
        "family_note": "P4_m is a key-secondary family; Holm within that family only"})}
    return {"outputs": outputs, "summary": {"H6": h6["verdict"]}}
