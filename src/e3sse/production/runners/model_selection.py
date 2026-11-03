"""model_selection: the frozen development-set rule over the complete deployed predictor.

``amendments.prospective.model_selection_objective`` and
``screening.prospective_contract.select_prospective_model``:

* convergence_rate: converged / attempted development ML-NEB units;
* independent_support_coverage: supported / converged on the label-free Q weight-fit
  slice (never the development set that defines tau_d);
* feasibility (relative 0.5 floor) first, then MAE/RMSE of the complete deployed
  predictor (support gate, family SRT or labelled raw, G1 contingency) on the
  per-system rate-limiting development hops converged for every feasible model.

Development barriers are read only through the pinned development manifest
(purpose ``prospective_model_selection``).  The selected model's development MAE is
winner's-curse biased and is never reported as a performance estimate.
"""

from __future__ import annotations

from typing import Any

from e3sse.production.runners.context import RunnerContext, RunnerValidationError

DEVELOPMENT_BARRIERS = "nebdft2k_development_barriers"


def model_summary(model: str, dev_records, slice_records, branch, descriptors, scales, *,
                  g1_passes: bool) -> dict[str, Any]:
    from e3sse.production.runners.deployed import (
        predict,
        support_decisions,
        support_rows,
        unit_family,
    )

    attempted = len(dev_records)
    converged = [r for r in dev_records if r["status"] == "converged"]
    dev_support = support_decisions(branch, support_rows(branch, converged, descriptors),
                                    development=True)
    predictions = {}
    for record in converged:
        decision = dev_support.get(record["unit_key"], {"supported": False})
        predictions[record["unit_id"]] = predict(
            record, family=unit_family(descriptors, record["unit_key"]), scales=scales,
            supported=decision["supported"], g1_passes=g1_passes)
    pool_slice = [r for r in slice_records if r["unit_key"].startswith("Q:")]
    pool_converged = [r for r in pool_slice if r["status"] == "converged"]
    slice_support = support_decisions(branch, support_rows(branch, pool_converged, descriptors))
    coverage = (sum(d["supported"] for d in slice_support.values()) / len(pool_converged)
                if pool_converged else 0.0)
    return {"model": model, "branch": branch.get("branch"), "attempted": attempted,
            "converged": len(converged),
            "convergence_rate": len(converged) / attempted if attempted else 0.0,
            "independent_support_coverage": coverage,
            "slice_converged": len(pool_converged), "slice_attempted": len(pool_slice),
            "predictions": predictions}


def run(ctx: RunnerContext) -> dict[str, Any]:
    import numpy as np

    from e3sse.production.runners.ml_neb_units import require_development_matches_freeze
    from e3sse.production.runners.representation import frozen_path_representation
    from e3sse.screening.prospective_contract import (
        MODEL_SELECTION_RELATIVE_FLOOR,
        select_prospective_model,
    )

    descriptors = ctx.upstream("descriptor_availability", "descriptors")["units"]
    representation = frozen_path_representation(ctx)["models"]
    scales = ctx.upstream("family_scale", "family_scales")["scales"]
    g1 = ctx.upstream("transfer", "g1_decision")
    manifest = ctx.upstream("development_population", "development_manifest")
    summaries = {}
    for model in sorted(representation):
        dev = require_development_matches_freeze(ctx, model)["records"]
        slices = ctx.upstream("weight_slice_ml_neb", "ml_neb_records", model=model)["records"]
        summaries[model] = model_summary(model, dev, slices, representation[model], descriptors,
                                         scales[model], g1_passes=bool(g1["passes"]))
    best_conv = max(s["convergence_rate"] for s in summaries.values())
    best_cov = max(s["independent_support_coverage"] for s in summaries.values())
    feasible = [m for m, s in summaries.items()
                if s["convergence_rate"] >= MODEL_SELECTION_RELATIVE_FLOOR * best_conv
                and s["independent_support_coverage"] >= MODEL_SELECTION_RELATIVE_FLOOR * best_cov]
    if not feasible:
        raise RunnerValidationError("no feasible model under the relative 0.5 floors")
    common = sorted(set.intersection(*[{h for h, p in summaries[m]["predictions"].items() if p}
                                       for m in feasible]))
    if not common:
        raise RunnerValidationError("no development hop has a prediction from every feasible model")
    ctx.require_outcome(role="test", outcome="development_dft_barriers",
                        outcome_kind=DEVELOPMENT_BARRIERS, unit_ids=common,
                        purpose="prospective_model_selection", development_manifest=manifest)
    from e3sse.production.runners.transfer import dft_barriers_unchecked

    reference = dft_barriers_unchecked(ctx, common)
    # Feasibility is decided once, over every model.  Only feasible models are scored,
    # all on the same common hops; infeasible models are never ranked or scored.
    records, contract_records = [], []
    for model, summary in summaries.items():
        record = {"model": model, "convergence_rate": summary["convergence_rate"],
                  "independent_support_coverage": summary["independent_support_coverage"],
                  "feasible": model in feasible}
        if model in feasible:
            errors = np.asarray([summary["predictions"][h]["prediction_eV"] - reference[h]
                                 for h in common])
            record.update(mae_common_eV=float(np.abs(errors).mean()),
                          rmse_common_eV=float(np.sqrt((errors ** 2).mean())), n_common=len(common))
            contract_records.append(record)
        else:
            record.update(mae_common_eV=None, rmse_common_eV=None, n_common=0,
                          status="infeasible_not_scored")
            # placeholder errors: the contract excludes infeasible models before ranking
            contract_records.append({**record, "mae_common_eV": 0.0, "rmse_common_eV": 0.0})
        records.append(record)
    selection = select_prospective_model(contract_records)
    if selection["feasible"] != sorted(feasible):
        raise RunnerValidationError("contract feasibility differs from the runner's")
    outputs = {"model_selection": ctx.publish("model_selection.json", {
        "selection": selection, "records": records, "common_development_hops": len(common),
        "g1_passes": g1["passes"],
        "summaries": {m: {k: v for k, v in s.items() if k != "predictions"}
                      for m, s in summaries.items()},
        "winners_curse": "the selected model's development MAE is not a performance estimate"})}
    return {"outputs": outputs, "summary": {"selected_model": selection["selected_model"]}}
