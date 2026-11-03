"""linear_recalibration_comparator: B5 = a * B0 + b fitted on training-partition barriers.

``amendments.transfer_benchmark.methods.B5_linear_recalibration``: a supervised,
label-using comparator, reported visually separated from the deployable methods.
Ordinary least squares on the published-train primary-population hops (fixed-geometry
raw ML barrier vs DFT barrier), evaluated on the published-test transfer hops on the
common population with B0.  It runs after model selection because its training
barriers belong to the conformal-calibration systems.
"""

from __future__ import annotations

from typing import Any

from e3sse.production.runners.context import RunnerContext

CALIBRATION_BARRIERS = "nebdft2k_conformal_calibration_barriers"
EVALUATION_BARRIERS = "nebdft2k_fixed_geometry_evaluation_barriers"


def fit_linear(x: list[float], y: list[float]) -> tuple[float, float]:
    import numpy as np

    design = np.column_stack([np.asarray(x), np.ones(len(x))])
    (a, b), *_ = np.linalg.lstsq(design, np.asarray(y), rcond=None)
    return float(a), float(b)


def run(ctx: RunnerContext) -> dict[str, Any]:
    from e3sse.barriers.fixed_geometry import common_population, metrics
    from e3sse.production.runners.transfer import dft_barriers, raw_barriers, universe_hops

    parent = ctx.parent_protocol
    thresholds = list(parent["study_2a_contract"]["thresholds_eV"])
    population_of = ctx.partition_manifest["primary_population"]["assignment_by_chemical_system"]
    hops = universe_hops(ctx.membership)
    primary = [h for h in hops if population_of.get(h["chemical_system_id"]) == "primary"]
    train = [h["hop_id"] for h in primary if h["published_split"] == "train"]
    evaluation = [h["hop_id"] for h in primary if h["published_split"] == "test"]
    y_train = dft_barriers(ctx, train, role="train", kind=CALIBRATION_BARRIERS)
    y_eval = dft_barriers(ctx, evaluation, role="test", kind=EVALUATION_BARRIERS)
    results = {}
    for model in sorted(ctx.models):
        single_points = ctx.upstream("transfer", "single_points", model=model)["hops"]
        raw_train = raw_barriers(single_points, train)
        raw_eval = raw_barriers(single_points, evaluation)
        fit_hops = sorted(raw_train)
        a, b = fit_linear([raw_train[h] for h in fit_hops], [y_train[h] for h in fit_hops])
        b5 = {h: a * raw_eval[h] + b for h in raw_eval}
        population = common_population({"B0_raw": list(raw_eval), "B5": list(b5)},
                                       label=f"{model}:B5")
        results[model] = {"a": a, "b": b, "n_fit": len(fit_hops), "population": population,
                          "raw": metrics(raw_eval, y_eval, population, thresholds),
                          "B5": metrics(b5, y_eval, population, thresholds),
                          "role": "supervised_comparator_visually_separated"}
    outputs = {"linear_recalibration": ctx.publish("linear_recalibration.json", results)}
    return {"outputs": outputs, "summary": {m: r["B5"]["mae_eV"] for m, r in results.items()}}
