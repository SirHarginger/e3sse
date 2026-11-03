"""density_ratio: covariate-shift weights w(x) = q_X(x) / p_X(x) for weighted selection.

``amendments.prospective.density_ratio``: the frozen L2-regularized logistic domain
classifier (``selection.density_ratio``, IRLS, deterministic, frozen l2 / max_iter /
tolerance, standardized on the fitting sample, no clipping) is fitted only on the
held-out weight-fit slices converged under the selected model: P-slice systems
(never calibration units) against Q-slice systems (never selected).  Its features
are the label-free descriptor set plus E*_3 and the TM/Hubbard-U indicator.  Weights are then computed for the calibration
units (deployment calibration) and the selectable pool units (discovery).  A unit
with a missing feature gets no weight and cannot enter weighted selection (fail
closed, counted).  Diagnostics: out-of-sample domain AUC on calibration vs pool
covariates, in-sample AUC, effective sample size, maximum weight, quantiles,
overlap fractions, class balance, the fraction of never-selectable test units, and
IRLS convergence.  Label-free.
"""

from __future__ import annotations

from typing import Any

from e3sse.production.runners.context import MissingEvidence, RunnerContext


def density_features(row) -> list[float] | None:
    """Label-free descriptors + E*_3 + TM/Hubbard-U indicator (Study-5 revision
    2026-11-03).  The unit-hop bottleneck gap is never a feature: every pool unit has
    gap 0, so it would zero the weight of every proxy calibration unit."""

    vector = (row or {}).get("vector")
    critical = (row or {}).get("critical_bvse_barrier_eV")
    tm = (row or {}).get("tm_hubbard_u")
    if vector is None or any(v is None for v in vector) or critical is None \
            or not isinstance(tm, bool):
        return None
    return [*vector, float(critical), float(tm)]


def _complete(descriptors, keys):
    rows, kept, missing = [], [], []
    for key in keys:
        features = density_features(descriptors.get(key))
        if features is None:
            missing.append(key)
        else:
            rows.append(features)
            kept.append(key)
    return rows, kept, missing


def run(ctx: RunnerContext) -> dict[str, Any]:
    import numpy as np

    from e3sse.production.runners.ml_neb_units import selected_model
    from e3sse.selection.density_ratio import (
        auc,
        fit_domain_classifier,
        predicted_q_probability,
        shift_weights,
        weight_diagnostics,
    )

    amendments = ctx.frozen_amendments()["prospective"]
    settings = amendments["density_ratio"]["settings"]
    alpha = float(amendments["alpha"])
    model = selected_model(ctx)
    descriptors = ctx.upstream("descriptor_availability", "descriptors")["units"]
    slices = ctx.upstream("weight_slice_ml_neb", "ml_neb_records", model=model)["records"]
    converged = [r["unit_key"] for r in slices if r["status"] == "converged"]
    p_rows, _, p_missing = _complete(descriptors, [k for k in converged if k.startswith("P:")])
    q_rows, _, q_missing = _complete(descriptors, [k for k in converged if k.startswith("Q:")])
    if not p_rows or not q_rows:
        raise MissingEvidence("no converged weight-fit slice units with complete descriptors")
    try:
        classifier = fit_domain_classifier(p_rows, q_rows, l2=float(settings["l2"]),
                                           max_iter=int(settings["max_iter"]),
                                           tolerance=float(settings["tolerance"]))
    except RuntimeError as exc:
        # Frozen estimator rule: non-convergence means no weights (fail closed).
        record = {"model": model, "classifier_status": "not_converged", "detail": str(exc),
                  "calibration_weights": {}, "test_weights": {}}
        return {"outputs": {"density_ratio": ctx.publish("density_ratio.json", record)},
                "summary": {"classifier_status": "not_converged"}}
    calibration = ctx.upstream("deployment_calibration", "deployed_units")["units"]
    pool = ctx.upstream("discovery", "deployed_units")["units"]
    c_rows, c_keys, c_missing = _complete(descriptors, [u["unit_key"] for u in calibration
                                                        if u["ml_neb_status"] == "converged"])
    t_rows, t_keys, t_missing = _complete(descriptors, [u["unit_key"] for u in pool
                                                        if u["ml_neb_status"] == "converged"])
    c_weights = shift_weights(classifier, c_rows)["weights"] if c_rows else np.zeros(0)
    t_weights = shift_weights(classifier, t_rows)["weights"] if t_rows else np.zeros(0)
    prob_c = predicted_q_probability(classifier, c_rows) if c_rows else np.zeros(0)
    prob_t = predicted_q_probability(classifier, t_rows) if t_rows else np.zeros(0)
    diagnostics = weight_diagnostics(np.asarray(c_weights), np.asarray(t_weights), prob_c, prob_t,
                                     q=alpha, overlap_epsilon=float(settings["overlap_epsilon"])) \
        if len(c_rows) and len(t_rows) else {}
    diagnostics["out_of_sample_auc_calibration_vs_pool"] = diagnostics.get("classifier_auc")
    diagnostics["in_sample_auc_slices"] = auc(predicted_q_probability(classifier, p_rows),
                                              predicted_q_probability(classifier, q_rows))
    diagnostics["class_balance"] = {"n_p_slice": len(p_rows), "n_q_slice": len(q_rows)}
    diagnostics["irls_iterations"] = classifier["iterations"]
    record = {"model": model, "classifier_status": "converged", "classifier": classifier,
              "clipping": "off (primary method)",
              "calibration_weights": dict(zip(c_keys, map(float, c_weights))),
              "test_weights": dict(zip(t_keys, map(float, t_weights))),
              "missing_features": {"p_slice": p_missing, "q_slice": q_missing,
                                   "calibration": c_missing, "pool": t_missing},
              "diagnostics": {k: (v.tolist() if hasattr(v, "tolist") else v)
                              for k, v in diagnostics.items()},
              "claim": "estimated weights give approximate control only (Theorem 3.5)"}
    outputs = {"density_ratio": ctx.publish("density_ratio.json", record)}
    return {"outputs": outputs, "summary": {
        "ess": diagnostics.get("calibration_effective_sample_size"),
        "auc": diagnostics.get("classifier_auc"),
        "unselectable_test_fraction": diagnostics.get("unselectable_test_fraction")}}
