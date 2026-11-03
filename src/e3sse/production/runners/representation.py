"""path_representation: per-model path vectors, acceptance and the support reference.

``amendments.representation.path_representation`` (frozen before any outcome):

* mapping rule: images ordered so the endpoint with the lower ML energy comes first
  (ties keep the BVSE order); saddle = interior image with the highest ML energy
  (ties: lowest index after orientation); z_path = [z(initial), z(saddle), z(final),
  mean over interior z, population SD over interior z] of the migrating-Li row;
* acceptance: every band image finite and deterministic for the pinned capture,
  atom order preserved, the capture environment-sensitive (pinned probe report), and
  the label-free degeneracy check on the development population: coefficient of
  variation of the leave-one-chemical-system-out k-NN distances >= 0.05 and of a
  1000-system-resample bootstrap of tau_d <= 0.10;
* otherwise the descriptor-space fallback (configs/descriptor_fallback_manifest.json)
  applies, and without descriptors reliability-gated prospective SRT is unavailable.

The support reference (z-scoring, reference rows, tau_d = Q_0.8 of the LOCSO k = 5
distances) is fitted on the development population only.  Label-free.
"""

from __future__ import annotations

from typing import Any

from e3sse.production.runners.context import (
    MissingEvidence,
    RunnerContext,
    RunnerValidationError,
    canonical_sha256,
)

SUPPORT_K = 5  # amendments.representation.support_metric (k = 5 nearest development units)
LOCSO_CV_MIN = 0.05
TAU_BOOTSTRAP_CV_MAX = 0.10
TAU_BOOTSTRAP_RESAMPLES = 1000


def path_vector(embeddings: list[list[float]], energies: list[float]) -> list[float]:
    """The frozen mapping rule (orientation, saddle, five blocks)."""

    import numpy as np

    z = np.asarray(embeddings, dtype=float)
    e = list(energies)
    if e[-1] < e[0]:  # lower-energy endpoint first; ties keep the BVSE order
        z, e = z[::-1], e[::-1]
    interior = list(range(1, len(e) - 1))
    if not interior:
        raise ValueError("a band needs interior images")
    saddle = max(interior, key=lambda i: (e[i], -i))
    inner = z[1:-1]
    return np.concatenate([z[0], z[saddle], z[-1], inner.mean(axis=0),
                           inner.std(axis=0, ddof=0)]).tolist()


def degeneracy(reference: dict[str, Any], seed: int) -> dict[str, Any]:
    """LOCSO k-NN distance CV and the tau_d system-bootstrap CV (frozen bounds)."""

    import numpy as np

    from e3sse.softening.frame_statistics import iter_bootstrap_indices

    loo = np.asarray(reference["loo_distances"], dtype=float)
    groups = np.asarray(reference["reference_chemical_systems"])
    systems = sorted(set(groups.tolist()))
    members = [np.nonzero(groups == system)[0] for system in systems]
    taus = []
    for block in iter_bootstrap_indices(seed, len(systems), TAU_BOOTSTRAP_RESAMPLES):
        for draw in np.atleast_2d(block):
            values = np.concatenate([loo[members[i]] for i in draw])
            taus.append(float(np.quantile(values, reference["quantile_level"], method="linear")))
    taus = np.asarray(taus)
    locso_cv = float(loo.std(ddof=0) / loo.mean()) if loo.mean() > 0 else 0.0
    # Collapsed embeddings (every distance 0) leave the CV undefined: recorded as None
    # with a reason and a failed check, never as a non-finite number.
    tau_cv = float(taus.std(ddof=0) / taus.mean()) if taus.mean() > 0 else None
    return {"locso_cv": locso_cv, "locso_cv_min": LOCSO_CV_MIN,
            "tau_bootstrap_cv": tau_cv, "tau_bootstrap_cv_max": TAU_BOOTSTRAP_CV_MAX,
            "tau_bootstrap_cv_undefined_reason": None if tau_cv is not None
            else "tau_d bootstrap mean is zero (collapsed embeddings)",
            "tau_bootstrap_resamples": int(taus.size),
            "passes": locso_cv >= LOCSO_CV_MIN and tau_cv is not None
            and tau_cv <= TAU_BOOTSTRAP_CV_MAX,
            "bootstrap_note": "systems resampled with replacement; tau_d* is the frozen quantile "
                              "of the resampled units' LOCSO distances (standardization held "
                              "fixed at the development fit)"}


def frozen_path_representation(ctx: RunnerContext) -> dict[str, Any]:
    """The pre-freeze path representation, accepted post-freeze only if it was built under
    the frozen representation amendment and gate quantile (never a stale definition)."""

    artifact = ctx.upstream("path_representation", "path_representation")
    frozen = ctx.frozen_amendments()
    expected = (canonical_sha256(frozen["representation"]),
                float(frozen["deployment_gate"]["quantile_level"]), SUPPORT_K)
    observed = (artifact.get("representation_amendment_sha256"),
                artifact.get("quantile_level"), artifact.get("support_k"))
    if observed != expected:
        raise RunnerValidationError("path_representation was built under a representation "
                                    "definition other than the frozen one")
    return artifact


def descriptor_reference(dev_keys: list[str], descriptors: dict[str, Any], quantile: float
                         ) -> tuple[dict[str, Any] | None, str | None]:
    from e3sse.screening.applicability_support import FAMILY_COLUMNS, fit_reference

    rows = [descriptors[key]["vector"] for key in dev_keys]
    usable = [(key, row) for key, row in zip(dev_keys, rows)
              if row is not None and all(v is not None for v in row)]
    if len(usable) <= SUPPORT_K:
        return None, "descriptor_rows_unavailable"
    try:
        return fit_reference([row for _, row in usable],
                             chemical_systems=[descriptors[k]["chemical_system"] for k, _ in usable],
                             k=SUPPORT_K, quantile_level=quantile,
                             unscaled_columns=FAMILY_COLUMNS) | {
            "unit_keys": [k for k, _ in usable], "source": "descriptor_space"}, None
    except ValueError as exc:
        return None, f"descriptor_reference: {exc}"


def model_branch(model: str, records: list[dict[str, Any]], pin: dict[str, Any],
                 descriptors: dict[str, Any], dev_keys: list[str], *, quantile: float,
                 seed: int, environment_sensitive: bool) -> dict[str, Any]:
    from e3sse.screening.applicability_support import fit_reference

    result: dict[str, Any] = {"model": model, "latent_status": pin["latent_support_status"]}
    latent_reasons: list[str] = []
    if pin["latent_support_status"] != "validated":
        latent_reasons.append("latent_representation_not_validated")
    else:
        converged = [r for r in records if r["status"] == "converged"]
        if not all(r.get("embedding_checks", {}).get("finite") and
                   r["embedding_checks"].get("deterministic") for r in converged):
            latent_reasons.append("band_image_embedding_not_finite_or_not_deterministic")
        if not environment_sensitive:
            latent_reasons.append("capture_not_environment_sensitive")
        rows = {r["unit_key"]: path_vector(r["migrating_li_embeddings"], r["energies_eV"])
                for r in converged if "migrating_li_embeddings" in r}
        systems = {d["unit_key"]: d["chemical_system"] for d in (
            {"unit_key": k, **descriptors.get(k, {})} for k in rows)}
        try:
            reference = fit_reference(list(rows.values()), chemical_systems=[systems[k] for k in rows],
                                      k=SUPPORT_K, quantile_level=quantile) | {
                "unit_keys": list(rows), "source": "frozen_model_representation"}
            result["degeneracy"] = degeneracy(reference, seed)
            if not result["degeneracy"]["passes"]:
                latent_reasons.append("degeneracy_check_failed")
        except ValueError as exc:
            reference = None
            latent_reasons.append(f"latent_reference: {exc}")
        if not latent_reasons:
            result.update(branch="latent_support", reference=reference, path_vectors=rows)
            return result
    result["latent_rejection_reasons"] = latent_reasons
    reference, reason = descriptor_reference(dev_keys, descriptors, quantile)
    if reference is not None:
        result.update(branch="descriptor_space", reference=reference)
    else:
        result.update(branch="unavailable", unavailable_reason=reason)
    return result


def run(ctx: RunnerContext) -> dict[str, Any]:
    from pathlib import Path

    from e3sse.production.runners.context import load_json
    from e3sse.protocol.partitions import analysis_seed

    amendments = ctx.active_protocol["amendments"]  # pre-freeze stage
    quantile = float(amendments["deployment_gate"]["quantile_level"])
    pins = amendments["representation"]["embedding_capability_evidence"]["reports"]
    descriptors = ctx.upstream("descriptor_availability", "descriptors")["units"]
    population = ctx.upstream("development_population", "population_manifest")
    dev_keys = sorted(u["unit_key"] for u in population["units"] if u["role"] == "development")
    base_seed = int(ctx.parent_protocol["statistical_plan"]["randomness"]["base_seed"])
    models = ctx.upstream_models("development_ml_neb")
    if not models:
        raise MissingEvidence("no development ML-NEB records")
    branches = {}
    for model in models:
        records = ctx.upstream("development_ml_neb", "ml_neb_records", model=model)["records"]
        report = load_json(Path(pins[model]["path"])) if pins[model]["latent_support_status"] == \
            "validated" else {}
        sensitive = all(c.get("environment_sensitive") for c in report.get("structure_checks", [])) \
            if report else False
        branches[model] = model_branch(
            model, records, pins[model], descriptors, dev_keys, quantile=quantile,
            seed=analysis_seed(base_seed, "cluster_bootstrap",
                               "path_representation_tau_bootstrap", model),
            environment_sensitive=sensitive)
    outputs = {"path_representation": ctx.publish("path_representation.json", {
        "support_k": SUPPORT_K, "quantile_level": quantile, "models": branches,
        "representation_amendment_sha256": canonical_sha256(amendments["representation"]),
        "label_free": True})}
    return {"outputs": outputs,
            "summary": {model: b["branch"] for model, b in branches.items()}}
