"""pilot: the training-only pilot amendment (seal 2) and its internal pilot record.

Parent rules (``repeated_test_pilot_overlap_rule``, ``mde_on_training_pilot``): the
pilot reads only published-train MPLiTrj non-path force information (here the
Study-1 estimates derived from it) and simulated outcomes; it may fix only
``study1_``/``study2_`` settings that no internal-freeze amendment already fixes; it
never reads a nebDFT2k barrier.  Its amendment, bound to this internal freeze by a
hashed record, unseals non-training outcomes for the later stages.

Frozen thresholds bound here are values the protocol already states (classification
thresholds c = 0.3 and 0.5 eV; one-sided alpha 0.05 of the P1 Holm family), so the
pilot tunes nothing from data.  The MDE analysis is the simulation the parent rule
names: per eligible Group-A model (s-bar <= 0.9), barriers of the published-test
evaluation hops are simulated from their BVSE barriers (a covariate) as
raw = s * B + noise, with s drawn from that model's supported published-train
material-scope estimates and s-hat perturbed by their median relative bootstrap
half-width; the distribution of the Eq. 16 rho_SRT is reported per noise level.
"""

from __future__ import annotations

import math
from typing import Any

from e3sse.production.runners.context import (
    MissingEvidence,
    RunnerContext,
)

NOISE_LEVELS_EV = (0.02, 0.05, 0.10)
SIMULATIONS = 2000
ELIGIBLE_S_BAR = 0.9


def study1_tables(ctx: RunnerContext) -> tuple[list[dict], list[dict]]:
    from e3sse.softening.study1_tables import read_csv

    manifest_path = ctx.upstream_path("estimate", "study1_analysis_manifest")
    root = manifest_path.parent
    return (read_csv(root / "derived" / "study1_softening_estimates.csv"),
            read_csv(root / "derived" / "study1_model_summary.csv"))


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def eligible_models(ctx: RunnerContext, summary: list[dict]) -> dict[str, float]:
    """Group-A models with s-bar <= 0.9 (primary population, frozen s_bar_definition)."""

    groups = {name: model["frozen_group"] for name, model in ctx.models.items()}
    eligible = {}
    for row in summary:
        s_bar = _float(row.get("s_bar"))
        if row.get("population") == "primary" and groups.get(row["model"]) == "A" and \
                s_bar is not None and s_bar <= ELIGIBLE_S_BAR:
            eligible[row["model"]] = s_bar
    return dict(sorted(eligible.items()))


def mde_simulation(scales: list[float], relative_noise: float, barriers: list[float], *,
                   noise_eV: float, seed: int, simulations: int = SIMULATIONS) -> dict[str, Any]:
    import numpy as np

    from e3sse.barriers.fixed_geometry import common_population, rho_srt

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    true = np.asarray(barriers, dtype=float)
    hops = [f"h{i}" for i in range(true.size)]
    population = common_population({"B0": hops, "B3": hops}, label="pilot_simulation")
    values = []
    for _ in range(simulations):
        s = rng.choice(np.asarray(scales), size=true.size, replace=True)
        raw = s * true + rng.normal(0.0, noise_eV, size=true.size)
        s_hat = s * np.exp(rng.normal(0.0, relative_noise, size=true.size))
        value = rho_srt(dict(zip(hops, raw)), dict(zip(hops, raw / s_hat)),
                        dict(zip(hops, true)), population)
        if value is not None:
            values.append(value)
    array = np.asarray(values)
    return {"noise_eV": noise_eV, "simulations": int(array.size),
            "rho_srt_median": float(np.median(array)),
            "rho_srt_q05": float(np.quantile(array, 0.05)),
            "rho_srt_q95": float(np.quantile(array, 0.95)),
            "fraction_rho_srt_ge_0.30": float((array >= 0.30).mean()),
            "fraction_rho_srt_ge_0.10": float((array >= 0.10).mean())}


def run(ctx: RunnerContext) -> dict[str, Any]:
    from e3sse.production.runners.gate1_tables import linkage_tables, pinned_g0_inputs
    from e3sse.protocol.contracts import build_pilot_amendment
    from e3sse.protocol.internal_freeze import build_internal_pilot_record
    from e3sse.protocol.partitions import analysis_seed

    parent = ctx.parent_protocol
    internal = ctx.internal_freeze()
    nonpath = sorted(r["frame_id"] for r in ctx.membership
                     if r.get("record_kind") == "mplitrj_frame" and r.get("published_split") == "train"
                     and r.get("raw_file_kind") in ("source", "target"))
    if not nonpath:
        raise MissingEvidence("no published-train non-path frames in the frozen membership")
    ctx.require_outcome(role="train", outcome="study1_derived_softening_estimates",
                        outcome_kind="mplitrj_published_train_dft_forces_energies",
                        unit_ids=nonpath)
    estimates, summary = study1_tables(ctx)
    eligible = eligible_models(ctx, summary)
    evaluation = [r for r in ctx.membership if r.get("record_kind") == "nebdft2k_hop"
                  and r.get("published_split") == "test" and r.get("eligibility_status") == "included"]
    bvse = {}
    for row in linkage_tables(ctx.data_root / "raw", pinned_g0_inputs(ctx.freeze_config))["nebBVSE122k"]:
        value = _float(row["bvse_migration_barrier_eV"])
        if value is not None:
            bvse[row["edge_id"]] = value
    barriers = [bvse[r["hop_id"]] for r in evaluation if r["hop_id"] in bvse]
    base_seed = int(parent["statistical_plan"]["randomness"]["base_seed"])
    mde: dict[str, Any] = {"study2_mde_design": (
        "rho_SRT (Eq. 16) simulated on the published-test evaluation hops: true barrier = "
        "their nebBVSE122k BVSE barrier (covariate), raw = s*B + N(0, noise), s from the "
        "model's supported published-train material-scope estimates, s-hat = s*exp(N(0, "
        "tau)), tau = median relative bootstrap half-width"),
        "study2_mde_evaluation_hops": len(barriers), "study2_mde_models": {}}
    for model in eligible:
        rows = [r for r in estimates if r["model"] == model and r["kind"] == "target_hop"
                and r["estimator_kind"] == "primary" and r["population"] == "primary"
                and r["component_set"] == "all" and r["status"] == "estimated"]
        scales = [_float(r["s_hat"]) for r in rows if _float(r["s_hat"])]
        widths = [(_float(r["ci_high"]) - _float(r["ci_low"])) / (2 * _float(r["s_hat"]))
                  for r in rows if _float(r["ci_high"]) is not None and _float(r["ci_low"])
                  is not None and _float(r["s_hat"])]
        if not scales or not barriers:
            mde["study2_mde_models"][model] = {"status": "insufficient_inputs"}
            continue
        tau = float(sorted(widths)[len(widths) // 2]) if widths else 0.0
        mde["study2_mde_models"][model] = {
            "s_bar": eligible[model], "n_scales": len(scales), "relative_scale_noise": tau,
            "by_noise": [mde_simulation(scales, tau, barriers, noise_eV=noise,
                                        seed=analysis_seed(base_seed, "pilot_mde", model, noise))
                         for noise in NOISE_LEVELS_EV]}
    thresholds = {
        "study2_classification_thresholds_eV": list(
            parent["study_2a_contract"]["thresholds_eV"]),
        "study2_p1_holm_one_sided_alpha": 0.05,
        "study2_p1_eligible_models": sorted(eligible),
    }
    amendment = build_pilot_amendment(parent, partition_records=ctx.membership,
                                      training_unit_ids=nonpath, frozen_thresholds=thresholds,
                                      mde_analysis=mde)
    record = build_internal_pilot_record(internal, parent, amendment)
    outputs = {"pilot_amendment": ctx.publish("pilot_amendment.json", amendment),
               "internal_pilot_record": ctx.publish("internal_pilot_record.json", record)}
    return {"outputs": outputs, "summary": {"amendment_id": amendment["amendment_id"],
                                            "eligible_models": sorted(eligible)}}
