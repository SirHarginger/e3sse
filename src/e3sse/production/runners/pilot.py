"""pilot: the training-only pilot amendment (seal 2) and its internal pilot record.

Parent rules (``repeated_test_pilot_overlap_rule``, ``mde_on_training_pilot``): the
pilot reads only published-train MPLiTrj non-path force information (here the
Study-1 estimates derived from it) and simulated outcomes; it may fix only
``study1_``/``study2_`` settings that no internal-freeze amendment already fixes; it
never reads a nebDFT2k barrier.  Its amendment, bound to this internal freeze by a
hashed record, unseals non-training outcomes for the later stages.

Frozen thresholds bound here are values the protocol already states (classification
thresholds c = 0.3 and 0.5 eV; one-sided alpha 0.05 of the P1 Holm family), so the
pilot tunes nothing from data.  The minimum-detectable-effect analysis is the
simulation the parent rule names (``mde_on_training_pilot``), per eligible Group-A
model (s-bar <= 0.9):

* scales s and their relative uncertainty come from the regime-matched training
  diagnostics (published-train primary hops, endpoint-relaxation frames only, one
  value per material: the deployment-gate threshold computation), read with purpose
  ``threshold_selection`` on non-path frames; tau = median relative 95% half-width / 1.96;
* true barriers are the published-test evaluation hops' nebBVSE122k BVSE barriers (a
  covariate, never a DFT barrier), clustered by chemical system;
* raw = s B + N(0, noise); the correction applied is (s-hat)^lambda with
  s-hat = s exp(N(0, tau)), so lambda = 0 is no correction and lambda = 1 full SRT;
* each simulated data set is tested as P1 is: one-sided cluster-bootstrap test of
  rho_SRT > 0 at the strictest Holm level alpha/m (m eligible models);
* power(lambda) and the true effect rho(lambda) (noise-free median) give the MDE: the
  smallest true rho_SRT with power >= 0.8.
"""

from __future__ import annotations

import math
from typing import Any

from e3sse.production.runners.context import (
    MissingEvidence,
    RunnerContext,
)
from e3sse.production.stage_graph import TRAIN_FORCES

NOISE_LEVELS_EV = (0.02, 0.05, 0.10)
SIMULATIONS = 400
BOOTSTRAP_RESAMPLES = 400
EFFECT_GRID = tuple(i / 10 for i in range(11))
TARGET_POWER = 0.8
CI95_Z = 1.959963984540054  # two-sided 95% normal quantile: half-width -> SD
ELIGIBLE_S_BAR = 0.9


def study1_tables(ctx: RunnerContext) -> tuple[list[dict], list[dict]]:
    """Study-1 derived tables (training forces of every published-train frame)."""

    from e3sse.softening.study1_tables import read_csv

    frames = sorted(r["frame_id"] for r in ctx.membership
                    if r.get("record_kind") == "mplitrj_frame" and r.get("published_split") == "train")
    ctx.require_outcome(role="train", outcome="study1_derived_softening_estimates",
                        outcome_kind=TRAIN_FORCES, unit_ids=frames)
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


def pilot_eligible_models(ctx: RunnerContext) -> dict[str, float]:
    """P1 eligibility as fixed by the pilot amendment (never recomputed downstream)."""

    thresholds = ctx.upstream("pilot", "pilot_amendment")["frozen_thresholds"]
    return dict(sorted(thresholds["study2_p1_eligible_s_bar"].items()))


def mde_simulation(scales: list[float], relative_sd: float, barriers: list[float],
                   clusters: list[str], *, noise_eV: float, alpha: float, seed: int,
                   simulations: int = SIMULATIONS, resamples: int = BOOTSTRAP_RESAMPLES
                   ) -> dict[str, Any]:
    """Power of the P1 test over partial-correction effects; MDE at TARGET_POWER."""

    from statistics import NormalDist

    import numpy as np

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    true = np.asarray(barriers, dtype=float)
    names = sorted(set(clusters))
    member = np.zeros((len(names), true.size))
    for j, cluster in enumerate(clusters):
        member[names.index(cluster), j] = 1.0
    z = NormalDist().inv_cdf(1.0 - alpha)
    grid = []
    for effect in EFFECT_GRID:
        rejected, estimates, noiseless = 0, [], []
        for _ in range(simulations):
            s = rng.choice(np.asarray(scales), size=true.size, replace=True)
            s_hat = s * np.exp(rng.normal(0.0, relative_sd, size=true.size))
            for noise, sink in ((noise_eV, estimates), (0.0, noiseless)):
                raw = s * true + (rng.normal(0.0, noise, size=true.size) if noise else 0.0)
                corrected = raw / s_hat ** effect
                raw_c, srt_c = member @ np.abs(raw - true), member @ np.abs(corrected - true)
                if raw_c.sum() == 0:
                    continue
                rho = 1.0 - srt_c.sum() / raw_c.sum()
                sink.append(rho)
                if noise:
                    weights = rng.multinomial(len(names), [1 / len(names)] * len(names),
                                              size=resamples)
                    denominators = weights @ raw_c
                    keep = denominators > 0
                    boot = 1.0 - (weights @ srt_c)[keep] / denominators[keep]
                    sd = float(np.std(boot, ddof=1)) if boot.size > 1 else 0.0
                    rejected += bool(sd > 0 and rho / sd > z)
        grid.append({"lambda": effect, "true_rho_srt": float(np.median(noiseless)),
                     "estimated_rho_srt_median": float(np.median(estimates)),
                     "power": rejected / simulations})
    detectable = [g["true_rho_srt"] for g in grid if g["power"] >= TARGET_POWER]
    return {"noise_eV": noise_eV, "simulations_per_effect": simulations,
            "bootstrap_resamples": resamples, "one_sided_alpha": alpha,
            "test": "one-sided cluster-bootstrap (normal approximation) at alpha/m",
            "target_power": TARGET_POWER, "power_curve": grid,
            "mde_rho_srt": min(detectable) if detectable else None}


def run(ctx: RunnerContext) -> dict[str, Any]:
    from e3sse.production.runners.gate1_tables import linkage_tables, pinned_g0_inputs
    from e3sse.production.runners.transfer import (
        primary_train_hops,
        training_endpoint_diagnostics,
        universe_hops,
    )
    from e3sse.protocol.contracts import build_pilot_amendment
    from e3sse.protocol.internal_freeze import build_internal_pilot_record
    from e3sse.protocol.partitions import analysis_seed
    from e3sse.softening.study1_inputs import statistical_constants, support_parameters

    parent = ctx.parent_protocol
    internal = ctx.internal_freeze()
    constants = statistical_constants(parent)
    constants["support"] = support_parameters(parent)
    train_frames = sorted(r["frame_id"] for r in ctx.membership
                          if r.get("record_kind") == "mplitrj_frame"
                          and r.get("published_split") == "train")
    if not train_frames:
        raise MissingEvidence("no published-train frames in the frozen membership")
    _, summary = study1_tables(ctx)  # s-bar (eligibility) only; access recorded there
    eligible = eligible_models(ctx, summary)
    alpha = 0.05
    evaluation = [h for h in universe_hops(ctx.membership, ("test",))]
    bvse = {}
    for row in linkage_tables(ctx.data_root / "raw", pinned_g0_inputs(ctx.freeze_config))["nebBVSE122k"]:
        value = _float(row["bvse_migration_barrier_eV"])
        if value is not None:
            bvse[row["edge_id"]] = value
    usable = [h for h in evaluation if h["hop_id"] in bvse]
    barriers = [bvse[h["hop_id"]] for h in usable]
    clusters = [h["chemical_system_id"] for h in usable]
    train_hops = primary_train_hops(ctx)
    base_seed = int(parent["statistical_plan"]["randomness"]["base_seed"])
    mde: dict[str, Any] = {
        "study2_mde_design": (
            "simulation-based minimum detectable effect of P1: published-test evaluation hops "
            "with true barrier = nebBVSE122k BVSE barrier (covariate), clustered by chemical "
            "system; raw = s*B + N(0, noise); corrected = raw / s_hat**lambda, s_hat = "
            "s*exp(N(0, tau)); s and tau from regime-matched training diagnostics (published-"
            "train primary hops, endpoint-relaxation frames, one value per material), tau = "
            "median relative 95% half-width / 1.96; one-sided cluster-bootstrap test at the "
            "strictest Holm level alpha/m; MDE = smallest true rho_SRT with power >= 0.8"),
        "study2_mde_evaluation_hops": len(barriers),
        "study2_mde_settings": {"noise_levels_eV": list(NOISE_LEVELS_EV),
                                "simulations_per_effect": SIMULATIONS,
                                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                                "effect_grid_lambda": list(EFFECT_GRID),
                                "target_power": TARGET_POWER},
        "study2_mde_models": {}}
    for model in eligible:
        diagnostics = training_endpoint_diagnostics(ctx, model, train_hops, constants)
        rows = [v for v in diagnostics["per_material"].values() if v["supported"]]
        scales = [v["s_hat"] for v in rows if v["s_hat"] is not None and v["s_hat"] > 0]
        widths = sorted(v["scale_uncertainty"] for v in rows if v["scale_uncertainty"] is not None)
        if not scales or not barriers:
            mde["study2_mde_models"][model] = {"status": "insufficient_inputs"}
            continue
        tau = widths[len(widths) // 2] / CI95_Z if widths else 0.0
        mde["study2_mde_models"][model] = {
            "s_bar": eligible[model], "n_scales": len(scales), "relative_scale_sd": tau,
            "endpoint_frames": len(diagnostics["frame_ids"]),
            "by_noise": [mde_simulation(scales, tau, barriers, clusters, noise_eV=noise,
                                        alpha=alpha / len(eligible),
                                        seed=analysis_seed(base_seed, "pilot_mde", model, noise))
                         for noise in NOISE_LEVELS_EV]}
    thresholds = {
        "study2_classification_thresholds_eV": list(
            parent["study_2a_contract"]["thresholds_eV"]),
        "study2_p1_holm_one_sided_alpha": alpha,
        "study2_p1_eligible_models": sorted(eligible),
        "study2_p1_eligible_s_bar": dict(sorted(eligible.items())),
        "study2_p1_eligibility_source": "Study-1 s-bar (primary population) from published-train "
                                        "frames only; no test frame or barrier",
    }
    # Training units read: every published-train frame (Study-1 tables), a superset of
    # the endpoint frames read for the MDE scales.
    amendment = build_pilot_amendment(parent, partition_records=ctx.membership,
                                      training_unit_ids=train_frames, frozen_thresholds=thresholds,
                                      mde_analysis=mde)
    record = build_internal_pilot_record(internal, parent, amendment)
    outputs = {"pilot_amendment": ctx.publish("pilot_amendment.json", amendment),
               "internal_pilot_record": ctx.publish("internal_pilot_record.json", record)}
    return {"outputs": outputs, "summary": {"amendment_id": amendment["amendment_id"],
                                            "eligible_models": sorted(eligible)}}
