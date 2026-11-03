"""controlled_selection: Study 4 conformal FDR validity and power on repeated partitions.

Every parent ``study_4_contract`` and ``statistical_plan`` rule is unchanged
(``amendments.selection.unchanged``): the ten repeated chemical-system partitions;
200 draws per partition of one hop per chemical system (``draw_one_hop_per_system``
with ``study4_draw_seed``); training-or-calibration side = calibration units, test
side = test units; conformal p-values with U_j = 1 (``conformal_pvalues``), BH with
the BY sensitivity, the naive comparator (select if predicted Y < c); alpha in
{0.1, 0.2}, c in {0.3, 0.5} eV; realised FDR as the mean FDP with its Monte-Carlo
standard error (``realized_fdr``; G4: realised FDR <= alpha + 2 MC s.e.).

Predictors are fixed without calibration or test outcomes
(``study4_predictor_calibration_independence``): raw fixed-geometry B0; SRT-global
and SRT-family with s pooled per repeat over MPLiTrj frames of that repeat's
training-side chemical systems without nebDFT2k hops (symmetric fallback on the
forces of all calibration and test units of the repeat when that stratum lacks
support); and the added SRT_gated_correct_or_raw (``amendments.selection``):
material-scope s (s_scope frames) under per-repeat symmetric, Y-free thresholds.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from e3sse.production.runners.context import RunnerContext

REPEATED_BARRIERS = "nebdft2k_repeated_partition_barriers"
PREDICTORS = ("raw", "SRT_global", "SRT_family", "SRT_gated_correct_or_raw")


def prepare(ctx: RunnerContext) -> dict[str, Any]:
    """Validation-split frame index and DFT reference (validation systems are Study-4 units)."""

    from e3sse.production.runners.transfer import prepare_frames

    return prepare_frames(ctx, "val")


def run_model(ctx: RunnerContext) -> dict[str, Any]:
    """Validation-hop single points and validation-frame statistics in the model runtime."""

    from e3sse.production.runners.transfer import model_frames_and_points

    return model_frames_and_points(ctx, splits=("val",), frame_split="val")


def frame_sources(ctx: RunnerContext, model: str) -> dict[str, Any]:
    """Per-split frame statistics: train (Study 1), test (transfer), val (this stage)."""

    from e3sse.production.runners.family_scale import study1_frame_stats
    from e3sse.production.runners.frame_scales import aligned_frame_stats
    from e3sse.production.runners.transfer import TEST_FORCES, heldout_source
    from e3sse.production.stage_graph import TRAIN_FORCES, VALIDATION_FORCES
    from e3sse.softening.study1_inference import verify_statistics_file

    train = study1_frame_stats(ctx, model)
    ctx.require_outcome(role="train", outcome="study4_training_frame_forces",
                        outcome_kind=TRAIN_FORCES, unit_ids=train.frame_ids)
    sources = {"train": train}
    access = {"test": ("test", TEST_FORCES), "val": ("validation", VALIDATION_FORCES)}
    for split, stage in (("test", "transfer"), ("val", "controlled_selection")):
        _, _, frames = heldout_source(ctx, split)
        ids = [r["frame_id"] for r in frames]
        role, kind = access[split]
        ctx.require_outcome(role=role, outcome=f"study4_{split}_frame_forces", outcome_kind=kind,
                            unit_ids=ids)
        _, reference = verify_statistics_file(ctx.upstream_path(
            stage, f"{split}_frame_statistics:dft_reference", phase="prepare").parent)
        _, arrays = verify_statistics_file(ctx.upstream_path(
            stage, f"{split}_frame_statistics", model=model).parent)
        sources[split] = aligned_frame_stats(ids, arrays, reference)
    return sources


class _Pooled:
    """Look up a frame in whichever split holds it; pooled estimates across splits."""

    def __init__(self, sources) -> None:
        self.sources = tuple(sources.values())
        self.row = {}
        for source in self.sources:
            for frame_id, index in source.row.items():
                self.row[frame_id] = (index, source)

    def estimate(self, frame_ids, *, materials, parameters):
        sums = {"sxy": 0.0, "sxx": 0.0, "syy": 0.0}
        from e3sse.softening.frame_statistics import eq17, support_status

        informative, failed, n = 0, 0, 0
        for frame_id in frame_ids:
            if frame_id not in self.row:
                continue
            i, source = self.row[frame_id]
            for c, key in enumerate(("sxy", "sxx", "syy")):
                sums[key] += float(source.sums["all"][i, c])
            informative += int(source.informative[i])
            failed += int(not source.success[i])
            n += 1
        reasons = support_status(n_frames=n, n_informative=informative,
                                 n_materials=len(set(materials)), pooled=True,
                                 parameters=parameters)
        if failed:
            reasons.append("model_frame_failure")
        s = eq17(sums["sxy"], sums["sxx"]) if sums["sxx"] > 0 else None
        return {"s_hat": s if not reasons else None, "unsupported_reasons": reasons, "n_frames": n}


def repeat_scales(pooled: _Pooled, membership, population_of, family_of, training_systems,
                  unit_frames, parameters) -> dict[str, Any]:
    """Global and per-family pooled s for one repeat, with the symmetric fallback.

    Frozen ``study4_predictor_calibration_independence``: if the training-side stratum
    lacks minimum support for a scope, s for that scope is fitted on forces only of all
    calibration and test units of the repeat in that scope (global: every unit; family:
    the units of that family), applied identically to both sides, never using Y.
    """

    from e3sse.production.runners.family_scale import scale_frames

    frames = scale_frames(membership, population_of, training_systems=training_systems)
    by_family = defaultdict(list)
    for row in frames:
        by_family[family_of[row["material_id"]]].append(row)
    result: dict[str, Any] = {"global": pooled.estimate(
        [r["frame_id"] for r in frames], materials=[r["material_id"] for r in frames],
        parameters=parameters), "family": {}}
    result["global"]["source"] = "training_side_systems_without_nebdft2k_hops"
    for family, rows in by_family.items():
        result["family"][family] = {**pooled.estimate(
            [r["frame_id"] for r in rows], materials=[r["material_id"] for r in rows],
            parameters=parameters), "source": "training_side_systems_without_nebdft2k_hops"}
    def fallback(pairs):
        frames_ = sorted({f for f, _ in pairs})
        return pooled.estimate(frames_, materials=sorted({m for _, m in pairs}),
                               parameters=parameters)

    every_unit = [pair for frames_ in unit_frames.values() for pair in frames_]
    if result["global"]["s_hat"] is None:
        result["global"] = {**fallback(every_unit), "source": "symmetric_fallback_all_units_forces"}
    unit_families = {family_of[m] for _, m in every_unit if m in family_of}
    for family in sorted(unit_families):
        if (result["family"].get(family) or {}).get("s_hat") is None:
            pairs = [(f, m) for f, m in every_unit if family_of.get(m) == family]
            result["family"][family] = {**fallback(pairs),
                                        "source": "symmetric_fallback_family_units_forces"}
    return result


def _repeat_se(rows, key: str) -> float | None:
    """Monte Carlo s.e. across partitions (the dependence unit): SD of repeat means / sqrt(R)."""

    import numpy as np

    by_repeat = defaultdict(list)
    for r in rows:
        if r[key] is not None:
            by_repeat[r["repeat"]].append(r[key])
    means = [float(np.mean(v)) for v in by_repeat.values()]
    return float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) >= 2 else None


def h5b_power_differences(records, contract, *, seed: int, resamples: int) -> dict[str, Any]:
    """H5b (SRT increases power; supported if paired CI > 0): per draw the power of an SRT
    predictor minus raw on the identical units, averaged per repeat; the CI resamples the
    repeats (partitions are the dependence unit)."""

    import numpy as np

    from e3sse.production.runners.inference_stats import cluster_bootstrap, percentile_interval

    key = lambda r: (r["repeat"], r["draw"], r["c_eV"], r["alpha"], r["method"])
    raw = {key(r): r["power"] for r in records if r["predictor"] == "raw"}
    out = {}
    for predictor in PREDICTORS[1:]:
        for c in contract["threshold_eV"]:
            for alpha in contract["alpha"]:
                for method in ("BH", "BY"):
                    by_repeat = defaultdict(list)
                    for r in records:
                        if (r["predictor"], r["c_eV"], r["alpha"], r["method"]) != (
                                predictor, c, alpha, method):
                            continue
                        base = raw.get(key(r))
                        if r["power"] is not None and base is not None:
                            by_repeat[r["repeat"]].append(r["power"] - base)
                    repeats = sorted(by_repeat)
                    means = np.asarray([np.mean(by_repeat[k]) for k in repeats])
                    interval = None
                    if len(repeats) >= 2:
                        interval = percentile_interval(cluster_bootstrap(
                            [str(k) for k in repeats], lambda idx, m=means: float(m[idx].mean()),
                            seed=seed, resamples=resamples))
                    lower = (interval or {}).get("low")
                    out[f"{predictor}-raw|c={c}|alpha={alpha}|{method}"] = {
                        "mean_power_difference": float(means.mean()) if means.size else None,
                        "ci95": interval, "repeats": len(repeats),
                        "supported": lower is not None and lower > 0}
    return out


def run(ctx: RunnerContext) -> dict[str, Any]:
    import numpy as np

    from e3sse.barriers.fixed_geometry import srt_barrier
    from e3sse.production.runners.frame_scales import hop_diagnostics, scope_frames
    from e3sse.production.runners.transfer import dft_barriers, raw_barriers, universe_hops
    from e3sse.protocol.partitions import analysis_seed
    from e3sse.selection.conformal import (
        benjamini_hochberg,
        benjamini_yekutieli,
        conformal_pvalues,
        draw_one_hop_per_system,
        realized_fdr,
        selection_outcome,
        study4_draw_seed,
    )
    from e3sse.softening.applicability import gate_decision, gate_thresholds
    from e3sse.softening.study1_inputs import statistical_constants, support_parameters

    parent = ctx.parent_protocol
    contract = parent["study_4_contract"]
    constants = statistical_constants(parent)
    parameters = support_parameters(parent)
    quantile = float(ctx.frozen_amendments()["deployment_gate"]["quantile_level"])
    population_of = ctx.partition_manifest["primary_population"]["assignment_by_chemical_system"]
    family_of = ctx.partition_manifest["anion_families"]["assignment_by_material"]
    partitions = ctx.partition_manifest["repeated_partitions"]["partitions"]
    hops = [h for h in universe_hops(ctx.membership, ("train", "val", "test"))
            if population_of.get(h["chemical_system_id"]) == "primary"]
    y = dft_barriers(ctx, [h["hop_id"] for h in hops], role="repeated_test", kind=REPEATED_BARRIERS)
    draws = int(contract["draws_per_partition"])
    results = {}
    for model in sorted(ctx.models):
        sources = frame_sources(ctx, model)
        pooled = _Pooled(sources)
        available = set(pooled.row)
        raw = raw_barriers(ctx.upstream("transfer", "single_points", model=model)["hops"],
                           [h["hop_id"] for h in hops])
        raw.update(raw_barriers(ctx.upstream("controlled_selection", "single_points", model=model)["hops"],
                                [h["hop_id"] for h in hops]))
        cache: dict[tuple, Any] = {}

        def diagnose(hop, frames, cache=cache, pooled=pooled):
            key = (hop["material_id"], tuple(frames))
            if key not in cache:
                source = pooled.row[frames[0]][1] if frames else None
                cache[key] = hop_diagnostics(
                    source, source.rows(frames), material=hop["material_id"],
                    seed=analysis_seed(constants["base_seed"], "deployment_gate_threshold",
                                       hop["material_id"]),
                    resamples=constants["bootstrap_resamples"],
                    parameters=parameters) if frames else None
            return cache[key]

        records, undefined_draws = [], 0
        for partition in partitions:
            repeat = int(partition["repeat_index"])
            calibration_systems = set(partition["training_or_calibration_chemical_systems"])
            test_systems = set(partition["test_chemical_systems"])
            members = [h for h in hops if h["hop_id"] in raw]
            unit_frames, diagnostics = {}, {}
            for side, systems in (("training_or_calibration", calibration_systems),
                                  ("test", test_systems)):
                side_hops = [h for h in members if h["chemical_system_id"] in systems]
                frames = scope_frames(protocol=parent, membership=ctx.membership, hops=side_hops,
                                      available=available, partition=repeat, side=side)
                for hop in side_hops:
                    unit_frames[hop["hop_id"]] = [(f, hop["material_id"]) for f in frames[hop["hop_id"]]]
                    diagnostics[hop["hop_id"]] = diagnose(hop, frames[hop["hop_id"]])
            scales = repeat_scales(pooled, ctx.membership, population_of, family_of,
                                   calibration_systems, unit_frames, parameters)
            materials = sorted({h["material_id"] for h in members})
            by_material = {}
            for hop in members:
                diag = diagnostics.get(hop["hop_id"])
                if diag and hop["material_id"] not in by_material:
                    by_material[hop["material_id"]] = diag
            thresholds = gate_thresholds(
                {"rho_F": [by_material[m]["rho_F"] if m in by_material else None for m in materials],
                 "scale_uncertainty": [by_material[m]["scale_uncertainty"] if m in by_material
                                       else None for m in materials]},
                [bool(m in by_material and by_material[m]["estimates"]["all"]["supported"])
                 for m in materials], quantile)["thresholds"]
            predictions: dict[str, dict[str, float]] = {p: {} for p in PREDICTORS}
            for hop in members:
                h = hop["hop_id"]
                family = family_of.get(hop["material_id"])
                predictions["raw"][h] = raw[h]
                for name, scale in (("SRT_global", scales["global"].get("s_hat")),
                                    ("SRT_family", scales["family"].get(family, {}).get("s_hat"))):
                    value = srt_barrier(raw[h], scale)
                    if value is not None:
                        predictions[name][h] = value
                diag = diagnostics.get(h)
                passes, _ = gate_decision(
                    supported=bool(diag and diag["estimates"]["all"]["supported"]),
                    diagnostics={"rho_F": diag["rho_F"] if diag else None,
                                 "scale_uncertainty": diag["scale_uncertainty"] if diag else None},
                    thresholds=thresholds)
                predictions["SRT_gated_correct_or_raw"][h] = (
                    srt_barrier(raw[h], diag["s_hat"]) if passes else raw[h])
            hops_by_system = defaultdict(list)
            for hop in members:
                hops_by_system[hop["chemical_system_id"]].append(hop["hop_id"])
            for draw in range(draws):
                drawn = draw_one_hop_per_system(hops_by_system,
                                                study4_draw_seed(constants["base_seed"], repeat, draw))
                cal = [h for s, h in sorted(drawn.items()) if s in calibration_systems]
                test = [h for s, h in sorted(drawn.items()) if s in test_systems]
                # paired_identical_units: every predictor on the same units of this draw
                cal_ok = [h for h in cal if all(h in predictions[q] for q in PREDICTORS)]
                test_ok = [h for h in test if all(h in predictions[q] for q in PREDICTORS)]
                dropped = (len(cal) - len(cal_ok), len(test) - len(test_ok))
                if not cal_ok or not test_ok:
                    undefined_draws += 1
                    continue
                for predictor in PREDICTORS:
                    p = predictions[predictor]
                    test_y = [y[h] for h in test_ok]
                    for c in contract["threshold_eV"]:
                        pvalues = conformal_pvalues([p[h] for h in cal_ok], [y[h] for h in cal_ok],
                                                    [p[h] for h in test_ok], c)
                        for alpha in contract["alpha"]:
                            for method, mask in (
                                    ("BH", benjamini_hochberg(pvalues, alpha)),
                                    ("BY", benjamini_yekutieli(pvalues, alpha)),
                                    ("naive", np.asarray([p[h] < c for h in test_ok]))):
                                records.append({"repeat": repeat, "draw": draw, "predictor": predictor,
                                                "c_eV": c, "alpha": alpha, "method": method,
                                                "dropped_calibration_units": dropped[0],
                                                "dropped_test_units": dropped[1],
                                                **selection_outcome(mask, test_y, c)})
        summary = {}
        for predictor in PREDICTORS:
            for c in contract["threshold_eV"]:
                for alpha in contract["alpha"]:
                    for method in ("BH", "BY", "naive"):
                        rows = [r for r in records if (r["predictor"], r["c_eV"], r["alpha"],
                                                       r["method"]) == (predictor, c, alpha, method)]
                        by_repeat = defaultdict(list)
                        for r in rows:
                            by_repeat[r["repeat"]].append(r["fdp"])
                        key = f"{predictor}|c={c}|alpha={alpha}|{method}"
                        fdr = realized_fdr(by_repeat, alpha) if len(by_repeat) >= 2 else None
                        power = [r["power"] for r in rows if r["power"] is not None]
                        summary[key] = {"realized_fdr": fdr,
                                        "mean_power": float(np.mean(power)) if power else None,
                                        "power_mc_se": _repeat_se(rows, "power"),
                                        "mean_selection_size": float(np.mean([r["selection_size"]
                                                                              for r in rows]))
                                        if rows else None,
                                        "zero_selection_fraction": float(np.mean(
                                            [r["selection_size"] == 0 for r in rows])) if rows else None}
        results[model] = {"summary": summary, "draw_records": records,
                          "undefined_draws": undefined_draws,
                          "h5b": h5b_power_differences(records, contract, seed=analysis_seed(
                              constants["base_seed"], "cluster_bootstrap", "study4", "H5b", model),
                              resamples=constants["bootstrap_resamples"])}
    outputs = {"study4_selection": ctx.publish("study4_selection.json", {
        "results": {m: r["summary"] for m, r in results.items()},
        "h5b": {m: r["h5b"] for m, r in results.items()},
        "undefined_draws": {m: r["undefined_draws"] for m, r in results.items()},
        "g4_rule": "realised FDR <= alpha + 2 MC s.e. (validity check, not superiority)",
        "predictor_independence": "pooled s from training-side systems without nebDFT2k hops "
                                  "per repeat; symmetric Y-free gate thresholds per repeat"}),
               "conformal_draw_records": ctx.publish("derived/conformal_draw_records.json",
                                                     {m: r["draw_records"] for m, r in results.items()})}
    return {"outputs": outputs, "summary": {"models": sorted(results)}}
