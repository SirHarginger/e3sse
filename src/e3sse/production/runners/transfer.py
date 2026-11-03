"""transfer: Study 2a, fixed-geometry raw and SRT barriers on unseen DFT-NEB images.

Phases (``amendments.transfer_benchmark``, parent ``study_2a_contract``):

1. ``prepare`` (project): byte-offset index of the published-test MPLiTrj source and
   the DFT-reference statistics of its frames (test-side forces, outcome-gated).
2. ``run_model`` (each model's frozen runtime):
   * fixed-geometry single points on every image of ``<edge_id>_relaxed.xyz`` for
     every included published-train and published-test nebDFT2k hop (reused by B5
     and Study 4); ML energies only;
   * per-frame Eq. 17 statistics on the published-test frames (``infer_shard``).
3. ``run`` (project): per model, B0 raw = max - min ML image energy; B1/B2 from the
   Study-1 pooled training-side scales; B3 material scope from the test-side
   non-path frames of the target material (s_scope); B4 = B3 under the deployment
   gate whose thresholds come from the published-train hops (s_scope frames, one
   value per material, quantile 0.8); B6 own-path oracle (labelled).  DFT barriers
   (max - min of DFT image energies) are read only here, after the pilot seal.
   Metrics, paired effects, P1 (rho_SRT, one-sided cluster-bootstrap p, Holm over
   eligible Group-A models), G1, element-resolved SRT and the label-budget downstream
   analysis.  Per-hop diagnostics go to the reliability stage.

No image is chosen by DFT energy; every barrier uses all images.  Leakage: s never
uses the target hop's own path except the labelled oracle.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from e3sse.production.runners.context import (
    MissingEvidence,
    RunnerContext,
    RunnerValidationError,
)

TEST_FORCES = "mplitrj_test_side_nonpath_forces"
OWN_PATH = "mplitrj_own_path_forces_oracle"
EVALUATION_BARRIERS = "nebdft2k_fixed_geometry_evaluation_barriers"
SINGLE_POINT_NAMESPACE = "transfer.single_points"
HOPS_PER_SHARD = 8
CODE_FILES = ("src/e3sse/production/runners/transfer.py",
              "src/e3sse/production/runners/frame_scales.py",
              "src/e3sse/softening/study1_inference.py", "src/e3sse/protocol/adapters.py")


# ===================================================================== inputs

def universe_hops(membership, splits: tuple[str, ...] = ("train", "test")) -> list[dict[str, Any]]:
    return sorted((dict(r) for r in membership if r.get("record_kind") == "nebdft2k_hop"
                   and r.get("eligibility_status") == "included"
                   and r.get("published_split") in splits), key=lambda r: r["hop_id"])


SOURCE_FILES = {"test": "test_source_file", "val": "validation_source_file"}
SOURCE_DEFAULTS = {"test": "MPLiTrj_test.xyz", "val": "MPLiTrj_val.xyz"}


def heldout_source(ctx: RunnerContext, split: str = "test"
                   ) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    """A published held-out MPLiTrj source and its frozen frame records."""

    from e3sse.softening.study1_inputs import mplitrj_source

    name = ctx.runner_settings("transfer").get(SOURCE_FILES[split], SOURCE_DEFAULTS[split])
    source = mplitrj_source(ctx.parent_protocol, name)
    if source.get("published_split") != split:
        raise RunnerValidationError(f"{name} is not the published-{split} source")
    frames = sorted((dict(r) for r in ctx.membership if r.get("record_kind") == "mplitrj_frame"
                     and r.get("published_split") == split
                     and str(r["frame_id"]).startswith(f"{name}:")),
                    key=lambda r: r["source_frame_index"])
    return source, (ctx.data_root / source["identifier"]).resolve(), frames


def test_source(ctx: RunnerContext):
    return heldout_source(ctx, "test")


OUTCOME_KINDS = {"test": (("test", TEST_FORCES), ("test", OWN_PATH)),
                 "val": (("validation", "mplitrj_validation_frame_forces"),)}


def _require_test_frames(ctx: RunnerContext, frames, split: str = "test") -> None:
    ids = [r["frame_id"] for r in frames]
    for role, kind in OUTCOME_KINDS[split]:
        ctx.require_outcome(role=role, outcome=f"mplitrj_{split}_frame_forces",
                            outcome_kind=kind, unit_ids=ids)


def _merge_statistics(ctx: RunnerContext, directory: str, results: dict, *, reference: bool) -> Path:
    import json

    from e3sse.softening import study1_inference as inference

    arrays = inference.merge_frame_statistics(results, reference=reference)
    target = ctx.work_dir / directory
    target.mkdir(parents=True, exist_ok=True)
    digest = inference.write_npz_once(target / "frame_statistics.npz", arrays)
    manifest = {"schema_version": inference.INFERENCE_MANIFEST_SCHEMA,
                "statistics_file": "frame_statistics.npz", "statistics_sha256": digest,
                "frames": len(arrays["frame_index"]),
                "frame_failures": [] if reference else inference.failures_of(results)}
    path = target / "frame_statistics_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RunnerValidationError(f"{path} differs from the recomputed merge")
    else:
        ctx.publish(f"{directory}/frame_statistics_manifest.json", manifest)
    return path


# ===================================================================== phase 1 (project)

def prepare_frames(ctx: RunnerContext, split: str) -> dict[str, Any]:
    """Frame index and DFT-reference statistics of one held-out split (outcome-gated)."""

    from e3sse.softening import study1_inference as inference

    source, source_path, frames = heldout_source(ctx, split)
    _require_test_frames(ctx, frames, split)
    index_dir = ctx.stage_dir / f"{split}_frame_index"
    manifest, index = inference.build_frame_index(source_path, source, frames, index_dir,
                                                  resume=True)
    shards = inference.plan_frame_shards(frames, index, ctx.runner_settings("transfer").get(
        "frames_per_shard", 512))
    identity = {"code_sha256": ctx.code_sha256(CODE_FILES), "source_sha256": source["sha256"],
                "frame_index_sha256": manifest["arrays_sha256"], "run": "dft_reference"}
    results = ctx.execute(namespace=f"{split}.frames.reference",
                          units=[(s.identity, s.payload) for s in shards],
                          task=inference.infer_shard, identity=identity,
                          setup=inference.inference_worker_setup,
                          setup_args=(None, None, str(source_path), "cpu"))
    path = _merge_statistics(ctx, f"{split}_frames/dft_reference", results, reference=True)
    return {"outputs": {f"{split}_frame_statistics:dft_reference": path,
                        f"{split}_frame_index": index_dir / "manifest.json"},
            "summary": {f"{split}_frames": len(frames)}}


def prepare(ctx: RunnerContext) -> dict[str, Any]:
    return prepare_frames(ctx, "test")


# ===================================================================== phase 2 (runtime)

def single_point_setup(model: dict, checkpoint: str) -> dict[str, Any]:
    from e3sse.protocol.adapters import build_calculator
    from e3sse.softening.study1_inference import _torch_parameters, requested_dtype

    calc = build_calculator(model, Path(checkpoint), device="cpu")
    if _torch_parameters(calc) != ("cpu", requested_dtype(model)):
        raise RuntimeError("effective device/dtype differs from the frozen request")
    return {"calc": calc}


def single_point_task(payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """ML total energy of every relaxed DFT-NEB image (fixed geometry; no labels)."""

    from ase import Atoms

    from e3sse.production.runners.hop_files import read_geometry

    calc = context["calc"]
    hops = []
    for hop_id, path in payload["hops"]:
        energies, status, detail = [], "success", None
        for frame in read_geometry(Path(path)):
            atoms = Atoms(numbers=frame["numbers"], positions=frame["positions"],
                          cell=frame["cell"], pbc=frame["pbc"])
            atoms.calc = calc
            try:
                energy = float(atoms.get_potential_energy())
            except (MemoryError, OSError):
                raise
            except Exception as exc:  # a model failure on an image is the hop's status
                status, detail = "failed", f"{type(exc).__name__}: {exc}"[:500]
                break
            if not math.isfinite(energy):
                status, detail = "failed", "non-finite ML energy"
                break
            energies.append(energy)
        hops.append({"hop_id": hop_id, "status": status, "detail": detail,
                     "ml_image_energies_eV": energies if status == "success" else None})
    return {"hops": hops}


def model_frames_and_points(ctx: RunnerContext, *, splits: tuple[str, ...],
                            frame_split: str) -> dict[str, Any]:
    """Single points on the relaxed images of ``splits`` hops; ``frame_split`` frame stats."""

    from e3sse.production.runners.hop_files import nebdft2k_files
    from e3sse.softening import study1_inference as inference

    model = ctx.model
    entry = ctx.models[model]
    checkpoint = str(ctx.checkpoint(model))
    files = nebdft2k_files(ctx.data_root / "raw")
    hops = universe_hops(ctx.membership, splits)
    missing = [h["hop_id"] for h in hops if "relaxed" not in files.get(h["hop_id"], {})]
    if missing:
        raise MissingEvidence(f"{len(missing)} hops lack <edge_id>_relaxed.xyz, e.g. {missing[0]}")
    identity = {"code_sha256": ctx.code_sha256(CODE_FILES), "weights_sha256": entry["weights_sha256"]}
    units = []
    for start in range(0, len(hops), HOPS_PER_SHARD):
        block = hops[start: start + HOPS_PER_SHARD]
        units.append(({"hop_ids": [h["hop_id"] for h in block]},
                      {"hops": [[h["hop_id"], str(files[h["hop_id"]]["relaxed"])] for h in block]}))
    results = ctx.execute(namespace=SINGLE_POINT_NAMESPACE, units=units, task=single_point_task,
                          identity=identity, setup=single_point_setup,
                          setup_args=(entry, checkpoint), configure_torch=True)
    single_points = {h["hop_id"]: h for r in results.values() for h in r["hops"]}
    source, source_path, frames = heldout_source(ctx, frame_split)
    _require_test_frames(ctx, frames, frame_split)
    index = inference.load_frame_index(ctx.stage_dir / f"{frame_split}_frame_index", source)
    shards = inference.plan_frame_shards(frames, index, ctx.runner_settings("transfer").get(
        "frames_per_shard", 512))
    frame_results = ctx.execute(
        namespace=f"{frame_split}.frames", units=[(s.identity, s.payload) for s in shards],
        task=inference.infer_shard, identity={**identity, "source_sha256": source["sha256"]},
        setup=inference.inference_worker_setup,
        setup_args=(entry, checkpoint, str(source_path), "cpu"), configure_torch=True)
    stats_path = _merge_statistics(ctx, f"{frame_split}_frames", frame_results, reference=False)
    outputs = {"single_points": ctx.publish("single_points.json", {
        "model": model, "images": "nebDFT2k <edge_id>_relaxed.xyz (all images)",
        "hops": dict(sorted(single_points.items()))}),
        f"{frame_split}_frame_statistics": stats_path}
    failed = sum(h["status"] != "success" for h in single_points.values())
    return {"outputs": outputs, "summary": {"hops": len(single_points), "failed_hops": failed}}


def run_model(ctx: RunnerContext) -> dict[str, Any]:
    return model_frames_and_points(ctx, splits=("train", "test"), frame_split="test")


# ===================================================================== phase 3 (statistics)

def raw_barriers(single_points: dict[str, Any], hop_ids) -> dict[str, float]:
    from e3sse.barriers.fixed_geometry import barrier

    return {h: barrier(single_points[h]["ml_image_energies_eV"]) for h in hop_ids
            if single_points.get(h, {}).get("status") == "success"}


def dft_barriers(ctx: RunnerContext, hop_ids: list[str], *, role: str, kind: str) -> dict[str, float]:
    """Max - min DFT image energy of the relaxed band: an outcome read, gated."""

    ctx.require_outcome(role=role, outcome="nebdft2k_relaxed_image_dft_energies",
                        outcome_kind=kind, unit_ids=hop_ids)
    return dft_barriers_unchecked(ctx, hop_ids)


def dft_barriers_unchecked(ctx: RunnerContext, hop_ids: list[str]) -> dict[str, float]:
    """The read itself; callers must have been granted ``require_outcome`` first."""

    from e3sse.barriers.fixed_geometry import barrier
    from e3sse.production.runners.hop_files import nebdft2k_files, read_labelled

    files = nebdft2k_files(ctx.data_root / "raw")
    return {h: barrier([f["energy_eV"] for f in read_labelled(files[h]["relaxed"])])
            for h in hop_ids}


def study1_pooled(estimates: list[dict], model: str) -> dict[str, Any]:
    """Study-1 training-side pooled global and family scales (primary, set 'all')."""

    def number(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    pooled = {"global": None, "family": {}}
    for row in estimates:
        if (row["model"], row["kind"], row["population"], row["estimator_kind"],
                row["component_set"]) != (model, "pooled", "primary", "primary", "all") or row["decile"]:
            continue
        s, rf = number(row["s_hat"]), number(row["residual_fraction"])
        low, high = number(row["ci_low"]), number(row["ci_high"])
        value = {"s_hat": s if row["status"] == "estimated" else None,
                 "rho_F": math.sqrt(rf) if rf is not None and rf >= 0 else None,
                 "scale_uncertainty": (high - low) / (2 * s) if None not in (s, low, high) and s
                 else None}
        if row["scope"] == "global":
            pooled["global"] = value
        elif row["scope"] == "family":
            pooled["family"][row["scope_target"]] = value
    return pooled


def gate_thresholds_from_training(ctx: RunnerContext, model: str, train_hops, quantile: float,
                                  constants) -> dict[str, Any]:
    """Regime-matched thresholds: published-train hops, s_scope frames, one value per material."""

    from e3sse.production.runners.family_scale import study1_frame_stats
    from e3sse.production.runners.frame_scales import hop_diagnostics, scope_frames
    from e3sse.protocol.partitions import analysis_seed
    from e3sse.softening.applicability import gate_thresholds

    stats = study1_frame_stats(ctx, model)
    frames = scope_frames(protocol=ctx.parent_protocol, membership=ctx.membership, hops=train_hops,
                          available=set(stats.row), partition="published",
                          side="training_or_calibration")
    per_material: dict[str, Any] = {}
    for hop in train_hops:
        material = hop["material_id"]
        if material in per_material or not frames[hop["hop_id"]]:
            continue
        diag = hop_diagnostics(stats, stats.rows(frames[hop["hop_id"]]), material=material,
                               seed=analysis_seed(constants["base_seed"], "deployment_gate_threshold",
                                                  material),
                               resamples=constants["bootstrap_resamples"],
                               parameters=constants["support"])
        per_material[material] = {"supported": diag["estimates"]["all"]["supported"],
                                  "rho_F": diag["rho_F"],
                                  "scale_uncertainty": diag["scale_uncertainty"]}
    materials = sorted(per_material)
    thresholds = gate_thresholds(
        {"rho_F": [per_material[m]["rho_F"] for m in materials],
         "scale_uncertainty": [per_material[m]["scale_uncertainty"] for m in materials]},
        [per_material[m]["supported"] for m in materials], quantile)
    return {**thresholds, "population": "published-train primary hops, s_scope frames, one value "
            "per material", "n_materials": len(materials)}


def paired_bootstrap(hops, raw, srt, reference, clusters, *, seed, resamples) -> dict[str, Any]:
    import numpy as np

    from e3sse.production.runners.inference_stats import (
        cluster_bootstrap,
        one_sided_p,
        percentile_interval,
    )

    r = np.asarray([raw[h] for h in hops])
    s = np.asarray([srt[h] for h in hops])
    y = np.asarray([reference[h] for h in hops])

    def rho(idx):
        denominator = np.abs(r[idx] - y[idx]).mean()
        return None if denominator == 0 else float(1 - np.abs(s[idx] - y[idx]).mean() / denominator)

    def mae_difference(idx):
        return float(np.abs(r[idx] - y[idx]).mean() - np.abs(s[idx] - y[idx]).mean())

    rho_values = cluster_bootstrap(clusters, rho, seed=seed, resamples=resamples)
    diff_values = cluster_bootstrap(clusters, mae_difference, seed=seed, resamples=resamples)
    return {"rho_srt_ci95": percentile_interval(rho_values),
            "rho_srt_one_sided_p": one_sided_p(rho_values),
            "mae_reduction_ci95": percentile_interval(diff_values)}


def label_budget_downstream(stats, frames_by_hop, hops, raw, reference, *, constants, settings) -> dict:
    """amendments.label_budget_downstream on the evaluation hops (N >= 60 frames)."""

    import numpy as np

    from e3sse.protocol.partitions import analysis_seed
    from e3sse.softening.frame_statistics import draw_without_replacement

    minimum_frames = 60
    eligible = [h for h in hops if len(frames_by_hop.get(h["hop_id"], [])) >= minimum_frames
                and h["hop_id"] in raw and h["hop_id"] in reference]
    rows = []
    for K in settings["k"]:
        per_hop = []
        for hop in eligible:
            frame_rows = stats.rows(frames_by_hop[hop["hop_id"]])
            sums = stats.sums["all"][frame_rows]
            full = sums[:, 1].sum()
            s_full = sums[:, 0].sum() / full if full > 0 else None
            errors, gaps, defined = [], [], 0
            for repeat in range(settings["repeats"]):
                seed = analysis_seed(constants["base_seed"], "study1_label_budget", "material",
                                     hop["material_id"], hop["hop_id"], K, repeat)
                drawn = sums[np.asarray(draw_without_replacement(seed, len(frame_rows), K))]
                sxx = drawn[:, 1].sum()
                s_k = drawn[:, 0].sum() / sxx if sxx > 0 else None
                if s_k is None or s_k <= 0:
                    continue
                defined += 1
                errors.append(abs(raw[hop["hop_id"]] / s_k - reference[hop["hop_id"]]))
                if s_full is not None:
                    gaps.append(abs(s_k - s_full))
            per_hop.append({"hop_id": hop["hop_id"], "defined_fraction": defined / settings["repeats"],
                            "mean_abs_error_eV": float(np.mean(errors)) if errors else None,
                            "mean_abs_s_gap": float(np.mean(gaps)) if gaps else None})
        values = [p["mean_abs_error_eV"] for p in per_hop if p["mean_abs_error_eV"] is not None]
        rows.append({"K": K, "hops": len(per_hop),
                     "mean_abs_srt_error_eV": float(np.mean(values)) if values else None,
                     "median_abs_srt_error_eV": float(np.median(values)) if values else None,
                     "mean_defined_fraction": float(np.mean([p["defined_fraction"] for p in per_hop]))
                     if per_hop else None, "per_hop": per_hop})
    return {"qualifying_hops": len(eligible), "descriptive_only": len(eligible) < 20, "rows": rows}


def run(ctx: RunnerContext) -> dict[str, Any]:
    from e3sse.barriers.fixed_geometry import common_population, metrics, rho_srt, srt_barrier
    from e3sse.production.runners.frame_scales import FrameStats, hop_diagnostics, scope_frames
    from e3sse.production.runners.inference_stats import holm
    from e3sse.production.runners.pilot import eligible_models, study1_tables
    from e3sse.protocol.partitions import analysis_seed
    from e3sse.softening.applicability import gate_decision
    from e3sse.softening.study1_inference import verify_statistics_file
    from e3sse.softening.study1_inputs import statistical_constants, support_parameters

    amendments = ctx.frozen_amendments()
    parent = ctx.parent_protocol
    constants = statistical_constants(parent)
    constants["support"] = support_parameters(parent)
    thresholds_eV = list(parent["study_2a_contract"]["thresholds_eV"])
    quantile = float(amendments["deployment_gate"]["quantile_level"])
    population_of = ctx.partition_manifest["primary_population"]["assignment_by_chemical_system"]
    family_of = ctx.partition_manifest["anion_families"]["assignment_by_material"]
    hops = universe_hops(ctx.membership)
    evaluation = [h for h in hops if h["published_split"] == "test"]
    train = [h for h in hops if h["published_split"] == "train"
             and population_of.get(h["chemical_system_id"]) == "primary"]
    estimates, summary = study1_tables(ctx)
    eligible = eligible_models(ctx, summary)
    reference = dft_barriers(ctx, [h["hop_id"] for h in evaluation], role="test",
                             kind=EVALUATION_BARRIERS)
    _, ref_arrays = verify_statistics_file(ctx.stage_dir / "test_frames" / "dft_reference")
    _, _, test_frames = test_source(ctx)
    frame_ids = [r["frame_id"] for r in test_frames]
    records, hop_records, results = [], [], {}
    for model in sorted(ctx.models):
        single_points = ctx.upstream("transfer", "single_points", model=model)["hops"]
        _, model_arrays = verify_statistics_file(ctx.stage_dir / "models" / model / "test_frames")
        stats = FrameStats(frame_ids, model_arrays, ref_arrays)
        b3_frames = scope_frames(protocol=parent, membership=ctx.membership, hops=evaluation,
                                 available=set(stats.row), partition="published", side="test")
        oracle_frames = scope_frames(protocol=parent, membership=ctx.membership, hops=evaluation,
                                     available=set(stats.row), partition="published", side="test",
                                     estimator_kind="oracle")
        raw = raw_barriers(single_points, reference)
        pooled = study1_pooled(estimates, model)
        gate = gate_thresholds_from_training(ctx, model, train, quantile, constants)
        predictions: dict[str, dict[str, float]] = defaultdict(dict)
        for hop in evaluation:
            h = hop["hop_id"]
            if h not in raw:
                continue
            family = family_of.get(hop["material_id"])
            diag = hop_diagnostics(stats, stats.rows(b3_frames[h]), material=hop["material_id"],
                                   seed=analysis_seed(constants["base_seed"], "transfer_material_scale",
                                                      h), resamples=constants["bootstrap_resamples"],
                                   parameters=constants["support"]) if b3_frames[h] else None
            oracle = stats.estimate(stats.rows(oracle_frames[h]), "all",
                                    materials=[hop["material_id"]], pooled=False,
                                    parameters=constants["support"]) if oracle_frames[h] else None
            s3 = diag["s_hat"] if diag else None
            passes, reasons = gate_decision(
                supported=bool(diag and diag["estimates"]["all"]["supported"]),
                diagnostics={"rho_F": diag["rho_F"] if diag else None,
                             "scale_uncertainty": diag["scale_uncertainty"] if diag else None},
                thresholds=gate["thresholds"])
            values = {"B0_raw": raw[h],
                      "B1_srt_global": srt_barrier(raw[h], (pooled["global"] or {}).get("s_hat")),
                      "B2_srt_family": srt_barrier(raw[h], pooled["family"].get(family, {}).get("s_hat")),
                      "B3_srt_material": srt_barrier(raw[h], s3),
                      "B4_srt_gated_selective": srt_barrier(raw[h], s3) if passes else None,
                      "B4_srt_gated_correct_or_raw": srt_barrier(raw[h], s3) if passes else raw[h],
                      "B6_oracle_own_path": srt_barrier(raw[h], oracle["s_hat"] if oracle else None),
                      "element_resolved_srt_li": srt_barrier(
                          raw[h], diag["estimates"]["Li"]["s_hat"] if diag else None)}
            for method, value in values.items():
                if value is not None:
                    predictions[method][h] = value
            hop_records.append({
                "model": model, "hop_id": h, "chemical_system": hop["chemical_system_id"],
                "material_id": hop["material_id"], "population": population_of.get(
                    hop["chemical_system_id"]), "raw_barrier_eV": raw[h],
                "dft_barrier_eV": reference[h], "s_material": s3,
                "srt_material_eV": values["B3_srt_material"], "gate_passes": passes,
                "gate_reasons": reasons,
                "scores": {k: diag[k] if diag else None for k in
                           ("rho_F", "scale_uncertainty", "li_host_disagreement", "energy_dependence")},
                "undefined_reasons": diag["undefined_reasons"] if diag else {"all": "no_s_scope_frames"},
                "n_s_scope_frames": len(b3_frames[h])})
        primary = {h["hop_id"] for h in evaluation if population_of.get(h["chemical_system_id"]) == "primary"}
        clusters_of = {h["hop_id"]: h["chemical_system_id"] for h in evaluation}
        per_method = {}
        for method, values in sorted(predictions.items()):
            if method == "B0_raw":
                continue
            population = common_population({"B0_raw": [h for h in predictions["B0_raw"] if h in primary],
                                            method: [h for h in values if h in primary]},
                                           label=f"{model}:{method}")
            if not population["hop_ids"]:
                per_method[method] = {"population": population, "status": "empty_common_population"}
                continue
            per_method[method] = {
                "population": population,
                "raw": metrics(predictions["B0_raw"], reference, population, thresholds_eV),
                "method": metrics(values, reference, population, thresholds_eV),
                "rho_srt": rho_srt(predictions["B0_raw"], values, reference, population),
                "role": ("oracle_upper_bound_never_primary" if method.startswith("B6")
                         else "secondary_diagnostic" if method.startswith("element") else "deployable")}
        p1 = per_method.get("B3_srt_material", {})
        if p1.get("population", {}).get("hop_ids"):
            hop_ids = p1["population"]["hop_ids"]
            p1["bootstrap"] = paired_bootstrap(
                hop_ids, predictions["B0_raw"], predictions["B3_srt_material"], reference,
                [clusters_of[h] for h in hop_ids],
                seed=analysis_seed(constants["base_seed"], "cluster_bootstrap", "study2a", "P1",
                                   "published"), resamples=constants["bootstrap_resamples"])
        results[model] = {"methods": per_method, "deployment_gate": gate,
                          "label_budget_downstream": label_budget_downstream(
                              stats, b3_frames, [h for h in evaluation if h["hop_id"] in primary],
                              raw, reference, constants=constants,
                              settings={"k": amendments["label_budget_downstream"]["k"],
                                        "repeats": int(parent["study_1_contract"][
                                            "label_budget_stability"]["repeat_count"])})}
        for method, values in predictions.items():
            for h, value in values.items():
                records.append({"model": model, "method": method, "hop_id": h,
                                "predicted_eV": value, "dft_eV": reference[h]})
    p_values = {m: results[m]["methods"].get("B3_srt_material", {}).get("bootstrap", {}).get(
        "rho_srt_one_sided_p") for m in eligible}
    adjusted = holm(p_values)
    endpoints = {m: {"rho_srt": results[m]["methods"].get("B3_srt_material", {}).get("rho_srt"),
                     "p_one_sided": p_values[m], "p_holm": adjusted[m], "s_bar": eligible[m]}
                 for m in eligible}
    g1 = any((e["rho_srt"] or -math.inf) >= 0.10 for e in endpoints.values())
    decision = {"gate": "G1", "rule": "rho_SRT >= 0.10 for at least one Group-A model with "
                "s-bar <= 0.9", "passes": g1, "eligible_models": sorted(eligible),
                "contingency_if_failed": "prospective selection proceeds with raw ML-NEB scores"}
    outputs = {"transfer_results": ctx.publish("transfer_results.json", {
                   "results": results, "P1": endpoints, "holm_family": "P1 over eligible Group-A",
                   "noise_floor": "not available: no symmetry-equivalent image energies in nebDFT2k"}),
               "transfer_predictions": ctx.publish("derived/transfer_predictions.json", records),
               "transfer_hop_records": ctx.publish("transfer_hop_records.json", hop_records),
               "g1_decision": ctx.publish("g1_decision.json", decision)}
    return {"outputs": outputs, "summary": {"G1_passes": g1, "P1": endpoints}}
