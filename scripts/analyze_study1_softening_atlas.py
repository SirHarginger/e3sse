#!/usr/bin/env python3
"""Study 1 (Softening Atlas) analysis: estimates, bootstrap, label budget, tables.

Consumes the merged frame statistics of the DFT reference and all six frozen models
(``run_study1_softening_atlas.py infer --all-models``); evaluates no model.  The
outcome seal is checked before any statistic is read.  Writes, under
``<output>/<protocol-id-prefix>/analysis/<analysis-fingerprint-prefix>/``:

  analysis_inputs/   consolidated per-frame sufficient statistics (.npy, mmap-shared)
  frozen/            eligibility units, energy-decile cut points, label-budget hop sets,
                     common-hop population (all fixed before any estimate or draw)
  units/             resumable shard checkpoints (e3sse.execution)
  derived/           full-precision tables; figures read only these
  tables/            rounded paper tables
  study1_analysis_manifest.json   identities and SHA-256 of every output

Rerun with ``--resume`` after an interruption; completed units are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from e3sse.execution.resource_planning import (
    GIB,
    WorkloadPolicy,
    observe_host_resources,
    parse_workers,
    plan_workers,
)
from e3sse.execution.sharded_execution import (
    ShardedRun,
    canonical_sha256,
    ensure_safe_output_dir,
    execute_shards,
    make_shard,
    publish_json_once,
)
from e3sse.softening import study1_analysis as analysis
from e3sse.softening import study1_inference as inference
from e3sse.softening import study1_inputs as inputs
from e3sse.softening import study1_tables as tables
from e3sse.softening.frame_statistics import DRAW_MAPPING

REPOSITORY = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPOSITORY), *args], text=True).strip()


def _contracts_at(commit: str) -> str | None:
    """SHA-256 of the leakage contract at the freeze commit (None if not in this clone)."""

    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY), "show", f"{commit}:src/e3sse/protocol/contracts.py"],
        capture_output=True, check=False,
    )
    return hashlib.sha256(completed.stdout).hexdigest() if completed.returncode == 0 else None


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _publish_or_verify(path: Path, value, root: Path) -> None:
    """Frozen intermediate: written once; a resumed run must reproduce it exactly."""

    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise SystemExit(f"{path} is not valid JSON; refusing to continue") from exc
        if stored != json.loads(json.dumps(value)):
            raise SystemExit(f"{path} differs from its reconstruction; refusing to continue")
        return
    publish_json_once(path, value, root=root)


def _save_inputs(directory: Path, arrays: dict, root: Path) -> dict[str, str]:
    """Write each consolidated array once (.npy, mmap-able); verify on resume."""

    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, value in sorted(arrays.items()):
        path = directory / f"{name}.npy"
        if path.exists() and not np.array_equal(np.load(path, allow_pickle=False), value):
            raise SystemExit(f"{path} differs from its reconstruction")
        if not path.exists():
            temporary = directory / f".{name}.{os.getpid()}.npy"
            np.save(temporary, value, allow_pickle=False)
            os.link(temporary, path)
            os.unlink(temporary)
        digests[name] = _sha(path)
    return digests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPOSITORY / "configs/study1_softening_atlas.json"))
    parser.add_argument("--protocol-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument("--memory-reserve-gib", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    import numpy as np

    config_path = Path(args.config).resolve()
    config, config_sha = inputs.load_study1_config(config_path)
    protocol_dir = Path(args.protocol_dir or config["frozen_protocol"]["directory"]).expanduser()
    protocol = inputs.load_frozen_protocol(config, directory=protocol_dir)
    server = inputs.load_json(inputs.resolve(config_path.parent, config["server_config"]))
    data_root = Path(server["paths"]["data_root"]).resolve()
    root = ensure_safe_output_dir(
        Path(args.output_root or config["output_root"]).expanduser(), [data_root]
    ) / protocol["protocol_id"][:16]
    inputs.refuse_tmpfs(root)
    git_sha, git_dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    if git_dirty and not args.allow_dirty:
        raise SystemExit("production Study-1 analysis requires a clean worktree")

    records = inputs.load_membership(protocol_dir, protocol)
    partition_manifest = inputs.load_partition_manifest(protocol_dir, protocol)
    source = inputs.mplitrj_source(protocol, config["inference"]["source_file"])
    frames = inputs.published_train_frames(records, source)
    try:
        inputs.require_study1_outcome_access(
            protocol, inputs.registration_addendum(protocol_dir), records,
            [row["frame_id"] for row in frames],
        )
    except inputs.Study1InputError as exc:
        raise SystemExit(f"stopped before reading any statistic: {exc}") from exc
    constants = inputs.statistical_constants(protocol)
    if protocol["inclusion_exclusion_rules"]["primary_tm_hubbard_u"].get(
        "sensitivity"
    ) != analysis.F_BLOCK_RULE:
        raise SystemExit("frozen f-block sensitivity rule differs from the implemented one")
    registry = inputs.frozen_models(protocol)
    groups = {name: model["frozen_group"] for name, model in registry.items()}

    # ---- inputs: reference and every model, same frozen identity
    runs = {}
    for run in [inference.REFERENCE_RUN, *sorted(registry)]:
        directory = root / "inference" / run
        if not (directory / "frame_statistics_manifest.json").is_file():
            raise SystemExit(f"missing Study-1 inference for {run}; run `infer --all-models`")
        runs[run] = inference.verify_statistics_file(directory)
    identities = {run: manifest["scientific_identity"] for run, (manifest, _) in runs.items()}
    current_code = inference.code_sha256(REPOSITORY, inference.INFERENCE_CODE_FILES)
    for run, identity in identities.items():
        shared = (identity["code_sha256"], identity["statistics_schema"],
                  identity["membership_sha256"], identity["frames_per_shard"],
                  identity["protocol_json_sha256"])
        if shared != (current_code, identities[inference.REFERENCE_RUN]["statistics_schema"],
                      protocol["partition_definition"]["partition_universe"]["membership_sha256"],
                      config["inference"]["frames_per_shard"],
                      config["frozen_protocol"]["protocol_json_sha256"]):
            raise SystemExit(f"{run}: inference code, schema, membership or shard plan differs "
                             "from this checkout; rerun that inference")
        if run != inference.REFERENCE_RUN:
            _, environment = inputs.model_runtime(protocol, run)
            if (identity["environment_lock_sha256"], identity["requested_dtype"]) != (
                environment["environment_lock"]["sha256"],
                inference.requested_dtype(registry[run]),
            ):
                raise SystemExit(f"{run}: runtime lock or dtype differs from the frozen protocol")
        expected = (protocol["protocol_id"], source["sha256"],
                    identities[inference.REFERENCE_RUN]["frame_index_sha256"])
        if (identity["protocol_id"], identity["source_file_sha256"],
                identity["frame_index_sha256"]) != expected:
            raise SystemExit(f"{run}: inference belongs to another protocol, source or index")
        if run != inference.REFERENCE_RUN and identity["model"]["weights_sha256"] != registry[run][
            "weights_sha256"
        ]:
            raise SystemExit(f"{run}: checkpoint differs from the frozen registry")
        if run != inference.REFERENCE_RUN:
            try:
                inference.verify_pilot_gate_binding(
                    root / "inference" / run, runs[run][0].get("pilot_gate"), identity
                )
            except inference.PilotGateBindingError as exc:
                raise SystemExit(f"{run}: {exc}") from exc
    consolidated = analysis.consolidate(
        frames, runs[inference.REFERENCE_RUN][1],
        {run: arrays for run, (_, arrays) in runs.items() if run != inference.REFERENCE_RUN},
    )
    runs = {run: (manifest, None) for run, (manifest, _) in runs.items()}  # free arrays
    metadata = analysis.frame_metadata(frames, consolidated["frame_id"].tolist(), partition_manifest)
    del frames
    material_codes = {m: i for i, m in enumerate(sorted({r["material_id"] for r in metadata}))}

    failures = {run: len(m["frame_failures"]) for run, (m, _) in runs.items()
                if run != inference.REFERENCE_RUN}
    if any(failures.values()):
        # No registered rule says how a failed frame enters paired comparisons; stop.
        raise SystemExit(f"model inference failed on some frames {failures}; the handling of "
                         "failed frames needs a recorded decision before analysis")
    identity = {
        "schema_version": analysis.ANALYSIS_SCHEMA,
        "pilot_amendment_ids": inputs.pilot_amendment_ids(protocol_dir),
        "protocol_id": protocol["protocol_id"],
        "protocol_json_sha256": config["frozen_protocol"]["protocol_json_sha256"],
        "membership_sha256": protocol["partition_definition"]["partition_universe"][
            "membership_sha256"
        ],
        "partition": inputs.PUBLISHED_PARTITION,
        "code_sha256": inference.code_sha256(REPOSITORY, analysis.ANALYSIS_CODE_FILES),
        "statistics_sha256": {run: m["statistics_sha256"] for run, (m, _) in runs.items()},
        "constants": constants,
    }
    fingerprint = canonical_sha256(identity)
    out = root / "analysis" / fingerprint[:16]
    out.mkdir(parents=True, exist_ok=True)
    print(f"analysis directory {out}", file=sys.stderr)

    input_digests = _save_inputs(out / "analysis_inputs", {
        "set_stats": consolidated["set_stats"],
        "set_counts": consolidated["set_counts"],
        "success": consolidated["success"],
        "material_code": np.array([material_codes[r["material_id"]] for r in metadata],
                                  dtype=np.int64),
        "element_row": consolidated["element_row"],
        "element_Z": consolidated["element_Z"],
        "element_counts": consolidated["element_counts"],
        "element_stats": consolidated["element_stats"],
    }, out)
    _publish_or_verify(out / "analysis_inputs" / "models.json", consolidated["models"], out)

    # ---- frozen intermediates (before any estimate or draw)
    hops = inputs.study1_target_hops(records)
    population_of_system = partition_manifest["primary_population"]["assignment_by_chemical_system"]
    units_path = out / "frozen" / "eligibility_units.json"
    frame_order_sha = canonical_sha256(consolidated["frame_id"].tolist())
    if units_path.exists():
        frozen_units = json.loads(units_path.read_text(encoding="utf-8"))
        if (frozen_units["membership_sha256"], frozen_units.get("frame_ids_sha256")) != (
            identity["membership_sha256"], frame_order_sha
        ):
            raise SystemExit("stale eligibility units")
        units, reasons = frozen_units["units"], frozen_units["reason_counts"]
    else:
        units, reasons = analysis.eligibility_units(
            protocol=protocol, partition_records=records, metadata=metadata,
            target_hops=hops, population_of_system=population_of_system,
        )
        publish_json_once(units_path, {
            "classification": "frozen leakage audit (research_design_3.6_v1), published split",
            "membership_sha256": identity["membership_sha256"],
            "frame_ids_sha256": frame_order_sha,
            "pooled_sentinel_target_hop": analysis.POOLED_SENTINEL_HOP,
            "units": units, "reason_counts": reasons,
        }, root=out)
    del records

    deciles = np.zeros(len(metadata), dtype=np.int64)
    decile_records = {}
    for population in analysis.POPULATIONS:
        record, assigned = analysis.energy_deciles(
            metadata, consolidated["n_atoms"], consolidated["dft_energy_eV"], population
        )
        decile_records[population] = record
        deciles = np.where(assigned > 0, assigned, deciles)
    _publish_or_verify(out / "frozen" / "study1_energy_deciles.json", {
        "classification": "frozen configurational_energy_deciles; DFT energies only",
        "rule": protocol["study_1_contract"]["configurational_energy_deciles"],
        "membership_sha256": identity["membership_sha256"],
        "reference_statistics_sha256": identity["statistics_sha256"][inference.REFERENCE_RUN],
        "populations": decile_records,
    }, out)

    informative = consolidated["set_counts"][:, 0, 1] > 0
    hop_sets = {}
    for population in analysis.POPULATIONS:
        hop_sets[population] = analysis.label_budget_hop_set(
            units, informative, population, constants["support"]["label_budget_min_population"]
        )
        _publish_or_verify(out / "frozen" / f"study1_label_budget_hop_set_{population}.json", {
            "classification": "frozen label_budget_stability.hop_set_rule; model-independent",
            **{k: v for k, v in hop_sets[population].items() if k != "hops"},
            "population_frame_ids_sha256": canonical_sha256({
                hop["hop_id"]: [metadata[i]["frame_id"] for i in hop["population_rows"]]
                for hop in hop_sets[population]["hops"]
            }),
        }, out)

    # ---- shards
    payloads = analysis.estimate_payloads(units, deciles, metadata, constants=constants)
    chemsys_by_hop = {hop["hop_id"]: hop["chemical_system_id"] for hop in hops}
    f_block_free = {h for h, system in chemsys_by_hop.items() if analysis.f_block_free(system)}
    common_hops = analysis.common_hop_population(
        payloads, consolidated["set_counts"],
        np.array([material_codes[r["material_id"]] for r in metadata], dtype=np.int64),
        f_block_free,
    )
    common_path = out / "frozen" / "study1_common_hop_population.json"
    _publish_or_verify(common_path, {
        "schema_version": analysis.COMMON_HOP_SCHEMA,
        "classification": "frozen paired-comparison hop population (estimator_minimum_support);"
                          " DFT support only, model-independent, fixed before any estimate",
        "purpose": "H3 group comparison and every paired cross-model s-bar (s_bar_common_hops)",
        "definition": "primary-estimator target hops whose all-atom material-scope estimate "
                      "meets the frozen support rule; f-block sensitivity = primary hops with "
                      "an f-block-free chemical system",
        "hop_identifier": "partition-membership hop_id",
        "protocol_id": identity["protocol_id"],
        "protocol_json_sha256": identity["protocol_json_sha256"],
        "membership_sha256": identity["membership_sha256"],
        "partition": identity["partition"],
        "analysis_code_sha256": identity["code_sha256"],
        "support_parameters": constants["support"],
        "frame_ids_sha256": frame_order_sha,
        "eligibility_units_sha256": _sha(units_path),
        "reference_statistics_sha256": identity["statistics_sha256"][inference.REFERENCE_RUN],
        "set_counts_sha256": input_digests["set_counts"],
        "populations": {
            population: {"hop_ids": ids, "n_hops": len(ids),
                         "hop_ids_sha256": canonical_sha256(ids)}
            for population, ids in common_hops.items()
        },
    }, out)
    common_artifact = json.loads(common_path.read_text(encoding="utf-8"))
    try:
        common_hops = analysis.validate_common_hop_population(common_artifact)
    except analysis.Study1AnalysisError as exc:
        raise SystemExit(f"{common_path}: {exc}") from exc
    for population in analysis.POPULATIONS:
        payloads += analysis.label_budget_payloads(hop_sets[population], constants)
    shards = []
    for payload in payloads:
        if payload["unit"] == "estimate":
            key = {"analysis_id": payload["analysis_id"], "sets": payload["component_sets"]}
        else:
            key = {"label_budget": payload["population"], "hop_id": payload["hop_id"]}
        key["rows_sha256"] = canonical_sha256(payload["rows"])
        key["selection_sha256"] = payload["selection_sha256"]
        shards.append(make_shard(f"study1.{payload['unit']}", key, payload))
    largest = max(len(payload["rows"]) for payload in payloads)
    per_worker = 512 * 1024**2 + largest * len(consolidated["models"]) * 3 * 3 * 8 * 4
    policy = WorkloadPolicy(name="study1_statistics", threads_per_worker=1,
                            per_worker_memory_bytes=per_worker,
                            memory_reserve_bytes=int(args.memory_reserve_gib * GIB))
    plan = plan_workers(args.workers, policy, observe_host_resources(), unit_count=len(shards))
    run = ShardedRun(
        output_dir=out / "units",
        scientific_identity={**identity, "analysis_inputs_sha256": input_digests},
        task=analysis.run_unit,
        protected_roots=[data_root],
        worker_setup=analysis.analysis_worker_setup,
        worker_setup_args=(str(out / "analysis_inputs"),),
    )
    result = execute_shards(run, shards, plan, resume=args.resume)

    # ---- tables: written to a staging directory, then moved into place together
    if (out / "study1_analysis_manifest.json").exists():
        print(f"analysis already complete: {out / 'study1_analysis_manifest.json'}")
        return 0
    if (out / "derived").exists() or (out / "tables").exists():
        raise SystemExit(f"{out} holds tables without a manifest (interrupted write); "
                         "inspect and move them aside before rerunning")
    staging = out / f"staging-{os.getpid()}"
    rows = tables.estimate_rows(result.results, groups)
    derived, paper = staging / "derived", staging / "tables"
    tables.write_csv(derived / "study1_softening_estimates.csv", rows, tables.ESTIMATE_COLUMNS)
    label_results = {
        population: sorted((r for r in result.results.values()
                            if r.get("population") == population and "draws_sha256" in r),
                           key=lambda r: r["hop_id"])
        for population in analysis.POPULATIONS
    }
    hop_rows, stability_rows = [], []
    for population, results in label_results.items():
        if not results:
            continue
        per_hop, aggregate = analysis.label_budget_summary(
            results, consolidated["models"], constants["label_budget_k"],
            constants["label_budget_plateau_sd"], constants["label_budget_supported_max_k"],
        )
        hop_rows += per_hop
        stability_rows += aggregate
        repeats = {"hop_ids": np.array([r["hop_id"] for r in results], dtype=str),
                   "k_values": np.array(constants["label_budget_k"], dtype=np.int64)}
        for model in consolidated["models"]:
            repeats[f"s_k__{model}"] = np.array([
                [r["models"][model]["s_k"][str(k)] if r["models"][model]["status"] == "complete"
                 else [np.nan] * constants["label_budget_repeats"]
                 for k in constants["label_budget_k"]]
                for r in results
            ])
        derived.mkdir(parents=True, exist_ok=True)
        inference.write_npz_once(derived / f"study1_label_budget_repeats_{population}.npz", repeats)
    tables.write_csv(derived / "study1_label_budget_hops.csv", hop_rows, (
        "population", "model", "hop_id", "material_id", "K", "population_frames", "s_all",
        "median_s_k", "sd_s_k", "sd_s_k_fpc", "abs_median_s_k_minus_s_all"))
    stability_columns = ("population", "model", "K", "complete", "hops", "materials",
                         "median_material_mean_sd_fpc", "median_material_mean_abs_bias",
                         "plateau_K", "h4_supported")
    tables.write_csv(derived / "study1_label_budget_stability.csv", stability_rows,
                     stability_columns)

    try:
        summary = tables.model_summary(rows, consolidated["models"], groups, f_block_free,
                                       common_hops)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    summary_columns = list(summary[0])
    tables.write_csv(derived / "study1_model_summary.csv", summary, summary_columns)
    coverage = tables.pretraining_coverage(summary, registry)
    publish_json_once(derived / "study1_pretraining_coverage.json", coverage, root=staging)

    accounting = tables.inclusion_accounting(metadata, units, reasons, rows, hop_sets, failures)
    tables.write_csv(derived / "study1_inclusion_accounting.csv", accounting,
                     ("category", "population", "model", "count"))
    compute = []
    for run_name in runs:
        for path in sorted((root / "inference" / run_name / "study1_runs").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            compute.append({
                "run": run_name, "invocation": path.name,
                "workers": record["plan"]["workers"],
                "threads_per_worker": record["plan"]["threads_per_worker"],
                "wall_seconds": record["wall_seconds"],
                "children_cpu_seconds": record["children_cpu_seconds"],
                "frames_evaluated": record["frames_evaluated"],
                "throughput_frames_per_second": record["throughput_frames_per_second"],
                "peak_worker_resident_bytes": record["peak_worker_resident_bytes"],
                "shards_computed": record["shards_computed"],
                "shards_reused": record["shards_reused"],
                "error": record["error"],
            })
    compute_columns = ("run", "invocation", "workers", "threads_per_worker", "wall_seconds",
                       "children_cpu_seconds", "frames_evaluated",
                       "throughput_frames_per_second", "peak_worker_resident_bytes",
                       "shards_computed", "shards_reused", "error")
    tables.write_csv(derived / "study1_compute_cost.csv", compute, compute_columns)
    _paper_tables(paper, rows, summary, stability_rows, accounting, compute)
    os.rename(derived, out / "derived")
    os.rename(paper, out / "tables")
    staging.rmdir()
    derived, paper = out / "derived", out / "tables"

    outputs = {
        str(path.relative_to(out)): _sha(path)
        for path in sorted([*derived.rglob("*"), *paper.rglob("*"), *(out / "frozen").rglob("*")])
        if path.is_file()
    }
    finished = {
        "schema_version": analysis.ANALYSIS_SCHEMA,
        "analysis_fingerprint": fingerprint,
        "identity": identity,
        "analysis_inputs_sha256": input_digests,
        "units_run_fingerprint": run.run_fingerprint,
        "units_merged_sha256": result.merged_sha256,
        "common_hop_population": {
            "file": str(common_path.relative_to(out)),
            "sha256": _sha(common_path),  # file bytes, as sha256sum reports
            "canonical_json_sha256": canonical_sha256(common_artifact),
        },
        "pilot_gates": {run: m["pilot_gate"] for run, (m, _) in runs.items()
                        if run != inference.REFERENCE_RUN},
        "execution_manifest": str(result.manifest_path),
        "implementation_git_sha": git_sha,
        "implementation_git_dirty": git_dirty,
        "study1_config_sha256": config_sha,
        "interpretation_notes_extra": {
            "label_budget_sd": "ddof=1; H4 decision for the primary population only",
            "f_block_sensitivity_scope": "s-bar and pooled global (all, Li, host), primary",
            "failed_frames": "analysis refuses to run while any model frame failed",
            "pooled_atlas_label": analysis.ATLAS_DESCRIPTIVE_LABEL,
        },
        "leakage_contract_sha256": {
            "at_freeze_commit": _contracts_at(protocol["git_sha"]),
            "now": _sha(REPOSITORY / "src/e3sse/protocol/contracts.py"),
            "note": "batched audit refactor; selection-equivalence test in "
                    "tests/test_study1_statistics.py",
        },
        "numpy_version": np.__version__,
        "draw_mapping": DRAW_MAPPING,
        "interpretation_notes": {
            "support_components": "support counts informative components over all atoms of "
                                  "the estimate's frames (frozen validate_softening_support); "
                                  "own-component count reported as n_informative_components",
            "intervals": "whole-frame percentile bootstrap, conditional on observed paths "
                         "and materials; no formal coverage guarantee",
            "pooled_atlas": f"pooled scopes audited against sentinel {analysis.POOLED_SENTINEL_HOP}"
                            "; they correct no hop",
            "h3": "descriptive, on the common hop set estimated for every model",
        },
        "estimate_status_counts": dict(Counter(row["status"] for row in rows)),
        "outputs_sha256": outputs,
        "schemas": {
            "derived/study1_softening_estimates.csv": list(tables.ESTIMATE_COLUMNS),
            "derived/study1_label_budget_stability.csv": list(stability_columns),
            "derived/study1_model_summary.csv": summary_columns,
            "derived/study1_compute_cost.csv": list(compute_columns),
            "analysis_inputs/set_stats.npy": "[model, frame (frame_id order), set (all, Li, "
                                             "host), (sxy, sxx, syy)] eV^2/angstrom^2",
            "analysis_inputs/set_counts.npy": "[frame, set, (n_atoms, n_informative)]",
            "analysis_inputs/element_stats.npy": "[model, element row, (sxy, sxx, syy)]",
        },
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    publish_json_once(out / "study1_analysis_manifest.json", finished, root=out)
    print(json.dumps({"analysis": str(out), "estimates": len(rows),
                      "status": finished["estimate_status_counts"]}))
    return 0


def _paper_tables(paper, rows, summary, stability, accounting, compute) -> None:
    primary = [r for r in summary if r["population"] == "primary"]
    tables.write_csv(paper / "table_study1_global_softening.csv", [
        {"model": r["model"], "group": r["model_group"], "frames": r["global_frames"],
         "materials": r["global_materials"], "s_global": tables.rounded(r["global_s"], 3),
         "s_global_95ci": tables.interval(r["global_s_ci_low"], r["global_s_ci_high"]),
         "s_bar": tables.rounded(r["s_bar"], 3), "hops_estimated": r["hops_estimated"],
         "hops_total": r["target_hops"]}
        for r in primary
    ], ("model", "group", "frames", "materials", "s_global", "s_global_95ci", "s_bar",
        "hops_estimated", "hops_total"))
    family = [r for r in rows if r["kind"] == "pooled" and r["scope"] == "family"]
    tables.write_csv(paper / "table_study1_family_softening.csv", [
        {"model": r["model"], "group": r["model_group"], "population": r["population"],
         "family": r["scope_target"], "frames": r["n_frames"], "materials": r["n_materials"],
         "status": r["status"], "s": tables.rounded(r["s_hat"], 3),
         "s_95ci": tables.interval(r["ci_low"], r["ci_high"])}
        for r in family
    ], ("model", "group", "population", "family", "frames", "materials", "status", "s", "s_95ci"))
    tables.write_csv(paper / "table_study1_material_support.csv", [
        {"model": r["model"], "group": r["model_group"], "population": r["population"],
         "target_hops": r["target_hops"], "estimated": r["hops_estimated"],
         "unsupported": r["hops_unsupported"], "incomplete_inference": r["hops_incomplete_inference"],
         "s_bar": tables.rounded(r["s_bar"], 3),
         "s_iqr": tables.interval(r["s_hop_q25"], r["s_hop_q75"])}
        for r in summary
    ], ("model", "group", "population", "target_hops", "estimated", "unsupported",
        "incomplete_inference", "s_bar", "s_iqr"))
    element = [r for r in rows if r["kind"] == "pooled" and r["scope"] == "element"
               and r["population"] == "primary"]
    tables.write_csv(paper / "table_study1_element_host_softening.csv", [
        {"model": r["model"], "group": r["model_group"], "s_li": tables.rounded(r["global_s_li"], 3),
         "s_li_95ci": tables.interval(r["global_s_li_ci_low"], r["global_s_li_ci_high"]),
         "s_host": tables.rounded(r["global_s_host"], 3),
         "s_host_95ci": tables.interval(r["global_s_host_ci_low"], r["global_s_host_ci_high"]),
         "median_hop_s_li_minus_s_host": tables.rounded(r["median_hop_s_li_minus_s_host"], 3)}
        for r in primary
    ] + [
        {"model": r["model"], "group": r["model_group"], "element": r["scope_target"],
         "frames": r["n_frames"], "status": r["status"], "s_element": tables.rounded(r["s_hat"], 3),
         "s_element_95ci": tables.interval(r["ci_low"], r["ci_high"])}
        for r in element
    ], ("model", "group", "s_li", "s_li_95ci", "s_host", "s_host_95ci",
        "median_hop_s_li_minus_s_host", "element", "frames", "status", "s_element",
        "s_element_95ci"))
    tables.write_csv(paper / "table_study1_label_budget.csv", [
        {"population": r["population"], "model": r["model"], "K": r["K"], "hops": r["hops"],
         "materials": r["materials"],
         "median_sd_fpc": tables.rounded(r["median_material_mean_sd_fpc"], 4),
         "median_abs_bias": tables.rounded(r["median_material_mean_abs_bias"], 4),
         "plateau_K": r["plateau_K"], "h4_supported": r["h4_supported"]}
        for r in stability
    ], ("population", "model", "K", "hops", "materials", "median_sd_fpc", "median_abs_bias",
        "plateau_K", "h4_supported"))
    tables.write_csv(paper / "table_study1_inclusion_accounting.csv", accounting,
                     ("category", "population", "model", "count"))
    tables.write_csv(paper / "table_study1_compute_cost.csv", [
        {"run": r["run"], "workers": r["workers"], "threads_per_worker": r["threads_per_worker"],
         "wall_hours": tables.rounded((r["wall_seconds"] or 0) / 3600, 2),
         "cpu_hours": tables.rounded((r["children_cpu_seconds"] or 0) / 3600, 1),
         "frames_evaluated": r["frames_evaluated"],
         "frames_per_second": tables.rounded(r["throughput_frames_per_second"], 1),
         "peak_worker_memory_gib": tables.rounded((r["peak_worker_resident_bytes"] or 0) / GIB, 2)}
        for r in compute
    ], ("run", "workers", "threads_per_worker", "wall_hours", "cpu_hours", "frames_evaluated",
        "frames_per_second", "peak_worker_memory_gib"))


if __name__ == "__main__":
    raise SystemExit(main())
