"""Frozen ML-NEB over population units, in each model's own runtime.

One runner serves four stages; only the unit roles and models differ:

=====================  ==========================================  ============
stage                  units (population manifest role)            models
=====================  ==========================================  ============
development_ml_neb     development                                 every model
weight_slice_ml_neb    calibration_weight_slice, pool_weight_slice every model
deployment_calibration calibration                                 selected
discovery              pool                                        selected
=====================  ==========================================  ============

Every unit runs ``barriers.ml_neb.run_ml_neb`` once on its frozen BVSE path, with
the frozen settings, the frozen per-model dtype and a distinct calculator per image
(a per-worker pool, reset between units).  This is equivalent to "one calculator per
image": each image gets its own instance, and ``reset()`` clears every cached result,
so no state crosses images or units; only the loaded weights are reused, and the
dtype of every floating-point parameter of every pooled calculator is checked.  No unit is retried: a failure is a
recorded category, and only infrastructure interruptions resume.  Units without a
usable BVSE path are recorded as not attempted (an initialisation failure).  For
models whose atom-level representation is validated, the migrating-Li embedding of
every converged band image is extracted through the pinned capture, twice, and its
determinism recorded.  Development additionally runs the 20-hop reproducibility
rerun and the LiTraj-exact sensitivity.  Nothing here reads a DFT outcome.
"""

from __future__ import annotations

import hashlib
from typing import Any

from e3sse.production.runners.context import (
    MissingEvidence,
    RunnerContext,
    RunnerValidationError,
    file_sha256,
    load_json,
)

STAGE_ROLES = {
    "development_ml_neb": ("development",),
    "weight_slice_ml_neb": ("calibration_weight_slice", "pool_weight_slice"),
    "deployment_calibration": ("calibration",),
    "discovery": ("pool",),
}
SELECTED_MODEL_STAGES = ("deployment_calibration", "discovery")
PRE_FREEZE_STAGES = ("development_ml_neb",)
REPRODUCIBILITY_SUBSET = 20
REPRODUCIBILITY_TOLERANCE_EV = 1e-4
CODE_FILES = ("src/e3sse/barriers/ml_neb.py", "src/e3sse/production/runners/ml_neb_units.py",
              "src/e3sse/representation/atom_embeddings.py",
              "src/e3sse/representation/embedding_capability.py", "src/e3sse/protocol/adapters.py",
              "src/e3sse/screening/prospective_contract.py")
ATOM_ORDER_ATOL = 1e-6  # permutation check of the capture (float noise only)


# --------------------------------------------------------------------- worker side

def ml_neb_worker_setup(model: dict, checkpoint: str, frozen_settings: dict, dtype: str,
                        capture: dict | None, provenance: dict) -> dict[str, Any]:
    """Calculator pool, the frozen settings and the pinned embedding extractor."""

    from pathlib import Path

    import torch

    from e3sse.barriers.ml_neb import settings_from_protocol
    from e3sse.protocol.adapters import build_calculator
    from e3sse.representation.atom_embeddings import embedding_extractor, torch_module

    settings = settings_from_protocol({"amendments": {"prospective": {"ml_neb": {
        "frozen_numeric_settings": frozen_settings}}}})

    def checked(calc: Any) -> Any:
        """Every floating-point parameter must have the frozen dtype (not just the first)."""

        found = {str(p.dtype).replace("torch.", "")
                 for p in torch_module(calc, model["adapter_name"]).parameters()
                 if p.is_floating_point()}
        if found != {dtype}:
            raise RuntimeError(f"{model['canonical_name']}: loaded floating dtypes {sorted(found)} "
                               f"are not the frozen ML-NEB dtype {dtype}")
        return calc

    load = lambda: checked(build_calculator(model, Path(checkpoint)))
    pool = [load()]  # the probe is the pool's first calculator
    extractor = None
    if capture is not None:
        extractor = embedding_extractor(model, load(), capture_mode=capture["capture_mode"],
                                        location=capture["location"])
    return {"load": load, "pool": pool, "settings": settings, "extractor": extractor,
            "provenance": {**provenance, "dtype": dtype,
                           "torch_intra_op_threads": torch.get_num_threads(),
                           "torch_inter_op_threads": torch.get_num_interop_threads()}}


def _atoms(frame: dict[str, Any]) -> Any:
    from ase import Atoms

    return Atoms(numbers=frame["numbers"], positions=frame["positions"], cell=frame["cell"],
                 pbc=frame["pbc"])


def ml_neb_task(payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from e3sse.barriers.ml_neb import litraj_exact_variant, run_ml_neb

    images = [_atoms(frame) for frame in payload["images"]]
    pool = context["pool"]
    while len(pool) < len(images):
        pool.append(context["load"]())
    handed = iter(pool)

    def factory() -> Any:  # one distinct calculator per image, reset between units
        calc = next(handed)
        calc.reset()
        return calc

    settings = context["settings"]
    if payload.get("variant") == "litraj_exact":
        settings = litraj_exact_variant(settings)
    record = run_ml_neb(images, factory, settings, unit_id=payload["unit_id"],
                        provenance=context["provenance"])
    record["unit_key"], record["variant"] = payload["unit_key"], payload.get("variant", "frozen")
    extractor = context["extractor"]
    if record["status"] == "converged" and extractor is not None and payload.get("embed", True):
        moving = record["migrating_atom_index"]
        rows, deterministic, finite, order_difference = [], True, True, None
        for position, frame_positions in enumerate(record["band_geometry"]["positions"]):
            atoms = _atoms({**record["band_geometry"], "positions": frame_positions})
            first, second = extractor(atoms), extractor(atoms)
            deterministic &= bool(np.array_equal(first, second))
            finite &= bool(np.isfinite(first).all())
            if first.shape[0] != len(atoms):
                raise RuntimeError("the pinned capture is not atom-level for this band")
            if position == 0:  # rows follow atom order: a reversed copy gives reversed rows
                order = np.arange(len(atoms))[::-1]
                permuted = np.asarray(extractor(atoms[order]))
                order_difference = float(np.abs(permuted[np.argsort(order)] - first).max())
            rows.append(first[moving].tolist())
        record["migrating_li_embeddings"] = rows
        record["embedding_checks"] = {
            "finite": finite, "deterministic": deterministic,
            "atom_order_max_abs_difference": order_difference,
            "atom_order_preserved": order_difference is not None
            and order_difference <= ATOM_ORDER_ATOL}
    return record


# --------------------------------------------------------------------- orchestration side

def population_units(ctx: RunnerContext, roles: tuple[str, ...]) -> list[dict[str, Any]]:
    population = ctx.upstream("development_population", "population_manifest")
    return [u for u in population["units"] if u["role"] in roles]


def selected_model(ctx: RunnerContext) -> str:
    return ctx.upstream("model_selection", "model_selection")["selection"]["selected_model"]


def pinned_capture(ctx: RunnerContext, model: str, amendments: dict[str, Any]) -> dict | None:
    """Capture of a validated latent representation; None for the descriptor branch."""

    pin = amendments["representation"]["embedding_capability_evidence"]["reports"][model]
    if pin["latent_support_status"] != "validated":
        return None
    from pathlib import Path

    if file_sha256(Path(pin["path"])) != pin["sha256"]:
        raise RunnerValidationError(f"{model}: pinned embedding report differs from its SHA-256")
    report = load_json(Path(pin["path"]))
    return {"capture_mode": report["capture_mode"], "location": report["location"],
            "report_sha256": pin["sha256"]}


def frozen_ml_neb_identity(ctx: RunnerContext, model: str) -> dict[str, Any]:
    """Settings hash, dtype and capture of ``model`` under the frozen protocol."""

    from e3sse.barriers.ml_neb import frozen_model_dtype, settings_from_protocol

    amendments = ctx.frozen_amendments()
    capture = pinned_capture(ctx, model, amendments)
    return {"settings_sha256": settings_from_protocol({"amendments": amendments}).settings_sha256,
            "dtype": frozen_model_dtype({"amendments": amendments}, model).split()[0],
            "capture_report_sha256": None if capture is None else capture["report_sha256"]}


def require_development_matches_freeze(ctx: RunnerContext, model: str) -> dict[str, Any]:
    """Development ML-NEB ran before the freeze under the active protocol; every
    post-freeze consumer requires its settings, dtype and capture to equal the frozen
    ones, so model selection and later stages never mix two ML-NEB definitions."""

    development = ctx.upstream("development_ml_neb", "ml_neb_records", model=model)
    expected = frozen_ml_neb_identity(ctx, model)
    observed = {key: development.get(key) for key in expected}
    if observed != expected:
        raise RunnerValidationError(f"{model}: development ML-NEB identity {observed} differs "
                                    f"from the frozen identity {expected}")
    return development


def reproducibility_subset(unit_ids: list[str]) -> list[str]:
    """First 20 development hops by SHA-256 rank of the hop ID (frozen rule), ranked
    among development hops with a usable BVSE path (the hops ML-NEB can run)."""

    return sorted(unit_ids, key=lambda h: (hashlib.sha256(h.encode()).hexdigest(), h)
                  )[:REPRODUCIBILITY_SUBSET]


def run_model(ctx: RunnerContext) -> dict[str, Any]:
    """Executed by the model's frozen interpreter (``--model``)."""

    from e3sse.barriers.ml_neb import (
        frozen_model_dtype,
        litraj_exact_variant,
        settings_from_protocol,
    )
    from e3sse.production.runners.bvse_paths import load_path

    stage, model = ctx.stage, ctx.model
    if stage in SELECTED_MODEL_STAGES and model != selected_model(ctx):
        raise RunnerValidationError(f"{stage} runs the selected model only, not {model}")
    amendments = (ctx.active_protocol["amendments"] if stage in PRE_FREEZE_STAGES
                  else ctx.frozen_amendments())
    frozen = amendments["prospective"]["ml_neb"]["frozen_numeric_settings"]
    settings = settings_from_protocol({"amendments": amendments})
    # The frozen table reads e.g. "float64" or "float32 native": the dtype is the first word.
    dtype = frozen_model_dtype({"amendments": amendments}, model).split()[0]
    entry = ctx.models[model]
    if entry["adapter_name"] == "mace" and entry.get("load_options", {}).get(
            "default_dtype", "float64") != dtype:
        raise RunnerValidationError("MACE load_options dtype differs from the frozen dtype")
    environments = ctx.freeze_config["inputs"]["production_evidence"]["runtime_environments"]
    environment = next((name for name, record in environments.items()
                        if model in record["models"]), None)
    if environment is None:
        raise RunnerValidationError(f"{model} is in no pinned runtime environment")
    if stage not in PRE_FREEZE_STAGES:
        require_development_matches_freeze(ctx, model)
    provenance = {"model": model, "weights_sha256": entry["weights_sha256"],
                  "environment": environment,
                  "environment_lock_sha256": environments[environment]["environment_lock"]["sha256"],
                  "device": "cpu"}
    capture = pinned_capture(ctx, model, amendments)
    index = ctx.upstream("bvse_reproduction", "bvse_path_index")
    units, not_attempted = [], {}
    for unit in population_units(ctx, STAGE_ROLES[stage]):
        entry_path = index.get(unit["unit_key"])
        if entry_path is None or not entry_path.get("usable_for_ml_neb"):
            not_attempted[unit["unit_key"]] = (entry_path or {}).get("status", "no_bvse_path")
            continue
        units.append((unit, load_path(entry_path)))
    identity = {"code_sha256": ctx.code_sha256(CODE_FILES), "settings_sha256": settings.settings_sha256,
                "dtype": dtype, "capture": capture, "provenance": provenance}
    checkpoint = str(ctx.checkpoint(model))
    setup_args = (entry, checkpoint, frozen, dtype, capture, provenance)

    def shards(selection, variant, embed=None):
        embed = variant == "frozen" if embed is None else embed
        return [({"unit_key": u["unit_key"], "variant": variant},
                 {"unit_key": u["unit_key"], "unit_id": u["unit_id"], "images": images,
                  "variant": variant, "embed": embed})
                for u, images in selection]

    records = list(ctx.execute(namespace=f"{stage}.ml_neb", units=shards(units, "frozen"),
                               task=ml_neb_task, identity=identity, setup=ml_neb_worker_setup,
                               setup_args=setup_args, configure_torch=True).values())
    extra: dict[str, Any] = {}
    if stage == "development_ml_neb":
        subset = set(reproducibility_subset([u["unit_id"] for u, _ in units]))
        rerun = ctx.execute(namespace=f"{stage}.reproducibility_rerun",
                            units=shards([x for x in units if x[0]["unit_id"] in subset], "frozen",
                                         embed=False),  # compares barriers, not embeddings
                            task=ml_neb_task, identity={**identity, "purpose": "rerun"},
                            setup=ml_neb_worker_setup, setup_args=setup_args, configure_torch=True)
        extra["reproducibility"] = compare_reruns(records, list(rerun.values()))
        sensitivity = ctx.execute(namespace=f"{stage}.litraj_exact",
                                  units=shards(units, "litraj_exact"), task=ml_neb_task,
                                  identity={**identity, "purpose": "litraj_exact_sensitivity"},
                                  setup=ml_neb_worker_setup, setup_args=setup_args,
                                  configure_torch=True)
        extra["litraj_exact_sensitivity"] = [strip(r) for r in sensitivity.values()]
        extra["litraj_exact_source_endpoint_cap"] = {
            "steps": litraj_exact_variant(settings).source_endpoint_steps,
            "units_reaching_cap": sum(bool(r.get("source_endpoint_cap_reached"))
                                      for r in sensitivity.values()),
            "note": "declared deviation: LiTraj's source-endpoint FIRE had no step limit"}
    summary_records = sorted((strip(r) for r in records), key=lambda r: r["unit_key"])
    outputs = {"ml_neb_records": ctx.publish("ml_neb_records.json", {
        "model": model, "stage": stage, "settings": settings.as_json(),
        "settings_sha256": settings.settings_sha256, "dtype": dtype,
        "capture_report_sha256": None if capture is None else capture["report_sha256"],
        "records": summary_records,
        "not_attempted": dict(sorted(not_attempted.items())), **extra})}
    statuses: dict[str, int] = {}
    for record in summary_records:
        key = record["failure_category"] or "converged"
        statuses[key] = statuses.get(key, 0) + 1
    return {"outputs": outputs, "summary": {"model": model, "attempted": len(summary_records),
                                            "not_attempted": len(not_attempted),
                                            "outcomes": statuses}}


def strip(record: dict[str, Any]) -> dict[str, Any]:
    """The record without band geometry (kept in the shard checkpoint)."""

    return {k: v for k, v in record.items() if k != "band_geometry"}


def compare_reruns(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit = {r["unit_key"]: r for r in first}
    rows = []
    for record in second:
        original = by_unit[record["unit_key"]]
        same_status = original["status"] == record["status"]
        delta = (abs(original["barrier_eV"] - record["barrier_eV"])
                 if original["barrier_eV"] is not None and record["barrier_eV"] is not None
                 else None)
        rows.append({"unit_key": record["unit_key"], "same_status": same_status,
                     "barrier_difference_eV": delta,
                     "passes": same_status and (delta is None or delta <= REPRODUCIBILITY_TOLERANCE_EV)})
    return {"subset_size": len(rows), "tolerance_eV": REPRODUCIBILITY_TOLERANCE_EV,
            "all_pass": all(r["passes"] for r in rows), "rows": rows}


def deployed_units(ctx: RunnerContext, model: str, records: list[dict[str, Any]]
                   ) -> list[dict[str, Any]]:
    """Support decision and deployed prediction per unit (label-free)."""

    from e3sse.production.runners.deployed import (
        predict,
        support_decisions,
        support_rows,
        unit_family,
    )
    from e3sse.production.runners.representation import frozen_path_representation

    descriptors = ctx.upstream("descriptor_availability", "descriptors")["units"]
    branch = frozen_path_representation(ctx)["models"][model]
    scales = ctx.upstream("family_scale", "family_scales")["scales"][model]
    g1 = bool(ctx.upstream("transfer", "g1_decision")["passes"])
    decisions = support_decisions(branch, support_rows(branch, records, descriptors))
    units = []
    for record in records:
        decision = decisions.get(record["unit_key"], {"supported": False, "distance": None,
                                                      "reason": "ml_neb_failed"})
        family = unit_family(descriptors, record["unit_key"])
        units.append({"unit_key": record["unit_key"], "unit_id": record["unit_id"],
                      "ml_neb_status": record["status"],
                      "failure_category": record["failure_category"],
                      "failure_origin": record.get("failure_origin"),
                      "supported": bool(decision["supported"]) if record["status"] == "converged"
                      else None, "support_distance": decision.get("distance"),
                      "anion_family": family,
                      "prediction": predict(record, family=family, scales=scales,
                                            supported=bool(decision["supported"]), g1_passes=g1)})
    return units


def relaxed_vs_fixed(ctx: RunnerContext, model: str, units: list[dict[str, Any]]) -> dict[str, Any]:
    """amendments.scope.relaxed_path_comparison: Delta E_ML-NEB - Delta E_fixed (ML only)."""

    from e3sse.production.runners.transfer import raw_barriers

    single_points = ctx.upstream("transfer", "single_points", model=model)["hops"]
    fixed = raw_barriers(single_points, [u["unit_id"] for u in units])
    rows = []
    for unit in units:
        prediction = unit["prediction"]
        if prediction is None or unit["unit_id"] not in fixed:
            continue
        scale = prediction["s_family"]
        rows.append({"unit_id": unit["unit_id"],
                     "raw_difference_eV": prediction["raw_eV"] - fixed[unit["unit_id"]],
                     "srt_difference_eV": None if not scale else
                     (prediction["raw_eV"] - fixed[unit["unit_id"]]) / scale})
    return {"n_hops": len(rows), "rows": rows,
            "note": "relaxed ML-NEB minus fixed-geometry ML barrier on the same hop; no DFT"}


def run(ctx: RunnerContext) -> dict[str, Any]:
    """Project-side reduction: every model's records; deployed predictions when selected."""

    from pathlib import Path

    models = ctx.upstream_models(ctx.stage)
    if not models:
        raise MissingEvidence(f"{ctx.stage}: no model completed")
    outputs: dict[str, Path] = {}
    summary: dict[str, Any] = {"models": models}
    for model in models:
        manifest = load_json(ctx.stage_dir / "models" / model / "model_manifest.json")
        outputs[f"ml_neb_records:{model}"] = Path(manifest["outputs"]["ml_neb_records"]["path"])
    if ctx.stage in SELECTED_MODEL_STAGES:
        model = selected_model(ctx)
        data = ctx.upstream(ctx.stage, "ml_neb_records", model=model)
        units = deployed_units(ctx, model, data["records"])
        record = {"model": model, "units": units, "not_attempted": data["not_attempted"]}
        if ctx.stage == "deployment_calibration":
            record["relaxed_path_comparison"] = relaxed_vs_fixed(ctx, model, units)
        outputs["deployed_units"] = ctx.publish("deployed_units.json", record)
        summary["converged"] = sum(u["ml_neb_status"] == "converged" for u in units)
        summary["supported"] = sum(bool(u["supported"]) for u in units)
    return {"outputs": outputs, "summary": summary}
