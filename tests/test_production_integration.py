"""Synthetic end-to-end tests of the outcome-reading production pathways.

No real data root exists and nothing outside the test's temporary directory (and the
checked-in configs) is read.  The runners, sharded execution, frame-eligibility audit,
manifests and resume logic are production code; the frozen inputs (protocol,
membership, frames, models) are synthetic (production_synthetic_universe).  Outcome
access goes through the real stage allowlist and the contract's training-unit and
tuning-purpose checks; the protocol seals themselves (freeze, pilot amendment, seal
2) are covered by the contract's own tests, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import production_synthetic_universe as su
import pytest

from e3sse.production.runners import transfer
from e3sse.production.stage_graph import CALIBRATION_BARRIERS, EVALUATION_BARRIERS
from e3sse.protocol.contracts import OutcomeAccessDenied
from e3sse.softening import study1_inference


@pytest.fixture
def synthetic_models(monkeypatch):
    monkeypatch.setattr(study1_inference, "inference_worker_setup", su.synthetic_inference_setup)
    monkeypatch.setattr(transfer, "single_point_setup", su.synthetic_single_point_setup)


def _run_transfer(universe, run_dir: Path, *, workers: int = 1) -> None:
    su.build_estimate_and_pilot(universe, run_dir)
    assert su.run_phase(universe, run_dir, "transfer", "prepare", workers=workers) == 0
    for model in su.MODELS:
        assert su.run_phase(universe, run_dir, "transfer", "model", model=model,
                            workers=workers) == 0
    assert su.run_phase(universe, run_dir, "transfer", "reduce", workers=workers) == 0


TRANSFER_OUTPUTS = ("transfer_results.json", "transfer_hop_records.json", "g1_decision.json",
                    "transfer_hop_sets.json", "derived/transfer_predictions.json")


@pytest.fixture
def transferred(tmp_path, synthetic_models):
    universe = su.build(tmp_path)
    run_dir = tmp_path / "run_a"
    _run_transfer(universe, run_dir)
    return universe, run_dir


# ======================================================================= transfer

def test_transfer_end_to_end_resume_order_and_outcome_boundary(transferred, tmp_path):
    universe, run_a = transferred
    stage_dir = run_a / "transfer"
    manifest = json.loads((stage_dir / "stage_manifest.json").read_text())
    assert manifest["phase"] == "reduce" and set(manifest["outputs"]) >= {
        "transfer_results", "transfer_hop_sets", "g1_decision"}
    results = json.loads((stage_dir / "transfer_results.json").read_text())

    # raw ML barrier = scale x DFT and s_material = scale, so SRT is exact.
    a1 = results["results"]["Synth-A1"]["methods"]
    assert a1["B3_srt_material"]["method"]["mae_eV"] < 1e-6 < a1["B3_srt_material"]["raw"]["mae_eV"]
    assert results["P1"]["Synth-A1"]["rho_srt"] == pytest.approx(1.0, abs=1e-6)
    # B1/B2 are scope-level switches, reported per scope, never per hop.
    gate = results["results"]["Synth-A1"]["gated_scope_switch"]
    assert "global" in gate and any(k.startswith("family:") for k in gate)
    assert "B1_srt_global_gated_scope" in a1 and "B2_srt_family_gated_scope" in a1
    selective = a1["B4_srt_gated_selective"]  # B4 against B3 on the same hops
    assert selective["population"]["hop_ids"]
    assert selective["b3_on_same_hops"]["n"] == len(selective["population"]["hop_ids"])
    assert 0 < selective["coverage_of_b3_population"] <= 1
    # A model without any supported training diagnostic gets a closed gate (raw), recorded.
    assert results["results"]["Synth-A3"]["deployment_gate"]["status"] == "undefined"

    # Synth-A3 fails on every structure: a recorded model status, so its P1 p-value is
    # undefined; it stays in the Holm family (m = 3) and enters as p = 1.
    single = json.loads((stage_dir / "models" / "Synth-A3" / "single_points.json").read_text())
    assert {h["status"] for h in single["hops"].values()} == {"failed"}
    p1 = results["P1"]
    assert p1["Synth-A3"]["p_one_sided"] is None and p1["Synth-A3"]["p_undefined_entered_as_1"]
    assert p1["Synth-A3"]["p_holm"] == 1.0 and len(p1) == 3
    defined = sorted(e["p_one_sided"] for e in p1.values() if e["p_one_sided"] is not None)
    assert p1["Synth-A1"]["p_holm"] >= min(1.0, 3 * defined[0])  # m stays 3

    # The hop sets are published and hashed before the evaluation barriers are read.
    access = su.access_log(run_a)
    barrier_reads = [a for a in access if a["outcome_kind"] == EVALUATION_BARRIERS]
    assert barrier_reads and all(a["hop_sets_published_before"] for a in barrier_reads)
    hop_sets = json.loads((stage_dir / "transfer_hop_sets.json").read_text())
    assert hop_sets["hop_sets_sha256"] == su.canonical(hop_sets["hop_sets"])

    # Forbidden outcomes fail closed at the stage boundary.
    ctx = su.context(universe, run_a, "transfer")
    with pytest.raises(OutcomeAccessDenied):
        ctx.require_outcome(role="train", outcome="x", outcome_kind=CALIBRATION_BARRIERS,
                            unit_ids=["h"])

    # Resume: every phase again reproduces byte-identical outputs and shard identities.
    before = su.sha_tree(stage_dir, TRANSFER_OUTPUTS)
    shards_before = sorted(p.name for p in stage_dir.rglob("shards/*/*") if p.is_file())
    assert shards_before
    _run_transfer(universe, run_a)
    assert su.sha_tree(stage_dir, TRANSFER_OUTPUTS) == before
    assert sorted(p.name for p in stage_dir.rglob("shards/*/*") if p.is_file()) == shards_before

    # Another run with two workers (different completion order): identical merged output
    # and identical shard identities.
    run_b = tmp_path / "run_b"
    _run_transfer(universe, run_b, workers=2)
    assert su.sha_tree(run_b / "transfer", TRANSFER_OUTPUTS) == before
    assert sorted(p.name for p in (run_b / "transfer").rglob("shards/*/*") if p.is_file()) \
        == shards_before


def test_transfer_reduce_fails_closed_without_barrier_permission(tmp_path, synthetic_models,
                                                                  monkeypatch):
    from e3sse.production import stage_graph

    universe = su.build(tmp_path)
    run = tmp_path / "run"
    su.build_estimate_and_pilot(universe, run)
    assert su.run_phase(universe, run, "transfer", "prepare") == 0
    for model in su.MODELS:
        assert su.run_phase(universe, run, "transfer", "model", model=model) == 0
    reads = []
    monkeypatch.setattr(transfer, "dft_barriers_unchecked",
                        lambda ctx, hops: reads.append(hops) or {})
    narrowed = dict(stage_graph.PERMITTED_OUTCOME_ACCESS)
    narrowed["transfer"] = narrowed["transfer"] - {EVALUATION_BARRIERS}
    monkeypatch.setattr(stage_graph, "PERMITTED_OUTCOME_ACCESS", narrowed)
    assert su.run_phase(universe, run, "transfer", "reduce") == 3  # validation failure
    assert not (run / "transfer" / "transfer_results.json").exists()
    assert reads == []  # the barrier reader itself was never reached


# ======================================================================= Study 4

def test_study4_repeat_thresholds_disjoint_from_evaluated_units(transferred, monkeypatch):
    from e3sse.production.runners import study4
    from e3sse.selection import conformal

    universe, run = transferred
    drawn_by_repeat, threshold_by_repeat, current = {}, {}, {}
    real_draw, real_thresholds = conformal.draw_one_hop_per_system, study4.repeat_thresholds
    real_split = study4.repeat_split
    # Pin one hop-bearing training-side system per repeat into the slice, so both
    # repeats test disjointness whatever the synthetic protocol hash is.
    pinned = {0: "Li-P-S", 1: "Cl-Li"}
    monkeypatch.setattr(study4, "threshold_slice_member",
                        lambda parent, repeat, system, fraction: pinned[repeat] == system)

    def recording_split(parent, repeat, *args):
        current["repeat"] = repeat
        return real_split(parent, repeat, *args)

    def recording_draw(hops_by_system, seed):
        drawn = real_draw(hops_by_system, seed)
        drawn_by_repeat.setdefault(current["repeat"], set()).update(drawn.values())
        return drawn

    def recording_thresholds(threshold_hops, diagnostics, quantile):
        threshold_by_repeat.setdefault(current["repeat"], set()).update(
            h["hop_id"] for h in threshold_hops)
        return real_thresholds(threshold_hops, diagnostics, quantile)

    monkeypatch.setattr(study4, "repeat_split", recording_split)
    monkeypatch.setattr(conformal, "draw_one_hop_per_system", recording_draw)
    monkeypatch.setattr(study4, "repeat_thresholds", recording_thresholds)
    assert su.run_phase(universe, run, "controlled_selection", "prepare") == 0
    for model in su.MODELS:
        assert su.run_phase(universe, run, "controlled_selection", "model", model=model) == 0
    assert su.run_phase(universe, run, "controlled_selection", "reduce") == 0
    output = json.loads((run / "controlled_selection" / "study4_selection.json").read_text())

    # Threshold units and evaluated (drawn calibration + test) units never overlap.
    parent_id = universe["protocol"]["protocol_id"]
    fraction = universe["active"]["amendments"]["deployment_gate"]["study4_threshold_slice"][
        "fraction"]
    assert drawn_by_repeat and threshold_by_repeat
    for repeat, assignment in enumerate(su.REPEATS):
        split = real_split(parent_id, repeat,
                           [s for s, r in assignment.items() if r == "training_or_calibration"],
                           [s for s, r in assignment.items() if r == "test"], fraction)
        assert split["threshold"] == {pinned[repeat]}
        assert not split["threshold"] & (split["calibration"] | split["test"])
        assert threshold_by_repeat[repeat] and drawn_by_repeat[repeat]
        assert not threshold_by_repeat[repeat] & drawn_by_repeat[repeat]
        drawn_materials = {h.split("_")[0] for h in drawn_by_repeat[repeat]}
        threshold_materials = {h.split("_")[0] for h in threshold_by_repeat[repeat]}
        assert not drawn_materials & threshold_materials
    for model, slices in output["threshold_slices"].items():
        assert [s["repeat"] for s in slices] == [0, 1]
        if model == "Synth-A3":  # no successful single point: no member unit at all
            assert all(r["threshold_units"] == r["evaluated_units"] == 0 for r in slices)
            continue
        for record in slices:
            assert record["evaluated_units"] > 0 and record["threshold_units"] > 0
            assert record["threshold_status"] == "defined"
            assert record["threshold_accounting"]["rho_F"]["n_used"] >= 1
    # the symmetric family fallback is used when the training stratum lacks a family
    sources = {v for s in output["threshold_slices"]["Synth-A1"] for v in s["scale_sources"].values()}
    assert "symmetric_fallback_family_units_forces" in sources

    # Paired H5b and Monte Carlo s.e. of power are reported per predictor and setting.
    h5b = output["h5b"]["Synth-A1"]
    assert {key.split("|")[0] for key in h5b} == {f"{p}-raw" for p in study4.PREDICTORS[1:]}
    # power (hence the paired difference) is defined only in repeats whose drawn test
    # units include a true low-barrier hop
    assert any(entry["repeats"] >= 1 and entry["mean_power_difference"] is not None
               for entry in h5b.values())
    assert output["undefined_draws"]["Synth-A1"] < len(su.REPEATS) * 4
    summary = output["results"]["Synth-A1"]
    assert all("power_mc_se" in entry for entry in summary.values())

    # Deterministic: a resumed reduce reproduces the same file.
    before = su.sha_tree(run / "controlled_selection", ["study4_selection.json"])
    monkeypatch.setattr(conformal, "draw_one_hop_per_system", real_draw)
    monkeypatch.setattr(study4, "repeat_thresholds", real_thresholds)
    monkeypatch.setattr(study4, "repeat_split", real_split)
    assert su.run_phase(universe, run, "controlled_selection", "reduce") == 0
    assert su.sha_tree(run / "controlled_selection", ["study4_selection.json"]) == before


# ======================================================================= discovery

FAMILIES = ("oxide", "sulfide", "halide", "nitride", "other")
ANIONS = ("O", "S", "Cl", "Br", "N", "Se", "F", "I")


def _descriptor(family: str, *, hop: float = 2.5, jitter: float = 0.0, far: bool = False):
    one_hot = [1.0 if f == family else 0.0 for f in FAMILIES]
    base = [0.3 + jitter, 8.0 + 3 * jitter, 3.0 + jitter, 12.0, 0.03, hop, 0.4 + jitter]
    if far:
        base = [0.9, 40.0, 15.0, 40.0, 0.2, hop, 3.0]
    return [*base, *one_hot]


def _unit(side: str, system: str, hop: str):
    return {"side": side, "chemical_system": system, "unit_edge_id": hop,
            "material_id": hop.split("_")[0], "rate_limiting_edge_ids": [hop]}


def test_discovery_keeps_every_attrition_category_distinct(tmp_path, monkeypatch):
    from e3sse.execution.sharded_execution import ShardedExecutionError
    from e3sse.production.runners import ml_neb_units, population, representation
    from e3sse.production.runners.ml_neb_units import deployed_units, ml_neb_task

    universe = su.build(tmp_path)
    run = tmp_path / "run"
    model = "Synth-A1"
    active = universe["active"]["amendments"]

    # -- synthetic prospective structures -> Gate 0/1 population (the production builder)
    systems = iter(f"Li-{a}-{b}" for i, a in enumerate(ANIONS) for b in ANIONS[i + 1:])
    train_hops = [h for _, _, split, h in universe["hops"] if split == "train"]
    dev = [_unit("development", next(systems), f"mp-d{i}_0_1_0_0_0") for i in range(7)]
    calibration = [_unit("P", next(systems), h) for h in train_hops]  # 12 calibration units
    pool_names = {"selected": "Li-O-S", "no_weight": "Li-O-Se", "abstained": "Cl-Li-O",
                  "ml_neb_failure": "Li-N-O", "weight_slice": "Br-Li-O", "gate0": "Ge-Li"}
    pool = {k: _unit("Q", s, f"mp-q{i}_0_1_0_0_0") for i, (k, s) in enumerate(pool_names.items())}
    units = [*dev, *calibration, *pool.values()]
    descriptors = {}
    for i, unit in enumerate(units):
        key = population.unit_key(unit)
        family = FAMILIES[i % 3]
        far = unit is pool["abstained"]
        descriptors[key] = {**unit, "vector": _descriptor(family, jitter=0.01 * (i % 7), far=far),
                            "chemical_system": unit["chemical_system"]}
    slice_systems = {calibration[0]["chemical_system"], pool["weight_slice"]["chemical_system"]}
    monkeypatch.setattr(population, "weight_fit_slice",
                        lambda parent, system, fraction: system in slice_systems)
    built, _ = population.build_population(units, descriptors, parent_protocol_id="p" * 64,
                                           fraction=0.2)
    roles = {u["unit_key"]: u["role"] for u in built["units"]}
    assert roles["Q:Ge-Li"] == "excluded_gate0"  # Ge is outside the reference elements
    assert roles["Q:Br-Li-O"] == "pool_weight_slice"
    assert sum(r.startswith("excluded") for r in roles.values()) == 1

    # -- ML-NEB: converged, scientific failure, infrastructure failure (real task code)
    ctx = su.context(universe, run, "discovery", model=model)
    band = su.li_vacancy_band()
    identity = {"code_sha256": ctx.code_sha256(ml_neb_units.CODE_FILES)}

    def ml_neb(kind, unit):
        key = population.unit_key(unit)
        return ctx.execute(namespace=f"discovery.{kind}", units=[(
            {"unit_key": key}, {"unit_key": key, "unit_id": unit["unit_edge_id"], "images": band,
                                "variant": "frozen", "embed": False})],
            task=ml_neb_task, identity=identity, setup=su.synthetic_ml_neb_setup,
            setup_args=(kind,))

    converged = next(iter(ml_neb("lj", pool["selected"]).values()))
    failed = next(iter(ml_neb("non_finite", pool["ml_neb_failure"]).values()))
    assert converged["status"] == "converged" and converged["failure_category"] is None
    assert failed["status"] != "converged" and failed["failure_category"] == "non_finite_value"
    with pytest.raises(ShardedExecutionError):  # host failure: never a scientific record
        ml_neb("allocator_failure", pool["no_weight"])
    shard_root = run / "discovery" / "models" / model / "shards"
    assert list((shard_root / "discovery.lj").rglob("shards/*/*.json"))  # positive control
    assert not list((shard_root / "discovery.allocator_failure").rglob("shards/*/*.json"))

    def like(record, unit, barrier):
        key = population.unit_key(unit)
        return {**record, "unit_key": key, "unit_id": unit["unit_edge_id"], "barrier_eV": barrier}

    pool_records = [converged, like(converged, pool["no_weight"], converged["barrier_eV"]),
                    like(converged, pool["abstained"], 0.9), failed]
    calibration_records = [like(converged, u, 0.8 * su.dft_barrier_of(u["unit_edge_id"]))
                           for u in calibration if roles[population.unit_key(u)] == "calibration"]
    slice_record = like(converged, pool["weight_slice"], 0.3)

    # -- upstream artifacts of the frozen pipeline (support reference, scales, G1, weights)
    dev_keys = [population.unit_key(u) for u in dev]
    reference, reason = representation.descriptor_reference(dev_keys, descriptors, 0.8)
    assert reason is None
    su.publish_upstream(run, "descriptor_availability", "descriptors", {"units": descriptors})
    su.publish_upstream(run, "path_representation", "path_representation", {
        "support_k": 5, "quantile_level": 0.8, "label_free": True,
        "models": {model: {"branch": "descriptor_space", "reference": reference}},
        "representation_amendment_sha256": su.canonical(active["representation"])})
    su.publish_upstream(run, "family_scale", "family_scales", {
        "scales": {model: {"family": {f: {"s_hat": 0.8} for f in FAMILIES}}}})
    su.publish_upstream(run, "transfer", "g1_decision", {"passes": True})
    su.publish_upstream(run, "model_selection", "model_selection",
                        {"selection": {"selected_model": model}})
    su.publish_upstream(run, "development_population", "population_manifest", built)
    su.publish_upstream(run, "bvse_reproduction", "bvse_path_index", {})
    su.publish_upstream(run, "weight_slice_ml_neb", "ml_neb_records", {"records": [slice_record]},
                        model=model)
    deployed_pool = deployed_units(ctx, model, pool_records)
    deployed_calibration = deployed_units(su.context(universe, run, "deployment_calibration",
                                                     model=model), model, calibration_records)
    su.publish_upstream(run, "discovery", "deployed_units", {"units": deployed_pool})
    su.publish_upstream(run, "deployment_calibration", "deployed_units",
                        {"units": deployed_calibration})
    weighted = {population.unit_key(pool[k]) for k in ("selected", "abstained", "ml_neb_failure")}
    su.publish_upstream(run, "density_ratio", "density_ratio", {
        "classifier_status": "converged",
        "calibration_weights": {u["unit_key"]: 1.0 for u in deployed_calibration},
        "test_weights": {k: 1.0 for k in weighted}})

    by_key = {u["unit_key"]: u for u in deployed_pool}
    abstained = by_key[population.unit_key(pool["abstained"])]
    assert abstained["supported"] is False  # applicability abstention: raw, labelled
    assert abstained["prediction"]["source"] != "srt_family"
    assert by_key[population.unit_key(pool["selected"])]["prediction"]["source"] == "srt_family"
    assert by_key[population.unit_key(pool["ml_neb_failure"])]["prediction"] is None

    # -- weighted conformal selection and the funnel (production stage entry point)
    assert su.run_phase(universe, run, "discovery_selection", "reduce") == 0
    result = json.loads((run / "discovery_selection" / "discovery_selection.json").read_text())
    counts = result["funnel"]["counts"]
    assert counts["gate1"] == 5  # Gate-0 exclusion is counted before Gate 1
    assert result["population_exclusions_before_gate1"]["excluded_gate0"] == 1
    assert counts["ml_neb_attempted"] == 5 and counts["ml_neb_converged"] == 4
    assert counts["failures_by_category"]["non_finite_value"] == 1  # scientific failure
    assert counts["weight_fit_holdout"] == 1
    assert counts["applicability_supported"] == 2  # selected + no_weight; abstained is not
    assert counts["no_density_weight"] == 1  # WCS-ineligible
    assert counts["wcs_run"] == 2 == result["n_test"]  # selected + abstained
    selected = [c for c in result["candidates"] if c["selected"]]
    assert [c["unit_key"] for c in selected] == [population.unit_key(pool["selected"])]
    assert counts["selected"] == 1
    assert "not a confirmed superionic conductor" in result["guarantee"]
    access = [a for a in su.access_log(run) if a["stage"] == "discovery_selection"]
    assert [a["outcome_kind"] for a in access] == [CALIBRATION_BARRIERS]
    assert access[0]["units"] == len(deployed_calibration)


# ======================================================================= RunnerAdapter

def _runner_layout(root: Path, monkeypatch):
    """A layout whose 'estimate' stage is the synthetic runner, driven by the real
    RunnerAdapter (real command line, child environment, manifests and records)."""

    import stat
    import sys

    from production_fixtures import adapters, make_layout, stage

    from e3sse.production import orchestrator
    from e3sse.production.runners import RUNNER_MODULES

    monkeypatch.setitem(RUNNER_MODULES, "estimate", "synthetic_runner_module")
    tests = Path(__file__).resolve().parent
    shim = root / "shim.py"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(tests)!r})\n"
        "from e3sse.production.runners import RUNNER_MODULES\n"
        "RUNNER_MODULES['estimate'] = 'synthetic_runner_module'\n"
        "sys.argv = sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n")
    interpreter = root / "project-python"
    interpreter.write_text(f'#!/bin/sh\nE3SSE_SYNTHETIC_INTERPRETER=shim exec "{sys.executable}" '
                           f'"{shim}" "$@"\n')
    interpreter.chmod(interpreter.stat().st_mode | stat.S_IXUSR)

    class ShimRunnerAdapter(orchestrator.RunnerAdapter):
        def _python(self, ctx):
            return interpreter

    stages = [stage("runtime_audit"), stage("resource_profile"),
              stage("preflight", ["runtime_audit", "resource_profile"]),
              stage("freeze", ["preflight"], adapter="fake_freeze"),
              stage("estimate", ["freeze"], adapter="shim_runner",
                    outcome_access=["mplitrj_published_train_dft_forces_energies"])]
    table = adapters(shim_runner=ShimRunnerAdapter())
    return make_layout(root, stages), table


def _execute(config, table, *, resume=False):
    from production_fixtures import REPOSITORY

    from e3sse.production.orchestrator import execute, load_context

    ctx = load_context(REPOSITORY, config, adapters=table, resume=resume)
    return ctx, {r.stage: r for r in execute(ctx, "all", table)}


@pytest.fixture
def freeze_binding(monkeypatch):
    from e3sse.production import orchestrator

    def binding(ctx):
        path = ctx.stage_dir("freeze") / orchestrator.FREEZE_BINDING_FILE
        if not path.is_file():
            raise orchestrator.StageBlocked(orchestrator.BLOCKED_MISSING_EVIDENCE, "no freeze")
        return json.loads(path.read_text())

    monkeypatch.setattr(orchestrator, "freeze_binding", binding)


def test_runner_adapter_invocation_scope_errors_identity_resume_and_corruption(
        tmp_path, monkeypatch, freeze_binding):
    from e3sse.production import orchestrator

    config, table = _runner_layout(tmp_path / "a", monkeypatch)
    ctx, results = _execute(config, table)
    assert results["estimate"].outcome == orchestrator.COMPLETED
    stage_dir = ctx.stage_dir("estimate")
    output = json.loads((stage_dir / "synthetic_result.json").read_text())
    assert output["interpreter"] == "shim"  # the adapter's interpreter ran the phase
    assert output["outcome_scope"] == ["mplitrj_published_train_dft_forces_energies"]
    assert [r["square"] for r in output["squares"]] == [0, 1, 4, 9, 16, 25]
    completion = json.loads((stage_dir / "stage_completion.json").read_text())
    assert {"synthetic_result", "manifest:reduce"} <= set(completion["outputs"])
    digest = completion["outputs"]["synthetic_result"]["sha256"]

    # Resume reuses the verified stage; a second layout reproduces the same output identity.
    _, resumed = _execute(config, table, resume=True)
    assert resumed["estimate"].outcome == orchestrator.REUSED
    other, again = _execute(_runner_layout(tmp_path / "b", monkeypatch)[0], table)
    assert again["estimate"].outcome == orchestrator.COMPLETED
    other_completion = json.loads((other.stage_dir("estimate") / "stage_completion.json").read_text())
    assert other_completion["outputs"]["synthetic_result"]["sha256"] == digest

    # A corrupted completion record is rejected on resume.
    record = stage_dir / "stage_completion.json"
    record.chmod(0o644)
    record.write_text("{not json")
    with pytest.raises(orchestrator.StaleStageError):
        _execute(config, table, resume=True)


@pytest.mark.parametrize(("mode", "outcome"), [
    ("missing", "blocked_missing_evidence"), ("invalid", "validation_failure"),
    ("forbidden_outcome", "validation_failure"), ("crash", "infrastructure_failure")])
def test_runner_adapter_propagates_child_failure_classes(tmp_path, monkeypatch, freeze_binding,
                                                         mode, outcome):
    from e3sse.production import orchestrator

    monkeypatch.setenv("E3SSE_SYNTHETIC_RUNNER_MODE", mode)
    config, table = _runner_layout(tmp_path, monkeypatch)
    ctx, results = _execute(config, table)
    expected = {"blocked_missing_evidence": orchestrator.BLOCKED_MISSING_EVIDENCE,
                "validation_failure": orchestrator.VALIDATION_FAILURE,
                "infrastructure_failure": orchestrator.INFRASTRUCTURE_FAILURE}[outcome]
    assert results["estimate"].outcome == expected
    assert not (ctx.stage_dir("estimate") / "stage_completion.json").exists()
    assert not (ctx.stage_dir("estimate") / "synthetic_result.json").exists()
