"""End-to-end Study-1 pipeline on a synthetic frozen universe (spawned workers)."""

from __future__ import annotations

import argparse
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import study1_fixtures as fx

from e3sse.execution.sharded_execution import StaleCheckpointError
from e3sse.softening import study1_inference, study1_inputs

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import analyze_study1_softening_atlas as analyzer
    import plot_study1_figures as plotter
    import run_study1_softening_atlas as runner

    def git(*args):
        return "c" * 40 if args[0] == "rev-parse" else ""

    monkeypatch.setattr(runner, "_git", git)
    monkeypatch.setattr(analyzer, "_git", git)
    monkeypatch.setattr(runner, "CONFIGURE_TORCH_FOR_MODELS", False)
    monkeypatch.setattr(study1_inference, "inference_worker_setup", fx.synthetic_worker_setup)
    monkeypatch.setattr(study1_inference, "verify_runtime",
                        lambda *args: {"recorded_dependencies": {"test": "1"}})
    monkeypatch.setattr(runner, "_lock_file", lambda *args: None)
    monkeypatch.setattr(study1_inputs, "refuse_tmpfs", lambda path: None)
    return runner, analyzer, plotter


def _run(main, argv):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert main(argv) == 0
    return buffer.getvalue()


def _infer(runner, config, run, *, workers=2, extra=()):
    return _run(runner.main, ["--config", str(config), "_runtime-infer", "--run", run,
                              "--workers", str(workers), *extra])


def _estimates(analysis: Path) -> list[dict]:
    with (analysis / "derived" / "study1_softening_estimates.csv").open() as handle:
        return list(csv.DictReader(handle))


def test_sealed_protocol_stops_before_any_label_is_read(tmp_path, scripts):
    runner, _, _ = scripts
    universe = fx.build_universe(tmp_path, register=False)
    plan = json.loads(_run(runner.main, ["--config", str(universe["config"]), "plan"]))
    assert plan["outcome_seal"]["execution_permitted"] is False
    assert "registered on OSF" in plan["outcome_seal"]["reason"]
    assert plan["published_train_frames"] == len(universe["frames"])
    _run(runner.main, ["--config", str(universe["config"]), "index-frames"])
    # The memory probe's structures are chosen and read from geometry alone.
    ctx = runner.Context(argparse.Namespace(config=str(universe["config"]), protocol_dir=None,
                                            output_root=None, checkpoint_map=None))
    probe = json.loads(runner._probe_structures(ctx).read_text())
    assert probe["structures"] and all(
        set(item) == {"frame_id", "numbers", "positions", "cell", "pbc"}
        for item in probe["structures"]
    )
    assert max(len(item["numbers"]) for item in probe["structures"]) == 8
    with pytest.raises(SystemExit, match="stopped before reading any label"):
        _infer(runner, universe["config"], study1_inference.REFERENCE_RUN)
    assert not (universe["output"] / "inference").exists()


def test_full_pipeline_recovers_known_scales_and_excludes_own_path(tmp_path, scripts):
    runner, analyzer, plotter = scripts
    universe = fx.build_universe(tmp_path)
    config = universe["config"]
    plan = json.loads(_run(runner.main, ["--config", str(config), "plan"]))
    assert plan["outcome_seal"]["execution_permitted"] is True
    assert plan["target_hops_by_population"] == {"primary": 9, "tm_hubbard_u_secondary": 2}
    _run(runner.main, ["--config", str(config), "index-frames"])
    for run in [study1_inference.REFERENCE_RUN, *fx.MODEL_SCALES]:
        _infer(runner, config, run)
    (pilot_path,) = (universe["output"] / "inference" / "CHGNet-0.3.0" / "pilot_gate").iterdir()
    pilot = json.loads(pilot_path.read_text())
    assert pilot["passed"] and all(pilot["checks"].values()) and pilot["informative_pairs"] >= 3

    _run(analyzer.main, ["--config", str(config), "--workers", "2"])
    (analysis,) = (universe["output"] / "analysis").iterdir()
    rows = _estimates(analysis)

    def one(**match):
        found = [r for r in rows if all(r[k] == v for k, v in match.items())]
        assert len(found) == 1, match
        return found[0]

    # Leakage stress test: the contaminated hop's own path has slope 0.5, every other
    # frame 0.8.  The primary estimate uses only admissible frames; the oracle only
    # the own path, labelled as an oracle upper bound.
    primary = one(model="CHGNet-0.3.0", target_hop_id=fx.CONTAMINATED_HOP,
                  estimator_kind="primary", component_set="all")
    oracle = one(model="CHGNet-0.3.0", target_hop_id=fx.CONTAMINATED_HOP,
                 estimator_kind="oracle", component_set="all")
    assert primary["status"] == "estimated"
    assert float(primary["s_hat"]) == pytest.approx(0.8, abs=1e-12)
    assert float(oracle["s_hat"]) == pytest.approx(0.5, abs=1e-12)
    assert oracle["analysis_label"] == "oracle_upper_bound"
    assert int(primary["eligible_frames"]) == 2 + 2 * 32  # own endpoints + two other hops
    # Uniform models: every admissible scale equals the model scale, element-resolved too
    # (the contaminated hop's admissible frames are all uncontaminated).
    for model, scale in fx.MODEL_SCALES.items():
        for component in ("all", "Li", "host"):
            row = one(model=model, target_hop_id=fx.CONTAMINATED_HOP, estimator_kind="primary",
                      component_set=component)
            assert float(row["s_hat"]) == pytest.approx(scale, abs=1e-12)
            assert float(row["ci_low"]) == pytest.approx(scale, abs=1e-12)
    # Support follows the frozen contract: all-atom informative components of the
    # frames used.  mp-2 has 2 S atoms per frame, so host's own count is 60 < 100, but
    # the all-atom count over the same 10 frames (180) supports it, as it does Li.
    host = one(model="SevenNet-omat", target_hop_id="mp-2_0_1", estimator_kind="primary",
               component_set="host")
    li = one(model="SevenNet-omat", target_hop_id="mp-2_0_1", estimator_kind="primary",
             component_set="Li")
    every = one(model="SevenNet-omat", target_hop_id="mp-2_0_1", estimator_kind="primary",
                component_set="all")
    assert int(host["n_informative_components"]) == 60
    assert host["n_support_informative_components"] == every["n_informative_components"]
    assert li["n_support_informative_components"] == every["n_informative_components"]
    assert host["status"] == "estimated" and host["unsupported_reasons"] == ""
    assert li["status"] == "estimated"
    # Small materials: 2 + 8 = 10 frames qualify; scopes never fall back to another scope.
    hop = one(model="SevenNet-omat", target_hop_id="mp-3_0_2", estimator_kind="primary",
              component_set="all")
    assert hop["status"] == "estimated" and int(hop["n_frames"]) == 10
    family = [r for r in rows if r["scope"] == "family" and r["scope_target"] == "halide"]
    assert {r["status"] for r in family} == {"unsupported"}  # one material < 3
    assert all(r["s_hat"] == "" and "fewer_than_3_materials" in r["unsupported_reasons"]
               for r in family)
    # The TM/Hubbard-U secondary population never enters the primary pooled scopes.
    pooled = one(model="CHGNet-0.3.0", kind="pooled", scope="global", population="primary",
                 component_set="all", decile="")
    assert int(pooled["n_materials"]) == 4
    secondary = [r for r in rows if r["population"] == "tm_hubbard_u_secondary"
                 and r["kind"] == "target_hop"]
    assert secondary and {r["material_id"] for r in secondary} == {"mp-5"}

    deciles = json.loads((analysis / "frozen" / "study1_energy_deciles.json").read_text())
    assert len(deciles["populations"]["primary"]["cut_points_eV_per_atom"]) == 9
    hop_set = json.loads(
        (analysis / "frozen" / "study1_label_budget_hop_set_primary.json").read_text())
    assert hop_set["hop_ids"] == ["mp-1_0_1", "mp-1_0_2", "mp-1_0_3"]
    stability = list(csv.DictReader(
        (analysis / "derived" / "study1_label_budget_stability.csv").open()))
    sevennet = [r for r in stability if r["model"] == "SevenNet-omat"]
    assert [int(r["K"]) for r in sevennet] == [1, 3, 5, 10, 30]
    coverage = json.loads((analysis / "derived" / "study1_pretraining_coverage.json").read_text())
    assert coverage["group_medians_strictly_ordered_A_lt_B1_lt_B2"] is True

    _run(plotter.main, ["--config", str(config)])
    for stem in ("study1_softening_atlas", "study1_label_budget_stability"):
        for suffix in (".pdf", ".png", ".figure.json"):
            assert (analysis / "figures" / f"{stem}{suffix}").stat().st_size > 0


def test_worker_count_resume_and_corruption(tmp_path, scripts):
    runner, _, _ = scripts
    universe = fx.build_universe(tmp_path)
    config = universe["config"]
    _run(runner.main, ["--config", str(config), "index-frames"])
    _infer(runner, config, study1_inference.REFERENCE_RUN, workers=1)
    run_dir = universe["output"] / "inference" / study1_inference.REFERENCE_RUN
    serial = json.loads((run_dir / "frame_statistics_manifest.json").read_text())

    other = tmp_path / "outputs" / "parallel"
    _run(runner.main, ["--config", str(config), "--output-root", str(other), "index-frames"])
    _run(runner.main, ["--config", str(config), "--output-root", str(other),
                       "_runtime-infer", "--run", study1_inference.REFERENCE_RUN, "--workers", "3"])
    parallel_dir = other / universe["output"].name / "inference" / study1_inference.REFERENCE_RUN
    parallel = json.loads((parallel_dir / "frame_statistics_manifest.json").read_text())
    assert parallel["statistics_sha256"] == serial["statistics_sha256"]
    assert parallel["merged_shard_sha256"] == serial["merged_shard_sha256"]

    # Interrupted after all shards but before the merge: every shard is reused.
    for name in ("frame_statistics_manifest.json", "frame_statistics.npz"):
        (run_dir / name).unlink()
    _infer(runner, config, study1_inference.REFERENCE_RUN, extra=("--resume",))
    record = max((run_dir / "study1_runs").glob("*.json"))
    assert json.loads(record.read_text())["shards_computed"] == 0
    assert json.loads((run_dir / "frame_statistics_manifest.json").read_text())[
        "statistics_sha256"] == serial["statistics_sha256"]

    # A corrupted checkpoint is refused, never silently recomputed.
    for name in ("frame_statistics_manifest.json", "frame_statistics.npz"):
        (run_dir / name).unlink()
    shard = next((run_dir / "shards").rglob("*.json"))
    payload = json.loads(shard.read_text())
    payload["result"]["frames"][0]["dft_energy_eV"] += 1.0
    shard.write_text(json.dumps(payload))
    with pytest.raises(StaleCheckpointError):
        _infer(runner, config, study1_inference.REFERENCE_RUN, extra=("--resume",))


def test_changed_identity_is_refused_and_failed_frames_stop_analysis(tmp_path, scripts):
    runner, analyzer, _ = scripts
    universe = fx.build_universe(tmp_path)
    config = universe["config"]
    _run(runner.main, ["--config", str(config), "index-frames"])
    for run in [study1_inference.REFERENCE_RUN, *fx.MODEL_SCALES]:
        _infer(runner, config, run)
    run_dir = universe["output"] / "inference" / "SevenNet-omat"
    manifest_path = run_dir / "frame_statistics_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tampered = dict(manifest, scientific_identity={**manifest["scientific_identity"],
                                                   "code_sha256": {"x": "0" * 64}})
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(SystemExit, match="different scientific identity"):
        _infer(runner, config, "SevenNet-omat")
    with pytest.raises(SystemExit, match="differs from this checkout"):
        _run(analyzer.main, ["--config", str(config), "--workers", "1"])
    manifest["frame_failures"] = [{"frame_id": "MPLiTrj_train.xyz:0", "error_type": "X",
                                   "error_message": "y"}]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="needs a recorded decision"):
        _run(analyzer.main, ["--config", str(config), "--workers", "1"])


# --------------------------------------------------------------------------- provenance


def _writable(path: Path) -> Path:
    path.chmod(0o644)
    return path


def _old_common_hops(rows: list[dict], models) -> dict[str, set[str]]:
    """The common-hop set as derived after estimation before it was persisted."""

    common = {}
    for population in ("primary", "tm_hubbard_u_secondary"):
        common[population] = set.intersection(*[
            {r["target_hop_id"] for r in rows
             if r["model"] == m and r["population"] == population and r["kind"] == "target_hop"
             and r["estimator_kind"] == "primary" and r["component_set"] == "all"
             and r["status"] == "estimated"}
            for m in models
        ])
    return common


def test_common_hop_population_is_persisted_hashed_and_unchanged(tmp_path, scripts):
    import hashlib

    from e3sse.execution.sharded_execution import canonical_json_bytes, canonical_sha256
    from e3sse.softening import study1_analysis as sa
    from e3sse.softening import study1_tables as st

    runner, analyzer, _ = scripts
    universe = fx.build_universe(tmp_path)
    config = universe["config"]
    _run(runner.main, ["--config", str(config), "index-frames"])
    for run in [study1_inference.REFERENCE_RUN, *fx.MODEL_SCALES]:
        _infer(runner, config, run)
    _run(analyzer.main, ["--config", str(config), "--workers", "2"])
    (analysis,) = (universe["output"] / "analysis").iterdir()
    path = analysis / "frozen" / "study1_common_hop_population.json"
    artifact = json.loads(path.read_text())
    hops = sa.validate_common_hop_population(artifact)
    assert artifact["protocol_id"] == universe["protocol"]["protocol_id"]
    for record in artifact["populations"].values():
        assert record["hop_ids"] == sorted(record["hop_ids"])
        assert record["n_hops"] == len(record["hop_ids"])
        assert record["hop_ids_sha256"] == canonical_sha256(record["hop_ids"])
    manifest = json.loads((analysis / "study1_analysis_manifest.json").read_text())
    assert manifest["common_hop_population"] == {
        "file": "frozen/study1_common_hop_population.json",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_json_sha256": canonical_sha256(artifact),
    }
    assert manifest["outputs_sha256"]["frozen/study1_common_hop_population.json"] == (
        manifest["common_hop_population"]["sha256"])

    # Persistence changes no population: equal to the post-estimation intersection.
    rows = _estimates(analysis)
    old = _old_common_hops(rows, fx.MODEL_SCALES)
    assert hops["primary"] and set(hops["primary"]) == old["primary"]
    assert set(hops["tm_hubbard_u_secondary"]) == old["tm_hubbard_u_secondary"]
    f_free = {h for h in old["primary"]
              if sa.f_block_free(fx.MATERIALS[h.split("_")[0]][0])}
    assert set(hops[sa.F_BLOCK_SENSITIVITY]) == old["primary"] & f_free
    summary = list(csv.DictReader((analysis / "derived" / "study1_model_summary.csv").open()))
    assert {int(r["common_hops"]) for r in summary if r["population"] == "primary"} == {
        len(hops["primary"])}

    # Deterministic: payload order does not matter; canonical bytes are identical.
    payloads = [{"unit": "estimate", "kind": "target_hop", "estimator_kind": "primary",
                 "decile": None, "population": "primary", "scope": "material",
                 "target_hop_id": h, "rows": [i],
                 "support": {"full_estimate_min_frames": 1,
                             "full_estimate_min_informative_components": 1,
                             "pooled_min_materials": 3}}
                for i, h in enumerate(["h-c", "h-a", "h-b"])]
    import numpy as np
    counts = np.ones((3, 3, 2), dtype=np.int64)
    codes = np.zeros(3, dtype=np.int64)
    forward = sa.common_hop_population(payloads, counts, codes, {"h-a", "h-b", "h-c"})
    backward = sa.common_hop_population(payloads[::-1], counts, codes, {"h-a", "h-b", "h-c"})
    assert forward["primary"] == ["h-a", "h-b", "h-c"]
    assert canonical_json_bytes(forward) == canonical_json_bytes(backward)

    # A disagreement between the persisted and the estimated population is explicit.
    with pytest.raises(ValueError, match="persisted common-hop population differs"):
        st.model_summary(
            [{**r, "s_hat": float(r["s_hat"]) if r["s_hat"] else None,
              "decile": r["decile"] or None} for r in rows],
            list(fx.MODEL_SCALES), {m: "A" for m in fx.MODEL_SCALES}, f_free,
            {**hops, "primary": hops["primary"][1:]},
        )
    with pytest.raises(sa.Study1AnalysisError, match="corrupt"):
        sa.validate_common_hop_population({**artifact, "populations": {
            **artifact["populations"],
            "primary": {**artifact["populations"]["primary"], "n_hops": 0}}})

    # Resume reuses a valid artifact unchanged; tampered or corrupt ones are refused.
    before = path.read_bytes()
    _run(analyzer.main, ["--config", str(config), "--workers", "1", "--resume"])
    assert path.read_bytes() == before
    tampered = json.loads(before)
    tampered["populations"]["primary"]["hop_ids"] = hops["primary"][1:]
    _writable(path).write_text(json.dumps(tampered))
    with pytest.raises(SystemExit, match="differs from its reconstruction"):
        _run(analyzer.main, ["--config", str(config), "--workers", "1", "--resume"])
    path.write_text("{not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        _run(analyzer.main, ["--config", str(config), "--workers", "1", "--resume"])


def test_pilot_gate_sha_is_bound_into_provenance_and_fails_closed(tmp_path, scripts):
    import hashlib

    from e3sse.execution.sharded_execution import canonical_sha256

    runner, analyzer, _ = scripts
    universe = fx.build_universe(tmp_path)
    config = universe["config"]
    _run(runner.main, ["--config", str(config), "index-frames"])
    for run in [study1_inference.REFERENCE_RUN, *fx.MODEL_SCALES]:
        _infer(runner, config, run)
    run_dir = universe["output"] / "inference" / "SevenNet-omat"
    manifest = json.loads((run_dir / "frame_statistics_manifest.json").read_text())
    gate = manifest["pilot_gate"]
    (report,) = (run_dir / "pilot_gate").iterdir()
    assert gate["report"] == f"pilot_gate/{report.name}"  # run-relative, no host path
    assert gate["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert gate["status"] == "passed"
    assert gate["identity_sha256"] == canonical_sha256(manifest["scientific_identity"])
    assert json.loads((run_dir / "pilot_gate_binding.json").read_text()) == gate
    record = json.loads(max((run_dir / "study1_runs").glob("*.json")).read_text())
    assert record["pilot_gate"] == gate
    assert record["protocol_id"] == universe["protocol"]["protocol_id"]
    assert record["implementation_git_sha"] == manifest["implementation_git_sha"] == "c" * 40
    # Provenance only: the gate never enters the scientific or protocol identity.
    identity = manifest["scientific_identity"]
    assert "pilot" not in json.dumps(identity)
    assert identity["protocol_id"] == universe["protocol"]["protocol_id"]
    reference = json.loads((universe["output"] / "inference" / study1_inference.REFERENCE_RUN
                            / "frame_statistics_manifest.json").read_text())
    assert reference["pilot_gate"] is None

    def drop_merged():
        for name in ("frame_statistics_manifest.json", "frame_statistics.npz"):
            (run_dir / name).unlink(missing_ok=True)

    # Resume validates the same gate and records the same binding.
    drop_merged()
    _infer(runner, config, "SevenNet-omat", extra=("--resume",))
    assert json.loads((run_dir / "frame_statistics_manifest.json").read_text())[
        "pilot_gate"] == gate

    # A gate report changed after binding stops inference, a completed run and analysis.
    original = report.read_bytes()
    content = json.loads(original)
    content["fd_step_angstrom"] = content["fd_step_angstrom"] * 2  # still "passed"
    _writable(report).write_text(json.dumps(content))
    with pytest.raises(SystemExit, match="changed after it was bound"):
        _infer(runner, config, "SevenNet-omat")
    with pytest.raises(SystemExit, match="changed after it was bound"):
        _run(analyzer.main, ["--config", str(config), "--workers", "1"])
    drop_merged()
    with pytest.raises(SystemExit, match="changed after it was bound"):
        _infer(runner, config, "SevenNet-omat", extra=("--resume",))
    assert not (run_dir / "frame_statistics_manifest.json").exists()

    # A missing bound report still fails closed.
    report.unlink()
    with pytest.raises(SystemExit, match="missing"):
        _infer(runner, config, "SevenNet-omat", extra=("--resume",))

    # A non-authorizing report is refused even without a prior binding.
    content = json.loads(original)
    content["passed"] = False
    report.write_text(json.dumps(content))
    _writable(run_dir / "pilot_gate_binding.json").unlink()
    with pytest.raises(SystemExit, match="pilot gate failed"):
        _infer(runner, config, "SevenNet-omat", extra=("--resume",))
    assert not (run_dir / "pilot_gate_binding.json").exists()
    with pytest.raises(study1_inference.PilotGateBindingError, match="no valid pilot-gate"):
        study1_inference.verify_pilot_gate_binding(run_dir, None, identity)
    # The binding is tied to the run's scientific identity and to a run-local report name.
    content["passed"] = True
    report.write_text(json.dumps(content))
    with pytest.raises(study1_inference.PilotGateBindingError, match="another run identity"):
        study1_inference.verify_pilot_gate_binding(
            run_dir, {**gate, "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()},
            {**identity, "device": "cuda"})
    with pytest.raises(study1_inference.PilotGateBindingError, match="invalid pilot-gate report"):
        study1_inference.pilot_gate_binding(run_dir, "../../escape.json",
                                            gate["gate_key_sha256"], identity)
