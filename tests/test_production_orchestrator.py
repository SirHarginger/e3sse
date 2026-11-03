"""Synthetic end-to-end tests of the production orchestrator (no model, no outcome)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from production_fixtures import REPOSITORY, adapters, make_layout, stage

from e3sse.execution.sharded_execution import ArtifactExistsError
from e3sse.production import orchestrator
from e3sse.production.orchestrator import (
    BLOCKED_BY_DEPENDENCY,
    BLOCKED_MISSING_EVIDENCE,
    BLOCKED_MISSING_RUNNER,
    COMPLETED,
    INFRASTRUCTURE_FAILURE,
    REUSED,
    StaleStageError,
    dry_run_plan,
    execute,
    exit_code,
    load_context,
)


@pytest.fixture
def fake_freeze(monkeypatch):
    """Outcome stages need a verified freeze; here: the binding file must exist."""

    def binding(ctx):
        path = ctx.stage_dir("freeze") / orchestrator.FREEZE_BINDING_FILE
        if not path.is_file():
            raise orchestrator.StageBlocked(BLOCKED_MISSING_EVIDENCE, "no materialized freeze")
        return json.loads(path.read_text())

    monkeypatch.setattr(orchestrator, "freeze_binding", binding)


def _context(config: Path, *, resume=False, **kwargs):
    return load_context(REPOSITORY, config, adapters=adapters(), resume=resume, **kwargs)


def _outcomes(results):
    return {result.stage: result.outcome for result in results}


def test_stage_all_runs_in_dependency_order_and_blocks_missing_runner(tmp_path, fake_freeze):
    config = make_layout(tmp_path)
    ctx = _context(config)
    results = execute(ctx, "all", adapters())
    assert [r.stage for r in results] == ["runtime_audit", "resource_profile", "preflight",
                                          "freeze", "estimate", "transfer"]
    outcomes = _outcomes(results)
    assert outcomes["estimate"] == COMPLETED
    assert outcomes["transfer"] == BLOCKED_MISSING_RUNNER
    assert "implement the transfer runner" in results[-1].detail
    assert exit_code(results) == 2
    record = json.loads((ctx.stage_dir("estimate") / "stage_completion.json").read_text())
    assert record["outcome_access"] == ["mplitrj_published_train_dft_forces_energies"]
    assert record["dependency_completion_sha256"].keys() == {"freeze"}
    assert len(list((ctx.run_dir / "invocations").glob("*.json"))) == 1


def test_resume_reuses_verified_stages_and_create_new_refuses_without_resume(tmp_path, fake_freeze):
    config = make_layout(tmp_path)
    execute(_context(config), "all", adapters())
    with pytest.raises(ArtifactExistsError):
        execute(_context(config), "all", adapters())
    results = execute(_context(config, resume=True), "all", adapters())
    outcomes = _outcomes(results)
    assert all(outcomes[name] == REUSED for name in
               ("runtime_audit", "resource_profile", "preflight", "freeze", "estimate"))
    assert outcomes["transfer"] == BLOCKED_MISSING_RUNNER


def test_single_stage_runs_only_missing_dependencies(tmp_path, fake_freeze):
    config = make_layout(tmp_path)
    results = execute(_context(config), "preflight", adapters())
    assert [r.stage for r in results] == ["runtime_audit", "resource_profile", "preflight"]
    results = execute(_context(config, resume=True), "estimate", adapters())
    assert _outcomes(results) == {"runtime_audit": REUSED, "resource_profile": REUSED,
                                  "preflight": REUSED, "freeze": COMPLETED, "estimate": COMPLETED}


def test_invalid_stage_is_refused(tmp_path):
    config = make_layout(tmp_path)
    with pytest.raises(ValueError, match="unknown stage"):
        execute(_context(config), "not_a_stage", adapters())


def test_corrupted_output_is_rejected_on_resume(tmp_path, fake_freeze):
    config = make_layout(tmp_path)
    ctx = _context(config)
    execute(ctx, "preflight", adapters())
    (ctx.stage_dir("runtime_audit") / "result.json").write_text('{"stage": "tampered"}')
    with pytest.raises(StaleStageError, match="changed"):
        execute(_context(config, resume=True), "preflight", adapters())


def test_corrupted_completion_record_is_rejected(tmp_path, fake_freeze):
    config = make_layout(tmp_path)
    ctx = _context(config)
    execute(ctx, "runtime_audit", adapters())
    (ctx.stage_dir("runtime_audit") / "stage_completion.json").write_text("{not json")
    with pytest.raises(StaleStageError, match="unreadable"):
        execute(_context(config, resume=True), "runtime_audit", adapters())


def test_stale_dependency_record_invalidates_dependents(tmp_path, fake_freeze):
    config = make_layout(tmp_path)
    ctx = _context(config)
    execute(ctx, "preflight", adapters())
    path = ctx.stage_dir("runtime_audit") / "stage_completion.json"
    record = json.loads(path.read_text())
    record["completed_at_utc"] = "rewritten"
    os.chmod(path, 0o644)
    path.write_text(json.dumps(record))
    with pytest.raises(StaleStageError, match="dependency runtime_audit changed"):
        execute(_context(config, resume=True), "preflight", adapters())


def test_run_directory_of_other_identity_is_refused(tmp_path, fake_freeze):
    config = make_layout(tmp_path)
    ctx = _context(config)
    execute(ctx, "runtime_audit", adapters())
    identity = ctx.run_dir / "run_identity.json"
    stored = json.loads(identity.read_text())
    stored["scientific_identity"]["git_commit"] = "0" * 40
    os.chmod(identity, 0o644)
    identity.write_text(json.dumps(stored))
    with pytest.raises(StaleStageError, match="different scientific identity"):
        execute(_context(config, resume=True), "runtime_audit", adapters())


def test_child_failure_is_infrastructure_and_blocks_only_dependents(tmp_path, fake_freeze):
    stages = [stage("runtime_audit", adapter="failing_child"), stage("resource_profile"),
              stage("preflight", ["runtime_audit", "resource_profile"]),
              stage("freeze", ["preflight"], adapter="fake_freeze")]
    config = make_layout(tmp_path, stages)
    ctx = _context(config)
    results = execute(ctx, "all", adapters())
    outcomes = _outcomes(results)
    assert outcomes == {"runtime_audit": INFRASTRUCTURE_FAILURE, "resource_profile": COMPLETED,
                        "preflight": BLOCKED_BY_DEPENDENCY, "freeze": BLOCKED_BY_DEPENDENCY}
    assert exit_code(results) == 4
    assert "exit 7" in results[0].detail
    assert not (ctx.stage_dir("runtime_audit") / "stage_completion.json").exists()
    logs = list((ctx.run_dir / "logs" / "runtime_audit").glob("*.log"))
    assert logs and "# exit 7" in logs[0].read_text()


def test_outcome_stage_never_runs_without_a_verified_freeze(tmp_path, fake_freeze):
    stages = [stage("runtime_audit"), stage("preflight", ["runtime_audit"]),
              stage("freeze", ["preflight"], adapter="missing_runner", status="missing_runner",
                    next_task="no freeze yet"),
              stage("estimate", ["freeze"],
                    outcome_access=["mplitrj_published_train_dft_forces_energies"])]
    config = make_layout(tmp_path, stages)
    ctx = _context(config)
    # Even when called directly, the stage gate refuses: no freeze binding exists.
    with pytest.raises(orchestrator.StageBlocked, match="no materialized freeze"):
        orchestrator._require_permitted(ctx, ctx.graph.stages["estimate"])
    assert _outcomes(execute(ctx, "all", adapters()))["estimate"] == BLOCKED_BY_DEPENDENCY


def test_dry_run_writes_nothing_and_reports_plan(tmp_path):
    config = make_layout(tmp_path)
    ctx = _context(config, dry_run=True)
    plan = dry_run_plan(ctx, "all", adapters())
    assert not ctx.run_dir.exists()
    assert plan["stage_order"][-1] == "transfer"
    assert plan["run_id"] == ctx.run_id
    assert set(plan["model_runtimes"]) == set(ctx.models)
    assert plan["preflight"]["freeze_permitted"] is False
    transfer = next(s for s in plan["stages"] if s["name"] == "transfer")
    assert transfer["commands"][0].startswith("BLOCKED (missing runner)")
    assert "statistics/statistics_numpy" not in plan["predicted_resources"]["missing_profiles"]


def test_run_identity_is_stable_and_ignores_operational_settings(tmp_path):
    config = make_layout(tmp_path)
    first, second = _context(config), _context(config)
    assert first.run_id == second.run_id
    data = json.loads(config.read_text())
    data["resource_policy"]["max_threads_per_worker"] = 3
    data["resource_profiling"]["process_grid"] = [1]
    config.write_text(json.dumps(data))
    assert _context(config).run_id == first.run_id
    data["stages"][0]["description"] = "reworded prose"
    data["stages"][-1]["next_task"] = "reworded next task"
    data["runtime"]["tmpdir"] = "elsewhere"
    config.write_text(json.dumps(data))
    assert _context(config).run_id == first.run_id  # prose and paths never orphan a run
    data["stages"][2]["depends_on"].append("resource_profile")
    data["stages"][1]["depends_on"] = []
    data["stages"][4]["outcome_access"] = []  # a real change to what a stage reads
    config.write_text(json.dumps(data))
    assert _context(config).run_id != first.run_id


def test_output_root_inside_data_root_is_refused(tmp_path):
    config = make_layout(tmp_path)
    server = config.parent / "server.json"
    data = json.loads(server.read_text())
    data["paths"]["outputs_root"] = "data/outputs"
    server.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cannot be written below"):
        _context(config)


def _bound_freeze(ctx, tmp_path, monkeypatch):
    """A real binding of a synthetic frozen artifact; the protocol-level validator is
    tested in test_internal_freeze_protocol and replaced here."""

    from e3sse.protocol import internal_freeze

    monkeypatch.setattr(internal_freeze, "validate_internal_freeze_protocol",
                        lambda internal, parent: None)
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    (parent_dir / "protocol.json").write_text("{}")
    ctx.active_protocol["parent_protocol"]["directory"] = str(parent_dir)
    artifact_dir = tmp_path / "freeze_artifact"
    artifact_dir.mkdir()
    artifact = artifact_dir / "internal_freeze_protocol.json"
    artifact.write_text(json.dumps({"protocol_id": "p" * 64, **{
        key: ctx.active_protocol[key] for key in ("amendments", "open_items", "governance")}}))
    from e3sse.production.run_identity import file_sha256

    return {"protocol_id": "p" * 64, "file_sha256": file_sha256(artifact),
            "directory": str(artifact_dir)}


def test_freeze_binding_detects_tampering_and_protocol_drift(tmp_path, monkeypatch):
    ctx = _context(make_layout(tmp_path))
    binding = _bound_freeze(ctx, tmp_path, monkeypatch)
    assert orchestrator.validate_freeze_binding(ctx, binding) == binding
    with pytest.raises(orchestrator.StageBlocked, match="differs from its binding"):
        orchestrator.validate_freeze_binding(ctx, {**binding, "file_sha256": "0" * 64})
    with pytest.raises(orchestrator.StageBlocked, match="protocol_id differs"):
        orchestrator.validate_freeze_binding(ctx, {**binding, "protocol_id": "q" * 64})
    ctx.active_protocol["amendments"]["deployment_gate"]["quantile_level"] = 0.9
    with pytest.raises(orchestrator.StageBlocked, match="differs from the frozen") as raised:
        orchestrator.validate_freeze_binding(ctx, binding)
    assert raised.value.outcome == orchestrator.VALIDATION_FAILURE


def test_threads_are_pinned_for_the_run_and_only_workers_follow_the_host(tmp_path, monkeypatch):
    from e3sse.execution.resource_planning import GIB, HostResources

    ctx = _context(make_layout(tmp_path))
    directory = ctx.evidence("resource_profile_dir") / "m"
    directory.mkdir(parents=True)
    (directory / "ml_neb_band-x.json").write_text(json.dumps({
        "schema_version": "e3sse.production.resource-profile",
        "identity": {"model": "m", "workload": "ml_neb_band", "memory_threads_per_worker": 8},
        "measurement": {"peak_resident_bytes": GIB},
        "benchmark": {"points": [{"processes": 2, "threads_per_process": 8,
                                  "status": "completed", "units_per_second": 9.0}]}}))

    def fake_host(cpus, gib):
        return HostResources(logical_cpus=cpus, usable_cpus=tuple(range(cpus)),
                             mem_total_bytes=gib * GIB, mem_available_bytes=gib * GIB,
                             swap_free_bytes=0)

    ctx.run_dir.mkdir(parents=True)
    monkeypatch.setattr(type(ctx), "host", lambda self: fake_host(80, 18))
    first = orchestrator.stage_allocation(ctx, "estimate", "m", "ml_neb_band")
    monkeypatch.setattr(type(ctx), "host", lambda self: fake_host(16, 18))
    later = orchestrator.stage_allocation(ctx, "transfer", "m", "ml_neb_band")
    assert first.threads_per_worker == later.threads_per_worker == 8
    assert later.workers * later.threads_per_worker <= 16
    assert (ctx.run_dir / "numerical_threads" / "m--ml_neb_band.json").is_file()


def test_malformed_completion_record_is_refused(tmp_path, fake_freeze):
    config = make_layout(tmp_path)
    ctx = _context(config)
    execute(ctx, "runtime_audit", adapters())
    path = ctx.stage_dir("runtime_audit") / "stage_completion.json"
    os.chmod(path, 0o644)
    path.write_text("{}")
    with pytest.raises(StaleStageError, match="malformed"):
        execute(_context(config, resume=True), "runtime_audit", adapters())


def test_children_inherit_only_their_stage_outcome_scope(tmp_path):
    import sys

    ctx = _context(make_layout(tmp_path))
    ctx.run_dir.mkdir(parents=True)
    code = "import os; print(os.environ['E3SSE_STAGE_OUTCOME_ACCESS'])"
    _, estimate = ctx.run_command("estimate", [sys.executable, "-c", code],
                                  ctx.child_environment(None, 1))
    _, audit = ctx.run_command("runtime_audit", [sys.executable, "-c", code],
                               ctx.child_environment(None, 1))
    assert json.loads(estimate.strip()) == ["mplitrj_published_train_dft_forces_energies"]
    assert json.loads(audit.strip()) == []


def test_unexpected_crash_is_infrastructure_and_the_invocation_is_recorded(tmp_path, fake_freeze):
    class Crashing(orchestrator.StageAdapter):
        def run(self, ctx, stage):
            raise KeyError("unexpected")

    config = make_layout(tmp_path, [stage("runtime_audit", adapter="crash"),
                                    stage("preflight", ["runtime_audit"]),
                                    stage("freeze", ["preflight"], adapter="fake_freeze")])
    table = adapters(crash=Crashing())
    ctx = load_context(REPOSITORY, config, adapters=table)
    results = execute(ctx, "all", table)
    assert results[0].outcome == INFRASTRUCTURE_FAILURE and "traceback" in results[0].detail
    assert list((ctx.run_dir / "logs" / "runtime_audit").glob("exception-*.log"))
    assert list((ctx.run_dir / "invocations").glob("*.json"))


def test_runner_exit_codes_map_to_distinct_failure_classes(tmp_path):
    import sys

    ctx = _context(make_layout(tmp_path))
    ctx.run_dir.mkdir(parents=True)
    env = ctx.child_environment(None, 1)
    outcomes = {}
    for code in (2, 3, 9):
        try:
            ctx.run_command("estimate", [sys.executable, "-c", f"raise SystemExit({code})"], env,
                            failure_outcomes={2: orchestrator.BLOCKED_MISSING_EVIDENCE,
                                              3: orchestrator.VALIDATION_FAILURE})
        except orchestrator.ChildCommandFailed as exc:
            outcomes[code] = exc.outcome
    assert outcomes == {2: orchestrator.BLOCKED_MISSING_EVIDENCE,
                        3: orchestrator.VALIDATION_FAILURE, 9: INFRASTRUCTURE_FAILURE}


def test_checked_in_graph_has_no_missing_runner():
    from e3sse.production.stage_graph import build_stage_graph

    config = json.loads((REPOSITORY / "configs" / "production_execution.json").read_text())
    graph = build_stage_graph(config["stages"], known_adapters=adapters())
    assert all(s.implementation_status == "implemented" for s in graph.stages.values())
    assert graph.order.index("pilot") < graph.order.index("transfer")
    assert graph.stages["pilot"].outcome_access == ("mplitrj_published_train_dft_forces_energies",)
