"""Parallel execution is bounded, resumable, and scientifically invariant to scheduling."""

from __future__ import annotations

import json
import multiprocessing
import random
from pathlib import Path

import parallel_execution_fixtures as fixtures
import pytest

from e3sse.execution import (
    HostResources,
    InsufficientResourcesError,
    ShardedExecutionError,
    ShardedRun,
    StaleCheckpointError,
    WorkloadPolicy,
    execute_shards,
    make_shard,
    plan_workers,
)
from e3sse.execution.resource_planning import GIB
from e3sse.execution.sharded_execution import (
    ArtifactExistsError,
    canonical_sha256,
    ensure_safe_output_dir,
    publish_json_once,
)
from e3sse.execution.thread_control import THREAD_ENVIRONMENT_VARIABLES, ThreadControlError

IDENTITY = {"code_sha256": "c" * 64, "config_sha256": "d" * 64,
            "model": {"canonical_name": "synthetic", "weights_sha256": "e" * 64}}


def _host(cpus: int = 8, available_gib: float = 16.0) -> HostResources:
    return HostResources(
        logical_cpus=cpus, usable_cpus=tuple(range(cpus)), mem_total_bytes=int(19 * GIB),
        mem_available_bytes=int(available_gib * GIB), swap_free_bytes=0, numa_nodes={},
    )


def _plan(workers: int, *, threads: int = 1, in_flight: int = 2, reserve_gib: float = 2.0):
    policy = WorkloadPolicy("synthetic-bootstrap", threads_per_worker=threads,
                            in_flight_per_worker=in_flight,
                            memory_reserve_bytes=int(reserve_gib * GIB))
    return plan_workers(workers, policy, _host())


def _shards(count: int = 6) -> list:
    shards = []
    for index in range(count):
        identity = {"unit_id": f"unit-{index:02d}", "replicate": index % 2}
        shards.append(make_shard("bootstrap", identity, {**identity, "n_values": 25,
                                                         "resamples": 50}))
    return shards


def _run(tmp_path: Path, name: str, task=fixtures.bootstrap_mean_task, **kwargs) -> ShardedRun:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return ShardedRun(output_dir=tmp_path / "outputs" / name,
                      scientific_identity=kwargs.pop("identity", IDENTITY), task=task,
                      protected_roots=[raw], **kwargs)


def _thread_env(monkeypatch, threads: int = 1) -> None:
    for name in THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, str(threads))


def _shard_bytes(run: ShardedRun) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted((run.output_dir / "shards").glob("*/*.json"))}


def test_serial_and_parallel_runs_are_identical_for_any_worker_count(tmp_path, monkeypatch):
    _thread_env(monkeypatch)
    serial = _run(tmp_path, "serial")
    reference = execute_shards(serial, _shards(), _plan(1), executor="in_process")
    for workers in (1, 2, 4):
        run = _run(tmp_path, f"workers-{workers}")
        result = execute_shards(run, _shards(), _plan(workers))
        assert result.results == reference.results
        assert result.merged_sha256 == reference.merged_sha256
        # Checkpoints are byte-identical: no worker, PID, plan or timing inside.
        assert _shard_bytes(run) == _shard_bytes(serial)
        assert (run.output_dir / "run_identity.json").read_bytes() == (
            serial.output_dir / "run_identity.json"
        ).read_bytes()


def test_payload_changes_invalidate_checkpoints(tmp_path):
    run = _run(tmp_path, "payload")
    execute_shards(run, _shards(), _plan(2))
    larger = [make_shard("bootstrap", s.identity, {**s.payload, "resamples": 75})
              for s in _shards()]
    with pytest.raises(StaleCheckpointError, match="fingerprint changed"):
        execute_shards(run, larger, _plan(2), resume=True)


def test_global_rng_use_cannot_depend_on_worker_history(tmp_path):
    shards = _shards(12)
    one = execute_shards(_run(tmp_path, "global-1", task=fixtures.global_random_task),
                         shards, _plan(1))
    many = execute_shards(_run(tmp_path, "global-4", task=fixtures.global_random_task),
                          shards, _plan(4))
    assert one.results == many.results


def test_shard_set_and_shard_ids_are_bound_to_the_run(tmp_path):
    from dataclasses import replace

    run = _run(tmp_path, "shard-set")
    execute_shards(run, _shards(6), _plan(2))
    with pytest.raises(StaleCheckpointError, match="different shard set"):
        execute_shards(run, _shards(4), _plan(2), resume=True)
    forged = replace(_shards(1)[0], identity={"unit_id": "other", "replicate": 0})
    with pytest.raises(ValueError, match="not derived from its identity"):
        execute_shards(_run(tmp_path, "forged"), [forged], _plan(1))


def test_one_orchestrator_per_run_directory(tmp_path):
    import fcntl
    import os

    run = _run(tmp_path, "locked")
    run.output_dir.mkdir(parents=True)
    descriptor = os.open(run.output_dir / ".run.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        with pytest.raises(RuntimeError, match="another process"):
            execute_shards(run, _shards(), _plan(1))
    finally:
        os.close(descriptor)
    execute_shards(run, _shards(), _plan(1))


def test_run_identity_must_bind_code(tmp_path):
    run = _run(tmp_path, "no-code", identity={"config_sha256": "d" * 64})
    with pytest.raises(ValueError, match="code_sha256, or git_sha"):
        execute_shards(run, _shards(), _plan(1))


def test_rng_draws_depend_only_on_canonical_identifiers(tmp_path):
    result = execute_shards(_run(tmp_path, "draws"), _shards(), _plan(3))
    for shard in _shards():
        direct = fixtures.bootstrap_mean_task(shard.payload, None)
        assert result.results[shard.shard_id] == direct


def test_completion_and_submission_order_do_not_change_the_merged_artifact(tmp_path):
    shuffled = _shards(8)
    random.Random(7).shuffle(shuffled)
    first = execute_shards(_run(tmp_path, "ordered"), _shards(8), _plan(4))
    second = execute_shards(_run(tmp_path, "shuffled"), shuffled, _plan(3))
    assert list(first.results) == sorted(first.results) == list(second.results)
    assert first.merged_sha256 == second.merged_sha256


def test_resume_reuses_verified_shards_and_refuses_overwrite(tmp_path):
    run = _run(tmp_path, "resume")
    first = execute_shards(run, _shards(), _plan(2))
    with pytest.raises(ArtifactExistsError):
        execute_shards(run, _shards(), _plan(2))
    again = execute_shards(run, _shards(), _plan(4), resume=True)
    assert again.computed == () and len(again.reused) == len(_shards())
    assert again.merged_sha256 == first.merged_sha256


def test_stale_fingerprints_and_identity_changes_are_rejected(tmp_path):
    run = _run(tmp_path, "stale")
    execute_shards(run, _shards(), _plan(2))
    # Model/runtime identity is part of the run identity.
    changed_model = {**IDENTITY, "model": {**IDENTITY["model"], "weights_sha256": "f" * 64}}
    with pytest.raises(StaleCheckpointError, match="different scientific run"):
        execute_shards(_run(tmp_path, "stale", identity=changed_model), _shards(),
                       _plan(2), resume=True)
    # Per-process numerical threads change CPU reductions: shards are not reused.
    with pytest.raises(StaleCheckpointError, match="fingerprint changed"):
        execute_shards(run, _shards(), _plan(2, threads=2), resume=True)
    # A tampered result fails its digest.
    path = next((run.output_dir / "shards").glob("*/*.json"))
    stored = json.loads(path.read_text())
    stored["result"]["seed"] += 1
    path.chmod(0o644)
    path.write_text(json.dumps(stored))
    with pytest.raises(StaleCheckpointError, match="digest"):
        execute_shards(run, _shards(), _plan(2), resume=True)


def test_failed_shard_keeps_completed_siblings_and_resume_finishes(tmp_path):
    run = _run(tmp_path, "failure", task=fixtures.failing_bootstrap_task)
    with pytest.raises(ShardedExecutionError) as caught:
        execute_shards(run, _shards(), _plan(2))
    assert len(caught.value.failures) == 1
    manifest = json.loads(caught.value.manifest_path.read_text())
    assert manifest["merged_sha256"] is None
    assert len(list((run.output_dir / "shards").glob("*/*.json"))) == len(_shards()) - 1
    fixed = _run(tmp_path, "failure")
    result = execute_shards(fixed, _shards(), _plan(2), resume=True)
    assert len(result.computed) == 1
    reference = execute_shards(_run(tmp_path, "failure-reference"), _shards(), _plan(1))
    assert result.merged_sha256 == reference.merged_sha256


def test_worker_crash_is_infrastructure_failure_not_a_result(tmp_path):
    run = _run(tmp_path, "crash", task=fixtures.crashing_bootstrap_task)
    with pytest.raises(ShardedExecutionError) as caught:
        execute_shards(run, _shards(), _plan(2))
    crashed = make_shard("bootstrap", {"unit_id": "unit-02", "replicate": 0}).shard_id
    assert crashed in caught.value.failures
    assert not list((run.output_dir / "shards").glob(f"*/{crashed}.json"))
    result = execute_shards(_run(tmp_path, "crash"), _shards(), _plan(2), resume=True)
    assert crashed in result.computed


def _publish_race(path: str, root: str, barrier, outcomes) -> None:
    barrier.wait()
    try:
        publish_json_once(Path(path), {"writer": multiprocessing.current_process().name},
                          root=Path(root))
    except ArtifactExistsError:
        outcomes.put("refused")
    else:
        outcomes.put("published")


def test_two_processes_cannot_publish_the_same_artifact(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier, outcomes = context.Barrier(2), context.Queue()
    target = tmp_path / "shards" / "bootstrap-0123456789abcdef0123.json"
    processes = [context.Process(target=_publish_race, args=(str(target), str(tmp_path),
                                                             barrier, outcomes))
                 for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
    assert sorted(outcomes.get(timeout=5) for _ in processes) == ["published", "refused"]
    assert "writer" in json.loads(target.read_text())


def test_outputs_can_never_overlap_protected_raw_data(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    with pytest.raises(PermissionError):
        ensure_safe_output_dir(raw / "results", [raw])
    with pytest.raises(PermissionError):
        ensure_safe_output_dir(tmp_path, [raw])  # output root containing raw data
    with pytest.raises(ValueError):
        ensure_safe_output_dir(tmp_path / "outputs", [])
    with pytest.raises(PermissionError):
        publish_json_once(raw / "x.json", {}, root=tmp_path / "outputs")
    inside = ShardedRun(output_dir=raw / "run", scientific_identity=IDENTITY,
                        task=fixtures.bootstrap_mean_task, protected_roots=[raw])
    with pytest.raises(PermissionError):
        execute_shards(inside, _shards(), _plan(1))
    assert list(raw.iterdir()) == []


def test_execution_plan_stays_out_of_scientific_identity(tmp_path):
    small = execute_shards(_run(tmp_path, "plan-a"), _shards(), _plan(1, reserve_gib=1))
    large = execute_shards(_run(tmp_path, "plan-b"), _shards(), _plan(4, in_flight=1,
                                                                      reserve_gib=6))
    assert small.merged_sha256 == large.merged_sha256
    assert _shard_bytes(_run(tmp_path, "plan-a")) == _shard_bytes(_run(tmp_path, "plan-b"))
    manifest = json.loads(large.manifest_path.read_text())
    assert "workers" in manifest["excluded_from_scientific_identity"]
    for checkpoint in _shard_bytes(_run(tmp_path, "plan-b")).values():
        assert b'"workers"' not in checkpoint and b"max_in_flight" not in checkpoint


def test_execution_manifest_is_strict_json_with_operational_metadata(tmp_path):
    result = execute_shards(_run(tmp_path, "manifest"), _shards(), _plan(2))
    text = result.manifest_path.read_text()
    manifest = json.loads(text, parse_constant=lambda name: pytest.fail(name))
    assert manifest["schema_version"] == "e3sse.execution.manifest.v1"
    assert manifest["classification"] == "operational_execution_metadata"
    assert manifest["start_method"] == "spawn"
    plan = manifest["plan"]
    assert plan["workers"] == 2 and plan["requested_workers"] == 2
    assert plan["threads_per_worker"] == 1 and plan["host"]["usable_cpu_count"] == 8
    assert manifest["shard_counts"] == {"requested": 6, "reused": 0, "computed": 6,
                                        "failed_or_not_run": 0}
    assert manifest["merged_sha256"] == result.merged_sha256


def test_submission_is_bounded_by_the_in_flight_window(tmp_path):
    plan = _plan(2, in_flight=2)
    result = execute_shards(_run(tmp_path, "bounded"), _shards(40), plan)
    assert plan.max_in_flight == 4
    assert 1 <= result.max_observed_in_flight <= plan.max_in_flight


def test_workers_inherit_explicit_thread_controls(tmp_path, monkeypatch):
    for name in THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    shard = make_shard("threads", {"probe": "environment"})
    result = execute_shards(_run(tmp_path, "threads", task=fixtures.thread_environment_task),
                            [shard], _plan(1, threads=2))
    observed = result.results[shard.shard_id]
    assert set(observed["environment"].values()) == {"2"}
    # The parent environment is restored after the pool closes.
    assert all(name not in __import__("os").environ for name in THREAD_ENVIRONMENT_VARIABLES)
    # Serial in-process execution needs the variables at interpreter start.
    with pytest.raises(ThreadControlError):
        execute_shards(_run(tmp_path, "threads-serial"), _shards(), _plan(1),
                       executor="in_process")


def test_torch_threads_are_fixed_in_the_worker_initializer(tmp_path):
    pytest.importorskip("torch")
    policy = WorkloadPolicy("synthetic-mlip", threads_per_worker=2, configure_torch=True)
    shard = make_shard("threads", {"probe": "torch"})
    result = execute_shards(_run(tmp_path, "torch", task=fixtures.thread_environment_task),
                            [shard], plan_workers(1, policy, _host()))
    assert result.results[shard.shard_id]["torch_threads"] == [2, 1]


def test_model_is_loaded_once_per_worker_and_reused(tmp_path):
    frames = [make_shard("frame", {"frame_id": f"f{index}", "n_atoms": index + 1})
              for index in range(5)]
    run = _run(tmp_path, "model", task=fixtures.scaled_energy_task,
               worker_setup=fixtures.load_synthetic_model, worker_setup_args=(-2.5,))
    result = execute_shards(run, frames, _plan(2))
    energies = {r["frame_id"]: r["energy"] for r in result.results.values()}
    assert energies == {f"f{index}": -2.5 * (index + 1) for index in range(5)}


def test_worker_planning_is_memory_and_cpu_bounded():
    policy = WorkloadPolicy("mlip", threads_per_worker=4, per_worker_memory_bytes=3 * GIB,
                            memory_reserve_bytes=2 * GIB)
    host = _host(cpus=80, available_gib=17)
    auto = plan_workers("auto", policy, host)
    assert auto.cpu_envelope_workers == 20
    assert auto.memory_envelope_workers == 5 and auto.workers == 5
    with pytest.raises(InsufficientResourcesError, match="memory envelope"):
        plan_workers(6, policy, host)
    with pytest.raises(InsufficientResourcesError, match="usable CPUs"):
        plan_workers(21, WorkloadPolicy("stats", threads_per_worker=4), host)
    with pytest.raises(InsufficientResourcesError, match="measured per-worker memory"):
        plan_workers("auto", WorkloadPolicy("stats"), host)
    assert plan_workers(40, WorkloadPolicy("stats"), host, unit_count=3).workers == 3
    tight = WorkloadPolicy("mlip", per_worker_memory_bytes=8 * GIB, memory_reserve_bytes=12 * GIB)
    with pytest.raises(InsufficientResourcesError, match="not even one worker"):
        plan_workers("auto", tight, host)


def test_host_observation_reads_affinity_meminfo_and_numa(tmp_path):
    from e3sse.execution.resource_planning import observe_host_resources

    proc, sys_root = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (proc / "meminfo").write_text(
        "MemTotal:       19922944 kB\nMemAvailable:   17825792 kB\nSwapFree:        8073216 kB\n"
    )
    for index, cpus in enumerate(("0-39", "40-79")):
        node = sys_root / "devices" / "system" / "node" / f"node{index}"
        node.mkdir(parents=True)
        (node / "cpulist").write_text(cpus + "\n")
        (node / "meminfo").write_text(f"Node {index} MemTotal:  9961472 kB\n")
    host = observe_host_resources(proc, sys_root)
    assert host.mem_available_bytes == 17825792 * 1024
    assert host.numa_nodes["node1"] == {"cpus": "40-79", "mem_total_bytes": 9961472 * 1024,
                                        "mem_free_bytes": None}
    assert canonical_sha256(host.as_json())  # strict-JSON serialisable


def test_worker_memory_report_must_match_model_and_threads(tmp_path):
    from e3sse.execution.resource_planning import per_worker_memory_from_report
    from e3sse.execution.worker_memory_probe import peak_resident_bytes, probe_supercell_repeats

    report = tmp_path / "report.json"
    identity = {"canonical_name": "synthetic", "weights_sha256": "e" * 64, "threads_per_worker": 2}
    report.write_text(json.dumps({"schema_version": "e3sse.execution.worker-memory.v1",
                                  "identity": identity,
                                  "fingerprint_sha256": canonical_sha256(identity),
                                  "measurement": {"peak_resident_bytes": 1000}}))
    expected = {"canonical_name": "synthetic", "weights_sha256": "e" * 64}
    assert per_worker_memory_from_report(report, expected_identity=expected,
                                         threads_per_worker=2) == 1500
    with pytest.raises(ValueError, match="identity differs"):
        per_worker_memory_from_report(report, expected_identity={"weights_sha256": "0" * 64},
                                      threads_per_worker=2)
    with pytest.raises(ValueError, match="thread count"):
        per_worker_memory_from_report(report, expected_identity=expected, threads_per_worker=1)
    forged = json.loads(report.read_text())
    forged["identity"]["threads_per_worker"] = 1
    report.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="does not match its identity"):
        per_worker_memory_from_report(report, expected_identity=expected, threads_per_worker=1)
    status = tmp_path / "status"
    status.write_text("VmPeak:  10 kB\nVmHWM:     2048 kB\nVmRSS: 1 kB\n")
    assert peak_resident_bytes(status) == 2048 * 1024
    assert probe_supercell_repeats(12) == 1 and probe_supercell_repeats(97) == 3
