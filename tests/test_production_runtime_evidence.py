"""Runtime audit, workload benchmark and memory-throttled sharding (synthetic only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk

from e3sse.execution.resource_planning import HostResources, WorkloadPolicy, plan_workers
from e3sse.execution.sharded_execution import ShardedRun, execute_shards, make_shard
from e3sse.execution.workload_benchmark import benchmark_grid, run_grid_point

REPOSITORY = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(REPOSITORY / "src"), "PATH": "/usr/bin:/bin"}


def _structures():
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(2)
    return [{"numbers": atoms.numbers.tolist(), "positions": atoms.positions.tolist(),
             "cell": atoms.cell.tolist(), "pbc": atoms.pbc.tolist()}]


def test_grid_point_measures_memory_and_throughput():
    point = run_grid_point({"kind": "synthetic_emt_test_only"}, "force_single_point",
                           {"structures": _structures()}, processes=2, threads=1,
                           evaluations=3, warmup=1, timeout_seconds=120)
    assert point["status"] == "completed" and point["units"] == 6
    assert point["units_per_second"] > 0 and point["peak_resident_bytes"] > 0


def test_ml_neb_band_workload_uses_the_frozen_settings():
    from e3sse.barriers.ml_neb import reference_emt_vacancy_band

    protocol = json.loads((REPOSITORY / "configs" /
                           "reliability_gated_internal_freeze_protocol.json").read_text())
    band = reference_emt_vacancy_band(3)
    payload = {"structures": [{"numbers": a.numbers.tolist(), "positions": a.positions.tolist(),
                               "cell": a.cell.tolist(), "pbc": a.pbc.tolist()} for a in band],
               "frozen_numeric_settings": protocol["amendments"]["prospective"]["ml_neb"][
                   "frozen_numeric_settings"]}
    point = run_grid_point({"kind": "synthetic_emt_test_only"}, "ml_neb_band", payload,
                           processes=1, threads=1, evaluations=2, warmup=0, timeout_seconds=120)
    assert point["status"] == "completed" and point["units"] == (2 + 1) * 3


def test_dead_worker_is_an_infrastructure_failure_not_a_result():
    point = run_grid_point({"kind": "synthetic_emt_test_only"}, "force_single_point",
                           {"structures": [{"numbers": [3], "positions": [[0, 0, 0]],
                                            "cell": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                                            "pbc": [True] * 3}]},  # EMT has no Li parameters
                           processes=1, threads=1, evaluations=1, warmup=0, timeout_seconds=60)
    assert point["status"] == "failed" and point["classification"] == "infrastructure"


def test_benchmark_grid_respects_cpu_and_memory():
    grid = benchmark_grid([1, 2, 4, 8], [1, 2, 4], usable_cpus=8,
                          memory_budget_bytes=10, planning_bytes_per_process=3)
    assert grid and all(p * t <= 8 and p * 3 <= 10 for p, t in grid)
    assert (4, 1) not in grid  # memory: 4 x 3 > 10


def test_runtime_audit_script_records_frozen_neb_identity(tmp_path):
    out = tmp_path / "audit"
    command = [sys.executable, str(REPOSITORY / "scripts" / "audit_ml_neb_runtime.py"),
               "--protocol-freeze-config", str(REPOSITORY / "configs" / "protocol_freeze.json"),
               "--active-protocol", str(REPOSITORY / "configs" /
                                        "reliability_gated_internal_freeze_protocol.json"),
               "--environment", "local", "--output", str(out),
               "--protected-root", str(tmp_path / "data"), "--synthetic-test-runtime"]
    first = subprocess.run(command, check=True, env=ENV, capture_output=True, text=True)
    path = Path(json.loads(first.stdout.strip().splitlines()[-1])["report"])
    assert path.parent == out and path.name.startswith("local-")
    report = json.loads(path.read_text())
    assert report["checks"]["failure_classification"] == {
        "raising_calculator": "calculator_exception", "non_finite_calculator": "non_finite_value"}
    assert report["passed"] is True
    assert report["checks"]["neb_constructor"]["method"] == "ImprovedTangentMethod"
    assert report["checks"]["neb_import_compatibility"] in {
        "native", "scipy_cumulative_trapezoid_alias"}
    assert report["checks"]["reference_neb"]["deterministic_repeat"] is True
    again = subprocess.run([*command, "--resume"], env=ENV, capture_output=True, text=True,
                     check=False)
    assert again.returncode == 0 and json.loads(again.stdout)["verified_existing"] is True
    refused = subprocess.run(command, env=ENV, capture_output=True, text=True,
                     check=False)
    assert refused.returncode != 0  # create-once: no silent overwrite


def test_profile_script_is_resumable(tmp_path):
    structures = tmp_path / "structures.json"
    structures.write_text(json.dumps({"structures": _structures()}))
    command = [sys.executable, str(REPOSITORY / "scripts" / "profile_production_resources.py"),
               "--protocol-freeze-config", str(REPOSITORY / "configs" / "protocol_freeze.json"),
               "--active-protocol", str(REPOSITORY / "configs" /
                                        "reliability_gated_internal_freeze_protocol.json"),
               "--model", "synthetic", "--workload", "force_single_point",
               "--structure-file", str(structures), "--memory-threads", "1",
               "--process-grid", "1,2", "--thread-grid", "1", "--evaluations", "2",
               "--memory-reserve-gib", "0.01", "--memory-safety-factor", "1.2",
               "--output", str(tmp_path / "profiles"), "--protected-root", str(tmp_path / "data"),
               "--synthetic-test-calculator"]
    subprocess.run(command, check=True, env=ENV, capture_output=True)
    from e3sse.production.resource_scheduler import load_profiles

    profiles = load_profiles((tmp_path / "profiles").glob("*/*.json"))
    profile = profiles[("synthetic", "force_single_point")]
    assert profile.benchmark and profile.measured_threads_per_worker == 1
    resumed = subprocess.run([*command, "--resume"], env=ENV, capture_output=True, text=True,
                     check=False)
    assert resumed.returncode == 0 and "verified existing" in resumed.stdout


def _square(payload, context):
    return {"unit": payload["unit"], "value": payload["unit"] ** 2}


def test_memory_floor_throttles_launches_without_changing_results(tmp_path):
    shards = [make_shard("unit", {"unit": i}) for i in range(8)]
    host = HostResources(logical_cpus=4, usable_cpus=(0, 1, 2, 3), mem_total_bytes=None,
                         mem_available_bytes=None, swap_free_bytes=None)
    plan = plan_workers(2, WorkloadPolicy("synthetic"), host)
    identity = {"code_sha256": "synthetic"}

    def run(directory, **kwargs):
        sharded = ShardedRun(tmp_path / directory, identity, _square, [tmp_path / "data"])
        return execute_shards(sharded, shards, plan, progress=lambda m: None, **kwargs)

    free = run("free")
    calls = []

    def tight():
        calls.append(1)
        return 0  # always under the floor: only one shard may be in flight at a time

    throttled = run("throttled", memory_floor_bytes=1, available_memory=tight)
    assert throttled.results == free.results and throttled.merged_sha256 == free.merged_sha256
    assert throttled.max_observed_in_flight == 1 and calls
    manifest = json.loads(throttled.manifest_path.read_text())
    assert manifest["memory_throttle_events"] > 0
    assert np.isclose(sum(r["value"] for r in throttled.results.values()), 140)


def test_lock_verification_canonicalizes_dotted_distribution_names(tmp_path, monkeypatch):
    """uv writes ``ruamel-yaml`` while metadata says ``ruamel.yaml``: not drift."""

    from importlib import metadata

    from e3sse.softening import study1_inference

    class Dist:
        def __init__(self, name, version):
            self.metadata, self.version = {"Name": name}, version

    installed = [Dist("ruamel.yaml", "0.18.6"), Dist("Zope.Interface", "7.0"),
                 Dist("typing_extensions", "4.12.2")]
    monkeypatch.setattr(metadata, "distributions", lambda: installed)
    lock = tmp_path / "lock.txt"
    lock.write_text("ruamel-yaml==0.18.6\nzope-interface==7.0\ntyping-extensions==4.12.2\n")
    environment = {"python_version": sys.version.split()[0], "pinned_packages": {}}
    assert study1_inference.verify_runtime(environment, None, lock)["lock_verified"] is True
    lock.write_text("ruamel-yaml==0.18.5\n")  # a real version difference is still drift
    import pytest

    with pytest.raises(RuntimeError, match="ruamel-yaml"):
        study1_inference.verify_runtime(environment, None, lock)
