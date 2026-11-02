"""Central CPU/RAM scheduler: discovery, envelopes, benchmark selection, serialization."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from e3sse.execution.resource_planning import GIB, HostResources
from e3sse.production.resource_scheduler import (
    RESOURCE_PROFILE_SCHEMA,
    BenchmarkPoint,
    ResourcePolicy,
    SchedulingError,
    WorkloadProfile,
    allocate_workload,
    discover_resources,
    load_profiles,
    memory_pressure_safe,
    predicted_schedule,
)

POLICY = ResourcePolicy(memory_reserve_bytes=2 * GIB, memory_reserve_fraction=0.1,
                        memory_safety_factor=1.25, max_threads_per_worker=16,
                        statistics_worker_memory_bytes=GIB // 2)


def host(cpus=80, available_gib=18.0, total_gib=19.0):
    return HostResources(logical_cpus=cpus, usable_cpus=tuple(range(cpus)),
                         mem_total_bytes=int(total_gib * GIB),
                         mem_available_bytes=int(available_gib * GIB), swap_free_bytes=8 * GIB,
                         numa_nodes={"node0": {"cpus": "0-39"}, "node1": {"cpus": "40-79"}})


def profile(peak_gib, *, threads=16, benchmark=()):
    return WorkloadProfile(model="m", workload="force_single_point",
                           peak_resident_bytes=int(peak_gib * GIB),
                           measured_threads_per_worker=threads, benchmark=tuple(benchmark))


def _fake_system(root: Path, cpulist: dict[str, str], meminfo: str) -> tuple[Path, Path]:
    proc, sys = root / "proc", root / "sys"
    proc.mkdir()
    (proc / "meminfo").write_text(meminfo)
    for node, cpus in cpulist.items():
        directory = sys / "devices" / "system" / "node" / node
        directory.mkdir(parents=True)
        (directory / "cpulist").write_text(cpus)
        (directory / "meminfo").write_text(f"Node {node[-1]} MemTotal: 1000 kB\n")
    return proc, sys


def test_discovery_uses_affinity_and_overrides_only_narrow(tmp_path, monkeypatch):
    proc, sys = _fake_system(tmp_path, {"node0": "0-39", "node1": "40-79"},
                             "MemTotal: 19922944 kB\nMemAvailable: 18874368 kB\nSwapFree: 8388608 kB\n")
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(8, 24)), raising=False)
    observed = discover_resources(proc_root=proc, sys_root=sys)
    assert observed.usable_cpus == tuple(range(8, 24))  # affinity mask, not 80
    assert observed.mem_available_bytes == 18 * GIB
    assert set(observed.numa_nodes) == {"node0", "node1"}
    narrowed = discover_resources(max_cpus=4, max_memory_gib=4.0, proc_root=proc, sys_root=sys)
    assert narrowed.usable_cpus == (8, 9, 10, 11)
    assert narrowed.mem_available_bytes == 4 * GIB
    widened = discover_resources(max_cpus=1000, max_memory_gib=1000.0, proc_root=proc, sys_root=sys)
    assert len(widened.usable_cpus) == 16 and widened.mem_available_bytes == 18 * GIB


def test_memory_bounds_workers_and_swap_is_never_capacity():
    allocation = allocate_workload(profile(4.0), host(), POLICY)
    budget = 18 * GIB - max(2 * GIB, int(0.1 * 19 * GIB))
    assert allocation.memory_budget_bytes == budget
    assert allocation.workers == budget // int(4.0 * 1.25 * GIB) == 3
    assert allocation.workers * allocation.planning_memory_per_worker_bytes <= budget
    assert allocation.basis == "unbenchmarked_floor_cpus_over_workers"
    assert allocation.threads_per_worker == 16  # floor(80 / 3) capped at 16


def test_threads_bounded_by_cpus_and_measurement():
    allocation = allocate_workload(profile(0.5, threads=4), host(cpus=80), POLICY)
    assert allocation.threads_per_worker <= 4  # never above the measured thread count
    assert allocation.workers * allocation.threads_per_worker <= 80


@pytest.mark.parametrize("cpus,available,peak", list(itertools.product(
    (1, 3, 8, 80), (3.0, 8.0, 18.0), (0.2, 1.4, 2.1, 5.3))))
def test_no_oversubscription_property(cpus, available, peak):
    try:
        allocation = allocate_workload(profile(peak), host(cpus=cpus, available_gib=available), POLICY)
    except SchedulingError:
        return
    assert allocation.workers >= 1 and allocation.threads_per_worker >= 1
    assert allocation.workers * allocation.threads_per_worker <= cpus
    assert allocation.workers * allocation.planning_memory_per_worker_bytes <= \
        allocation.memory_budget_bytes


def test_measured_benchmark_selects_highest_feasible_throughput():
    points = [BenchmarkPoint(1, 16, 10.0), BenchmarkPoint(2, 8, 18.0),
              BenchmarkPoint(4, 4, 30.0), BenchmarkPoint(8, 2, 55.0),  # infeasible: memory
              BenchmarkPoint(4, 8, 30.0)]  # same throughput, more threads -> not chosen
    allocation = allocate_workload(profile(3.0, benchmark=points), host(), POLICY)
    assert (allocation.workers, allocation.threads_per_worker) == (4, 4)
    assert allocation.basis == "measured_throughput_benchmark"
    assert allocation.expected_units_per_second == 30.0


def test_pending_units_cap_workers():
    allocation = allocate_workload(profile(0.5), host(), POLICY, pending_units=2)
    assert allocation.workers == 2


def test_infeasible_worker_is_refused():
    with pytest.raises(SchedulingError, match="one worker needs"):
        allocate_workload(profile(20.0), host(), POLICY)


def test_memory_pressure_guard():
    assert memory_pressure_safe(host(available_gib=10.0), POLICY)
    assert not memory_pressure_safe(host(available_gib=1.0), POLICY)


def test_profile_serialization_round_trip_and_duplicates(tmp_path):
    report = {"schema_version": RESOURCE_PROFILE_SCHEMA,
              "identity": {"model": "CHGNet-0.3.0", "workload": "ml_neb_band",
                           "memory_threads_per_worker": 8},
              "measurement": {"peak_resident_bytes": 3 * GIB},
              "benchmark": {"points": [
                  {"processes": 2, "threads_per_process": 4, "status": "completed",
                   "units_per_second": 7.5},
                  {"processes": 8, "threads_per_process": 8, "status": "failed"}]}}
    path = tmp_path / "CHGNet-0.3.0" / "ml_neb_band-abc.json"
    path.parent.mkdir()
    path.write_text(json.dumps(report))
    profiles = load_profiles([path])
    loaded = profiles[("CHGNet-0.3.0", "ml_neb_band")]
    assert loaded.peak_resident_bytes == 3 * GIB and loaded.measured_threads_per_worker == 8
    assert loaded.benchmark == (BenchmarkPoint(2, 4, 7.5),)  # failed points never count
    allocation = allocate_workload(loaded, host(), POLICY).as_json()
    assert json.loads(json.dumps(allocation))["workers"] == 2
    second = tmp_path / "CHGNet-0.3.0" / "ml_neb_band-def.json"
    second.write_text(json.dumps(report))
    with pytest.raises(SchedulingError, match="two resource profiles"):
        load_profiles([path, second])


def test_predicted_schedule_lists_missing_profiles_instead_of_guessing():
    schedule = predicted_schedule(host(), POLICY, {}, [("m", "force_single_point"),
                                                      ("statistics", "statistics_numpy")])
    assert schedule["missing_profiles"] == ["m/force_single_point"]
    stats = schedule["allocations"]["statistics/statistics_numpy"]
    assert stats["threads_per_worker"] == 1


def test_memory_cap_also_caps_the_fractional_reserve(tmp_path, monkeypatch):
    proc, sys = _fake_system(tmp_path, {"node0": "0-3"},
                             "MemTotal: 268435456 kB\nMemAvailable: 260046848 kB\n")
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(4)), raising=False)
    capped = discover_resources(max_memory_gib=24.0, proc_root=proc, sys_root=sys)
    policy = ResourcePolicy(memory_reserve_bytes=int(2.5 * GIB), memory_reserve_fraction=0.12,
                            memory_safety_factor=1.35, max_threads_per_worker=16,
                            statistics_worker_memory_bytes=GIB)
    assert capped.mem_total_bytes == 24 * GIB
    assert memory_pressure_safe(capped, policy)  # reserve is 12% of 24 GiB, not of 256 GiB


def test_pinned_threads_fix_threads_and_only_workers_vary():
    points = [BenchmarkPoint(4, 4, 30.0), BenchmarkPoint(2, 8, 40.0)]
    big = allocate_workload(profile(1.0, benchmark=points), host(), POLICY, fixed_threads=4)
    small = allocate_workload(profile(1.0, benchmark=points), host(cpus=8, available_gib=6.0),
                              POLICY, fixed_threads=4)
    assert big.threads_per_worker == small.threads_per_worker == 4
    assert small.workers <= 2 and big.basis == "pinned_threads_run_level"
    with pytest.raises(SchedulingError, match="not covered"):
        allocate_workload(profile(1.0, threads=2), host(), POLICY, fixed_threads=4)


def test_duplicate_profiles_are_dropped_in_lenient_mode(tmp_path):
    report = {"schema_version": RESOURCE_PROFILE_SCHEMA,
              "identity": {"model": "m", "workload": "ml_neb_band", "memory_threads_per_worker": 1},
              "measurement": {"peak_resident_bytes": GIB}, "benchmark": {"points": []}}
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    ambiguous: list[str] = []
    assert load_profiles(paths, strict=False, ambiguous=ambiguous) == {}
    assert ambiguous == ["m/ml_neb_band"]
