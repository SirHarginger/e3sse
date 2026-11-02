"""One central CPU/RAM scheduler for every production workload.

Inputs are observations and measurements, never assumed hardware:

* schedulable CPUs from the process affinity mask (``os.sched_getaffinity``), which
  respects cgroup/taskset limits, not the machine-wide CPU count;
* ``MemAvailable`` from ``/proc/meminfo`` (swap is never capacity);
* a measured resource profile per (model, workload class): peak resident memory of
  one worker at a given thread count, and optionally a throughput benchmark over
  (concurrent processes x threads per process).

For one workload the allocation satisfies, with explicit headroom:

    workers * planning_memory_per_worker <= MemAvailable - reserve
    workers * threads_per_worker         <= schedulable CPUs

``planning_memory_per_worker`` is the measured peak times a safety factor, measured
at a thread count at least as large as the one planned.  Among feasible benchmarked
configurations the highest measured aggregate throughput wins (ties: fewer
processes, then fewer threads).  Without a benchmark the rule is the PyTorch upper
bound ``threads = floor(cpus / workers)`` capped by ``max_threads_per_worker``, and
the allocation is labelled ``unbenchmarked``.

Worker count is operational.  Threads per worker can change CPU floating-point
reductions, so the stage records it and resumes must keep it (shard fingerprints
already bind it).  Placement (NUMA, worker order) never changes results.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from e3sse.execution.resource_planning import GIB, HostResources, observe_host_resources

RESOURCE_PROFILE_SCHEMA = "e3sse.production.resource-profile"
ALLOCATION_SCHEMA = "e3sse.production.workload-allocation"
WORKLOAD_CLASSES = ("force_single_point", "ml_neb_band", "embedding_inference",
                    "statistics_numpy")
# Statistical workloads run plain numpy at one thread; they never load a model.
STATISTICS_WORKLOAD = "statistics_numpy"


class SchedulingError(RuntimeError):
    """No allocation fits the observed resources and the measured evidence."""


@dataclass(frozen=True)
class ResourcePolicy:
    """Operational policy from the production config; never scientific identity."""

    memory_reserve_bytes: int
    memory_reserve_fraction: float
    memory_safety_factor: float
    max_threads_per_worker: int
    statistics_worker_memory_bytes: int

    def __post_init__(self) -> None:
        if self.memory_reserve_bytes < 0 or not 0 <= self.memory_reserve_fraction < 1:
            raise ValueError("memory reserve must be non-negative and the fraction in [0, 1)")
        if self.memory_safety_factor < 1:
            raise ValueError("memory_safety_factor must be at least 1")
        if self.max_threads_per_worker < 1 or self.statistics_worker_memory_bytes < 1:
            raise ValueError("thread cap and statistics worker memory must be positive")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ResourcePolicy:
        return cls(
            memory_reserve_bytes=int(float(config["memory_reserve_gib"]) * GIB),
            memory_reserve_fraction=float(config["memory_reserve_fraction"]),
            memory_safety_factor=float(config["memory_safety_factor"]),
            max_threads_per_worker=int(config["max_threads_per_worker"]),
            statistics_worker_memory_bytes=int(float(config["statistics_worker_memory_gib"]) * GIB),
        )

    def reserve_for(self, host: HostResources) -> int:
        total = host.mem_total_bytes or 0
        return max(self.memory_reserve_bytes, math.ceil(self.memory_reserve_fraction * total))


@dataclass(frozen=True)
class BenchmarkPoint:
    processes: int
    threads_per_process: int
    units_per_second: float


@dataclass(frozen=True)
class WorkloadProfile:
    """Measured evidence for one (model, workload class)."""

    model: str
    workload: str
    peak_resident_bytes: int
    measured_threads_per_worker: int
    benchmark: tuple[BenchmarkPoint, ...] = ()
    evidence_path: str | None = None
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.workload not in WORKLOAD_CLASSES:
            raise ValueError(f"unknown workload class {self.workload!r}")
        if self.peak_resident_bytes <= 0 or self.measured_threads_per_worker < 1:
            raise ValueError("profile needs a positive peak and thread count")


@dataclass(frozen=True)
class WorkloadAllocation:
    model: str
    workload: str
    workers: int
    threads_per_worker: int
    planning_memory_per_worker_bytes: int
    memory_budget_bytes: int
    usable_cpus: int
    basis: str
    expected_units_per_second: float | None
    evidence_path: str | None
    evidence_sha256: str | None
    constraints: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {"schema_version": ALLOCATION_SCHEMA, **asdict(self)}


def discover_resources(
    *,
    max_cpus: int | None = None,
    max_memory_gib: float | None = None,
    proc_root: Path = Path("/proc"),
    sys_root: Path = Path("/sys"),
) -> HostResources:
    """Observed resources, optionally narrowed (never widened) by operator overrides.

    ``max_cpus`` keeps the lowest-numbered schedulable CPUs, so a narrowed set stays
    on as few NUMA nodes as possible.
    """

    host = observe_host_resources(proc_root, sys_root)
    usable = host.usable_cpus
    if max_cpus is not None:
        if max_cpus < 1:
            raise ValueError("max_cpus must be positive")
        usable = tuple(sorted(usable)[:max_cpus])
    available, total = host.mem_available_bytes, host.mem_total_bytes
    if max_memory_gib is not None:
        if max_memory_gib <= 0:
            raise ValueError("max_memory_gib must be positive")
        cap = int(max_memory_gib * GIB)
        available = cap if available is None else min(available, cap)
        # The fractional reserve then applies to the capped amount, not the whole host.
        total = cap if total is None else min(total, cap)
    return replace(host, usable_cpus=usable, mem_available_bytes=available, mem_total_bytes=total)


def memory_budget(host: HostResources, policy: ResourcePolicy) -> int:
    if host.mem_available_bytes is None:
        raise SchedulingError("MemAvailable is unobservable; refusing to schedule")
    return max(0, host.mem_available_bytes - policy.reserve_for(host))


def allocate_workload(
    profile: WorkloadProfile,
    host: HostResources,
    policy: ResourcePolicy,
    *,
    pending_units: int | None = None,
    fixed_threads: int | None = None,
) -> WorkloadAllocation:
    """Choose workers and threads for one workload from measured evidence.

    ``fixed_threads``: the run's pinned threads per worker for this (model, workload);
    only the worker count is then chosen.
    """

    cpus = len(host.usable_cpus)
    if cpus < 1:
        raise SchedulingError("no schedulable CPUs")
    budget = memory_budget(host, policy)
    per_worker = math.ceil(profile.peak_resident_bytes * policy.memory_safety_factor)
    memory_workers = budget // per_worker
    if memory_workers < 1:
        raise SchedulingError(
            f"{profile.model}/{profile.workload}: one worker needs {per_worker} B but only "
            f"{budget} B are available after the reserve"
        )
    unit_cap = max(1, pending_units) if pending_units is not None else None
    constraints = {"memory_envelope_workers": int(memory_workers), "cpu_count": cpus,
                   "pending_units": pending_units,
                   "measured_threads_per_worker": profile.measured_threads_per_worker}

    def feasible(processes: int, threads: int) -> bool:
        return (processes <= memory_workers and processes * threads <= cpus
                and threads <= policy.max_threads_per_worker
                # The memory measurement must cover per-thread buffers.
                and threads <= profile.measured_threads_per_worker
                and (unit_cap is None or processes <= unit_cap))

    points = [p for p in profile.benchmark if feasible(p.processes, p.threads_per_process)]
    if fixed_threads is not None:
        if not 1 <= fixed_threads <= profile.measured_threads_per_worker:
            raise SchedulingError(f"pinned {fixed_threads} threads are not covered by the "
                                  f"memory measurement at {profile.measured_threads_per_worker}")
        if fixed_threads > cpus:
            raise SchedulingError(f"pinned {fixed_threads} threads exceed {cpus} usable CPUs")
        threads = fixed_threads
        workers = int(min(memory_workers, cpus // threads, unit_cap or memory_workers))
        match = [p.units_per_second for p in profile.benchmark
                 if (p.processes, p.threads_per_process) == (workers, threads)]
        expected, basis = (match[0] if match else None), "pinned_threads_run_level"
    elif points:
        best = min(points, key=lambda p: (-p.units_per_second, p.processes,
                                          p.threads_per_process))
        workers, threads, expected = best.processes, best.threads_per_process, best.units_per_second
        basis = "measured_throughput_benchmark"
    else:
        workers = int(min(memory_workers, cpus, unit_cap or memory_workers))
        threads = max(1, min(policy.max_threads_per_worker, profile.measured_threads_per_worker,
                             cpus // workers))
        expected, basis = None, "unbenchmarked_floor_cpus_over_workers"
    if workers * threads > cpus or workers * per_worker > budget:
        raise SchedulingError("internal error: allocation exceeds the envelope")
    return WorkloadAllocation(
        model=profile.model, workload=profile.workload, workers=int(workers),
        threads_per_worker=int(threads), planning_memory_per_worker_bytes=per_worker,
        memory_budget_bytes=budget, usable_cpus=cpus, basis=basis,
        expected_units_per_second=expected, evidence_path=profile.evidence_path,
        evidence_sha256=profile.evidence_sha256, constraints=constraints,
    )


def statistics_profile(policy: ResourcePolicy) -> WorkloadProfile:
    """Plain numpy statistics: one thread per worker and a small fixed footprint."""

    return WorkloadProfile(model="statistics", workload=STATISTICS_WORKLOAD,
                           peak_resident_bytes=policy.statistics_worker_memory_bytes,
                           measured_threads_per_worker=1)


def memory_pressure_safe(host: HostResources, policy: ResourcePolicy) -> bool:
    """False when MemAvailable has fallen into the reserve: launch nothing new."""

    return host.mem_available_bytes is not None and host.mem_available_bytes > policy.reserve_for(host)


# ----------------------------------------------------------------- profile files


def _sha256_bytes(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def load_profiles(paths: Iterable[Path], *, strict: bool = True,
                  ambiguous: list[str] | None = None) -> dict[tuple[str, str], WorkloadProfile]:
    """Resource-profile JSON files keyed by (model, workload).

    ``strict``: two profiles for one key raise.  Otherwise the key is dropped (never
    guessed) and reported in ``ambiguous``.
    """

    profiles: dict[tuple[str, str], WorkloadProfile] = {}
    dropped: set[tuple[str, str]] = set()
    for path in sorted(Path(p) for p in paths):
        raw = path.read_bytes()
        report = json.loads(raw)
        if report.get("schema_version") != RESOURCE_PROFILE_SCHEMA:
            continue
        identity, measurement = report["identity"], report["measurement"]
        benchmark = tuple(
            BenchmarkPoint(int(row["processes"]), int(row["threads_per_process"]),
                           float(row["units_per_second"]))
            for row in report.get("benchmark", {}).get("points", [])
            if row.get("status") == "completed"
        )
        profile = WorkloadProfile(
            model=identity["model"], workload=identity["workload"],
            peak_resident_bytes=int(measurement["peak_resident_bytes"]),
            measured_threads_per_worker=int(identity["memory_threads_per_worker"]),
            benchmark=benchmark, evidence_path=str(path), evidence_sha256=_sha256_bytes(raw),
        )
        key = (profile.model, profile.workload)
        if key in profiles or key in dropped:
            if strict:
                raise SchedulingError(f"two resource profiles for {key}: "
                                      f"{profiles[key].evidence_path} and {path}")
            profiles.pop(key, None)
            dropped.add(key)
            continue
        profiles[key] = profile
    if ambiguous is not None:
        ambiguous.extend(f"{model}/{workload}" for model, workload in sorted(dropped))
    return profiles


def predicted_schedule(
    host: HostResources,
    policy: ResourcePolicy,
    profiles: Mapping[tuple[str, str], WorkloadProfile],
    required: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Allocation per required (model, workload); missing evidence is listed, not guessed."""

    allocations, missing, infeasible = {}, [], {}
    for model, workload in required:
        key = f"{model}/{workload}"
        if workload == STATISTICS_WORKLOAD:
            profile = statistics_profile(policy)
        elif (model, workload) in profiles:
            profile = profiles[(model, workload)]
        else:
            missing.append(key)
            continue
        try:
            allocations[key] = allocate_workload(profile, host, policy).as_json()
        except SchedulingError as exc:
            infeasible[key] = str(exc)
    return {"host": host.as_json(), "memory_budget_bytes": memory_budget(host, policy),
            "memory_reserve_bytes": policy.reserve_for(host),
            "allocations": allocations, "missing_profiles": missing, "infeasible": infeasible,
            "numa_policy": "kernel_default_local_allocation_no_binding"}
