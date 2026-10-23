"""Observe host resources and plan a bounded worker count for one workload.

Memory, not CPU count, bounds MLIP inference on the production host, so ``auto``
requires a measured per-worker peak resident memory (see
``e3sse.execution.worker_memory_probe``) and never guesses it.  An explicit worker
count is still checked against the CPU and memory envelopes and refused rather than
oversubscribed.  Plans are operational metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

GIB = 1024**3
DEFAULT_MEMORY_RESERVE_BYTES = 2 * GIB
WORKER_MEMORY_REPORT_SCHEMA = "e3sse.execution.worker-memory.v1"


class InsufficientResourcesError(RuntimeError):
    """The requested execution cannot fit the observed CPU or memory envelope."""


@dataclass(frozen=True)
class HostResources:
    """A point-in-time observation; available memory changes during a run."""

    logical_cpus: int | None
    usable_cpus: tuple[int, ...]
    mem_total_bytes: int | None
    mem_available_bytes: int | None
    swap_free_bytes: int | None
    numa_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["usable_cpus"] = _cpu_ranges(self.usable_cpus)
        payload["usable_cpu_count"] = len(self.usable_cpus)
        return payload


@dataclass(frozen=True)
class WorkloadPolicy:
    """Per-workload execution policy; MLIP and statistical workloads differ."""

    name: str
    threads_per_worker: int = 1
    per_worker_memory_bytes: int | None = None
    memory_reserve_bytes: int = DEFAULT_MEMORY_RESERVE_BYTES
    in_flight_per_worker: int = 2
    configure_torch: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("workload policy needs a name")
        for name in ("threads_per_worker", "in_flight_per_worker"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.per_worker_memory_bytes is not None and self.per_worker_memory_bytes <= 0:
            raise ValueError("per_worker_memory_bytes must be positive")
        if self.memory_reserve_bytes < 0:
            raise ValueError("memory_reserve_bytes must be non-negative")


@dataclass(frozen=True)
class ExecutionPlan:
    workload: str
    requested_workers: int | str
    workers: int
    threads_per_worker: int
    max_in_flight: int
    cpu_envelope_workers: int
    memory_envelope_workers: int | None
    per_worker_memory_bytes: int | None
    memory_reserve_bytes: int
    configure_torch: bool
    host: HostResources

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["host"] = self.host.as_json()
        return payload


def _parse_cpu_list(text: str) -> tuple[int, ...]:
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return tuple(sorted(set(cpus)))


def _cpu_ranges(cpus: tuple[int, ...]) -> str:
    ranges: list[str] = []
    for cpu in sorted(cpus):
        if ranges and cpu == int(ranges[-1].split("-")[-1]) + 1:
            ranges[-1] = f"{ranges[-1].split('-')[0]}-{cpu}"
        else:
            ranges.append(str(cpu))
    return ",".join(ranges)


def _meminfo_bytes(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError:
        return values
    for line in lines:
        # Per-node meminfo lines carry a "Node N " prefix before the key.
        key, _, rest = line.rpartition(":")
        fields = rest.split()
        if fields and fields[0].isdigit():
            scale = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
            values[key.split()[-1]] = int(fields[0]) * scale
    return values


def observe_host_resources(
    proc_root: Path = Path("/proc"), sys_root: Path = Path("/sys")
) -> HostResources:
    """Usable CPUs (affinity mask), MemAvailable, swap and NUMA topology."""

    if hasattr(os, "sched_getaffinity"):
        usable = tuple(sorted(os.sched_getaffinity(0)))
    else:  # pragma: no cover - non-Linux fallback
        usable = tuple(range(os.cpu_count() or 1))
    meminfo = _meminfo_bytes(proc_root / "meminfo")
    nodes: dict[str, dict[str, Any]] = {}
    node_root = sys_root / "devices" / "system" / "node"
    for node in sorted(node_root.glob("node[0-9]*")):
        try:
            cpus = _parse_cpu_list((node / "cpulist").read_text(encoding="ascii"))
        except (OSError, ValueError):
            continue
        node_memory = _meminfo_bytes(node / "meminfo")
        nodes[node.name] = {
            "cpus": _cpu_ranges(cpus),
            "mem_total_bytes": node_memory.get("MemTotal"),
            "mem_free_bytes": node_memory.get("MemFree"),
        }
    return HostResources(
        logical_cpus=os.cpu_count(),
        usable_cpus=usable,
        mem_total_bytes=meminfo.get("MemTotal"),
        mem_available_bytes=meminfo.get("MemAvailable"),
        swap_free_bytes=meminfo.get("SwapFree"),
        numa_nodes=nodes,
    )


def parse_workers(value: str | int) -> int | str:
    """``--workers`` accepts a positive integer or ``auto``."""

    if value == "auto":
        return "auto"
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("workers must be a positive integer or 'auto'") from exc
    if number < 1 or isinstance(value, bool):
        raise argparse.ArgumentTypeError("workers must be a positive integer or 'auto'")
    return number


def plan_workers(
    requested: int | str,
    policy: WorkloadPolicy,
    host: HostResources,
    *,
    unit_count: int | None = None,
) -> ExecutionPlan:
    """Deterministic worker count from the observed envelopes; refuse oversubscription.

    workers * threads_per_worker never exceeds the usable CPUs, and when the
    per-worker peak memory is known, workers * peak never exceeds
    MemAvailable - reserve (swap is never counted as capacity).
    """

    requested = parse_workers(requested)
    cpu_envelope = len(host.usable_cpus) // policy.threads_per_worker
    if cpu_envelope < 1:
        raise InsufficientResourcesError(
            f"{policy.threads_per_worker} threads per worker exceed "
            f"{len(host.usable_cpus)} usable CPUs"
        )
    memory_envelope = None
    if policy.per_worker_memory_bytes is not None:
        if host.mem_available_bytes is None:
            raise InsufficientResourcesError("MemAvailable is unobservable; cannot plan memory")
        budget = host.mem_available_bytes - policy.memory_reserve_bytes
        memory_envelope = max(0, budget // policy.per_worker_memory_bytes)
    if requested == "auto":
        if memory_envelope is None:
            raise InsufficientResourcesError(
                "--workers auto needs a measured per-worker memory (worker-memory report); "
                "give an explicit --workers instead of guessing"
            )
        workers = min(cpu_envelope, memory_envelope)
    else:
        if requested > cpu_envelope:
            raise InsufficientResourcesError(
                f"{requested} workers x {policy.threads_per_worker} threads exceed "
                f"{len(host.usable_cpus)} usable CPUs"
            )
        if memory_envelope is not None and requested > memory_envelope:
            raise InsufficientResourcesError(
                f"{requested} workers exceed the memory envelope of {memory_envelope} "
                f"(MemAvailable minus reserve over the measured per-worker peak)"
            )
        workers = requested
    if unit_count is not None:
        # Idle processes would hold memory without doing work.
        workers = min(workers, max(1, unit_count))
    if workers < 1:
        raise InsufficientResourcesError(
            "not even one worker fits MemAvailable minus the memory reserve"
        )
    return ExecutionPlan(
        workload=policy.name,
        requested_workers=requested,
        workers=workers,
        threads_per_worker=policy.threads_per_worker,
        max_in_flight=workers * policy.in_flight_per_worker,
        cpu_envelope_workers=cpu_envelope,
        memory_envelope_workers=memory_envelope,
        per_worker_memory_bytes=policy.per_worker_memory_bytes,
        memory_reserve_bytes=policy.memory_reserve_bytes,
        configure_torch=policy.configure_torch,
        host=host,
    )


def per_worker_memory_from_report(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    threads_per_worker: int,
    safety_factor: float = 1.5,
) -> int:
    """Planning memory from a worker-memory probe report for the same model and threads."""

    if safety_factor < 1.0:
        raise ValueError("safety_factor must be at least 1")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema_version") != WORKER_MEMORY_REPORT_SCHEMA:
        raise ValueError("not a worker-memory probe report")
    identity = report.get("identity", {})
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if report.get("fingerprint_sha256") != digest:
        raise ValueError("worker-memory report fingerprint does not match its identity")
    mismatched = sorted(k for k, v in expected_identity.items() if identity.get(k) != v)
    if mismatched:
        raise ValueError(f"worker-memory report identity differs: {mismatched}")
    if identity.get("threads_per_worker") != threads_per_worker:
        raise ValueError("worker-memory report was measured with a different thread count")
    peak = report.get("measurement", {}).get("peak_resident_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise ValueError("worker-memory report lacks a positive peak_resident_bytes")
    return math.ceil(peak * safety_factor)


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Operational CLI options shared by parallel production scripts."""

    group = parser.add_argument_group("execution (operational; never scientific identity)")
    group.add_argument("--workers", type=parse_workers, default=1, help="positive integer or auto")
    group.add_argument("--threads-per-worker", type=int, default=1)
    group.add_argument(
        "--worker-memory-report",
        type=Path,
        default=None,
        help="worker-memory probe JSON for this model and thread count (required for auto)",
    )
    group.add_argument("--memory-reserve-gib", type=float, default=DEFAULT_MEMORY_RESERVE_BYTES / GIB)
    group.add_argument("--resume", action="store_true")
