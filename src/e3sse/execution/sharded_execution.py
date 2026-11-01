"""Bounded, resumable execution of independent shards with parent-only publication.

Contract:

* A shard is identified by a content-derived ID (``<namespace>-<sha256 prefix>`` of
  its canonical scientific identity), never by worker, chunk or queue position.
* A shard fingerprint binds the run's scientific identity (code, config, model,
  inputs), the shard identity, and the per-process numerical thread configuration.
  Worker count, in-flight bound, memory reserve and NUMA layout are excluded.
* Workers only compute and return JSON-able results; the parent alone writes, each
  checkpoint once (create-new link, never replace), in the configured output root.
* Submission is a sliding window of at most ``plan.max_in_flight`` shards.
* ``resume`` reuses checkpoints whose fingerprint and result digest verify, reruns
  shards with no checkpoint, and refuses a stale checkpoint or a run directory
  belonging to another scientific identity.
* Infrastructure failures (exceptions, dead workers) are never saved as results; a
  run with any missing shard raises after its completed siblings are kept, and no
  merged digest is produced.
* Results are returned in canonical shard-ID order, whatever the completion order.
  Shard-ID order is a hash order: reducers that sum or stack results must order
  them by scientific keys (e.g. ascending identifier) as the protocol specifies.
* A shard is a whole scientific unit.  Never split one frozen RNG stream (e.g. the
  resamples of one bootstrap analysis) across shards, and never let a shard span
  partitions, chemical systems, or calibration and test sides.

Tasks receive ``(payload, context)`` where ``context`` is the per-worker object built
once by ``worker_setup`` (for example a loaded model); tasks derive any randomness
from canonical identifiers in the payload (``e3sse.protocol.partitions.analysis_seed``),
never from worker identity, PID, order or time.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import random
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .resource_planning import ExecutionPlan
from .thread_control import (
    THREAD_ENVIRONMENT_VARIABLES,
    ThreadControlError,
    child_thread_environment,
    configure_process_threads,
    numerical_thread_identity,
)

SHARD_SCHEMA = "e3sse.execution.shard.v1"
RUN_IDENTITY_SCHEMA = "e3sse.execution.run-identity.v1"
MANIFEST_SCHEMA = "e3sse.execution.manifest.v1"
# Spawn gives each worker a fresh interpreter: no inherited torch/OpenMP thread
# pools or locks (fork is unsafe once those exist), identical on every platform.
START_METHOD = "spawn"
NUMA_STRATEGY = "kernel_default_local_allocation_no_binding"
_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9_.]*")
_SHARD_ID = re.compile(r"[a-z0-9][a-z0-9_.]*-[0-9a-f]{20}")


class StaleCheckpointError(RuntimeError):
    """A checkpoint or run directory belongs to a different scientific identity."""


class ArtifactExistsError(FileExistsError):
    """A published artifact already exists; artifacts are never replaced."""


class ShardedExecutionError(RuntimeError):
    def __init__(self, failures: Mapping[str, str], manifest_path: Path) -> None:
        self.failures = dict(failures)
        self.manifest_path = manifest_path
        super().__init__(
            f"{len(failures)} shard(s) not completed; completed shards were kept; "
            f"see {manifest_path}"
        )


def canonical_json_bytes(value: Any) -> bytes:
    # Strict JSON: NaN/inf must be represented explicitly by the task, not silently.
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class WorkShard:
    shard_id: str
    identity: Any
    payload: Any


def make_shard(namespace: str, identity: Any, payload: Any = None) -> WorkShard:
    """Content-derived shard; ``identity`` must be canonical scientific identifiers."""

    if _NAMESPACE.fullmatch(namespace) is None:
        raise ValueError(f"invalid shard namespace {namespace!r}")
    shard_id = f"{namespace}-{canonical_sha256(identity)[:20]}"
    return WorkShard(shard_id, identity, identity if payload is None else payload)


@dataclass(frozen=True)
class ShardedRun:
    """One scientific run: identity, task, optional per-worker setup, output root."""

    output_dir: Path
    scientific_identity: Mapping[str, Any]
    task: Callable[[Any, Any], Any]
    protected_roots: Sequence[Path]
    worker_setup: Callable[..., Any] | None = None
    worker_setup_args: tuple[Any, ...] = field(default_factory=tuple)

    @property
    def run_fingerprint(self) -> str:
        return canonical_sha256(dict(self.scientific_identity))


@dataclass(frozen=True)
class ShardedExecutionResult:
    results: dict[str, Any]
    merged_sha256: str
    manifest_path: Path
    reused: tuple[str, ...]
    computed: tuple[str, ...]
    max_observed_in_flight: int


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def ensure_safe_output_dir(output_dir: Path, protected_roots: Iterable[Path]) -> Path:
    """Generated material never lives in (or contains) protected source data."""

    roots = [Path(root) for root in protected_roots]
    if not roots:
        raise ValueError("protected_roots must name the immutable source-data roots")
    resolved = Path(output_dir).resolve()
    for root in roots:
        if _within(resolved, root) or _within(root, resolved):
            raise PermissionError(f"output {resolved} overlaps protected data root {root}")
    return resolved


def publish_json_once(path: Path, value: Any, *, root: Path) -> None:
    """Atomically create ``path`` (fsynced temp file + hard link); never replace."""

    path = Path(path)
    if not _within(path.parent, root):
        raise PermissionError(f"{path} is outside the run output directory {root}")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ArtifactExistsError(f"refusing to replace existing artifact: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


# Per-worker state, built once by the initializer (e.g. a loaded model).
_WORKER_CONTEXT: Any = None


def _initialize_worker(
    threads_per_worker: int,
    configure_torch: bool,
    setup: Callable[..., Any] | None,
    setup_args: tuple[Any, ...],
) -> None:
    global _WORKER_CONTEXT
    configure_process_threads(threads_per_worker, configure_torch=configure_torch)
    # Imported before any task so per-shard global-RNG resets cover every shard,
    # not only those after a task's first import (which depends on scheduling).
    import numpy  # noqa: F401
    _WORKER_CONTEXT = setup(*setup_args) if setup is not None else None


def _reset_global_random_state(shard_id: str) -> None:
    """Make any (discouraged) process-global RNG use depend only on the shard.

    Without this, global draws would depend on which shards a worker ran earlier and
    hence on worker count.  Tasks should still use analysis_seed-keyed generators.
    """

    seed = int(shard_id.rsplit("-", 1)[1][:8], 16)
    random.seed(seed)
    if "numpy" in sys.modules:
        sys.modules["numpy"].random.seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def _run_task(task: Callable[[Any, Any], Any], shard_id: str, payload: Any) -> Any:
    _reset_global_random_state(shard_id)
    return task(payload, _WORKER_CONTEXT)


def _shard_fingerprint(run_fingerprint: str, shard: WorkShard, threads: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": SHARD_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "shard_id": shard.shard_id,
            "shard_identity": shard.identity,
            # The task may draw on payload fields, so they bind the checkpoint too.
            "payload_sha256": canonical_sha256(shard.payload),
            "numerical_threads": threads,
        }
    )


def _load_verified(path: Path, fingerprint: str) -> Any:
    """The stored result, or raise if the checkpoint is stale or corrupt."""

    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Published checkpoints are complete by construction (create-new link), so
        # an unreadable one needs inspection rather than silent replacement.
        raise StaleCheckpointError(f"unreadable checkpoint {path}: {exc}") from exc
    if stored.get("schema_version") != SHARD_SCHEMA or stored.get("shard_fingerprint") != fingerprint:
        raise StaleCheckpointError(f"stale shard checkpoint (fingerprint changed): {path}")
    if canonical_sha256(stored.get("result")) != stored.get("result_sha256"):
        raise StaleCheckpointError(f"shard checkpoint result digest mismatch: {path}")
    return stored["result"]


def _shard_path(shard_dir: Path, shard_id: str) -> Path:
    # Two-hex fan-out keeps directories small for very large shard sets.
    return shard_dir / shard_id.rsplit("-", 1)[1][:2] / f"{shard_id}.json"


@contextmanager
def _exclusive_run_lock(output_dir: Path) -> Iterator[None]:
    """One orchestrator per run directory; a second concurrent resume is refused."""

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_dir / ".run.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process is executing {output_dir}") from exc
        yield
    finally:
        os.close(descriptor)


def _bind_run_directory(
    run: ShardedRun, output_dir: Path, shard_ids: Sequence[str], *, resume: bool
) -> None:
    identity_path = output_dir / "run_identity.json"
    record = {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "run_fingerprint": run.run_fingerprint,
        "scientific_identity": dict(run.scientific_identity),
        # A merged digest over a different shard set would look complete.
        "shard_set_sha256": canonical_sha256(sorted(shard_ids)),
        "shard_count": len(shard_ids),
    }
    if identity_path.exists():
        if not resume:
            raise ArtifactExistsError(f"{output_dir} already holds a run; use --resume")
        stored = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored.get("run_fingerprint") != run.run_fingerprint:
            raise StaleCheckpointError(f"{output_dir} belongs to a different scientific run")
        if stored.get("shard_set_sha256") != record["shard_set_sha256"]:
            raise StaleCheckpointError(f"{output_dir} was started with a different shard set")
        return
    publish_json_once(identity_path, record, root=output_dir)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def execute_shards(
    run: ShardedRun,
    shards: Sequence[WorkShard],
    plan: ExecutionPlan,
    *,
    resume: bool = False,
    executor: str = "process",
    progress: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
) -> ShardedExecutionResult:
    """Run every shard once under ``plan``; see the module docstring for the contract."""

    if executor not in {"process", "in_process"}:
        raise ValueError("executor must be 'process' or 'in_process'")
    if executor == "in_process":
        if plan.workers != 1:
            raise ValueError("in_process execution is serial: plan.workers must be 1")
        # Libraries already loaded here read thread counts at start-up, so the
        # interpreter must have been launched with the planned values.
        require_thread_environment_at_start(plan.threads_per_worker)
    output_dir = ensure_safe_output_dir(run.output_dir, run.protected_roots)
    ordered = sorted(shards, key=lambda shard: shard.shard_id)
    ids = [shard.shard_id for shard in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate shard IDs")
    for shard in ordered:
        namespace = shard.shard_id.rsplit("-", 1)[0]
        if (
            _SHARD_ID.fullmatch(shard.shard_id) is None
            or make_shard(namespace, shard.identity).shard_id != shard.shard_id
        ):
            raise ValueError(f"shard ID is not derived from its identity: {shard.shard_id}")
        canonical_json_bytes(shard.payload)
    identity = run.scientific_identity
    if "code_sha256" not in identity and not (
        "git_sha" in identity and identity.get("git_dirty") is False
    ):
        # An edited task under an unchanged identity would silently reuse old results.
        raise ValueError("scientific_identity must include code_sha256, or git_sha with "
                         "git_dirty false")

    with _exclusive_run_lock(output_dir):
        return _execute_locked(run, ordered, plan, output_dir, resume, executor, progress)


def _execute_locked(
    run: ShardedRun,
    ordered: Sequence[WorkShard],
    plan: ExecutionPlan,
    output_dir: Path,
    resume: bool,
    executor: str,
    progress: Callable[[str], None],
) -> ShardedExecutionResult:
    ids = [shard.shard_id for shard in ordered]
    started = _utc_now()
    _bind_run_directory(run, output_dir, ids, resume=resume)
    threads = numerical_thread_identity(plan.threads_per_worker, configure_torch=plan.configure_torch)
    shard_dir = output_dir / "shards"
    fingerprints = {s.shard_id: _shard_fingerprint(run.run_fingerprint, s, threads) for s in ordered}

    results: dict[str, Any] = {}
    reused: list[str] = []
    pending: list[WorkShard] = []
    rejected: list[str] = []
    for shard in ordered:
        path = _shard_path(shard_dir, shard.shard_id)
        if path.exists():
            if not resume:
                raise ArtifactExistsError(f"{path} exists; use --resume")
            try:
                results[shard.shard_id] = _load_verified(path, fingerprints[shard.shard_id])
            except StaleCheckpointError as exc:
                rejected.append(str(exc))
                continue
            reused.append(shard.shard_id)
        else:
            pending.append(shard)
    if rejected:
        raise StaleCheckpointError(
            f"{len(rejected)} checkpoint(s) rejected (numerical threads now {threads}); "
            f"first: {rejected[0]}"
        )

    failures: dict[str, str] = {}
    computed: list[str] = []
    timings: dict[str, float] = {}

    def settle(shard: WorkShard, outcome: Any, elapsed: float) -> None:
        try:
            record = {
                "schema_version": SHARD_SCHEMA,
                "shard_id": shard.shard_id,
                "shard_fingerprint": fingerprints[shard.shard_id],
                "result": outcome,
                "result_sha256": canonical_sha256(outcome),
            }
            publish_json_once(_shard_path(shard_dir, shard.shard_id), record, root=output_dir)
        except Exception as exc:  # e.g. non-JSON result or disk full: not a result
            failures[shard.shard_id] = f"{type(exc).__name__}: {exc}"
            progress(f"{shard.shard_id}: FAILED {failures[shard.shard_id]}")
            return
        results[shard.shard_id] = outcome
        computed.append(shard.shard_id)
        timings[shard.shard_id] = round(elapsed, 3)

    max_in_flight_observed = 0
    with child_thread_environment(plan.threads_per_worker) as environment:
        if executor == "in_process":
            _initialize_worker(
                plan.threads_per_worker, plan.configure_torch, run.worker_setup, run.worker_setup_args
            )
            for shard in pending:
                begin = time.perf_counter()
                max_in_flight_observed = 1
                try:
                    outcome = _run_task(run.task, shard.shard_id, shard.payload)
                except Exception as exc:
                    failures[shard.shard_id] = f"{type(exc).__name__}: {exc}"
                    progress(f"{shard.shard_id}: FAILED {failures[shard.shard_id]}")
                    continue
                settle(shard, outcome, time.perf_counter() - begin)
        elif pending:
            context = multiprocessing.get_context(START_METHOD)
            pool = ProcessPoolExecutor(
                max_workers=plan.workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(
                    plan.threads_per_worker,
                    plan.configure_torch,
                    run.worker_setup,
                    run.worker_setup_args,
                ),
            )
            try:
                queue = iter(pending)
                in_flight: dict[Future, tuple[WorkShard, float]] = {}
                broken = False
                while True:
                    while not broken and len(in_flight) < plan.max_in_flight:
                        shard = next(queue, None)
                        if shard is None:
                            break
                        try:
                            future = pool.submit(_run_task, run.task, shard.shard_id, shard.payload)
                        except BrokenProcessPool as exc:
                            broken = True
                            failures[shard.shard_id] = f"not run: {exc}"
                            break
                        in_flight[future] = (shard, time.perf_counter())
                    max_in_flight_observed = max(max_in_flight_observed, len(in_flight))
                    if not in_flight:
                        break
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda item: in_flight[item][0].shard_id):
                        shard, begin = in_flight.pop(future)
                        try:
                            outcome = future.result()
                        except BrokenProcessPool:
                            # A dead worker (e.g. OOM kill) breaks every pending future;
                            # the culprit cannot be identified, so none is blamed.
                            broken = True
                            failures[shard.shard_id] = "not completed: worker pool broken"
                        except Exception as exc:
                            failures[shard.shard_id] = f"{type(exc).__name__}: {exc}"
                            progress(f"{shard.shard_id}: FAILED {failures[shard.shard_id]}")
                        else:
                            settle(shard, outcome, time.perf_counter() - begin)
                for shard in queue:
                    failures[shard.shard_id] = "not run: worker pool broken"
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

    complete = not failures and set(results) == set(ids)
    canonical = {shard_id: results[shard_id] for shard_id in ids if shard_id in results}
    merged = (
        canonical_sha256(
            {
                "run_fingerprint": run.run_fingerprint,
                "shards": [
                    {"shard_id": shard_id, "result_sha256": canonical_sha256(canonical[shard_id])}
                    for shard_id in ids
                ],
            }
        )
        if complete
        else None
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "classification": "operational_execution_metadata",
        "run_fingerprint": run.run_fingerprint,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "command": list(sys.argv),
        "python_version": sys.version.split()[0],
        "executor": executor,
        "start_method": START_METHOD if executor == "process" else None,
        "plan": plan.as_json(),
        "numerical_threads": threads,
        "child_thread_environment": environment,
        "numa_strategy": NUMA_STRATEGY,
        "tmpdir": os.environ.get("TMPDIR"),
        "max_observed_in_flight": max_in_flight_observed,
        "shard_counts": {
            "requested": len(ids),
            "reused": len(reused),
            "computed": len(computed),
            "failed_or_not_run": len(failures),
        },
        "failures": dict(sorted(failures.items())),
        "shard_elapsed_seconds": dict(sorted(timings.items())),
        "merged_sha256": merged,
        "excluded_from_scientific_identity": [
            "workers",
            "max_in_flight",
            "memory_reserve_bytes",
            "per_worker_memory_bytes",
            "host",
            "numa_strategy",
            "timestamps",
        ],
    }
    manifest_path = (
        output_dir
        / "execution_manifests"
        / f"{started.replace(':', '').replace('-', '')}-{os.getpid()}.json"
    )
    publish_json_once(manifest_path, manifest, root=output_dir)
    if not complete:
        raise ShardedExecutionError(failures, manifest_path)
    return ShardedExecutionResult(
        results=canonical,
        merged_sha256=merged,
        manifest_path=manifest_path,
        reused=tuple(reused),
        computed=tuple(sorted(computed)),
        max_observed_in_flight=max_in_flight_observed,
    )


def require_thread_environment_at_start(threads_per_worker: int) -> None:
    """Fail early when a serial in-process run was launched without the thread variables."""

    expected = str(threads_per_worker)
    actual = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES}
    if any(value != expected for value in actual.values()):
        raise ThreadControlError(
            f"launch with {', '.join(f'{name}={expected}' for name in THREAD_ENVIRONMENT_VARIABLES)} "
            "for in-process execution, or use the process executor"
        )
