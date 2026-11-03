"""Thin, resumable production orchestrator over the tested E3-SSE stage runners.

Layout (``<outputs_root>/<production_subdirectory>/<run_id>/``)::

    run_identity.json              scientific identity (create-once; defines run_id)
    <stage>/stage_completion.json  create-once record: outputs and their SHA-256,
                                   dependency records, resource allocation
    <stage>/...                    stage-local artifacts (preflight report, freeze
                                   binding, derived configuration)
    logs/<stage>/<utc>-<pid>-<n>.log   combined output of every child command
    invocations/<utc>-<pid>.json   operational record of each orchestrator call

Semantics:

* ``--stage X`` runs X and every dependency that has no completion record yet;
  ``--stage all`` runs the whole graph in topological order.  A stage that cannot
  complete stops its dependents (``blocked_by_dependency``).  Independent stages
  still run.
* ``--resume`` reuses a completion record only after verifying the run ID, the
  stage definition, the dependency records and every output digest.  A stale or
  corrupt record is refused (``StaleStageError``), never overwritten.  Without
  ``--resume`` an existing run directory is refused (create-new semantics).
* Scientific computation happens only in child commands, which carry their own
  shard-level resume (``e3sse.execution``) and outcome seals.  The orchestrator
  enforces stage-level outcome boundaries: a stage that reads an outcome runs only
  after a verified, frozen internal-freeze binding exists.
* Failure classes stay distinct: missing evidence, missing runner, deferred,
  validation failure, infrastructure failure.  A scientific unit failure (e.g. a
  non-converged ML-NEB) is a recorded status inside a stage's results, never a
  stage failure, and is never retried.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from e3sse.execution.resource_planning import GIB
from e3sse.execution.sharded_execution import ArtifactExistsError, publish_json_once
from e3sse.execution.thread_control import THREAD_ENVIRONMENT_VARIABLES
from e3sse.production import preflight as checks
from e3sse.production.model_runtimes import (
    ModelRuntime,
    assert_no_package_installation,
    frozen_models,
    model_runtimes,
    runtime_environment,
)
from e3sse.production.resource_scheduler import (
    ResourcePolicy,
    SchedulingError,
    WorkloadAllocation,
    allocate_workload,
    discover_resources,
    load_profiles,
    memory_pressure_safe,
    predicted_schedule,
    statistics_profile,
)
from e3sse.production.run_identity import (
    build_scientific_identity,
    canonical_sha256,
    file_sha256,
    git_state,
    operational_identity,
    project_environment_sha256,
    run_id,
)
from e3sse.production.stage_graph import (
    FREEZE_STAGE,
    ProductionStage,
    StageGraph,
    build_stage_graph,
)
from e3sse.protocol.contracts import STAGE_OUTCOME_ACCESS_ENV

PRODUCTION_CONFIG_SCHEMA = "e3sse.production-execution.config"
COMPLETION_SCHEMA = "e3sse.production.stage-completion"
INVOCATION_SCHEMA = "e3sse.production.invocation"
FREEZE_BINDING_FILE = "internal_freeze_binding.json"

COMPLETED = "completed"
REUSED = "reused"
BLOCKED_MISSING_EVIDENCE = "blocked_missing_evidence"
BLOCKED_MISSING_RUNNER = "blocked_missing_runner"
BLOCKED_BY_DEPENDENCY = "blocked_by_dependency"
DEFERRED = "deferred"
VALIDATION_FAILURE = "validation_failure"
INFRASTRUCTURE_FAILURE = "infrastructure_failure"
SUCCESS = (COMPLETED, REUSED)
EXIT_CODES = {"success": 0, "blocked": 2, VALIDATION_FAILURE: 3, INFRASTRUCTURE_FAILURE: 4}


class StaleStageError(RuntimeError):
    """A completion record belongs to a different identity or its outputs changed."""


class StageBlocked(Exception):
    """A stage could not complete; ``outcome`` says why (never a scientific result)."""

    def __init__(self, outcome: str, detail: str) -> None:
        super().__init__(detail)
        self.outcome, self.detail = outcome, detail


class ChildCommandFailed(StageBlocked):
    def __init__(self, command: Sequence[str], returncode: int, log: Path,
                 outcome: str = INFRASTRUCTURE_FAILURE) -> None:
        super().__init__(outcome, f"exit {returncode}: {' '.join(command)} (log {log})")
        self.returncode = returncode


@dataclass
class StageResult:
    stage: str
    outcome: str
    detail: str
    outputs: dict[str, dict[str, str]] = field(default_factory=dict)
    commands: list[list[str]] = field(default_factory=list)
    allocations: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {"stage": self.stage, "outcome": self.outcome, "detail": self.detail,
                "outputs": self.outputs, "commands": self.commands,
                "allocations": self.allocations}


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _compact_utc() -> str:
    return _utc().replace(":", "").replace("-", "")


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def output_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


# ===================================================================== context


@dataclass
class RunContext:
    """Resolved configuration, identities and paths for one orchestrator call."""

    repository: Path
    config_path: Path
    config: dict[str, Any]
    graph: StageGraph
    active_protocol_path: Path
    active_protocol: dict[str, Any]
    freeze_config_path: Path
    freeze_config: dict[str, Any]
    study1_config_path: Path
    server_config_path: Path
    server_config: dict[str, Any]
    data_root: Path
    outputs_root: Path
    evidence_root: Path
    run_root: Path
    git: dict[str, Any]
    scientific_identity: dict[str, Any]
    run_id: str
    policy: ResourcePolicy
    runtimes: dict[str, ModelRuntime]
    models: dict[str, dict[str, Any]]
    project_python: Path
    max_cpus: int | None = None
    max_memory_gib: float | None = None
    resume: bool = False
    dry_run: bool = False
    progress: Callable[[str], None] = lambda message: print(message, file=sys.stderr)
    _command_counter: int = 0

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_id

    def stage_dir(self, stage: str) -> Path:
        return self.run_dir / stage

    def evidence(self, key: str) -> Path:
        """Evidence locations always lie under outputs_root (never data, never elsewhere)."""

        path = _resolve(self.outputs_root, self.config["evidence"][key])
        try:
            path.relative_to(self.outputs_root)
        except ValueError as exc:
            raise ValueError(f"evidence path {key} = {path} is outside {self.outputs_root}") from exc
        return path

    def write_roots(self) -> list[Path]:
        """Every root this run writes to, for the preflight safety check."""

        roots = [self.run_root, self.evidence("ml_neb_runtime_audit_dir"),
                 self.evidence("resource_profile_dir"), self.evidence("gate1_population_audit_dir")]
        study1 = _load_json(self.study1_config_path)
        roots.append(_resolve(self.study1_config_path.parent, study1["output_root"]))
        if self.config["runtime"].get("tmpdir"):
            roots.append(_resolve(self.outputs_root, self.config["runtime"]["tmpdir"]))
        return roots

    def host(self):
        return discover_resources(max_cpus=self.max_cpus, max_memory_gib=self.max_memory_gib)

    def stage_outputs(self, stage: str) -> dict[str, dict[str, str]] | None:
        """Outputs of a verified completion record of this run, or None."""

        if stage not in self.graph.stages:
            return None
        record = verified_completion(self, self.graph.stages[stage])
        return None if record is None else record["outputs"]

    def profiles(self, ambiguous: list[str] | None = None):
        """This run's measured profiles (from its resource_profile record when complete).

        Before that record exists (e.g. a dry run), every profile in the evidence
        directory is read; a (model, workload) with several profiles is dropped and
        listed in ``ambiguous``, never guessed.
        """

        outputs = self.stage_outputs("resource_profile")
        if outputs is not None:
            paths = [Path(o["path"]) for k, o in outputs.items() if k.startswith("resource_profile:")]
            return load_profiles(paths)
        directory = self.evidence("resource_profile_dir")
        if not directory.is_dir():
            return {}
        return load_profiles(directory.glob("*/*.json"), strict=False, ambiguous=ambiguous)

    def checkpoint_map(self) -> Path:
        return Path(self.config["runtime"]["checkpoint_map"]).expanduser()

    def checkpoint(self, model: str) -> Path:
        from e3sse.production.checkpoints import CheckpointInventoryError, load_inventory

        try:
            return load_inventory(self.checkpoint_map(), self.models)[model]
        except CheckpointInventoryError as exc:
            raise StageBlocked(BLOCKED_MISSING_EVIDENCE, str(exc)) from exc

    def child_environment(self, runtime: ModelRuntime | None, threads: int) -> dict[str, str]:
        env = runtime_environment(runtime, self.repository)
        env.update({name: str(threads) for name in THREAD_ENVIRONMENT_VARIABLES})
        tmpdir = self.config["runtime"].get("tmpdir")
        if tmpdir:
            from e3sse.execution.sharded_execution import ensure_safe_output_dir

            # Python silently falls back to RAM-backed /tmp when TMPDIR does not exist.
            path = ensure_safe_output_dir(_resolve(self.outputs_root, tmpdir), [self.data_root])
            path.mkdir(parents=True, exist_ok=True)
            env["TMPDIR"] = str(path)
        return env

    def run_command(self, stage: str, command: Sequence[str], env: Mapping[str, str], *,
                    accept_returncodes: Sequence[int] = (0,),
                    failure_outcomes: Mapping[int, str] | None = None) -> tuple[int, str]:
        """Run one child command, streaming its output to the stage log.

        The child inherits the stage's permitted outcome kinds
        (``E3SSE_STAGE_OUTCOME_ACCESS``), which the outcome contract enforces.  A
        non-accepted exit code maps to ``failure_outcomes`` (default: infrastructure).
        """

        command = [str(part) for part in command]
        env = dict(env)
        env[STAGE_OUTCOME_ACCESS_ENV] = json.dumps(sorted(self.graph.stages[stage].outcome_access))
        assert_no_package_installation(command)
        self.wait_for_memory(stage)
        self._command_counter += 1
        log = self.run_dir / "logs" / stage / f"{_compact_utc()}-{os.getpid()}-{self._command_counter}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.progress(f"[{stage}] {' '.join(command)}")
        lines: list[str] = []
        with log.open("x", encoding="utf-8") as handle:
            handle.write(f"# {_utc()} {json.dumps(command)}\n")
            process = subprocess.Popen(command, env=dict(env), cwd=self.repository,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                lines.append(line)
                sys.stderr.write(line)
            returncode = process.wait()
            handle.write(f"# exit {returncode} at {_utc()}\n")
        if returncode not in accept_returncodes:
            outcome = (failure_outcomes or {}).get(returncode, INFRASTRUCTURE_FAILURE)
            raise ChildCommandFailed(command, returncode, log, outcome)
        return returncode, "".join(lines)

    def wait_for_memory(self, stage: str) -> None:
        """Launch nothing while MemAvailable is inside the reserve (no OOM by design)."""

        limit = float(self.config["resource_policy"].get("memory_pressure_wait_seconds", 600))
        waited = 0.0
        while not memory_pressure_safe(self.host(), self.policy):
            if waited >= limit:
                raise StageBlocked(INFRASTRUCTURE_FAILURE,
                                   f"{stage}: MemAvailable stayed inside the reserve for "
                                   f"{limit:.0f} s; nothing new was launched")
            time.sleep(10)
            waited += 10


def load_context(repository: Path, config_path: Path, *, adapters: Mapping[str, Any],
                 max_cpus: int | None = None, max_memory_gib: float | None = None,
                 resume: bool = False, dry_run: bool = False,
                 project_python: Path | None = None) -> RunContext:
    from e3sse.barriers.ml_neb import settings_from_protocol

    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != PRODUCTION_CONFIG_SCHEMA:
        raise ValueError("unsupported production configuration")
    base = config_path.parent
    bindings = config["scientific_bindings"]
    paths = {key: _resolve(base, value) for key, value in bindings.items()}
    active = _load_json(paths["active_protocol_config"])
    freeze_config = _load_json(paths["protocol_freeze_config"])
    from e3sse.config import load_config

    # Resolves relative_to and refuses an output root under the data root.
    server = load_config(paths["server_config"])
    data_root = Path(server["paths"]["data_root"])
    outputs_root = Path(server["paths"]["outputs_root"])
    graph = build_stage_graph(config["stages"], known_adapters=adapters)
    git = git_state(repository)
    identity = build_scientific_identity(
        git=git, production_config=config, bound_files=paths, active_protocol=active,
        freeze_config=freeze_config,
        ml_neb_settings_sha256=settings_from_protocol(active).settings_sha256,
        project_environment_sha256=project_environment_sha256(),
    )
    runtimes = model_runtimes(freeze_config, Path(config["runtime"]["model_runtime_root"]))
    return RunContext(
        repository=repository, config_path=config_path, config=config, graph=graph,
        active_protocol_path=paths["active_protocol_config"], active_protocol=active,
        freeze_config_path=paths["protocol_freeze_config"], freeze_config=freeze_config,
        study1_config_path=paths["study1_config"], server_config_path=paths["server_config"],
        server_config=server, data_root=data_root, outputs_root=outputs_root,
        evidence_root=outputs_root,
        run_root=outputs_root / config["output"]["production_subdirectory"],
        git=git, scientific_identity=identity, run_id=run_id(identity),
        policy=ResourcePolicy.from_config(config["resource_policy"]), runtimes=runtimes,
        models=frozen_models(freeze_config),
        project_python=Path(project_python or sys.executable),
        max_cpus=max_cpus, max_memory_gib=max_memory_gib, resume=resume, dry_run=dry_run,
    )


# ===================================================================== completion


def _stage_definition_sha(stage: ProductionStage) -> str:
    return canonical_sha256(stage.definition())


def completion_path(ctx: RunContext, stage: str) -> Path:
    return ctx.stage_dir(stage) / "stage_completion.json"


def verified_completion(ctx: RunContext, stage: ProductionStage) -> dict[str, Any] | None:
    """The stage's completion record if present and valid; raise if stale or corrupt."""

    path = completion_path(ctx, stage.name)
    if not path.is_file():
        return None
    try:
        record = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise StaleStageError(f"unreadable completion record {path}: {exc}") from exc
    if not isinstance(record, dict) or not isinstance(record.get("outputs"), dict) or \
            not isinstance(record.get("dependency_completion_sha256"), dict):
        raise StaleStageError(f"malformed completion record {path}")
    if record.get("schema_version") != COMPLETION_SCHEMA or record.get("run_id") != ctx.run_id:
        raise StaleStageError(f"{path} belongs to another run")
    if record.get("stage_definition_sha256") != _stage_definition_sha(stage):
        raise StaleStageError(f"{path}: the stage definition changed since completion")
    for dependency in stage.depends_on:
        dep_path = completion_path(ctx, dependency)
        if not dep_path.is_file() or record["dependency_completion_sha256"].get(
                dependency) != file_sha256(dep_path):
            raise StaleStageError(f"{path}: dependency {dependency} changed since completion")
    for name, output in record["outputs"].items():
        if not isinstance(output, dict) or not {"path", "sha256"} <= set(output):
            raise StaleStageError(f"{path}: malformed output entry {name}")
        target = Path(output["path"])
        if not target.is_file():
            raise StaleStageError(f"{path}: output {name} ({target}) is missing")
        if file_sha256(target) != output["sha256"]:
            raise StaleStageError(f"{path}: output {name} ({target}) changed")
    return record


def publish_completion(ctx: RunContext, stage: ProductionStage, result: StageResult) -> Path:
    record = {
        "schema_version": COMPLETION_SCHEMA,
        "run_id": ctx.run_id,
        "stage": stage.name,
        "public_name": stage.public_name,
        "stage_definition_sha256": _stage_definition_sha(stage),
        "dependency_completion_sha256": {
            dep: file_sha256(completion_path(ctx, dep)) for dep in stage.depends_on},
        "outcome_access": list(stage.outcome_access),
        "outputs": result.outputs,
        "allocations": result.allocations,
        "commands": result.commands,
        "completed_at_utc": _utc(),
    }
    path = completion_path(ctx, stage.name)
    publish_json_once(path, record, root=ctx.run_dir)
    return path


def publish_or_verify(path: Path, value: Any, *, root: Path) -> None:
    """Create-once artifact; an existing one must be identical."""

    if path.exists():
        if _load_json(path) != value:
            raise StaleStageError(f"{path} exists with different content")
        return
    publish_json_once(path, value, root=root)


# ===================================================================== freeze gate


FROZEN_FIELDS_MATCHED = ("amendments", "open_items", "governance")


def validate_freeze_binding(ctx: RunContext, binding: Mapping[str, Any]) -> dict[str, Any]:
    """The bound artifact is a valid frozen internal freeze of exactly the active protocol.

    Beyond content hash, parent and frozen status, the active protocol configuration
    must still equal the frozen copy (amendments, open items, governance), so a
    protocol edit after the freeze can never run under the old freeze.
    """

    from e3sse.protocol.internal_freeze import validate_internal_freeze_protocol

    artifact = Path(binding["directory"]) / "internal_freeze_protocol.json"
    if not artifact.is_file() or file_sha256(artifact) != binding["file_sha256"]:
        raise StageBlocked(VALIDATION_FAILURE, "internal-freeze artifact differs from its binding")
    internal = _load_json(artifact)
    parent = _load_json(Path(ctx.active_protocol["parent_protocol"]["directory"]) / "protocol.json")
    try:
        validate_internal_freeze_protocol(internal, parent)
    except ValueError as exc:
        raise StageBlocked(VALIDATION_FAILURE, f"internal freeze invalid: {exc}") from exc
    if internal["protocol_id"] != binding["protocol_id"]:
        raise StageBlocked(VALIDATION_FAILURE, "internal-freeze protocol_id differs from binding")
    drift = [key for key in FROZEN_FIELDS_MATCHED
             if canonical_sha256(internal.get(key)) != canonical_sha256(ctx.active_protocol.get(key))]
    if drift:
        raise StageBlocked(VALIDATION_FAILURE, f"active protocol config differs from the frozen "
                           f"internal freeze in {drift}; a changed protocol needs a new freeze")
    return dict(binding)


def freeze_binding(ctx: RunContext) -> dict[str, Any]:
    """The verified frozen internal freeze, required by every outcome-reading stage."""

    path = ctx.stage_dir(FREEZE_STAGE) / FREEZE_BINDING_FILE
    if not path.is_file():
        raise StageBlocked(BLOCKED_MISSING_EVIDENCE, "no materialized internal freeze")
    return validate_freeze_binding(ctx, _load_json(path))


def freeze_is_bound(ctx: RunContext) -> bool:
    try:
        freeze_binding(ctx)
    except StageBlocked:
        return False
    return True


# ===================================================================== allocation


def stage_allocation(ctx: RunContext, stage: str, model: str, workload: str,
                     *, pending_units: int | None = None) -> WorkloadAllocation:
    """Workers for one stage from measured evidence, with run-level pinned threads.

    Threads per worker can change CPU floating-point reductions (and so a borderline
    ML-NEB convergence), so the first stage that needs a (model, workload) pins its
    thread count for the whole run in ``numerical_threads/``.  Every later stage and
    every resume uses that count.  Only the worker count follows the current host.
    """

    host = ctx.host()
    if workload == "statistics_numpy":
        profile = statistics_profile(ctx.policy)
    else:
        profile = ctx.profiles().get((model, workload))
        if profile is None:
            raise StageBlocked(BLOCKED_MISSING_EVIDENCE,
                               f"no measured resource profile for {model}/{workload}")
    pin_path = ctx.run_dir / "numerical_threads" / f"{model}--{workload}.json"
    try:
        if pin_path.is_file():
            threads = int(_load_json(pin_path)["threads_per_worker"])
        else:
            chosen = allocate_workload(profile, host, ctx.policy, pending_units=pending_units)
            threads = chosen.threads_per_worker
            publish_json_once(pin_path, {
                "classification": "numerical_identity_fixed_for_the_run",
                "model": model, "workload": workload, "threads_per_worker": threads,
                "chosen_by": chosen.as_json(), "chosen_in_stage": stage,
                "pinned_at_utc": _utc()}, root=ctx.run_dir)
        return allocate_workload(profile, host, ctx.policy, pending_units=pending_units,
                                 fixed_threads=threads)
    except SchedulingError as exc:
        raise StageBlocked(INFRASTRUCTURE_FAILURE, str(exc)) from exc


# ===================================================================== adapters


class StageAdapter:
    """Interface: ``plan`` lists commands without side effects; ``run`` executes."""

    def plan(self, ctx: RunContext, stage: ProductionStage) -> list[str]:
        return []

    def run(self, ctx: RunContext, stage: ProductionStage) -> StageResult:
        raise NotImplementedError


class MissingRunnerAdapter(StageAdapter):
    def plan(self, ctx, stage):
        return [f"BLOCKED (missing runner): {stage.next_task}"]

    def run(self, ctx, stage):
        raise StageBlocked(BLOCKED_MISSING_RUNNER, stage.next_task or "no runner")


class DeferredAdapter(StageAdapter):
    def plan(self, ctx, stage):
        return [f"DEFERRED: {stage.next_task}"]

    def run(self, ctx, stage):
        raise StageBlocked(DEFERRED, stage.next_task or "deferred by the active protocol")


def _scripts(ctx: RunContext, name: str) -> Path:
    return ctx.repository / "scripts" / name


def _last_json(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise StageBlocked(INFRASTRUCTURE_FAILURE, "child printed no JSON result line")


def _last_path(output: str) -> Path:
    lines = [line for line in output.strip().splitlines() if line.strip()]
    if not lines:
        raise StageBlocked(INFRASTRUCTURE_FAILURE, "child printed no output path")
    return Path(lines[-1].split()[-1])


def _require_interpreters(ctx: RunContext) -> None:
    missing = sorted({str(rt.python) for rt in ctx.runtimes.values() if not rt.python.is_file()})
    if missing:
        raise StageBlocked(BLOCKED_MISSING_EVIDENCE, f"frozen interpreter(s) absent: {missing}; "
                           "production never installs packages")


def _environments(ctx: RunContext) -> dict[str, ModelRuntime]:
    chosen: dict[str, ModelRuntime] = {}
    for runtime in ctx.runtimes.values():
        chosen.setdefault(runtime.environment, runtime)
    return dict(sorted(chosen.items()))


class RuntimeAuditAdapter(StageAdapter):
    """ASE/FIRE/NEB audit in every frozen runtime (evidence only; preflight judges)."""

    def _command(self, ctx, environment, runtime):
        return [str(runtime.python), str(_scripts(ctx, "audit_ml_neb_runtime.py")),
                "--protocol-freeze-config", str(ctx.freeze_config_path),
                "--active-protocol", str(ctx.active_protocol_path),
                "--environment", environment,
                "--output", str(ctx.evidence("ml_neb_runtime_audit_dir")),
                "--protected-root", str(ctx.data_root), "--resume"]

    def plan(self, ctx, stage):
        return [" ".join(self._command(ctx, env, rt)) for env, rt in _environments(ctx).items()]

    def run(self, ctx, stage):
        result = StageResult(stage.name, COMPLETED, "runtime audits recorded")
        for environment, runtime in _environments(ctx).items():
            if not runtime.python.is_file():
                raise StageBlocked(BLOCKED_MISSING_EVIDENCE,
                                   f"frozen interpreter {runtime.python} absent")
            command = self._command(ctx, environment, runtime)
            # Exit 3 = the audit ran and recorded a failed check (preflight judges it);
            # any other non-zero exit is a crash.
            _, output = ctx.run_command(stage.name, command, ctx.child_environment(runtime, 1),
                                        accept_returncodes=(0, 3))
            result.commands.append(command)
            report = Path(_last_json(output).get("report", ""))
            if not report.is_file():
                raise StageBlocked(INFRASTRUCTURE_FAILURE, f"audit wrote no report for {environment}")
            result.outputs[f"ml_neb_runtime_audit:{environment}"] = output_record(report)
        return result


class ResourceProfileAdapter(StageAdapter):
    """Measured memory and throughput per (model, workload) in each frozen runtime."""

    def _settings(self, ctx):
        return ctx.config["resource_profiling"]

    def _study1(self, ctx, *args):
        return [str(ctx.project_python), str(_scripts(ctx, "run_study1_softening_atlas.py")),
                "--config", str(ctx.study1_config_path),
                "--checkpoint-map", str(ctx.checkpoint_map()), *args]

    def _profile_command(self, ctx, model, workload, structures: str, min_atoms: str):
        settings, runtime = self._settings(ctx), ctx.runtimes[model]
        grid_threads = [t for t in settings["thread_grid"] if t <= ctx.policy.max_threads_per_worker]
        command = [str(runtime.python), str(_scripts(ctx, "profile_production_resources.py")),
                   "--protocol-freeze-config", str(ctx.freeze_config_path),
                   "--active-protocol", str(ctx.active_protocol_path),
                   "--model", model, "--checkpoint", str(ctx.checkpoint(model)
                                                         if not ctx.dry_run else "<checkpoint>"),
                   "--workload", workload, "--memory-threads", str(max(grid_threads)),
                   "--process-grid", ",".join(map(str, settings["process_grid"])),
                   "--thread-grid", ",".join(map(str, grid_threads)),
                   "--evaluations", str(settings["evaluations"][workload]),
                   "--warmup", str(settings["warmup"]),
                   "--memory-reserve-gib", str(ctx.policy.reserve_for(ctx.host()) / GIB),
                   "--memory-safety-factor", str(ctx.policy.memory_safety_factor),
                   "--timeout-seconds", str(settings["timeout_seconds"]),
                   "--output", str(ctx.evidence("resource_profile_dir")),
                   "--protected-root", str(ctx.data_root), "--resume"]
        if ctx.max_cpus is not None:
            command += ["--max-cpus", str(ctx.max_cpus)]
        if ctx.max_memory_gib is not None:
            command += ["--max-memory-gib", str(ctx.max_memory_gib)]
        if workload == "force_single_point":
            command += ["--structure-file", structures]
        else:
            command += ["--ml-neb-min-atoms", min_atoms,
                        "--ml-neb-interior-images", str(settings["ml_neb_interior_images"])]
        return command

    def _probe_structures(self, ctx) -> Path:
        study1 = _load_json(ctx.study1_config_path)
        root = (_resolve(ctx.study1_config_path.parent, study1["output_root"])
                / study1["frozen_protocol"]["protocol_id"][:16] / "resources" / "probe_structures")
        files = sorted(root.glob("*.json"))
        if len(files) != 1:
            raise StageBlocked(INFRASTRUCTURE_FAILURE,
                               f"expected one Study-1 probe structure file in {root}, found {len(files)}")
        return files[0]

    def plan(self, ctx, stage):
        settings = self._settings(ctx)
        lines = [" ".join(self._study1(ctx, "index-frames", "--resume")),
                 " ".join(self._study1(ctx, "probe", "--all-models", "--threads-per-worker",
                                       str(ctx.policy.max_threads_per_worker)))]
        for model in sorted(ctx.models):
            for workload in settings["workloads"]:
                lines.append(" ".join(self._profile_command(
                    ctx, model, workload, "<study1 probe structures>", "<largest probe frame>")))
        return lines

    def run(self, ctx, stage):
        _require_interpreters(ctx)
        settings = self._settings(ctx)
        result = StageResult(stage.name, COMPLETED, "resource profiles measured")
        env = ctx.child_environment(None, 1)
        for args in (("index-frames", "--resume"),
                     ("probe", "--all-models", "--threads-per-worker",
                      str(ctx.policy.max_threads_per_worker))):
            command = self._study1(ctx, *args)
            ctx.run_command(stage.name, command, env)
            result.commands.append(command)
        structures = self._probe_structures(ctx)
        result.outputs["probe_structures"] = output_record(structures)
        largest = max(len(item["numbers"]) for item in _load_json(structures)["structures"])
        for model in sorted(ctx.models):
            runtime = ctx.runtimes[model]
            for workload in settings["workloads"]:
                command = self._profile_command(ctx, model, workload, str(structures), str(largest))
                _, output = ctx.run_command(stage.name, command, ctx.child_environment(runtime, 1))
                result.commands.append(command)
                path = _last_path(output)
                if not path.is_file():
                    raise StageBlocked(INFRASTRUCTURE_FAILURE, f"no profile written for {model}")
                result.outputs[f"resource_profile:{model}/{workload}"] = output_record(path)
        return result


class Gate1PopulationAdapter(StageAdapter):
    def _command(self, ctx):
        return [str(ctx.project_python), str(_scripts(ctx, "audit_gate1_structural_populations.py")),
                "--config", str(ctx.server_config_path),
                "--protocol-config", str(ctx.freeze_config_path),
                "--study1-config", str(ctx.study1_config_path),
                "--output-dir", str(ctx.evidence("gate1_population_audit_dir")), "--resume"]

    def plan(self, ctx, stage):
        return [" ".join(self._command(ctx))]

    def run(self, ctx, stage):
        command = self._command(ctx)
        _, output = ctx.run_command(stage.name, command, ctx.child_environment(None, 1))
        path = _last_path(output)
        if not path.is_file():
            raise StageBlocked(INFRASTRUCTURE_FAILURE, f"{path} was not written")
        return StageResult(stage.name, COMPLETED, "structural populations recorded",
                           outputs={"gate1_structural_population_audit": output_record(path)},
                           commands=[command])


def collect_preflight(ctx: RunContext, *, rehash_datasets: bool) -> list[checks.PreflightCheck]:
    """Every preflight check; nothing here reads a scientific outcome."""

    from e3sse.production.resource_scheduler import WORKLOAD_CLASSES

    config_dir = ctx.freeze_config_path.parent
    g0 = Path(ctx.freeze_config["inputs"]["g0_input_manifest"]["path"])
    existing = ctx.config.get("freeze", {}).get("existing_binding")
    if existing:
        try:
            validate_freeze_binding(ctx, existing)
            freeze_bound = True
        except StageBlocked:
            freeze_bound = False
    else:
        freeze_bound = freeze_is_bound(ctx)
    code = {name: file_sha256(ctx.repository / name) for name in
            ("scripts/audit_ml_neb_runtime.py", "src/e3sse/barriers/ml_neb.py",
             "src/e3sse/screening/prospective_contract.py")}
    from e3sse.barriers.ml_neb import settings_from_protocol

    required = sorted({workload for stage in ctx.graph.stages.values()
                       if stage.implementation_status == "implemented"
                       for workload in stage.workloads})
    ambiguous: list[str] = []
    expanded = sorted({(model, workload) for selector, workload in required
                       for model in (sorted(ctx.models) if selector == "each_model" else [selector])
                       if workload in WORKLOAD_CLASSES})
    result = [
        checks.check_git_identity(ctx.git, require_clean=ctx.config["runtime"]["require_clean_worktree"],
                                  expected_commit=ctx.config["runtime"].get("expected_commit")),
        *[checks.check_output_root(root, [ctx.data_root]) for root in ctx.write_roots()],
        checks.check_scope(ctx.active_protocol),
        checks.check_bvse_pool_tolerance(ctx.active_protocol, config_dir=config_dir),
        checks.check_outcome_access_declaration(ctx.study1_config_path, freeze_bound=freeze_bound),
        *checks.check_freeze_input_pins(ctx.freeze_config, config_dir=config_dir),
        checks.check_dataset_hashes(g0, rehash=rehash_datasets),
        checks.check_gate1_linkage(ctx.evidence("gate1_linkage_audit")),
        checks.check_gate1_structural_populations(_population_audit(ctx),
                                                  ctx.evidence("gate1_linkage_audit")),
        *checks.check_embedding_evidence(ctx.active_protocol, ctx.models),
        *checks.check_runtimes(ctx.runtimes),
        *checks.check_checkpoints(ctx.models, ctx.checkpoint_map()),
        *checks.check_ml_neb_runtime_audits(
            ctx.evidence("ml_neb_runtime_audit_dir"),
            [runtime.environment for runtime in ctx.runtimes.values()],
            expected_code_sha256=code,
            expected_settings_sha256=settings_from_protocol(ctx.active_protocol).settings_sha256),
        *checks.check_resource_profiles(ctx.profiles(ambiguous), expanded, ambiguous=ambiguous),
        *checks.check_protocol_open_items(ctx.active_protocol),
        checks.check_outcome_runners_implemented(
            {name: stage.implementation_status for name, stage in ctx.graph.stages.items()
             if stage.outcome_access},
            required=bool(ctx.config.get("freeze", {}).get("require_outcome_runners_implemented",
                                                           True))),
    ]
    return result


def _population_audit(ctx: RunContext) -> Path:
    """This run's audit from its completion record, else the single one on disk."""

    outputs = ctx.stage_outputs("gate1_population_audit")
    if outputs is not None:
        return Path(outputs["gate1_structural_population_audit"]["path"])
    directory = ctx.evidence("gate1_population_audit_dir")
    found = sorted(directory.glob("gate1_structural_population_audit-*.json"))
    return found[0] if len(found) == 1 else directory / "gate1_structural_population_audit-<run>.json"


class PreflightAdapter(StageAdapter):
    def plan(self, ctx, stage):
        return ["in-process: aggregate outcome-free evidence (e3sse.production.preflight)"]

    def run(self, ctx, stage):
        report = checks.summarize(collect_preflight(
            ctx, rehash_datasets=ctx.config["preflight"]["rehash_datasets"]))
        report.update(run_id=ctx.run_id, created_at_utc=_utc())
        attempt = ctx.stage_dir(stage.name) / "attempts" / f"{_compact_utc()}-{os.getpid()}.json"
        publish_json_once(attempt, report, root=ctx.run_dir)
        if report["blocking"]:
            failed = [c["check_id"] for c in report["checks"]
                      if c["blocking"] and c["status"] == "failed"]
            outcome = VALIDATION_FAILURE if failed else BLOCKED_MISSING_EVIDENCE
            raise StageBlocked(outcome, f"{len(report['blocking'])} blocking check(s): "
                               f"{', '.join(report['blocking'])} (report {attempt})")
        return StageResult(stage.name, COMPLETED, "every blocking preflight check passed",
                           outputs={"preflight_report": output_record(attempt)})


class FreezeAdapter(StageAdapter):
    """Materialize (or bind an existing) frozen internal-freeze protocol."""

    def _command(self, ctx):
        return [str(ctx.project_python), str(_scripts(ctx, "build_internal_freeze_protocol.py")),
                "build", "--config", str(ctx.active_protocol_path), "--git-sha", ctx.git["commit"],
                "--study1-config", str(ctx.study1_config_path),
                "--server-config", str(ctx.server_config_path), "--resume"]

    def plan(self, ctx, stage):
        existing = ctx.config.get("freeze", {}).get("existing_binding")
        if existing:
            return [f"verify existing internal freeze {existing['protocol_id']}"]
        return [" ".join(self._command(ctx))]

    def run(self, ctx, stage):
        existing = ctx.config.get("freeze", {}).get("existing_binding")
        commands = []
        if existing:
            binding = {key: existing[key] for key in ("protocol_id", "file_sha256", "directory")}
        else:
            command = self._command(ctx)
            # Exit 1: the builder refused (dirty tree, falsified declaration, parent
            # mismatch...): a validation failure, not infrastructure.
            _, output = ctx.run_command(stage.name, command, ctx.child_environment(None, 1),
                                        failure_outcomes={1: VALIDATION_FAILURE})
            commands.append(command)
            line = next((row for row in output.splitlines()
                         if row.startswith("Internal-freeze protocol: ")), None)
            if line is None:
                raise StageBlocked(INFRASTRUCTURE_FAILURE, "freeze builder printed no artifact path")
            artifact = Path(line.split(": ", 1)[1].strip())
            internal = _load_json(artifact)
            binding = {"protocol_id": internal["protocol_id"],
                       "file_sha256": file_sha256(artifact), "directory": str(artifact.parent)}
        # Validate first: an invalid binding is never published (create-once files persist).
        validate_freeze_binding(ctx, binding)
        path = ctx.stage_dir(stage.name) / FREEZE_BINDING_FILE
        publish_or_verify(path, binding, root=ctx.run_dir)
        artifact = Path(binding["directory"]) / "internal_freeze_protocol.json"
        return StageResult(stage.name, COMPLETED, f"frozen internal protocol {binding['protocol_id']}",
                           outputs={"internal_freeze_binding": output_record(path),
                                    "internal_freeze_protocol": output_record(artifact)},
                           commands=commands)


class EstimateAdapter(StageAdapter):
    """ESTIMATE: the existing Study-1 runner, analysis and figures under the freeze."""

    def _config_path(self, ctx):
        return ctx.stage_dir("estimate") / "study1_softening_atlas_internal_freeze_config.json"

    def _derived_config(self, ctx, binding):
        config = _load_json(ctx.study1_config_path)
        config["server_config"] = str(ctx.server_config_path)
        config["internal_freeze_protocol"] = binding
        return config

    def _runner(self, ctx, config, *args):
        return [str(ctx.project_python), str(_scripts(ctx, "run_study1_softening_atlas.py")),
                "--config", str(config), "--checkpoint-map", str(ctx.checkpoint_map()), *args]

    def plan(self, ctx, stage):
        config = self._config_path(ctx)
        lines = [" ".join(self._runner(ctx, config, "index-frames", "--resume")),
                 " ".join(self._runner(ctx, config, "infer", "--reference", "--workers", "<stats>",
                                       "--threads-per-worker", "1", "--resume"))]
        for model in sorted(ctx.models):
            lines.append(" ".join(self._runner(
                ctx, config, "infer", "--model", model, "--workers", "<scheduled>",
                "--threads-per-worker", "<scheduled>", "--resume")))
        lines += [" ".join(self._runner(ctx, config, "verify")),
                  (f"{ctx.project_python} scripts/analyze_study1_softening_atlas.py "
                   f"--config {config} --workers <stats> --resume"),
                  (f"{ctx.project_python} scripts/plot_study1_figures.py --config {config} "
                   "--analysis-dir <analysis>")]
        return lines

    def run(self, ctx, stage):
        binding = freeze_binding(ctx)
        config_path = self._config_path(ctx)
        publish_or_verify(config_path, self._derived_config(ctx, binding), root=ctx.run_dir)
        study1 = _load_json(config_path)
        result = StageResult(stage.name, COMPLETED, "ESTIMATE complete")
        result.outputs["study1_config"] = output_record(config_path)
        reserve = f"{ctx.policy.reserve_for(ctx.host()) / GIB:.3f}"
        env = ctx.child_environment(None, 1)

        def run(command):
            ctx.run_command(stage.name, command, env)
            result.commands.append(command)

        run(self._runner(ctx, config_path, "index-frames", "--resume"))
        stats = stage_allocation(ctx, stage.name, "statistics", "statistics_numpy")
        result.allocations["reference_and_analysis"] = stats.as_json()
        run(self._runner(ctx, config_path, "infer", "--reference", "--workers", str(stats.workers),
                         "--threads-per-worker", "1", "--max-threads-per-worker", "1",
                         "--memory-reserve-gib", reserve, "--resume"))
        for model in sorted(ctx.models):
            allocation = stage_allocation(ctx, stage.name, model, "force_single_point")
            result.allocations[model] = allocation.as_json()
            threads = str(allocation.threads_per_worker)
            run(self._runner(ctx, config_path, "infer", "--model", model,
                             "--workers", str(allocation.workers), "--threads-per-worker", threads,
                             "--max-threads-per-worker", threads,
                             "--memory-reserve-gib", reserve, "--resume"))
        verify = self._runner(ctx, config_path, "verify")
        ctx.run_command(stage.name, verify, env, failure_outcomes={1: VALIDATION_FAILURE})
        result.commands.append(verify)
        root = (_resolve(config_path.parent, study1["output_root"])
                / study1["frozen_protocol"]["protocol_id"][:16])
        for name in ["dft_reference", *sorted(ctx.models)]:
            manifest = root / "inference" / name / "frame_statistics_manifest.json"
            if manifest.is_file():
                result.outputs[f"frame_statistics:{name}"] = output_record(manifest)
        analyze = [str(ctx.project_python), str(_scripts(ctx, "analyze_study1_softening_atlas.py")),
                   "--config", str(config_path), "--workers", str(stats.workers),
                   "--memory-reserve-gib", reserve, "--resume"]
        _, output = ctx.run_command(stage.name, analyze, env)
        result.commands.append(analyze)
        analysis = self._analysis_dir(output)
        result.outputs["study1_analysis_manifest"] = output_record(
            analysis / "study1_analysis_manifest.json")
        self._figures(ctx, stage, analysis, config_path, env, result)
        return result

    @staticmethod
    def _analysis_dir(output: str) -> Path:
        for line in reversed(output.splitlines()):
            if line.startswith("analysis already complete: "):
                return Path(line.split(": ", 1)[1]).parent
            if line.startswith("{") and '"analysis"' in line:
                return Path(json.loads(line)["analysis"])
        raise StageBlocked(INFRASTRUCTURE_FAILURE, "analysis printed no output directory")

    @staticmethod
    def _figures(ctx, stage, analysis, config_path, env, result):
        stems = ("study1_softening_atlas", "study1_label_budget_stability")
        manifests = [analysis / "figures" / f"{stem}.figure.json" for stem in stems]
        present = [path.is_file() for path in manifests]
        if not any(present):
            command = [str(ctx.project_python), str(_scripts(ctx, "plot_study1_figures.py")),
                       "--config", str(config_path), "--analysis-dir", str(analysis)]
            ctx.run_command(stage.name, command, env)
            result.commands.append(command)
        elif not all(present):
            raise StageBlocked(VALIDATION_FAILURE, f"partial figure set in {analysis / 'figures'}; "
                               "inspect it before rebuilding (figures are never overwritten)")
        for path in manifests:
            figure = _load_json(path)
            for name, digest in figure["outputs_sha256"].items():
                if file_sha256(path.parent / name) != digest:
                    raise StageBlocked(VALIDATION_FAILURE, f"{name} differs from its manifest")
            result.outputs[f"figure:{path.stem}"] = output_record(path)


class RunnerAdapter(StageAdapter):
    """A stage implemented by ``e3sse.production.runners`` (scripts/run_production_stage.py).

    Phases: ``prepare`` (project, if the runner has one), ``model`` (each model's frozen
    runtime: every model, or only the selected model), ``reduce`` (project, or the BVSE
    runtime).  Workers and threads come from the central scheduler; threads are the
    run-level pinned value per (model, workload).  Exit 2 = missing evidence, 3 =
    validation failure, anything else = infrastructure.
    """

    def __init__(self, *, models: str | None = None, model_workload: str | None = None,
                 interpreter: str = "project") -> None:
        if models not in (None, "each", "selected"):
            raise ValueError("models must be None, 'each' or 'selected'")
        self.models, self.model_workload, self.interpreter = models, model_workload, interpreter

    def _module(self, stage):
        import importlib

        from e3sse.production.runners import RUNNER_MODULES

        return importlib.import_module(RUNNER_MODULES[stage.name])

    def _python(self, ctx) -> Path:
        if self.interpreter == "bvse":
            path = Path(ctx.config.get("runners", {}).get("bvse", {}).get("python", "")).expanduser()
            if not path.is_file():
                raise StageBlocked(BLOCKED_MISSING_EVIDENCE,
                                   f"BVSE runtime interpreter {path} absent (runners.bvse.python); "
                                   "production never installs packages")
            return path
        return ctx.project_python

    def _command(self, ctx, stage, phase, python, *, model=None, workers=1, threads=1):
        command = [str(python), str(_scripts(ctx, "run_production_stage.py")),
                   "--config", str(ctx.config_path), "--run-dir", str(ctx.run_dir),
                   "--stage", stage.name, "--phase", phase, "--workers", str(workers),
                   "--threads", str(threads),
                   "--memory-reserve-gib", f"{ctx.policy.reserve_for(ctx.host()) / GIB:.3f}"]
        return command + (["--model", model] if model else [])

    def _models(self, ctx) -> list[str]:
        if self.models == "each":
            return sorted(ctx.models)
        if self.models == "selected":
            stage_manifest = ctx.stage_dir("model_selection") / "stage_manifest.json"
            if not stage_manifest.is_file():
                raise StageBlocked(BLOCKED_MISSING_EVIDENCE, "no model selection in this run")
            output = _load_json(stage_manifest)["outputs"].get("model_selection")
            path = Path(output["path"]) if output else None
            if path is None or not path.is_file() or file_sha256(path) != output["sha256"]:
                raise StageBlocked(VALIDATION_FAILURE, "model_selection.json differs from its "
                                                       "stage manifest")
            return [_load_json(path)["selection"]["selected_model"]]
        return []

    def plan(self, ctx, stage):
        module = self._module(stage)
        lines = []
        if hasattr(module, "prepare"):
            lines.append(" ".join(self._command(ctx, stage, "prepare", "<project python>",
                                                workers="<stats>")))
        if self.models:
            label = "every model" if self.models == "each" else "the selected model"
            lines.append(f"<frozen runtime python> scripts/run_production_stage.py --stage "
                         f"{stage.name} --phase model --model <{label}> --workers <scheduled> "
                         f"--threads <run-pinned> ({self.model_workload})")
        lines.append(" ".join(self._command(ctx, stage, "reduce", "<project python>"
                                            if self.interpreter == "project" else "<BVSE python>",
                                            workers="<stats>")))
        return lines

    def _run_phase(self, ctx, stage, phase, python, env, result, **kwargs):
        command = self._command(ctx, stage, phase, python, **kwargs)
        _, output = ctx.run_command(stage.name, command, env, failure_outcomes={
            2: BLOCKED_MISSING_EVIDENCE, 3: VALIDATION_FAILURE})
        result.commands.append(command)
        manifest = Path(_last_json(output)["manifest"])
        for name, item in _load_json(manifest)["outputs"].items():
            key = f"{name}@{kwargs.get('model')}" if kwargs.get("model") else name
            if key in result.outputs:
                raise StageBlocked(VALIDATION_FAILURE, f"{stage.name}: output {key!r} produced "
                                                       "by two phases")
            result.outputs[key] = {"path": item["path"], "sha256": item["sha256"]}
        result.outputs[f"manifest:{phase}" + (f":{kwargs['model']}" if kwargs.get("model") else "")] = \
            output_record(manifest)

    def run(self, ctx, stage):
        module = self._module(stage)
        result = StageResult(stage.name, COMPLETED, f"{stage.name} runner complete")
        stats = stage_allocation(ctx, stage.name, "statistics", "statistics_numpy")
        result.allocations["statistics"] = stats.as_json()
        python = self._python(ctx)
        env = ctx.child_environment(None, 1)
        if hasattr(module, "prepare"):
            self._run_phase(ctx, stage, "prepare", python, env, result, workers=stats.workers)
        for model in self._models(ctx):
            runtime = ctx.runtimes[model]
            if not runtime.python.is_file():
                raise StageBlocked(BLOCKED_MISSING_EVIDENCE, f"frozen interpreter {runtime.python} absent")
            allocation = stage_allocation(ctx, stage.name, model, self.model_workload)
            result.allocations[model] = allocation.as_json()
            self._run_phase(ctx, stage, "model", runtime.python,
                            ctx.child_environment(runtime, allocation.threads_per_worker), result,
                            model=model, workers=allocation.workers,
                            threads=allocation.threads_per_worker)
        self._run_phase(ctx, stage, "reduce", python, env, result, workers=stats.workers)
        return result


DEFAULT_ADAPTERS: dict[str, StageAdapter] = {
    "runtime_audit": RuntimeAuditAdapter(),
    "resource_profile": ResourceProfileAdapter(),
    "gate1_population_audit": Gate1PopulationAdapter(),
    "preflight": PreflightAdapter(),
    "freeze": FreezeAdapter(),
    "estimate": EstimateAdapter(),
    "missing_runner": MissingRunnerAdapter(),
    "deferred": DeferredAdapter(),
    "runner_project": RunnerAdapter(),
    "runner_bvse": RunnerAdapter(interpreter="bvse"),
    "runner_each_model_ml_neb": RunnerAdapter(models="each", model_workload="ml_neb_band"),
    "runner_each_model_single_points": RunnerAdapter(models="each",
                                                     model_workload="force_single_point"),
    "runner_selected_model_ml_neb": RunnerAdapter(models="selected", model_workload="ml_neb_band"),
}


# ===================================================================== execution


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(run_dir / ".orchestrator.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another orchestrator is running {run_dir}") from exc
        yield
    finally:
        os.close(descriptor)


def _bind_run(ctx: RunContext) -> None:
    path = ctx.run_dir / "run_identity.json"
    record = {"run_id": ctx.run_id, "scientific_identity": ctx.scientific_identity}
    if path.exists():
        if not ctx.resume:
            raise ArtifactExistsError(f"{ctx.run_dir} already holds this run; use --resume")
        if _load_json(path) != record:
            raise StaleStageError(f"{path} records a different scientific identity")
        return
    publish_json_once(path, record, root=ctx.run_dir)


def _require_permitted(ctx: RunContext, stage: ProductionStage) -> None:
    if stage.outcome_access:
        freeze_binding(ctx)
    if ctx.config["runtime"]["require_clean_worktree"] and ctx.git["dirty"] and \
            stage.implementation_status == "implemented":
        raise StageBlocked(VALIDATION_FAILURE, "production stages require a clean worktree")


def execute(ctx: RunContext, requested: str, adapters: Mapping[str, StageAdapter]) -> list[StageResult]:
    """Run the requested stages; see the module docstring for the semantics."""

    order = ctx.graph.resolve(requested)
    results: list[StageResult] = []
    with _run_lock(ctx.run_dir):
        _bind_run(ctx)
        started = _utc()
        settled: dict[str, str] = {}
        try:
            for name in order:
                stage = ctx.graph.stages[name]
                blockers = [dep for dep in stage.depends_on if settled.get(dep) not in SUCCESS]
                if blockers:
                    result = StageResult(name, BLOCKED_BY_DEPENDENCY, f"waiting for {blockers}")
                else:
                    result = _execute_stage(ctx, stage, adapters[stage.adapter])
                settled[name] = result.outcome
                results.append(result)
                ctx.progress(f"[{name}] {result.outcome}: {result.detail}")
        finally:
            # Written even when a stale record stops the run, so every call leaves a trace.
            try:
                host = ctx.host().as_json()
            except Exception as exc:  # the record must still be written
                host = {"unobservable": str(exc)}
            invocation = {
                "schema_version": INVOCATION_SCHEMA, "run_id": ctx.run_id,
                "requested_stage": requested, "resume": ctx.resume, "started_at_utc": started,
                "finished_at_utc": _utc(),
                "operational": operational_identity(host=host,
                                                    resource_policy=ctx.config["resource_policy"],
                                                    command=list(sys.argv)),
                "git": ctx.git, "results": [r.as_json() for r in results],
                "completed_normally": len(results) == len(order),
            }
            publish_json_once(ctx.run_dir / "invocations" / f"{_compact_utc()}-{os.getpid()}.json",
                              invocation, root=ctx.run_dir)
    return results


def _execute_stage(ctx: RunContext, stage: ProductionStage, adapter: StageAdapter) -> StageResult:
    existing = completion_path(ctx, stage.name)
    if existing.is_file():
        if not ctx.resume:
            raise ArtifactExistsError(f"{existing} exists; use --resume")
        record = verified_completion(ctx, stage)
        assert record is not None
        return StageResult(stage.name, REUSED, "completion record verified",
                           outputs=record["outputs"])
    try:
        _require_permitted(ctx, stage)
        result = adapter.run(ctx, stage)
    except StageBlocked as blocked:
        return StageResult(stage.name, blocked.outcome, blocked.detail)
    except StaleStageError as exc:
        # e.g. a derived artifact exists with different content: fail closed, keep a record.
        return StageResult(stage.name, VALIDATION_FAILURE, str(exc))
    except Exception as exc:  # an unexpected crash is infrastructure; keep the traceback
        log = ctx.run_dir / "logs" / stage.name / f"exception-{_compact_utc()}-{os.getpid()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
        return StageResult(stage.name, INFRASTRUCTURE_FAILURE,
                           f"{type(exc).__name__}: {exc} (traceback {log})")
    publish_completion(ctx, stage, result)
    return result


def exit_code(results: Sequence[StageResult]) -> int:
    outcomes = {result.outcome for result in results}
    if INFRASTRUCTURE_FAILURE in outcomes:
        return EXIT_CODES[INFRASTRUCTURE_FAILURE]
    if VALIDATION_FAILURE in outcomes:
        return EXIT_CODES[VALIDATION_FAILURE]
    if outcomes - set(SUCCESS):
        return EXIT_CODES["blocked"]
    return EXIT_CODES["success"]


# ===================================================================== dry run


def dry_run_plan(ctx: RunContext, requested: str, adapters: Mapping[str, StageAdapter]) -> dict[str, Any]:
    """Resolved DAG, inputs, runtimes, predicted resources, outputs, blockers, commands.

    Writes nothing, reads no outcome and launches no computation.
    """

    from e3sse.production.resource_scheduler import WORKLOAD_CLASSES

    order = ctx.graph.resolve(requested)
    host = ctx.host()
    profiles = ctx.profiles()
    stages = []
    required: set[tuple[str, str]] = set()
    for name in order:
        stage = ctx.graph.stages[name]
        for selector, workload in stage.workloads:
            if workload in WORKLOAD_CLASSES:
                for model in (sorted(ctx.models) if selector == "each_model" else [selector]):
                    required.add((model, workload))
        try:
            state = "complete" if verified_completion(ctx, stage) else "pending"
        except StaleStageError as exc:
            state = f"stale: {exc}"
        try:
            commands = adapters[stage.adapter].plan(ctx, stage)
        except StageBlocked as blocked:
            commands = [f"BLOCKED: {blocked.detail}"]
        stages.append({**stage.as_json(), "state": state,
                       "output_directory": str(ctx.stage_dir(name)), "commands": commands})
    preflight_checks = checks.summarize(collect_preflight(ctx, rehash_datasets=False))
    return {
        "classification": "production_dry_run_no_outcome_no_computation",
        "run_id": ctx.run_id,
        "run_directory": str(ctx.run_dir),
        "run_directory_exists": ctx.run_dir.exists(),
        "requested_stage": requested,
        "stage_order": order,
        "scientific_identity": ctx.scientific_identity,
        "inputs": {"production_config": str(ctx.config_path),
                   "active_protocol_config": str(ctx.active_protocol_path),
                   "protocol_freeze_config": str(ctx.freeze_config_path),
                   "study1_config": str(ctx.study1_config_path),
                   "server_config": str(ctx.server_config_path),
                   "data_root_protected": str(ctx.data_root),
                   "outputs_root": str(ctx.outputs_root)},
        "model_runtimes": {name: rt.as_json() for name, rt in sorted(ctx.runtimes.items())},
        "predicted_resources": predicted_schedule(host, ctx.policy, profiles, sorted(required)),
        "stages": stages,
        "preflight": {key: preflight_checks[key] for key in
                      ("freeze_permitted", "production_compute_permitted", "counts", "blocking")},
        "preflight_checks": preflight_checks["checks"],
    }
