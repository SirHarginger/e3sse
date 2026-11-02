#!/usr/bin/env python3
"""Study 1 (Softening Atlas) production runner: plan, index, probe, infer, verify.

Subcommands (run from the repository root with the project environment):

  plan          non-outcome dry run: identities, counts, shard plan, seal status
  index-frames  structural byte-offset index of MPLiTrj_train.xyz (no labels parsed)
  probe         operational worker-memory probe per model on the largest real frames
  infer         DFT reference and/or model inference, each model in its own frozen
                runtime (~/.venvs/<env>), one model at a time; always resumable
  verify        re-hash every Study-1 inference artifact against its manifest

Scientific constants come from the frozen protocol; CLI options are operational.
Inference reads DFT labels, so it runs only when the repository outcome seal allows
it (frozen protocol registered on OSF; ``registration_registered.json``).
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from e3sse.execution.resource_planning import (
    GIB,
    WORKER_MEMORY_REPORT_SCHEMA,
    WorkloadPolicy,
    observe_host_resources,
    parse_workers,
    plan_workers,
)
from e3sse.execution.sharded_execution import (
    START_METHOD,
    _initialize_worker,
    canonical_sha256,
    ensure_safe_output_dir,
    publish_json_once,
)
from e3sse.execution.thread_control import child_thread_environment
from e3sse.softening import study1_inference as inference
from e3sse.softening import study1_inputs as inputs

REPOSITORY = Path(__file__).resolve().parents[1]

REFERENCE_WORKER_MEMORY_BYTES = 512 * 1024**2  # numpy/ASE parsing only, one frame at a time
# Every frozen model package runs on torch; its intra/inter-op threads are fixed per worker.
CONFIGURE_TORCH_FOR_MODELS = True
PROBE_CANDIDATES = 3


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPOSITORY), *args], text=True).strip()


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Context:
    """Resolved configuration, frozen protocol and paths shared by the subcommands."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.config_path = Path(args.config).resolve()
        self.config, self.config_sha256 = inputs.load_study1_config(self.config_path)
        base = self.config_path.parent
        self.protocol_dir = Path(
            args.protocol_dir or self.config["frozen_protocol"]["directory"]
        ).expanduser()
        self.protocol = inputs.load_frozen_protocol(self.config, directory=self.protocol_dir)
        self.protocol_sha256 = self.config["frozen_protocol"]["protocol_json_sha256"]
        self.internal_freeze = inputs.internal_freeze_binding(self.config, self.protocol)
        server = inputs.load_json(inputs.resolve(base, self.config["server_config"]))
        self.data_root = Path(server["paths"]["data_root"]).resolve()
        self.protected = [self.data_root]
        root = Path(args.output_root or self.config["output_root"]).expanduser()
        self.output = ensure_safe_output_dir(root, self.protected) / self.protocol["protocol_id"][:16]
        inputs.refuse_tmpfs(self.output)
        settings = self.config["inference"]
        self.settings = settings
        self.source = inputs.mplitrj_source(self.protocol, settings["source_file"])
        self.source_path = (self.data_root / self.source["identifier"]).resolve()
        self.source_path.relative_to(self.data_root)
        self.models = inputs.frozen_models(self.protocol)
        self.runtime_root = Path(self.config["model_runtime_root"]).expanduser()
        self.checkpoint_map_path = Path(args.checkpoint_map or self.config["checkpoint_map"])
        self.index_dir = self.output / "frame_index" / self.source["sha256"][:16]
        self._records = None

    @property
    def records(self) -> list[dict]:
        if self._records is None:
            self._records = inputs.load_membership(self.protocol_dir, self.protocol)
        return self._records

    def release_records(self) -> None:
        self._records = None
        gc.collect()

    def train_frames(self) -> list[dict]:
        return inputs.published_train_frames(self.records, self.source)

    def runtime_python(self, name: str) -> Path:
        environment, _ = inputs.model_runtime(self.protocol, name)
        return self.runtime_root / environment / "bin" / "python"

    def checkpoint(self, name: str) -> Path:
        if not self.checkpoint_map_path.is_file():
            raise SystemExit(f"missing checkpoint map {self.checkpoint_map_path}")
        mapping = json.loads(self.checkpoint_map_path.read_text(encoding="utf-8"))
        if name not in mapping:
            raise SystemExit(f"checkpoint map lacks {name}")
        return Path(mapping[name]).expanduser()

    def run_dir(self, run: str) -> Path:
        return self.output / "inference" / run

    def probe_dir(self) -> Path:
        return self.output / "resources"


def _seal(ctx: Context, frame_ids: list[str]) -> tuple[bool, str]:
    addendum = inputs.registration_addendum(ctx.protocol_dir)
    try:
        inputs.require_study1_outcome_access(ctx.protocol, addendum, ctx.records, frame_ids,
                                             internal_freeze_protocol=ctx.internal_freeze)
    except inputs.Study1InputError as exc:
        return False, str(exc)
    if ctx.internal_freeze is not None:
        return True, ("seal 1 satisfied: internal-freeze protocol "
                      f"{ctx.internal_freeze['protocol_id'][:16]} (pre-specified, not "
                      "pre-registered); published-train units verified")
    return True, "seal 1 satisfied: frozen protocol registered; published-train units verified"


# --------------------------------------------------------------------------- plan


def command_plan(ctx: Context, args: argparse.Namespace) -> int:
    frames = ctx.train_frames()
    hops = inputs.study1_target_hops(ctx.records)
    manifest = inputs.load_partition_manifest(ctx.protocol_dir, ctx.protocol)
    population = manifest["primary_population"]["assignment_by_chemical_system"]
    allowed, reason = _seal(ctx, [row["frame_id"] for row in frames])
    shards = -(-len(frames) // ctx.settings["frames_per_shard"])
    models = {}
    for name, model in ctx.models.items():
        environment, record = inputs.model_runtime(ctx.protocol, name)
        reports = sorted((ctx.probe_dir() / "worker_memory" / name).glob("*.json"))
        models[name] = {
            "group": model["frozen_group"],
            "runtime_environment": environment,
            "runtime_python": str(ctx.runtime_python(name)),
            "runtime_python_exists": ctx.runtime_python(name).is_file(),
            "environment_lock_sha256": record["environment_lock"]["sha256"],
            "weights_sha256": model["weights_sha256"],
            "worker_memory_reports": [str(path) for path in reports],
            "inference_complete": (ctx.run_dir(name) / "frame_statistics_manifest.json").is_file(),
        }
    plan = {
        "classification": "non_outcome_dry_run",
        "protocol_id": ctx.protocol["protocol_id"],
        "protocol_json_sha256": ctx.protocol_sha256,
        "freeze_git_sha": ctx.protocol["git_sha"],
        "implementation_git_sha": _git("rev-parse", "HEAD"),
        "implementation_git_dirty": bool(_git("status", "--porcelain")),
        "source": {k: ctx.source[k] for k in ("identifier", "sha256", "frames")},
        "provenance_index_id": ctx.source["provenance_index_id"],
        "membership_sha256": ctx.protocol["partition_definition"]["partition_universe"][
            "membership_sha256"
        ],
        "published_train_frames": len(frames),
        "materials": len({row["material_id"] for row in frames}),
        "chemical_systems_by_population": {
            name: sum(
                1 for system in {row["chemical_system_id"] for row in frames}
                if population.get(system) == name
            )
            for name in ("primary", "tm_hubbard_u_secondary")
        },
        "target_hops": len(hops),
        "target_hops_by_population": {
            name: sum(1 for hop in hops if population.get(hop["chemical_system_id"]) == name)
            for name in ("primary", "tm_hubbard_u_secondary")
        },
        "frames_per_shard": ctx.settings["frames_per_shard"],
        "shards_per_model": shards,
        "frame_index_present": (ctx.index_dir / "manifest.json").is_file(),
        "checkpoint_map_present": ctx.checkpoint_map_path.is_file(),
        "output_root": str(ctx.output),
        "outcome_seal": {"execution_permitted": allowed, "reason": reason},
        "resource_probe_required": [n for n, m in models.items() if not m["worker_memory_reports"]],
        "models": models,
        "host": observe_host_resources().as_json(),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- index


def command_index(ctx: Context, args: argparse.Namespace) -> int:
    manifest, _ = inference.build_frame_index(
        ctx.source_path, ctx.source, ctx.train_frames(), ctx.index_dir, resume=args.resume
    )
    print(json.dumps({"frame_index": str(ctx.index_dir), "frames": manifest["frames"],
                      "arrays_sha256": manifest["arrays_sha256"]}))
    return 0


def _index(ctx: Context) -> tuple[dict, dict]:
    manifest_path = ctx.index_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("frame index missing: run `index-frames` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = inference.load_frame_index(ctx.index_dir, ctx.source)
    stat = ctx.source_path.stat()
    if (manifest["source_bytes"], manifest["source_mtime_ns"]) != (stat.st_size, stat.st_mtime_ns):
        raise SystemExit("source file changed since the frame index was built")
    return manifest, arrays


# --------------------------------------------------------------------------- probe


def _probe_structures(ctx: Context) -> Path:
    """Largest and densest published-train frames, chosen from structure only."""

    from e3sse.softening.extxyz_frames import parse_frame_geometry, read_frame_text

    _, index = _index(ctx)
    rows = [int(row["source_frame_index"]) for row in ctx.train_frames()]
    atoms = {i: int(index["n_atoms"][i]) for i in rows}
    volume = {i: float(index["cell_volume"][i]) for i in rows}
    # Atom count; atoms^2/V tracks pair (two-body) edges; atoms^3/V^2 tracks triplets
    # (three-body terms of M3GNet and CHGNet's bond graph).
    keys = (lambda i: atoms[i], lambda i: atoms[i] ** 2 / volume[i],
            lambda i: atoms[i] ** 3 / volume[i] ** 2)
    chosen = sorted({i for key in keys
                     for i in sorted(rows, key=lambda i, k=key: (-k(i), i))[:PROBE_CANDIDATES]})
    structures = []
    with ctx.source_path.open("rb") as handle:
        for i in chosen:
            text = read_frame_text(handle, int(index["byte_offset"][i]), int(index["byte_length"][i]))
            geometry = parse_frame_geometry(text)
            structures.append({
                "frame_id": f"{ctx.settings['source_file']}:{i}",
                "numbers": geometry["numbers"],
                "positions": geometry["positions"].tolist(),
                "cell": geometry["cell"],
                "pbc": geometry["pbc"],
            })
    payload = {
        "classification": "operational_probe_geometry_only_no_labels",
        "selection": f"top {PROBE_CANDIDATES} by atoms, atoms^2/volume and atoms^3/volume^2",
        "structures": structures,
    }
    path = ctx.probe_dir() / "probe_structures" / f"{canonical_sha256(payload)[:16]}.json"
    if not path.exists():
        publish_json_once(path, payload, root=ctx.output)
    return path


def command_probe(ctx: Context, args: argparse.Namespace) -> int:
    structures = _probe_structures(ctx)
    ctx.release_records()
    names = sorted(ctx.models) if args.all_models else [args.model]
    for name in names:
        command = [
            str(ctx.runtime_python(name)), str(REPOSITORY / "scripts/probe_model_worker_memory.py"),
            "--config", str(REPOSITORY / "configs/protocol_freeze.json"),
            "--model", name, "--checkpoint", str(ctx.checkpoint(name)),
            "--structure-file", str(structures),
            "--threads-per-worker", str(args.threads_per_worker),
            "--output", str(ctx.probe_dir() / "worker_memory"),
            "--protected-root", str(ctx.data_root), "--resume",
        ]
        _run_in_runtime(ctx, name, command)
    return 0


def _runtime_environment(ctx: Context, name: str) -> dict[str, str]:
    _, record = inputs.model_runtime(ctx.protocol, name)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPOSITORY / "src")
    env["PYTHONNOUSERSITE"] = "1"  # model packages only from the frozen runtime
    env.update(record.get("environment_variables", {}))
    return env


def _run_in_runtime(ctx: Context, name: str, command: list[str]) -> None:
    print(f"[{name}] {' '.join(command)}", file=sys.stderr)
    completed = subprocess.run(command, env=_runtime_environment(ctx, name), check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{name}: runtime command failed with exit {completed.returncode}")


# --------------------------------------------------------------------------- infer


def command_infer(ctx: Context, args: argparse.Namespace) -> int:
    """Run the reference, then each model in its own runtime, strictly one at a time.

    Each model's pool is sized to the whole memory envelope, so running models
    concurrently could only split the same RAM; sequential runs keep each pool at
    its measured optimum and isolate failures.
    """

    runs = []
    if args.reference or args.all_models:
        runs.append(inference.REFERENCE_RUN)
    if args.all_models:
        runs += sorted(ctx.models)
    elif args.model:
        runs.append(args.model)
    if not runs:
        raise SystemExit("choose --reference, --model or --all-models")
    ctx.release_records()
    for run in runs:
        python = sys.executable if run == inference.REFERENCE_RUN else str(ctx.runtime_python(run))
        command = [python, str(Path(__file__).resolve()), "--config", str(ctx.config_path)]
        for flag, value in (("--protocol-dir", args.protocol_dir),
                            ("--output-root", args.output_root),
                            ("--checkpoint-map", args.checkpoint_map)):
            if value:
                command += [flag, str(value)]
        command += [
            "_runtime-infer", "--run", run,
            "--workers", str(args.workers), "--threads-per-worker", str(args.threads_per_worker),
            "--max-threads-per-worker", str(args.max_threads_per_worker),
            "--memory-reserve-gib", str(args.memory_reserve_gib),
        ]
        if args.resume:
            command.append("--resume")
        if run == inference.REFERENCE_RUN:
            print(f"[{run}] {' '.join(command)}", file=sys.stderr)
            env = dict(os.environ, PYTHONPATH=str(REPOSITORY / "src"))
            if subprocess.run(command, env=env, check=False).returncode != 0:
                raise SystemExit(f"{run}: inference failed")
        else:
            _run_in_runtime(ctx, run, command)
    return 0


def _probe_memory(ctx: Context, model: dict, max_threads: int) -> tuple[int, Path]:
    """Planning bytes (1.5 x measured VmHWM) from this model's probe report.

    Only reports measured with at least the largest thread count the run may use
    count, so per-thread buffers are covered.
    """

    import importlib.metadata

    directory = ctx.probe_dir() / "worker_memory" / model["canonical_name"]
    version = importlib.metadata.version(model["package_name"])
    matches = []
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        identity = report.get("identity", {})
        if (
            report.get("schema_version") == WORKER_MEMORY_REPORT_SCHEMA
            and report.get("fingerprint_sha256") == canonical_sha256(identity)
            and identity.get("weights_sha256") == model["weights_sha256"]
            and identity.get("package_version") == version
            and identity.get("structure_file_sha256")
            and identity.get("threads_per_worker", 0) >= max_threads
        ):
            matches.append((report["measurement"]["peak_resident_bytes"], path))
    if not matches:
        raise SystemExit(
            f"no worker-memory report for {model['canonical_name']}: run `probe` first, "
            "or pass an explicit --workers"
        )
    peak, path = max(matches)
    return int(peak * 1.5), path


def _execution_settings(ctx: Context, run_dir: Path, args, per_worker: int | None) -> int:
    """Threads per worker: stored on first start and reused on every resume."""

    path = run_dir / "execution_settings.json"
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))["threads_per_worker"]
        if args.threads_per_worker not in ("auto", str(stored)):
            raise SystemExit(f"this run uses {stored} threads per worker; resume with that value")
        return int(stored)
    host = observe_host_resources()
    if args.threads_per_worker != "auto":
        threads = int(args.threads_per_worker)
    elif per_worker is None or host.mem_available_bytes is None:
        threads = 1
    else:
        budget = host.mem_available_bytes - int(args.memory_reserve_gib * GIB)
        memory_workers = max(1, budget // per_worker)
        threads = max(1, min(args.max_threads_per_worker,
                             len(host.usable_cpus) // memory_workers))
    publish_json_once(path, {
        "classification": "operational_execution_setting",
        "threads_per_worker": threads,
        "rule": "auto: floor(usable CPUs / memory-envelope workers), capped by "
                "--max-threads-per-worker; fixed for the life of the run directory",
        "chosen_at_utc": _utc(),
    }, root=run_dir)
    return threads


def command_runtime_infer(ctx: Context, args: argparse.Namespace) -> int:
    from e3sse.execution.sharded_execution import ShardedRun, execute_shards

    started_wall, started_utc = time.perf_counter(), _utc()
    reference = args.run == inference.REFERENCE_RUN
    model = None if reference else ctx.models[args.run]
    if reference:
        environment_name, environment = "project", None
        runtime = {"python_version": sys.version.split()[0],
                   "recorded_dependencies": inference.installed_versions(
                       inference.RECORDED_DEPENDENCIES)}
    else:
        environment_name, environment = inputs.model_runtime(ctx.protocol, args.run)
        runtime = inference.verify_runtime(environment, model, _lock_file(environment_name,
                                                                          environment))
    git_sha, git_dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    if git_dirty and not args.allow_dirty:
        raise SystemExit("production Study-1 inference requires a clean worktree")
    run_dir = ctx.run_dir(args.run)
    index_manifest, index = _index(ctx)  # structural only
    device = ctx.settings["device"]
    identity = inference.run_identity(
        protocol=ctx.protocol,
        protocol_json_sha256=ctx.protocol_sha256,
        code=inference.code_sha256(REPOSITORY, inference.INFERENCE_CODE_FILES),
        source=ctx.source,
        frame_index_sha256=index_manifest["arrays_sha256"],
        frames_per_shard=ctx.settings["frames_per_shard"],
        model=model,
        environment_name=environment_name,
        environment_lock_sha256=None if reference else environment["environment_lock"]["sha256"],
        device=device,
        runtime_versions=runtime["recorded_dependencies"],
    )
    merged_manifest = run_dir / "frame_statistics_manifest.json"
    if merged_manifest.is_file():
        manifest, _ = inference.verify_statistics_file(run_dir)
        if manifest["scientific_identity"] != identity:
            raise SystemExit(f"[{args.run}] existing statistics belong to a different "
                             "scientific identity (code, runtime or inputs changed)")
        if not reference:
            try:
                inference.verify_pilot_gate_binding(run_dir, manifest.get("pilot_gate"), identity)
            except inference.PilotGateBindingError as exc:
                raise SystemExit(f"[{args.run}] {exc}") from exc
        print(f"[{args.run}] complete and verified: {manifest['statistics_sha256']}")
        return 0

    frames = ctx.train_frames()
    allowed, reason = _seal(ctx, [row["frame_id"] for row in frames])
    if not allowed:
        raise SystemExit(f"[{args.run}] stopped before reading any label: {reason}")
    ctx.release_records()
    shards = inference.plan_frame_shards(frames, index, ctx.settings["frames_per_shard"])
    checkpoint = None if reference else str(ctx.checkpoint(args.run))
    n_frames = len(frames)

    gate = None if reference else _pilot_gate(
        ctx, run_dir, model, checkpoint, frames, index, device, identity
    )
    del frames
    gc.collect()
    if not reference:
        per_worker, memory_report = (
            _probe_memory(ctx, model, args.max_threads_per_worker) if args.workers == "auto" else (None, None)
        )
    else:
        per_worker, memory_report = REFERENCE_WORKER_MEMORY_BYTES, None
    threads = _execution_settings(ctx, run_dir, args, per_worker)
    policy = WorkloadPolicy(
        name=f"study1_inference_{args.run}",
        threads_per_worker=threads,
        per_worker_memory_bytes=per_worker,
        memory_reserve_bytes=int(args.memory_reserve_gib * GIB),
        configure_torch=not reference and CONFIGURE_TORCH_FOR_MODELS,
    )
    plan = plan_workers(args.workers, policy, observe_host_resources(), unit_count=len(shards))
    run = ShardedRun(
        output_dir=run_dir,
        scientific_identity=identity,
        task=inference.infer_shard,
        protected_roots=ctx.protected,
        worker_setup=inference.inference_worker_setup,
        worker_setup_args=(model, checkpoint, str(ctx.source_path), device),
    )
    error, statistics = None, {}
    shards_started = time.perf_counter()
    try:
        result = execute_shards(run, shards, plan, resume=args.resume,
                                memory_floor_bytes=int(args.memory_reserve_gib * GIB))
    except Exception as exc:  # recorded in the run manifest, then re-raised
        error, result = exc, None
    shard_wall = time.perf_counter() - shards_started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        if result is not None:
            statistics = _merge(run_dir, run, result, identity, reference, git_sha, gate)
    except Exception as exc:
        error = exc
    wall = time.perf_counter() - started_wall
    evaluated = 0 if result is None else sum(
        len(result.results[shard_id]["frames"]) for shard_id in result.computed
    )
    record = {
        "schema_version": "e3sse.study1.inference.run-record.v1",
        "classification": "provenance_and_compute_accounting",
        "run": args.run,
        "protocol_id": ctx.protocol["protocol_id"],
        "protocol_json_sha256": ctx.protocol_sha256,
        "implementation_git_sha": git_sha,
        "implementation_git_dirty": git_dirty,
        "run_fingerprint": run.run_fingerprint,
        "model": identity["model"],
        "runtime_environment": environment_name,
        "environment_lock_sha256": identity["environment_lock_sha256"],
        "runtime": runtime,
        "source_file_sha256": ctx.source["sha256"],
        "provenance_index_id": ctx.source["provenance_index_id"],
        "membership_sha256": identity["membership_sha256"],
        "partition": inputs.PUBLISHED_PARTITION,
        "frame_index_sha256": index_manifest["arrays_sha256"],
        "pilot_gate": gate,
        "plan": plan.as_json(),
        "worker_memory_report": None if memory_report is None else str(memory_report),
        "frames": n_frames,
        "shards": len(shards),
        "shards_reused": None if result is None else len(result.reused),
        "shards_computed": None if result is None else len(result.computed),
        "execution_manifest": None if result is None else str(result.manifest_path),
        "started_at_utc": started_utc,
        "finished_at_utc": _utc(),
        "wall_seconds": round(wall, 3),
        "shard_execution_wall_seconds": round(shard_wall, 3),
        "children_cpu_seconds": round(usage.ru_utime + usage.ru_stime, 3),
        "peak_worker_resident_bytes": usage.ru_maxrss * 1024,
        "frames_evaluated": evaluated,
        "throughput_frames_per_second": round(evaluated / shard_wall, 3) if shard_wall > 0
        else None,
        "merged_statistics": statistics or None,
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "command": sys.argv,
    }
    publish_json_once(
        run_dir / "study1_runs" / f"{started_utc.replace(':', '').replace('-', '')}-{os.getpid()}.json",
        record, root=run_dir,
    )
    if error is not None:
        raise error
    print(json.dumps({"run": args.run, "statistics_sha256": statistics.get("statistics_sha256"),
                      "frames": statistics.get("frames"),
                      "frame_failures": len(statistics.get("frame_failures", []))}))
    return 0


def _merge(run_dir, run, result, identity, reference, git_sha, gate) -> dict:
    """Merged frame statistics and their manifest (idempotent after a crash)."""

    arrays = inference.merge_frame_statistics(result.results, reference=reference)
    statistics = {
        "schema_version": inference.INFERENCE_MANIFEST_SCHEMA,
        "run_fingerprint": run.run_fingerprint,
        "scientific_identity": identity,
        "merged_shard_sha256": result.merged_sha256,
        "statistics_file": "frame_statistics.npz",
        "statistics_sha256": inference.write_npz_once(run_dir / "frame_statistics.npz", arrays),
        "frames": len(arrays["frame_index"]),
        "frame_failures": [] if reference else inference.failures_of(result.results),
        "implementation_git_sha": git_sha,
        "pilot_gate": gate,
    }
    publish_json_once(run_dir / "frame_statistics_manifest.json", statistics, root=run_dir)
    return statistics


def _lock_file(environment_name: str, environment: dict) -> Path:
    """The frozen environment lock (path from the tracked freeze config, SHA from protocol)."""

    config = json.loads((REPOSITORY / "configs/protocol_freeze.json").read_text(encoding="utf-8"))
    record = config["inputs"]["production_evidence"]["runtime_environments"][environment_name]
    path = Path(record["environment_lock"]["path"])
    if inputs.file_sha256(path) != environment["environment_lock"]["sha256"]:
        raise SystemExit(f"environment lock {path} differs from the frozen protocol")
    return path


def _pilot_gate(ctx, run_dir, model, checkpoint, frames, index, device, identity) -> dict:
    """Frozen pilot gate; its report is keyed by the run identity and the gate settings.

    Returns the report's SHA-256 binding.  The first start publishes it once as
    ``pilot_gate_binding.json``; every resume must find the same report bytes.
    """

    chosen = inference.pilot_frames(frames, ctx.settings["pilot_gate_frames"])
    entries = [
        [row["frame_id"], int(row["source_frame_index"]),
         int(index["byte_offset"][row["source_frame_index"]]),
         int(index["byte_length"][row["source_frame_index"]]),
         int(index["n_atoms"][row["source_frame_index"]]), row["material_id"]]
        for row in chosen
    ]
    key = canonical_sha256({"identity": identity, "entries": entries,
                            "schema": inference.PILOT_GATE_SCHEMA,
                            "fd_step": ctx.settings["pilot_gate_fd_step_angstrom"]})
    report_name = f"pilot_gate/{key[:16]}.json"
    path = run_dir / report_name
    binding_path = run_dir / "pilot_gate_binding.json"
    if binding_path.is_file():
        try:
            bound = json.loads(binding_path.read_text(encoding="utf-8"))
            if not isinstance(bound, dict) or (
                bound.get("report"), bound.get("gate_key_sha256")
            ) != (report_name, key):
                raise inference.PilotGateBindingError(
                    "this run directory is bound to a different pilot gate")
            inference.verify_pilot_gate_binding(run_dir, bound, identity)
        except (ValueError, inference.PilotGateBindingError) as exc:
            raise SystemExit(f"pilot gate for {model['canonical_name']}: {exc}") from exc
        return bound
    if not path.is_file():
        with child_thread_environment(1), ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context(START_METHOD),
            initializer=_initialize_worker,
            initargs=(1, CONFIGURE_TORCH_FOR_MODELS, None, ()),
        ) as pool:
            report = pool.submit(
                _pilot_in_worker, inference.inference_worker_setup, model, checkpoint,
                str(ctx.source_path), entries,
                ctx.settings["pilot_gate_fd_step_angstrom"], device,
            ).result()
        report["gate_key_sha256"] = key
        publish_json_once(path, report, root=run_dir)
    try:
        binding = inference.pilot_gate_binding(run_dir, report_name, key, identity)
    except inference.PilotGateBindingError as exc:
        raise SystemExit(f"pilot gate for {model['canonical_name']}: {exc}") from exc
    if binding["status"] != "passed":
        report = json.loads(path.read_text(encoding="utf-8"))
        raise SystemExit(f"pilot gate failed for {model['canonical_name']}: {report.get('checks')}")
    publish_json_once(binding_path, binding, root=run_dir)
    return binding


def _pilot_in_worker(setup, model, checkpoint, source_path, entries, fd_step, device):
    context = setup(model, checkpoint, source_path, device)
    try:
        return inference.run_pilot_gate(
            model, context["calc"], context["handle"], entries, fd_step=fd_step, device=device
        )
    finally:
        context["handle"].close()


# --------------------------------------------------------------------------- verify


def command_verify(ctx: Context, args: argparse.Namespace) -> int:
    report = {"frame_index": None, "runs": {}}
    _index(ctx)
    report["frame_index"] = "verified"
    if args.rehash_source:
        report["source_sha256_matches"] = inputs.file_sha256(ctx.source_path) == ctx.source["sha256"]
    for run in [inference.REFERENCE_RUN, *sorted(ctx.models)]:
        directory = ctx.run_dir(run)
        if not (directory / "frame_statistics_manifest.json").is_file():
            report["runs"][run] = "missing"
            continue
        manifest, _ = inference.verify_statistics_file(directory)
        report["runs"][run] = {
            "statistics_sha256": manifest["statistics_sha256"],
            "frames": manifest["frames"],
            "frame_failures": len(manifest["frame_failures"]),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    ok = all(value != "missing" for value in report["runs"].values()) and report.get(
        "source_sha256_matches", True
    )
    return 0 if ok else 1


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPOSITORY / "configs/study1_softening_atlas.json"))
    parser.add_argument("--protocol-dir", default=None, help="override the frozen protocol dir")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--checkpoint-map", default=None, help="JSON {canonical_name: path}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("index-frames").add_argument("--resume", action="store_true")
    probe = sub.add_parser("probe")
    target = probe.add_mutually_exclusive_group(required=True)
    target.add_argument("--model")
    target.add_argument("--all-models", action="store_true")
    probe.add_argument("--threads-per-worker", type=int, default=8,
                       help="measure at the largest production thread count (default cap 8)")
    for name in ("infer", "_runtime-infer"):
        command = sub.add_parser(name)
        if name == "infer":
            command.add_argument("--model")
            command.add_argument("--all-models", action="store_true")
            command.add_argument("--reference", action="store_true")
        else:
            command.add_argument("--run", required=True)
            command.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
        command.add_argument("--workers", type=parse_workers, default="auto")
        command.add_argument("--threads-per-worker", default="auto")
        command.add_argument("--max-threads-per-worker", type=int, default=8)
        command.add_argument("--memory-reserve-gib", type=float, default=2.0)
        command.add_argument("--resume", action="store_true")
    sub.add_parser("verify").add_argument("--rehash-source", action="store_true")
    args = parser.parse_args(argv)
    ctx = Context(args)
    handlers = {
        "plan": command_plan, "index-frames": command_index, "probe": command_probe,
        "infer": command_infer, "_runtime-infer": command_runtime_infer, "verify": command_verify,
    }
    return handlers[args.command](ctx, args)


if __name__ == "__main__":
    raise SystemExit(main())
