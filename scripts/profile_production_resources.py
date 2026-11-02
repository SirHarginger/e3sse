#!/usr/bin/env python3
"""Measure one frozen model's worker memory and throughput for one workload class.

Operational evidence for the production scheduler (``e3sse.production.
resource_scheduler``), never a scientific result.  Run it with the model's own
frozen interpreter (the production orchestrator's ``resource_profile`` stage does
this for every model).  Inputs are geometry only: ``--structure-file`` for
``force_single_point`` (structures chosen from structure alone, e.g. by
``run_study1_softening_atlas.py probe``), or a deterministic synthetic Li2O vacancy
band for ``ml_neb_band``.  No label or outcome is read.

Procedure: one worker at ``--memory-threads`` measures the peak resident memory.
Then every (processes, threads) grid point that fits the CPU set and the memory
budget (MemAvailable minus reserve, with the measured peak times the safety factor
per process) is timed.  Memory pressure is re-checked before each point, and a point
that would enter the reserve is skipped.  The report is published once to
``<output>/<model>/<workload>-<fingerprint>.json``.  ``--resume`` reuses it only
when its identity (model, package, code, inputs, grid) is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path

from e3sse.execution.resource_planning import GIB
from e3sse.execution.sharded_execution import (
    canonical_sha256,
    ensure_safe_output_dir,
    publish_json_once,
)
from e3sse.execution.workload_benchmark import (
    BENCHMARK_WORKLOADS,
    benchmark_grid,
    run_grid_point,
    synthetic_li_vacancy_band,
)
from e3sse.production.resource_scheduler import discover_resources

REPOSITORY = Path(__file__).resolve().parents[1]
RESOURCE_PROFILE_SCHEMA = "e3sse.production.resource-profile"
CODE_FILES = ("scripts/profile_production_resources.py",
              "src/e3sse/execution/workload_benchmark.py",
              "src/e3sse/barriers/ml_neb.py", "src/e3sse/protocol/adapters.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_entry(freeze_config: dict, name: str) -> dict:
    for group in freeze_config["protocol"]["model_groups"].values():
        for model in group:
            if model["canonical_name"] == name:
                return model
    raise SystemExit(f"model {name!r} is not in the frozen registry")


def _grid(text: str) -> list[int]:
    values = sorted({int(item) for item in text.split(",") if item.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("grid values must be positive integers")
    return values


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPOSITORY), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol-freeze-config", type=Path, required=True,
                        help="configs/protocol_freeze.json (frozen model registry)")
    parser.add_argument("--active-protocol", type=Path, required=True,
                        help="active protocol config (frozen ML-NEB settings)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--workload", choices=BENCHMARK_WORKLOADS, required=True)
    parser.add_argument("--structure-file", type=Path, default=None)
    parser.add_argument("--ml-neb-min-atoms", type=int, default=None)
    parser.add_argument("--ml-neb-interior-images", type=int, default=5)
    parser.add_argument("--memory-threads", type=int, required=True)
    parser.add_argument("--process-grid", type=_grid, required=True)
    parser.add_argument("--thread-grid", type=_grid, required=True)
    parser.add_argument("--evaluations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--memory-reserve-gib", type=float, required=True)
    parser.add_argument("--memory-safety-factor", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--max-cpus", type=int, default=None,
                        help="operator cap forwarded by the orchestrator (only narrows)")
    parser.add_argument("--max-memory-gib", type=float, default=None,
                        help="operator cap forwarded by the orchestrator (only narrows)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protected-root", type=Path, action="append", required=True)
    parser.add_argument("--synthetic-test-calculator", action="store_true",
                        help=argparse.SUPPRESS)  # tests only: EMT instead of a frozen model
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    freeze_config = json.loads(args.protocol_freeze_config.read_text(encoding="utf-8"))
    active = json.loads(args.active_protocol.read_text(encoding="utf-8"))
    frozen_neb = active["amendments"]["prospective"]["ml_neb"]["frozen_numeric_settings"]
    if args.synthetic_test_calculator:
        spec = {"kind": "synthetic_emt_test_only"}
        model_identity = {"canonical_name": args.model, "kind": spec["kind"]}
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for a frozen model")
        model = _model_entry(freeze_config, args.model)
        installed = importlib.metadata.version(model["package_name"])
        if installed != model["package_version"]:
            raise SystemExit(f"{model['package_name']} {installed} is not the frozen "
                             f"{model['package_version']}: run with the model's frozen runtime")
        spec = {"kind": "frozen_model", "model": model, "checkpoint": str(args.checkpoint)}
        model_identity = {"canonical_name": args.model, "kind": spec["kind"],
                          "weights_sha256": model["weights_sha256"],
                          "package": f"{model['package_name']}=={installed}",
                          "adapter_version": model["adapter_version"]}

    if args.workload == "force_single_point":
        if args.structure_file is None:
            raise SystemExit("force_single_point needs --structure-file (geometry only)")
        raw = args.structure_file.read_bytes()
        payload = {"structures": json.loads(raw)["structures"]}
        inputs = {"structure_file_sha256": hashlib.sha256(raw).hexdigest(),
                  "structures": len(payload["structures"])}
    else:
        if args.ml_neb_min_atoms is None:
            raise SystemExit("ml_neb_band needs --ml-neb-min-atoms")
        band = synthetic_li_vacancy_band(args.ml_neb_min_atoms, args.ml_neb_interior_images)
        payload = {"structures": [{"numbers": a.numbers.tolist(), "positions": a.positions.tolist(),
                                   "cell": a.cell.tolist(), "pbc": a.pbc.tolist()} for a in band],
                   "frozen_numeric_settings": frozen_neb}
        inputs = {"synthetic_band": "li2o_fluorite_single_li_vacancy_linear",
                  "n_atoms": len(band[0]), "images": len(band),
                  "frozen_ml_neb_settings_sha256": canonical_sha256(frozen_neb)}
    identity = {
        "model": args.model, "workload": args.workload, "model_identity": model_identity,
        "inputs": inputs, "memory_threads_per_worker": args.memory_threads,
        "process_grid": args.process_grid, "thread_grid": args.thread_grid,
        "evaluations": args.evaluations, "warmup": args.warmup,
        "runtime_python": sys.version.split()[0],
        "code_sha256": {name: _sha256(REPOSITORY / name) for name in CODE_FILES},
    }
    fingerprint = canonical_sha256(identity)
    output = ensure_safe_output_dir(args.output, args.protected_root)
    path = output / args.model / f"{args.workload}-{fingerprint[:16]}.json"
    if path.exists():
        if not args.resume:
            raise SystemExit(f"{path} exists; use --resume")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("fingerprint_sha256") != fingerprint or canonical_sha256(
                stored.get("identity")) != fingerprint:
            raise SystemExit(f"stale resource profile {path}: identity changed")
        print(f"verified existing resource profile {path}")
        return 0

    common = {"evaluations": args.evaluations, "warmup": args.warmup,
              "timeout_seconds": args.timeout_seconds}
    memory_point = run_grid_point(spec, args.workload, payload, processes=1,
                                  threads=args.memory_threads, **common)
    if memory_point["status"] != "completed":
        raise SystemExit(f"memory measurement failed (infrastructure): {memory_point}")
    peak = int(memory_point["peak_resident_bytes"])
    planning = math.ceil(peak * args.memory_safety_factor)
    reserve = int(args.memory_reserve_gib * GIB)
    host = discover_resources(max_cpus=args.max_cpus, max_memory_gib=args.max_memory_gib)
    budget = max(0, (host.mem_available_bytes or 0) - reserve)
    grid = benchmark_grid(args.process_grid, args.thread_grid, usable_cpus=len(host.usable_cpus),
                          memory_budget_bytes=budget, planning_bytes_per_process=planning)
    points = []
    for processes, threads in grid:
        current = discover_resources(max_cpus=args.max_cpus, max_memory_gib=args.max_memory_gib)
        if (current.mem_available_bytes or 0) - reserve < processes * planning:
            points.append({"processes": processes, "threads_per_process": threads,
                           "status": "skipped_memory_pressure"})
            continue
        point = run_grid_point(spec, args.workload, payload, processes=processes,
                               threads=threads, **common)
        print(json.dumps({key: point.get(key) for key in
                          ("processes", "threads_per_process", "status", "units_per_second")}),
              file=sys.stderr)
        points.append(point)
    report = {
        "schema_version": RESOURCE_PROFILE_SCHEMA,
        "classification": "operational_resource_measurement_not_scientific",
        "identity": identity,
        "fingerprint_sha256": fingerprint,
        "measurement": {"peak_resident_bytes": peak, "memory_point": memory_point,
                        "planning_bytes_per_worker": planning,
                        "memory_safety_factor": args.memory_safety_factor},
        "benchmark": {"memory_budget_bytes": budget, "memory_reserve_bytes": reserve,
                      "points": points,
                      "selection_rule": "highest measured aggregate units_per_second among "
                                        "points that fit the run-time envelope"},
        "host": host.as_json(),
        "installed_versions": {name: _version(name) for name in ("ase", "numpy", "torch")},
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unavailable"),
        "command": sys.argv,
    }
    publish_json_once(path, report, root=output)
    print(path)
    return 0


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
