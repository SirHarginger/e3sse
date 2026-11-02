"""Operational memory and throughput measurement for one (model, workload class).

Runs inside the model's own frozen runtime.  It reads no dataset label and no
scientific outcome.  Its inputs are geometry-only structures chosen from structure
alone (e.g. the largest and densest published-train frames that
``run_study1_softening_atlas.py probe`` selects) or a deterministic synthetic Li2O
vacancy band.

Workload classes:

* ``force_single_point``: energy + forces of each structure (ESTIMATE, TRANSFER);
* ``ml_neb_band``: the frozen NEB + FIRE on a synthetic band, a few steps, one
  calculator per image as in production (DEPLOYMENT CALIBRATION, DISCOVERY).

One grid point launches ``processes`` fresh spawned interpreters, each configured
with ``threads`` (environment variables before start, torch setters before any
model work).  Each loads its calculator, warms up, waits at a barrier and then
evaluates.  Aggregate throughput = total units / wall span between the first start
and the last finish.  Peak resident memory is the largest per-process VmHWM.  A
dead or timed-out process makes the point ``failed``.  That is infrastructure
evidence, never a result.
"""

from __future__ import annotations

import math
import multiprocessing
import queue as queue_module
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sharded_execution import START_METHOD
from .thread_control import child_thread_environment, configure_process_threads
from .worker_memory_probe import peak_resident_bytes

BENCHMARK_WORKLOADS = ("force_single_point", "ml_neb_band")
CALCULATOR_KINDS = ("frozen_model", "synthetic_emt_test_only")


def synthetic_li_vacancy_band(min_atoms: int, interior_images: int) -> list[Any]:
    """Deterministic Li2O supercell with one Li vacancy hop (linear, minimum image)."""

    import numpy as np
    from ase.build import bulk

    from .worker_memory_probe import probe_supercell_repeats

    repeats = probe_supercell_repeats(min_atoms + 1)
    host = bulk("Li2O", "fluorite", a=4.62, cubic=True).repeat((repeats,) * 3)
    lithium = [i for i, s in enumerate(host.get_chemical_symbols()) if s == "Li"]
    vacancy = lithium[0]
    vectors = host.get_distances(vacancy, lithium, mic=True, vector=True)
    lengths = np.linalg.norm(vectors, axis=1)
    lengths[lithium.index(vacancy)] = np.inf
    hopping = lithium[int(np.argmin(lengths))]
    initial, final = host.copy(), host.copy()
    final.positions[hopping] = host.positions[hopping] - vectors[lithium.index(hopping)]
    del initial[vacancy]
    del final[vacancy]
    band = [initial]
    for index in range(1, interior_images + 1):
        image = initial.copy()
        image.positions = initial.positions + (final.positions - initial.positions) * (
            index / (interior_images + 1))
        band.append(image)
    band.append(final)
    return band


def _calculator_factory(spec: Mapping[str, Any]) -> Any:
    kind = spec["kind"]
    if kind == "frozen_model":
        from e3sse.protocol.adapters import build_calculator

        model, checkpoint = dict(spec["model"]), Path(spec["checkpoint"])
        return lambda: build_calculator(model, checkpoint)
    if kind == "synthetic_emt_test_only":
        from ase.calculators.emt import EMT

        return EMT
    raise ValueError(f"unknown calculator kind {kind!r}")


def _structures(payload: Mapping[str, Any]) -> list[Any]:
    from ase import Atoms

    return [Atoms(numbers=item["numbers"], positions=item["positions"], cell=item["cell"],
                  pbc=item["pbc"]) for item in payload["structures"]]


def _setup(spec: Mapping[str, Any], workload: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    factory = _calculator_factory(spec)
    if workload == "force_single_point":
        calc = factory()
        return {"calc": calc, "structures": _structures(payload)}
    if workload == "ml_neb_band":
        from e3sse.barriers.ml_neb import settings_from_protocol

        # The frozen protocol section itself, parsed by the same strict reader as production.
        settings = settings_from_protocol({"amendments": {"prospective": {"ml_neb": {
            "frozen_numeric_settings": payload["frozen_numeric_settings"]}}}})
        band = _structures(payload)
        for image in band:
            image.calc = factory()  # one calculator per image, as in production
        return {"band": band, "settings": settings}
    raise ValueError(f"unknown workload {workload!r}")


def _evaluate(context: Mapping[str, Any], workload: str, evaluations: int) -> int:
    """Run ``evaluations`` repetitions; return the number of structure evaluations."""

    if workload == "force_single_point":
        units = 0
        for _ in range(evaluations):
            for atoms in context["structures"]:
                context["calc"].reset()  # never time a cached result
                atoms.calc = context["calc"]
                atoms.get_potential_energy()
                atoms.get_forces()
                units += 1
        return units
    from e3sse.barriers.ml_neb import build_neb, fire_optimizer

    band = [image.copy() for image in context["band"]]
    for copy, image in zip(band, context["band"]):
        copy.calc = image.calc
    neb = build_neb(band, context["settings"])
    optimizer = fire_optimizer(neb, context["settings"])
    # One FIRE step evaluates every interior image once; endpoints are evaluated once.
    optimizer.run(fmax=1e-12, steps=evaluations)
    # Each interior image is evaluated once at the start and once per step.
    return (int(optimizer.get_number_of_steps()) + 1) * (len(band) - 2)


def _benchmark_process(spec, workload, payload, threads, evaluations, warmup, barrier, results,
                       configure_torch):
    try:
        configure_process_threads(threads, configure_torch=configure_torch)
        after_imports = peak_resident_bytes()
        load_start = time.perf_counter()
        context = _setup(spec, workload, payload)
        load_seconds = time.perf_counter() - load_start
        # A model library must not have reset the thread pools while loading.
        configure_process_threads(threads, configure_torch=configure_torch)
        after_load = peak_resident_bytes()
        if warmup:
            _evaluate(context, workload, warmup)
        barrier.wait(timeout=3600)
        start_wall, start = time.time(), time.perf_counter()
        units = _evaluate(context, workload, evaluations)
        elapsed = time.perf_counter() - start
        results.put({"status": "completed", "units": units, "elapsed_seconds": elapsed,
                     "start_wall": start_wall, "end_wall": start_wall + elapsed,
                     "load_seconds": load_seconds,
                     "peak_resident_bytes": peak_resident_bytes(),
                     "peak_resident_bytes_after_imports": after_imports,
                     "peak_resident_bytes_after_model_load": after_load})
    except BaseException as exc:  # reported to the parent as infrastructure evidence
        try:
            barrier.abort()
        finally:
            results.put({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})


def run_grid_point(
    spec: Mapping[str, Any],
    workload: str,
    payload: Mapping[str, Any],
    *,
    processes: int,
    threads: int,
    evaluations: int,
    warmup: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if workload not in BENCHMARK_WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}")
    if processes < 1 or threads < 1 or evaluations < 1:
        raise ValueError("processes, threads and evaluations must be positive")
    context = multiprocessing.get_context(START_METHOD)
    configure_torch = spec["kind"] == "frozen_model"
    with child_thread_environment(threads):
        barrier = context.Barrier(processes)
        results = context.Queue()
        workers = [context.Process(target=_benchmark_process,
                                   args=(dict(spec), workload, dict(payload), threads, evaluations,
                                         warmup, barrier, results, configure_torch))
                   for _ in range(processes)]
        started = []
        try:
            for worker in workers:
                worker.start()
                started.append(worker)
        except BaseException:
            for worker in started:
                worker.kill()
                worker.join()
            raise
    deadline = time.monotonic() + timeout_seconds
    reports = []
    try:
        while len(reports) < processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                reports.append(results.get(timeout=min(remaining, 2.0)))
            except queue_module.Empty:
                dead = [w for w in workers if not w.is_alive() and w.exitcode not in (0, None)]
                if dead:
                    # e.g. an OOM kill before the barrier: release the waiting siblings.
                    barrier.abort()
                if all(not worker.is_alive() for worker in workers) and results.empty():
                    break
    finally:
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.kill()
                worker.join()
    exit_codes = [worker.exitcode for worker in workers]
    point = {"processes": processes, "threads_per_process": threads, "evaluations": evaluations,
             "warmup": warmup, "exit_codes": exit_codes}
    completed = [r for r in reports if r["status"] == "completed"]
    if len(completed) != processes:
        errors = sorted({r.get("error", "") for r in reports if r["status"] != "completed"})
        point.update(status="failed", classification="infrastructure",
                     reason=errors or [(f"{processes - len(completed)} process(es) did not "
                                        f"report (exit codes {exit_codes}; e.g. OOM kill or "
                                        "timeout)")])
        return point
    span = max(r["end_wall"] for r in completed) - min(r["start_wall"] for r in completed)
    units = sum(r["units"] for r in completed)
    point.update(
        status="completed", units=units, wall_span_seconds=span,
        units_per_second=units / span if span > 0 else math.inf,
        peak_resident_bytes=max(r["peak_resident_bytes"] for r in completed),
        peak_resident_bytes_after_imports=max(r["peak_resident_bytes_after_imports"]
                                              for r in completed),
        peak_resident_bytes_after_model_load=max(r["peak_resident_bytes_after_model_load"]
                                                 for r in completed),
        model_load_seconds=max(r["load_seconds"] for r in completed),
    )
    return point


def benchmark_grid(processes: Sequence[int], threads: Sequence[int], *, usable_cpus: int,
                   memory_budget_bytes: int, planning_bytes_per_process: int) -> list[tuple[int, int]]:
    """Grid points that fit the CPU and memory envelopes, smallest load first."""

    points = []
    for p in sorted(set(processes)):
        for t in sorted(set(threads)):
            if p * t <= usable_cpus and p * planning_bytes_per_process <= memory_budget_bytes:
                points.append((p, t))
    return sorted(points, key=lambda item: (item[0] * item[1], item[0]))
