"""bvse_reproduction: frozen BVSE initial paths (ions) and their reproduction check.

Runs with the dedicated BVSE runtime interpreter (``runners.bvse.python``), whose
``ions`` version must equal the configured pin (``runners.bvse.ions_version``; the
candidate 0.4.1 is the release matching LiTraj c3ca5c2 ``notebooks/bvse_neb.ipynb``).
The recipe is that notebook's, applied to the unit's own rate-limiting edge:

    Decorator().decorate(atoms); collect_bvse_params(atoms, "Li", 1)
    Edge(atoms, source, target, offset).superedge(8.0).interpolate(n_images=5)
    SaddleFinder().bvse_neb(images); FIRE(neb, logfile=None).run(fmax=0.1, steps=100)

For development and P units the regenerated path is compared with nebDFT2k
``<edge_id>_init.xyz`` (geometry only): same image count, atom count and order, cell
within ``cell_A`` and every minimum-image position within
``max_atom_displacement_A`` (``ml_neb.frozen_numeric_settings.bvse_reproduction_tolerance``).
A mismatch excludes the unit and is counted; an oxidation-state decoration failure
is its own initialisation-failure category.  Pool units cannot be checked.
"""

from __future__ import annotations

from typing import Any

from e3sse.production.runners.context import (
    MissingEvidence,
    RunnerContext,
    RunnerValidationError,
)

BVSE_NAMESPACE = "bvse.path"
CODE_FILES = ("src/e3sse/production/runners/bvse_paths.py",
              "src/e3sse/production/runners/structures.py",
              "src/e3sse/production/runners/hop_files.py")
INFRASTRUCTURE = (MemoryError, OSError, KeyboardInterrupt, SystemExit)


def recipe(protocol_amendments: dict[str, Any]) -> dict[str, Any]:
    ml_neb = protocol_amendments["prospective"]["ml_neb"]
    tolerance = ml_neb["frozen_numeric_settings"]["bvse_reproduction_tolerance"]
    return {"mobile_symbol": "Li", "mobile_charge": 1, "superedge_cutoff_A": 8.0,
            "n_images": 5, "fire_fmax": 0.1, "fire_steps": 100,
            "cell_tolerance_A": float(tolerance["cell_A"]),
            "position_tolerance_A": float(tolerance["max_atom_displacement_A"])}


def compare_paths(generated: list[dict[str, Any]], reference: list[dict[str, Any]],
                  settings: dict[str, Any]) -> dict[str, Any]:
    """Frozen reproduction tolerance: count/order identical, cell and positions close."""

    import numpy as np
    from ase.geometry import find_mic

    if len(generated) != len(reference):
        return {"matches": False, "reason": "image_count_differs",
                "n_generated": len(generated), "n_reference": len(reference)}
    worst_cell = worst_position = 0.0
    for mine, theirs in zip(generated, reference):
        if mine["numbers"] != theirs["numbers"]:
            return {"matches": False, "reason": "atom_count_or_order_differs"}
        cell = np.asarray(mine["cell"])
        worst_cell = max(worst_cell, float(np.abs(cell - np.asarray(theirs["cell"])).max()))
        delta = np.asarray(mine["positions"]) - np.asarray(theirs["positions"])
        _, lengths = find_mic(delta, cell, mine["pbc"])
        worst_position = max(worst_position, float(np.max(lengths)) if len(lengths) else 0.0)
    matches = (worst_cell <= settings["cell_tolerance_A"]
               and worst_position <= settings["position_tolerance_A"])
    return {"matches": matches, "reason": None if matches else "geometry_outside_tolerance",
            "max_cell_difference_A": worst_cell, "max_position_difference_A": worst_position}


def _geometry(atoms: Any) -> dict[str, Any]:
    return {"numbers": [int(z) for z in atoms.numbers], "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist() if hasattr(atoms.cell, "array") else
            [list(map(float, row)) for row in atoms.cell],
            "pbc": [bool(flag) for flag in atoms.pbc]}


def bvse_path_task(payload: dict[str, Any], context: Any) -> dict[str, Any]:
    """One unit: decorate, BVSE parameters, superedge path, BVSE NEB, compare."""

    from ase import Atoms
    from ase.optimize import FIRE

    settings = payload["recipe"]
    result: dict[str, Any] = {"unit_key": payload["unit_key"], "unit_id": payload["unit_id"]}
    structure = payload["structure"]
    atoms = Atoms(numbers=structure["numbers"], positions=structure["positions"],
                  cell=structure["cell"], pbc=structure["pbc"])
    try:
        from ions import Decorator

        atoms = Decorator().decorate(atoms)
    except INFRASTRUCTURE:
        raise
    except Exception as exc:  # the oxidation-state decoration is its own failure category
        return {**result, "status": "decorator_failure", "detail": f"{type(exc).__name__}: {exc}"}
    try:
        from ions.geom.edge import Edge
        from ions.tools import SaddleFinder
        from ions.utils import collect_bvse_params

        atoms = collect_bvse_params(atoms, settings["mobile_symbol"], settings["mobile_charge"])
        edge = Edge(atoms, payload["edge"]["source"], payload["edge"]["target"],
                    payload["edge"]["offset"])
        images = edge.superedge(settings["superedge_cutoff_A"]).interpolate(
            n_images=settings["n_images"])
        finder = SaddleFinder()
        neb = finder.bvse_neb(images)
        FIRE(neb, logfile=None).run(fmax=settings["fire_fmax"], steps=settings["fire_steps"])
        bvse_fmax = float(abs(neb.get_forces()).max())
        bvse_barrier = float(finder.get_barrier(images))
    except INFRASTRUCTURE:
        raise
    except Exception as exc:  # an ions failure on this structure: initialisation failure
        return {**result, "status": "ions_failure", "detail": f"{type(exc).__name__}: {exc}"}
    geometry = [_geometry(image) for image in images]
    result.update(status="generated", images=geometry, bvse_fmax_eV_per_A=bvse_fmax,
                  bvse_barrier_eV=bvse_barrier)
    if payload.get("reference") is not None:
        result["reproduction"] = compare_paths(geometry, payload["reference"], settings)
    return result


def run(ctx: RunnerContext) -> dict[str, Any]:
    import importlib.metadata

    from e3sse.execution.sharded_execution import _shard_path, canonical_sha256, make_shard
    from e3sse.production.runners.gate1_tables import pinned_g0_inputs
    from e3sse.production.runners.hop_files import nebdft2k_files, read_geometry
    from e3sse.production.runners.structures import StructureSource
    from e3sse.screening.rate_limiting_hop import parse_edge_id

    bvse = ctx.runner_settings("bvse")
    try:
        installed = importlib.metadata.version("ions")
    except importlib.metadata.PackageNotFoundError as exc:
        raise MissingEvidence("ions is not installed in this interpreter; create the pinned "
                              "BVSE runtime (runners.bvse.python)") from exc
    if installed != bvse.get("ions_version"):
        raise RunnerValidationError(f"ions {installed} is not the pinned {bvse.get('ions_version')}")
    population = ctx.upstream("development_population", "population_manifest")
    settings = recipe(ctx.active_protocol["amendments"])
    structures = StructureSource(ctx.runner_settings("structures").get("source"), ctx.data_root,
                                 pinned_g0_inputs(ctx.freeze_config))
    files = nebdft2k_files(ctx.data_root / "raw")
    units, failures = [], {}
    for unit in population["units"]:
        if unit["role"] == "excluded_gate0":
            continue
        try:
            atoms = structures.load(unit["material_id"])
        except MissingEvidence as exc:
            failures[unit["unit_key"]] = {"status": "structure_missing", "detail": str(exc)}
            continue
        reference = None
        if unit["side"] in ("development", "P"):
            path = files.get(unit["unit_id"], {}).get("init")
            reference = read_geometry(path) if path is not None else None
            if reference is None:
                failures[unit["unit_key"]] = {"status": "init_path_missing"}
                continue
        payload = {"unit_key": unit["unit_key"], "unit_id": unit["unit_id"],
                   "edge": dict(parse_edge_id(unit["unit_id"])), "structure": _geometry(atoms),
                   "reference": reference, "recipe": settings}
        units.append(({"unit_key": unit["unit_key"], "unit_id": unit["unit_id"]}, payload))
    identity = {"code_sha256": ctx.code_sha256(CODE_FILES), "ions_version": installed,
                "recipe": settings, "structure_source": structures.identity()}
    results = ctx.execute(namespace=BVSE_NAMESPACE, units=units, task=bvse_path_task,
                          identity=identity)
    shard_dir = ctx.work_dir / "shards" / BVSE_NAMESPACE / "shards"
    index, counts = {}, {}
    for unit_identity, _ in units:
        shard_id = make_shard(BVSE_NAMESPACE, unit_identity).shard_id
        result = results[shard_id]
        status = result["status"]
        reproduced = result.get("reproduction", {}).get("matches")
        usable = status == "generated" and reproduced is not False
        index[unit_identity["unit_key"]] = {
            "status": status, "usable_for_ml_neb": usable,
            "reproduction": result.get("reproduction"),
            "checkpoint": str(_shard_path(shard_dir, shard_id)),
            "result_sha256": canonical_sha256(result)}
        key = status if status != "generated" else (
            "generated" if reproduced is None else
            ("reproduced" if reproduced else "reproduction_mismatch"))
        counts[key] = counts.get(key, 0) + 1
    for key, failure in failures.items():
        index[key] = {**failure, "usable_for_ml_neb": False}
        counts[failure["status"]] = counts.get(failure["status"], 0) + 1
    checked = counts.get("reproduced", 0) + counts.get("reproduction_mismatch", 0)
    report = {"ions_version": installed, "recipe": settings, "counts": dict(sorted(counts.items())),
              "calibration_reproduction_rate": counts.get("reproduced", 0) / checked
              if checked else None,
              "policy": "mismatch -> excluded and counted; decorator failure is its own "
                        "initialisation-failure category; pool units are not checkable"}
    outputs = {"bvse_path_index": ctx.publish("bvse_path_index.json", index),
               "bvse_reproduction_report": ctx.publish("bvse_reproduction_report.json", report)}
    return {"outputs": outputs, "summary": report}


def load_path(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """A unit's verified BVSE path images from its shard checkpoint."""

    import json

    from e3sse.execution.sharded_execution import canonical_sha256

    if not entry.get("usable_for_ml_neb"):
        raise ValueError("this unit has no usable BVSE path")
    with open(entry["checkpoint"], encoding="utf-8") as handle:
        stored = json.load(handle)
    if canonical_sha256(stored["result"]) != entry["result_sha256"]:
        raise RunnerValidationError("BVSE path checkpoint differs from its index")
    return stored["result"]["images"]
