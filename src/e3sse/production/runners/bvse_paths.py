"""bvse_reproduction: frozen BVSE initial paths (ions) and their reproduction check.

Runs with the dedicated BVSE runtime interpreter (``runners.bvse.python``), whose
``ions`` version must equal the configured pin (``runners.bvse.ions_version``).  The
recipe is the historical nebDFT2k generation method reconstructed from the archive
(docs/research/bvse_reproduction_reconstruction.md), applied to the unit's own edge:

    oxidation states: pymatgen BVAnalyzer().get_valences (``decoration``)
    collect_bvse_params(atoms, "Li", 1)
    edge = Edge(atoms, source, target, offset)
    edge.superedge(8.0).interpolate(n_images=7 if edge.length >= 4.5 else 5)
    neb = SaddleFinder().bvse_neb(images, k=2.0, default=True, distort=True, gm=False)
    SaddleFinder().run(neb, fmax=0.1, steps=100, relax_endpoints=False)

``run`` relaxes the band in the seed-42 perturbed framework, subtracts the
perturbation and relaxes again (ions' documented procedure); the archived paths carry
no perturbation and unrelaxed endpoints, hence ``relax_endpoints=False``.  The image
count rule (``image_count``) is the archive's observed convention, adopted under the
frozen ``image_count`` provision; it depends only on the structure and the edge.

For development and P units the regenerated path is compared with nebDFT2k
``<edge_id>_init.xyz`` (geometry only): same image count, atom count and order, cell
within ``cell_A`` and every minimum-image position within
``max_atom_displacement_A`` (``ml_neb.frozen_numeric_settings.bvse_reproduction_tolerance``).
A mismatch excludes the unit and is counted; an oxidation-state decoration failure
is its own initialisation-failure category.  Pool units cannot be checked against
``_init.xyz``; for every unit the regenerated BVSE barrier is compared with the
nebBVSE122k barrier of the same edge (label-free) and recorded, and excluded only
under a frozen ``pool_barrier_tolerance_eV``.

Every ``SaddleFinder.bvse_neb`` option is passed explicitly from the frozen
``ml_neb.bvse_recipe`` (``k=2.0``, ``default=True`` which builds
``NEB(climb=False, method='improvedtangent')``, ``distort=True`` which adds a
seed-42 framework perturbation of at most 0.2 A that ``run`` removes before its final
relaxation, ``gm=False``), as are the decoration and the ``run`` driver options.  The
runtime's package versions are verified against its pinned lock
(``runners.bvse.environment_lock``) and enter the shard identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from e3sse.production.runners.context import (
    MissingEvidence,
    RunnerContext,
    RunnerValidationError,
)

BVSE_NAMESPACE = "bvse.path"
CODE_FILES = ("src/e3sse/production/runners/bvse_paths.py",
              "src/e3sse/production/runners/structures.py",
              "src/e3sse/production/runners/hop_files.py",
              "src/e3sse/softening/extxyz_frames.py")
INFRASTRUCTURE = (MemoryError, OSError, KeyboardInterrupt, SystemExit)
RUNTIME_PACKAGES = ("ions", "ase", "numpy", "scipy", "spglib", "pymatgen", "networkx", "PyYAML")
BVSE_NEB_OPTIONS = ("k", "default", "distort", "gm")
DECORATIONS = ("pymatgen_bvanalyzer", "ions_decorator")


POOL_TOLERANCE_EVIDENCE_SOURCES = ("bvse_reproduction_server_audit",
                                   "numerical_repeatability_report")
STAGE_REPORT_NAME = "bvse_reproduction_report.json"


def pool_tolerance_evidence_problem(evidence: Any) -> str | None:
    """Why pinned pool-tolerance evidence is unacceptable (None if acceptable).

    The evidence must be the predetermined server reproduction audit or a numerical-
    repeatability run, pinned by path, SHA-256 and the commit that produced it; never
    this stage's own report, whose pool-difference distribution includes calibration
    units (a tolerance read off it would let calibration covariates build the filter).
    """

    import re

    if not isinstance(evidence, dict):
        return "evidence must be {source, path, sha256, git_commit}"
    if evidence.get("source") not in POOL_TOLERANCE_EVIDENCE_SOURCES:
        return f"evidence source must be one of {POOL_TOLERANCE_EVIDENCE_SOURCES}"
    if not evidence.get("path") or Path(str(evidence["path"])).name == STAGE_REPORT_NAME:
        return "evidence path is missing or is this stage's own report"
    if not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256", ""))):
        return "evidence sha256 must be a SHA-256 digest"
    if not re.fullmatch(r"[0-9a-f]{40}", str(evidence.get("git_commit", ""))):
        return "evidence git_commit must name the commit that produced it"
    return None


def recipe(protocol_amendments: dict[str, Any]) -> dict[str, Any]:
    """The frozen BVSE recipe (protocol ``ml_neb.bvse_recipe``), never code constants."""

    ml_neb = protocol_amendments["prospective"]["ml_neb"]
    tolerance = ml_neb["frozen_numeric_settings"]["bvse_reproduction_tolerance"]
    frozen = ml_neb["bvse_recipe"]
    options = frozen["bvse_neb_options"]
    if sorted(options) != sorted(BVSE_NEB_OPTIONS):
        raise RunnerValidationError(f"bvse_neb_options must name exactly {BVSE_NEB_OPTIONS}")
    if frozen.get("decoration") not in DECORATIONS:
        raise RunnerValidationError(f"bvse_recipe.decoration must be one of {DECORATIONS}")
    count = frozen.get("image_count") or {}
    if (sorted(count) != ["long_hop_images", "long_hop_min_length_A", "short_hop_images"]
            or not all(isinstance(count[k], int) and count[k] >= 3 and count[k] % 2
                       for k in ("short_hop_images", "long_hop_images"))
            or not float(count["long_hop_min_length_A"]) > 0):
        raise RunnerValidationError("bvse_recipe.image_count must be {short_hop_images, "
                                    "long_hop_images (odd ints >= 3), long_hop_min_length_A}")
    driver = frozen.get("driver") or {}
    if driver.get("name") != "SaddleFinder.run" or not isinstance(driver.get("relax_endpoints"), bool):
        raise RunnerValidationError("bvse_recipe.driver must be {name: 'SaddleFinder.run', "
                                    "relax_endpoints: <bool>}")
    pool_tolerance = frozen.get("pool_barrier_tolerance_eV")
    if pool_tolerance is not None:
        problem = pool_tolerance_evidence_problem(frozen.get("pool_barrier_tolerance_evidence"))
        if problem:
            raise RunnerValidationError(f"pool BVSE tolerance evidence: {problem}")
    return {"decoration": frozen["decoration"], "relax_endpoints": driver["relax_endpoints"],
            "mobile_symbol": frozen["mobile_symbol"], "mobile_charge": int(frozen["mobile_charge"]),
            "superedge_cutoff_A": float(frozen["superedge_cutoff_A"]),
            "image_count": {"short_hop_images": count["short_hop_images"],
                            "long_hop_images": count["long_hop_images"],
                            "long_hop_min_length_A": float(count["long_hop_min_length_A"])},
            "bvse_neb_options": {"k": float(options["k"]), "default": bool(options["default"]),
                                 "distort": bool(options["distort"]), "gm": bool(options["gm"])},
            "fire_fmax": float(frozen["fire"]["fmax"]), "fire_steps": int(frozen["fire"]["steps"]),
            "pool_barrier_tolerance_eV": None if pool_tolerance is None else float(pool_tolerance),
            "cell_tolerance_A": float(tolerance["cell_A"]),
            "position_tolerance_A": float(tolerance["max_atom_displacement_A"])}


def _lock_versions(path: Path) -> dict[str, str]:
    """``name==version`` pins of a requirements-style lock (comments and hashes ignored)."""

    import re

    pins = {}
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        match = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;\\]+)", line)
        if match:
            pins[re.sub(r"[-_.]+", "-", match[1]).lower()] = match[2]
    return pins


def runtime_versions(bvse: dict[str, Any]) -> dict[str, str]:
    """Installed versions of the packages the BVSE paths depend on, verified against the
    pinned lock: an unpinned lock is missing evidence, a difference a validation failure."""

    import hashlib
    import importlib.metadata
    import platform
    import re

    lock = bvse.get("environment_lock") or {}
    if not lock.get("path") or not lock.get("sha256") or lock["sha256"] == "server_evidence_pending":
        raise MissingEvidence("BVSE runtime lock not pinned (runners.bvse.environment_lock "
                              "{path, sha256}); build the runtime from a lock and pin it")
    path = Path(lock["path"]).expanduser()
    if hashlib.sha256(path.read_bytes()).hexdigest() != lock["sha256"]:
        raise RunnerValidationError(f"{path}: SHA-256 differs from runners.bvse.environment_lock")
    pins = _lock_versions(path)
    versions = {"python": platform.python_version()}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise MissingEvidence(f"{name} is not installed in the BVSE runtime") from exc
        pinned = pins.get(re.sub(r"[-_.]+", "-", name).lower())
        if pinned != versions[name]:
            raise RunnerValidationError(f"{name} {versions[name]} differs from the lock ({pinned})")
    if versions["ions"] != bvse.get("ions_version"):
        raise RunnerValidationError(f"ions {versions['ions']} is not the pinned {bvse.get('ions_version')}")
    return versions


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


def n_images(edge_length_A: float, rule: dict[str, Any]) -> int:
    """Frozen image count: long hops (unwrapped Edge.length) get more images."""

    return (rule["long_hop_images"] if float(edge_length_A) >= rule["long_hop_min_length_A"]
            else rule["short_hop_images"])


def _geometry(atoms: Any) -> dict[str, Any]:
    return {"numbers": [int(z) for z in atoms.numbers], "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist() if hasattr(atoms.cell, "array") else
            [list(map(float, row)) for row in atoms.cell],
            "pbc": [bool(flag) for flag in atoms.pbc]}


def bvse_path_task(payload: dict[str, Any], context: Any) -> dict[str, Any]:
    """One unit: decorate, BVSE parameters, superedge path, BVSE NEB, compare."""

    from ase import Atoms

    settings = payload["recipe"]
    result: dict[str, Any] = {"unit_key": payload["unit_key"], "unit_id": payload["unit_id"]}
    structure = payload["structure"]
    atoms = Atoms(numbers=structure["numbers"], positions=structure["positions"],
                  cell=structure["cell"], pbc=structure["pbc"])
    try:
        if settings["decoration"] == "pymatgen_bvanalyzer":
            import numpy as np
            from pymatgen.analysis.bond_valence import BVAnalyzer
            from pymatgen.io.ase import AseAtomsAdaptor

            valences = BVAnalyzer().get_valences(AseAtomsAdaptor.get_structure(atoms))
            atoms.set_array("oxi_states", np.array(valences))  # the array ions reads
        else:
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
            n_images=n_images(edge.length, settings["image_count"]))
        finder = SaddleFinder()
        neb = finder.bvse_neb(images, **settings["bvse_neb_options"])
        perturbation = max((float(abs(image.info["perturbation"]).max()) for image in images
                            if "perturbation" in image.info), default=0.0)
        # band in the perturbed framework, perturbation subtracted, band relaxed again
        finder.run(neb, steps=settings["fire_steps"], fmax=settings["fire_fmax"],
                   relax_endpoints=settings["relax_endpoints"], logfile=None)
        bvse_fmax = float(abs(neb.get_forces()).max())
        bvse_barrier = float(finder.get_barrier(images))
    except INFRASTRUCTURE:
        raise
    except Exception as exc:  # an ions failure on this structure: initialisation failure
        return {**result, "status": "ions_failure", "detail": f"{type(exc).__name__}: {exc}"}
    geometry = [_geometry(image) for image in images]
    result.update(status="generated", images=geometry, bvse_fmax_eV_per_A=bvse_fmax,
                  bvse_converged=bvse_fmax <= settings["fire_fmax"],
                  max_framework_perturbation_A=perturbation, bvse_barrier_eV=bvse_barrier)
    if payload.get("reference") is not None:
        result["reproduction"] = compare_paths(geometry, payload["reference"], settings)
    return result


def run(ctx: RunnerContext) -> dict[str, Any]:
    from e3sse.execution.sharded_execution import _shard_path, canonical_sha256, make_shard
    from e3sse.production.runners.gate1_tables import linkage_module, pinned_g0_inputs
    from e3sse.production.runners.hop_files import (
        INIT,
        bvse_barriers,
        nebdft2k_files,
        read_geometry,
        verified_hop_file,
    )
    from e3sse.production.runners.structures import StructureSource, check_edge_sites
    from e3sse.screening.rate_limiting_hop import parse_edge_id

    versions = runtime_versions(ctx.runner_settings("bvse"))
    population = ctx.upstream("development_population", "population_manifest")
    settings = recipe(ctx.active_protocol["amendments"])
    g0_inputs = pinned_g0_inputs(ctx.freeze_config)
    structures = StructureSource(ctx.runner_settings("structures").get("source"), ctx.data_root,
                                 g0_inputs)
    files = nebdft2k_files(ctx.data_root / "raw")
    dataset_barriers = bvse_barriers(ctx.data_root / "raw", g0_inputs)
    units, failures, reference_sha256 = [], {}, {}
    for unit in population["units"]:
        if unit["role"] in ("excluded_gate0", "excluded_missing_covariate"):
            continue
        try:
            atoms = structures.load(unit["material_id"])
        except MissingEvidence as exc:
            failures[unit["unit_key"]] = {"status": "structure_missing", "detail": str(exc)}
            continue
        except (RunnerValidationError, KeyError, ValueError) as exc:
            failures[unit["unit_key"]] = {"status": "structure_unusable",
                                          "detail": f"{type(exc).__name__}: {exc}"}
            continue
        edge = parse_edge_id(unit["unit_id"])
        try:
            check_edge_sites(atoms, edge)
        except RunnerValidationError as exc:
            failures[unit["unit_key"]] = {"status": "edge_sites_not_li", "detail": str(exc)}
            continue
        reference = None
        if unit["side"] in ("development", "P"):
            path = files.get(unit["unit_id"], {}).get(INIT)
            if path is None:
                failures[unit["unit_key"]] = {"status": "init_path_missing"}
                continue
            try:
                digest = verified_hop_file(path, g0_inputs)
            except linkage_module().AuditError as exc:
                raise RunnerValidationError(str(exc)) from exc
            reference = read_geometry(path, digest)
            reference_sha256[unit["unit_key"]] = digest
        payload = {"unit_key": unit["unit_key"], "unit_id": unit["unit_id"],
                   "edge": dict(edge), "structure": _geometry(atoms),
                   "reference": reference, "recipe": settings}
        units.append(({"unit_key": unit["unit_key"], "unit_id": unit["unit_id"],
                       "init_sha256": reference_sha256.get(unit["unit_key"])}, payload))
    identity = {"code_sha256": ctx.code_sha256(CODE_FILES), "runtime_versions": versions,
                "recipe": settings, "structure_source": structures.identity()}
    results = ctx.execute(namespace=BVSE_NAMESPACE, units=units, task=bvse_path_task,
                          identity=identity)
    shard_dir = ctx.work_dir / "shards" / BVSE_NAMESPACE / "shards"
    index, counts, differences = {}, {}, []
    for unit_identity, _ in units:
        shard_id = make_shard(BVSE_NAMESPACE, unit_identity).shard_id
        result = results[shard_id]
        status = result["status"]
        reproduced = result.get("reproduction", {}).get("matches")
        dataset_barrier = dataset_barriers.get(unit_identity["unit_id"])
        difference = (None if status != "generated" or dataset_barrier is None
                      else result["bvse_barrier_eV"] - dataset_barrier)
        tolerance = settings["pool_barrier_tolerance_eV"]
        barrier_ok = tolerance is None or (difference is not None and abs(difference) <= tolerance)
        usable = status == "generated" and reproduced is not False and barrier_ok
        if status == "generated" and not barrier_ok:
            status = "bvse_barrier_mismatch"
        differences.append(difference)
        index[unit_identity["unit_key"]] = {
            "status": status, "usable_for_ml_neb": usable,
            "bvse_converged": result.get("bvse_converged"),
            "max_framework_perturbation_A": result.get("max_framework_perturbation_A"),
            "bvse_barrier_minus_nebbvse122k_eV": difference,
            "init_sha256": unit_identity["init_sha256"],
            "reproduction": result.get("reproduction"),
            "checkpoint": str(_shard_path(shard_dir, shard_id)),
            "result_sha256": canonical_sha256(result)}
        if result.get("bvse_converged") is False:
            counts["bvse_not_converged"] = counts.get("bvse_not_converged", 0) + 1
        key = status if status != "generated" else (
            "generated" if reproduced is None else
            ("reproduced" if reproduced else "reproduction_mismatch"))
        counts[key] = counts.get(key, 0) + 1
    for key, failure in failures.items():
        index[key] = {**failure, "usable_for_ml_neb": False}
        counts[failure["status"]] = counts.get(failure["status"], 0) + 1
    checked = counts.get("reproduced", 0) + counts.get("reproduction_mismatch", 0)
    position_differences = sorted(
        entry["reproduction"]["max_position_difference_A"] for entry in index.values()
        if (entry.get("reproduction") or {}).get("max_position_difference_A") is not None)
    barrier_differences = sorted(abs(d) for d in differences if d is not None)
    report = {"runtime_versions": versions, "recipe": settings,
              "counts": dict(sorted(counts.items())),
              "max_position_difference_A_distribution": _quantiles(position_differences),
              "abs_bvse_barrier_difference_eV_distribution": _quantiles(barrier_differences),
              "calibration_reproduction_rate": counts.get("reproduced", 0) / checked
              if checked else None,
              "policy": "_init.xyz mismatch -> excluded and counted (development and P); decorator "
                        "failure is its own initialisation-failure category; every unit's BVSE "
                        "barrier is compared with nebBVSE122k and excluded only under a frozen "
                        "pool_barrier_tolerance_eV (null: recorded only)"}
    outputs = {"bvse_path_index": ctx.publish("bvse_path_index.json", index),
               "bvse_reproduction_report": ctx.publish(STAGE_REPORT_NAME, report)}
    return {"outputs": outputs, "summary": report}


def _quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]
    return {"n": len(values), "min": values[0], "median": pick(0.5), "p90": pick(0.9),
            "max": values[-1]}


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
