"""Frozen ML-NEB engine: one relaxed band per hop, no retry, classified failures.

Every setting comes from the active protocol
(``amendments.prospective.ml_neb.frozen_numeric_settings``); no ASE default defines
the method.  The NEB algorithm (``method``), climbing image, spring constant, FIRE
parameters, force thresholds and step budgets are passed explicitly.  ASE's NEB
docstring marks the default change from ``aseneb`` to ``improvedtangent`` as
``versionchanged:: 3.27.1``; no 3.27.1 release exists (GitLab tags, PyPI), 3.27.0 still
defaults to ``aseneb`` and 3.28.0 is the first released code with the new default.
The frozen runtimes pin ASE 3.22.1 and 3.24.0, so nothing may rely on a default.

Procedure (LiTraj c3ca5c2 with the two declared deviations):

1. validate the input path (same atoms, order, cell and pbc in every image; finite
   positions; minimum interatomic distance);
2. relax both endpoints with FIRE at fixed cell (``fmax`` and ``steps`` frozen);
3. relax the band (endpoints fixed, interior images moved) with ``NEB`` + FIRE, one
   calculator per image (``allow_shared_calculator`` false);
4. success iff max |force component| is below the frozen threshold on the band
   (NEB-projected forces) and on both relaxed endpoints, every energy and force is
   finite, and the relaxed geometry passes the frozen checks;
5. barrier = max - min of the relaxed band's ML energies.

A failure is final: the unit receives no ML-NEB prediction and keeps its place in
every denominator.  Categories reuse ``screening.prospective_contract``.  Memory and
operating-system errors are infrastructure failures and are re-raised, never stored
as a scientific result.  Wall-clock timeouts are handled by the caller.

This module runs inside the frozen model runtimes (Python 3.11, ASE 3.22-3.24), so
it uses only APIs present in all of them and imports ASE lazily.  Per-runtime ASE versions are
retained (owner decision); preflight requires cross-runtime EMT equivalence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from e3sse.screening.prospective_contract import FAILURE_CATEGORIES

ML_NEB_RECORD_SCHEMA = "e3sse.ml-neb.unit-record"
ML_NEB_ENGINE_ID = "e3sse-frozen-ml-neb-engine"
FIRE_PARAMETERS = ("dt", "maxstep", "dtmax", "Nmin", "finc", "fdec", "astart", "fa", "a",
                   "downhill_check")
NEB_CONSTRUCTOR_KEYS = ("k", "climb", "method", "parallel", "remove_rotation_and_translation",
                        "allow_shared_calculator", "scale_fmax")
_CONVERGENCE_TEXT = re.compile(r"max \|force component\| < ([0-9]+(?:\.[0-9]+)?) eV/A")
# Infrastructure, never a scientific result: the caller decides whether to rerun.
INFRASTRUCTURE_EXCEPTIONS = (MemoryError, OSError, KeyboardInterrupt, SystemExit)


class MLNEBSettingsError(ValueError):
    """The protocol does not define a complete, explicit ML-NEB rule."""


@dataclass(frozen=True)
class FrozenMLNEBSettings:
    k: float
    climb: bool
    method: str
    parallel: bool
    remove_rotation_and_translation: bool
    allow_shared_calculator: bool
    scale_fmax: float
    fire: tuple[tuple[str, Any], ...]
    endpoint_steps: int
    band_steps: int
    endpoint_fmax: float
    band_fmax: float
    convergence_max_force_component: float
    min_interatomic_distance_A: float
    site_collapse_A: float
    interior_below_endpoints_tolerance_eV: float

    @property
    def fire_parameters(self) -> dict[str, Any]:
        return dict(self.fire)

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fire"] = self.fire_parameters
        return payload

    @property
    def settings_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.as_json(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


def _number(value: Any, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MLNEBSettingsError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise MLNEBSettingsError(f"{name} must be positive")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MLNEBSettingsError(f"{name} must be a positive integer")
    return value


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise MLNEBSettingsError(f"{name} must be a boolean")
    return value


def settings_from_protocol(protocol: Mapping[str, Any]) -> FrozenMLNEBSettings:
    """Frozen settings from an internal-freeze config or materialized protocol.

    Every key must be present with the right type; nothing falls back to ASE.
    """

    try:
        frozen = protocol["amendments"]["prospective"]["ml_neb"]["frozen_numeric_settings"]
    except (KeyError, TypeError) as exc:
        raise MLNEBSettingsError("protocol lacks amendments.prospective.ml_neb settings") from exc
    constructor = frozen.get("neb_constructor") or {}
    if set(constructor) != set(NEB_CONSTRUCTOR_KEYS):
        raise MLNEBSettingsError(f"neb_constructor keys must be exactly {NEB_CONSTRUCTOR_KEYS}")
    method = constructor["method"]
    if not isinstance(method, str) or not method:
        raise MLNEBSettingsError("neb_constructor.method must be an explicit method name")
    fire = frozen.get("fire") or {}
    missing = [name for name in FIRE_PARAMETERS if name not in fire]
    if missing:
        raise MLNEBSettingsError(f"fire lacks explicit parameters {missing}")
    fire_values: list[tuple[str, Any]] = []
    for name in FIRE_PARAMETERS:
        if name == "downhill_check":
            fire_values.append((name, _flag(fire[name], "fire.downhill_check")))
        elif name == "Nmin":
            fire_values.append((name, _integer(fire[name], "fire.Nmin")))
        else:
            fire_values.append((name, _number(fire[name], f"fire.{name}")))
    match = _CONVERGENCE_TEXT.search(str(frozen.get("convergence", "")))
    if match is None:
        raise MLNEBSettingsError("convergence must state 'max |force component| < X eV/A'")
    checks = frozen.get("geometry_checks") or {}
    interior = frozen.get("interior_below_endpoints_eV") or {}
    steps = frozen.get("steps") or {}
    fmax = frozen.get("fmax_eV_per_A") or {}
    scale_fmax = _number(constructor["scale_fmax"], "neb_constructor.scale_fmax", positive=False)
    if scale_fmax != 0.0:
        # ASE's NEB deprecates scale_fmax (a DyNEB option); only the inert value is supported.
        raise MLNEBSettingsError("only scale_fmax = 0 is supported by plain NEB")
    allow_shared = _flag(constructor["allow_shared_calculator"], "allow_shared_calculator")
    parallel = _flag(constructor["parallel"], "parallel")
    if parallel:
        raise MLNEBSettingsError("image-parallel NEB is not part of the frozen rule")
    return FrozenMLNEBSettings(
        k=_number(constructor["k"], "neb_constructor.k"),
        climb=_flag(constructor["climb"], "neb_constructor.climb"),
        method=method,
        parallel=parallel,
        remove_rotation_and_translation=_flag(constructor["remove_rotation_and_translation"],
                                              "remove_rotation_and_translation"),
        allow_shared_calculator=allow_shared,
        scale_fmax=scale_fmax,
        fire=tuple(fire_values),
        endpoint_steps=_integer(steps.get("endpoint_relaxation"), "steps.endpoint_relaxation"),
        band_steps=_integer(steps.get("band"), "steps.band"),
        endpoint_fmax=_number(fmax.get("endpoints"), "fmax_eV_per_A.endpoints"),
        band_fmax=_number(fmax.get("band"), "fmax_eV_per_A.band"),
        convergence_max_force_component=float(match.group(1)),
        min_interatomic_distance_A=_number(checks.get("min_interatomic_distance_A"),
                                           "geometry_checks.min_interatomic_distance_A"),
        site_collapse_A=_number(checks.get("site_collapse_A"), "geometry_checks.site_collapse_A"),
        interior_below_endpoints_tolerance_eV=_number(interior.get("tolerance"),
                                                      "interior_below_endpoints_eV.tolerance"),
    )


def neb_class() -> tuple[type, str]:
    """ASE's NEB class and the module it came from (moved to ase.mep in 3.23)."""

    try:
        from ase.mep import NEB

        return NEB, "ase.mep"
    except ImportError:
        from ase.neb import NEB

        return NEB, "ase.neb"


def build_neb(images: Sequence[Any], settings: FrozenMLNEBSettings) -> Any:
    """NEB with every algorithmic choice explicit; verifies inert DyNEB options."""

    NEB, _ = neb_class()
    neb = NEB(
        list(images),
        k=settings.k,
        climb=settings.climb,
        parallel=settings.parallel,
        remove_rotation_and_translation=settings.remove_rotation_and_translation,
        method=settings.method,
        allow_shared_calculator=settings.allow_shared_calculator,
    )
    if getattr(neb, "scale_fmax", 0.0) != settings.scale_fmax:
        raise MLNEBSettingsError("installed NEB has a non-zero scale_fmax default")
    if getattr(neb, "dynamic_relaxation", False):
        raise MLNEBSettingsError("installed NEB enables dynamic relaxation by default")
    if getattr(neb, "climb", settings.climb) != settings.climb:
        raise MLNEBSettingsError("installed NEB did not accept the frozen climb setting")
    return neb


def fire_optimizer(target: Any, settings: FrozenMLNEBSettings) -> Any:
    from ase.optimize import FIRE

    return FIRE(target, logfile=None, **settings.fire_parameters)


class _Failure(Exception):
    def __init__(self, category: str, stage: str, detail: str) -> None:
        if category not in FAILURE_CATEGORIES:
            raise ValueError(f"unknown failure category {category!r}")
        super().__init__(detail)
        self.category, self.stage, self.detail = category, stage, detail


def _minimum_image(vector: Any, cell: Any, pbc: Any) -> Any:
    from ase.geometry import find_mic

    _, lengths = find_mic([vector], cell, pbc)
    return float(lengths[0])


def migrating_atom_index(first: Any, last: Any, species: str) -> int:
    """The ``species`` atom with the largest minimum-image displacement (ties: lowest index)."""

    import numpy as np

    candidates = [i for i, symbol in enumerate(first.get_chemical_symbols()) if symbol == species]
    if not candidates:
        raise _Failure("geometry_validity", "input", f"no {species} atom in the path")
    displacement = np.asarray(last.positions) - np.asarray(first.positions)
    lengths = [(_minimum_image(displacement[i], first.cell, first.pbc), i) for i in candidates]
    best = max(length for length, _ in lengths)
    return min(i for length, i in lengths if length == best)


def minimum_interatomic_distance(atoms: Any, cutoff: float) -> float | None:
    """Smallest periodic pair distance below ``cutoff`` (None when none is closer)."""

    from ase.neighborlist import neighbor_list

    distances = neighbor_list("d", atoms, cutoff)
    return float(distances.min()) if len(distances) else None


def _validate_path(images: Sequence[Any], settings: FrozenMLNEBSettings) -> None:
    import numpy as np

    if len(images) < 3:
        raise _Failure("geometry_validity", "input", "a band needs at least one interior image")
    first = images[0]
    for index, image in enumerate(images):
        if list(image.numbers) != list(first.numbers):
            raise _Failure("geometry_validity", "input",
                           f"image {index}: atom count or order differs")
        if not np.allclose(np.asarray(image.cell), np.asarray(first.cell), rtol=0, atol=1e-8):
            raise _Failure("geometry_validity", "input", f"image {index}: cell differs")
        if list(image.pbc) != list(first.pbc):
            raise _Failure("geometry_validity", "input", f"image {index}: pbc differs")
        if not np.isfinite(np.asarray(image.positions)).all():
            raise _Failure("geometry_validity", "input", f"image {index}: non-finite positions")
        closest = minimum_interatomic_distance(image, settings.min_interatomic_distance_A)
        if closest is not None:
            raise _Failure("geometry_validity", "input",
                           f"image {index}: interatomic distance {closest:.4f} A below "
                           f"{settings.min_interatomic_distance_A} A")


def _max_component(forces: Any) -> float:
    import numpy as np

    array = np.asarray(forces, dtype=float)
    if not np.isfinite(array).all():
        raise _Failure("non_finite_value", "forces", "non-finite force component")
    return float(np.abs(array).max()) if array.size else 0.0


_ALLOCATOR_OOM_TEXT = ("out of memory", "can't allocate memory", "defaultcpuallocator",
                       "cannot allocate memory")


def _is_allocator_oom(exc: BaseException) -> bool:
    """PyTorch reports allocation failure as RuntimeError / torch.OutOfMemoryError."""

    text = str(exc).lower()
    return type(exc).__name__ == "OutOfMemoryError" or any(m in text for m in _ALLOCATOR_OOM_TEXT)


def _non_finite_state(work: Sequence[Any]) -> bool:
    import numpy as np

    for image in work:
        if not np.isfinite(np.asarray(image.positions, dtype=float)).all():
            return True
        results = getattr(getattr(image, "calc", None), "results", None) or {}
        for key in ("energy", "forces"):
            if key in results and not np.isfinite(np.asarray(results[key], dtype=float)).all():
                return True
    return False


def _calculator_call(stage: str, action: Callable[[], Any], work: Sequence[Any]) -> Any:
    """Run model work; classify a failure as model, non-finite, or infrastructure."""

    try:
        return action()
    except _Failure:
        raise
    except INFRASTRUCTURE_EXCEPTIONS:
        raise
    except FloatingPointError as exc:
        raise _Failure("non_finite_value", stage, f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        if _is_allocator_oom(exc):
            # Memory exhaustion depends on concurrency, not on the unit: infrastructure.
            raise MemoryError(f"allocator out of memory during {stage}: {exc}") from exc
        if _non_finite_state(work):
            # A NaN force moved atoms to NaN positions and the model then raised.
            raise _Failure("non_finite_value", stage, f"{type(exc).__name__} after a "
                           f"non-finite state: {exc}") from exc
        raise _Failure("calculator_exception", stage, f"{type(exc).__name__}: {exc}") from exc


REQUIRED_PROVENANCE = ("model", "weights_sha256", "environment_lock_sha256", "dtype", "device",
                       "torch_intra_op_threads", "torch_inter_op_threads")


def frozen_model_dtype(protocol: Mapping[str, Any], model: str) -> str:
    """The ML-NEB dtype from runtime_requirements.dtype, the only source (fail closed)."""

    table = protocol["amendments"]["prospective"]["ml_neb"]["runtime_requirements"]["dtype"]
    if model not in table:
        raise MLNEBSettingsError(f"no frozen ML-NEB dtype for {model}")
    return str(table[model])


def reference_emt_vacancy_band(interior_images: int = 5) -> list[Any]:
    """Runtime-audit reference: a Cu vacancy hop under ASE's EMT (no data, no model).

    Every runtime solves this identical problem with the frozen settings, so ASE
    implementations can be compared bit for bit across runtimes.
    """

    import numpy as np
    from ase.build import bulk

    host = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(2)
    vectors = host.get_distances(1, range(len(host)), mic=True, vector=True)
    lengths = np.linalg.norm(vectors, axis=1)
    lengths[1] = np.inf
    hopping = int(np.argmin(lengths))
    initial, final = host.copy(), host.copy()
    final.positions[hopping] = host.positions[hopping] - vectors[hopping]
    del initial[1]
    del final[1]
    band = [initial]
    for index in range(1, interior_images + 1):
        image = initial.copy()
        image.positions = initial.positions + (final.positions - initial.positions) * (
            index / (interior_images + 1))
        band.append(image)
    band.append(final)
    return band


def run_ml_neb(
    images: Sequence[Any],
    calculator_factory: Callable[[], Any],
    settings: FrozenMLNEBSettings,
    *,
    unit_id: str,
    provenance: Mapping[str, Any],
    migrating_species: str = "Li",
) -> dict[str, Any]:
    """Run the frozen ML-NEB once on ``images`` (endpoints included); never retries.

    ``calculator_factory`` returns a fresh calculator per image.  ``provenance`` binds
    the record to the model, checkpoint, environment lock, dtype, device and thread
    counts (keys ``REQUIRED_PROVENANCE``).  The input images are not modified.

    ``failure_origin`` separates a malformed input path (``input_path``: identical
    for every model, an initialisation failure) from a failure of the model's run
    (``model_run``).  ``failure_checks`` lists every final check that failed; the
    primary category follows the fixed order non-finite, geometry, convergence.
    There is no wall-clock limit: the frozen step budgets bound the work.
    """

    import ase
    import numpy as np

    missing = [key for key in REQUIRED_PROVENANCE if key not in provenance]
    if missing:
        raise ValueError(f"ML-NEB provenance lacks {missing}")
    _, module = neb_class()
    record: dict[str, Any] = {
        "schema_version": ML_NEB_RECORD_SCHEMA,
        "engine": ML_NEB_ENGINE_ID,
        "unit_id": unit_id,
        "provenance": dict(provenance),
        "settings_sha256": settings.settings_sha256,
        "neb_method": settings.method,
        "neb_module": module,
        "ase_version": ase.__version__,
        "retry_policy": "none",
        "n_images": len(images),
        "n_atoms": len(images[0]) if images else 0,
        "status": "failed",
        "failure_category": None,
        "failure_origin": None,
        "failure_stage": None,
        "failure_detail": None,
        "failure_checks": [],
        "barrier_eV": None,
    }
    work = [image.copy() for image in images]
    try:
        try:
            _validate_path(work, settings)
            moving = migrating_atom_index(work[0], work[-1], migrating_species)
        except _Failure:
            record["failure_origin"] = "input_path"
            raise
        record["failure_origin"] = "model_run"
        record["migrating_atom_index"] = moving

        endpoint_steps = []
        for position in (0, len(work) - 1):
            endpoint = work[position]
            endpoint.calc = calculator_factory()
            stage = f"endpoint_{position}"
            # ASE 3.22 evaluates the model while constructing the optimizer.
            optimizer = _calculator_call(stage, lambda e=endpoint: fire_optimizer(e, settings),
                                         work)
            _calculator_call(stage, lambda o=optimizer: o.run(fmax=settings.endpoint_fmax,
                                                              steps=settings.endpoint_steps), work)
            endpoint_steps.append(int(optimizer.get_number_of_steps()))
        for image in work[1:-1]:
            image.calc = calculator_factory()
        record["endpoint_optimizer_steps"] = endpoint_steps

        neb = build_neb(work, settings)
        band_optimizer = _calculator_call("band", lambda: fire_optimizer(neb, settings), work)
        _calculator_call("band", lambda: band_optimizer.run(fmax=settings.band_fmax,
                                                            steps=settings.band_steps), work)
        record["band_optimizer_steps"] = int(band_optimizer.get_number_of_steps())

        band_force = _max_component(_calculator_call("band_forces", neb.get_forces, work))
        endpoint_force = max(
            _max_component(_calculator_call("endpoint_forces", image.get_forces, work))
            for image in (work[0], work[-1])
        )
        energies = [float(_calculator_call("energies", image.get_potential_energy, work))
                    for image in work]
        if not all(math.isfinite(value) for value in energies):
            raise _Failure("non_finite_value", "energies", "non-finite image energy")
        record["max_force_component_eV_per_A"] = {"band": band_force, "endpoints": endpoint_force}
        record["energies_eV"] = energies

        triggered: list[tuple[str, str, str]] = []
        for index, image in enumerate(work):
            closest = minimum_interatomic_distance(image, settings.min_interatomic_distance_A)
            if closest is not None:
                triggered.append(("geometry_validity", "relaxed_band",
                                  f"image {index}: interatomic distance {closest:.4f} A"))
                break
        separation = _minimum_image(
            np.asarray(work[-1].positions[moving]) - np.asarray(work[0].positions[moving]),
            work[0].cell, work[0].pbc,
        )
        record["migrating_atom_endpoint_separation_A"] = separation
        if separation < settings.site_collapse_A:
            triggered.append(("geometry_validity", "relaxed_endpoints",
                              f"migrating atom endpoints {separation:.4f} A apart (site collapse)"))
        threshold = settings.convergence_max_force_component
        if band_force >= threshold or endpoint_force >= threshold:
            triggered.append(("optimizer_non_convergence", "convergence_check",
                              (f"max force component band {band_force:.4f}, endpoints "
                               f"{endpoint_force:.4f} eV/A (threshold {threshold})")))
        if triggered:
            record["failure_checks"] = [{"category": c, "stage": s, "detail": d}
                                        for c, s, d in triggered]
            raise _Failure(*triggered[0])
    except _Failure as failure:
        record.update(failure_category=failure.category, failure_stage=failure.stage,
                      failure_detail=failure.detail)
        if not record["failure_checks"]:
            record["failure_checks"] = [{"category": failure.category, "stage": failure.stage,
                                         "detail": failure.detail}]
        return record

    lowest_endpoint = min(energies[0], energies[-1])
    record.update(
        status="converged",
        failure_origin=None,
        barrier_eV=max(energies) - min(energies),
        # Unoriented: the path-representation mapping rule orients the band itself.
        highest_energy_image_index=int(np.argmax(energies)),
        flags={"interior_below_endpoints": any(
            value < lowest_endpoint - settings.interior_below_endpoints_tolerance_eV
            for value in energies[1:-1])},
        band_geometry={
            "numbers": [int(z) for z in work[0].numbers],
            "cell": np.asarray(work[0].cell).tolist(),
            "pbc": [bool(flag) for flag in work[0].pbc],
            "positions": [np.asarray(image.positions).tolist() for image in work],
        },
    )
    return record
