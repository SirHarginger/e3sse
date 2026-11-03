"""Operational per-worker peak resident memory for one frozen model.

The probe is not a scientific measurement and not a smoke-test rerun: it loads one
SHA-verified checkpoint in a spawned worker with the planned thread configuration,
evaluates a synthetic Li2O supercell of at least ``min_atoms`` atoms (take this from
atom counts in the frozen input manifest, never from DFT labels), and reports the
process high-water mark (VmHWM).  ``plan_workers`` uses it to bound worker counts.
Alternatively it evaluates explicit structures (geometry only: atomic numbers,
positions, cell, pbc), e.g. the largest and densest real frames selected from
structure alone, because a small synthetic cell can underestimate graph memory.
"""

from __future__ import annotations

import math
import multiprocessing
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .sharded_execution import START_METHOD, _initialize_worker
from .thread_control import child_thread_environment

PROBE_VERSION = "e3sse-worker-memory-probe-v1"
_LI2O_CONVENTIONAL_ATOMS = 12
# Conventional cubic fluorite Li2O, written out because ``ase.build.bulk(..., cubic=True)``
# raises for fluorite in ASE 3.22.1.  Order and fractional coordinates equal what later
# ASE releases build, so the probe geometry is identical in every frozen runtime.
_LI2O_FRACTIONAL = (
    ("Li", (0.0, 0.0, 0.0)), ("Li", (0.25, 0.25, 0.25)), ("O", (0.75, 0.75, 0.75)),
    ("Li", (0.0, 0.5, 0.5)), ("Li", (0.25, 0.75, 0.75)), ("O", (0.75, 0.25, 0.25)),
    ("Li", (0.5, 0.0, 0.5)), ("Li", (0.75, 0.25, 0.75)), ("O", (0.25, 0.75, 0.25)),
    ("Li", (0.5, 0.5, 0.0)), ("Li", (0.75, 0.75, 0.25)), ("O", (0.25, 0.25, 0.75)),
)


def li2o_conventional_cell(a: float = 4.62) -> Any:
    """The 12-atom conventional cubic Li2O cell (lattice constant ``a`` in A)."""

    from ase import Atoms

    symbols = [symbol for symbol, _ in _LI2O_FRACTIONAL]
    return Atoms(symbols, scaled_positions=[xyz for _, xyz in _LI2O_FRACTIONAL],
                 cell=[a, a, a], pbc=True)


def peak_resident_bytes(status_path: Path = Path("/proc/self/status")) -> int:
    """VmHWM ("peak resident set size") of the current process, in bytes."""

    for line in status_path.read_text(encoding="ascii").splitlines():
        if line.startswith("VmHWM:"):
            value, unit = line.split()[1:3]
            if unit != "kB":
                raise ValueError(f"unexpected VmHWM unit {unit!r}")
            return int(value) * 1024
    raise ValueError("VmHWM not reported")


def probe_supercell_repeats(min_atoms: int) -> int:
    """Smallest cubic repeat of the 12-atom Li2O cell with at least ``min_atoms`` atoms."""

    if isinstance(min_atoms, bool) or not isinstance(min_atoms, int) or min_atoms < 1:
        raise ValueError("min_atoms must be a positive integer")
    repeats = max(1, math.ceil((min_atoms / _LI2O_CONVENTIONAL_ATOMS) ** (1 / 3)))
    while _LI2O_CONVENTIONAL_ATOMS * repeats**3 < min_atoms:
        repeats += 1
    return repeats


def _measure_in_worker(
    model: Mapping[str, Any],
    checkpoint: str,
    min_atoms: int | None,
    evaluations: int,
    structures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import numpy as np
    from ase import Atoms

    from e3sse.protocol.adapters import build_calculator

    after_imports = peak_resident_bytes()
    calc = build_calculator(model, Path(checkpoint))
    after_load = peak_resident_bytes()
    if structures is not None:
        for _ in range(evaluations):
            for item in structures:
                atoms = Atoms(numbers=item["numbers"], positions=item["positions"],
                              cell=item["cell"], pbc=item["pbc"])
                atoms.calc = calc
                atoms.get_potential_energy()
                atoms.get_forces()
        return {
            "peak_resident_bytes": peak_resident_bytes(),
            "peak_resident_bytes_after_imports": after_imports,
            "peak_resident_bytes_after_model_load": after_load,
            "n_atoms": max(len(item["numbers"]) for item in structures),
            "structures": len(structures),
            "evaluations": evaluations,
        }
    repeats = probe_supercell_repeats(min_atoms)
    reference = li2o_conventional_cell(4.62).repeat((repeats,) * 3)
    rng = np.random.default_rng(2026)  # fixed operational geometry, not a scientific draw
    for _ in range(evaluations):
        atoms = reference.copy()
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        atoms.calc = calc
        atoms.get_potential_energy()
        atoms.get_forces()
    return {
        "peak_resident_bytes": peak_resident_bytes(),
        "peak_resident_bytes_after_imports": after_imports,
        "peak_resident_bytes_after_model_load": after_load,
        "n_atoms": len(reference),
        "supercell_repeats": repeats,
        "evaluations": evaluations,
    }


def measure_worker_memory(
    model: Mapping[str, Any],
    checkpoint: Path,
    *,
    min_atoms: int | None,
    threads_per_worker: int,
    evaluations: int = 3,
    structures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the probe in one fresh spawned worker configured exactly like production."""

    if evaluations < 1:
        raise ValueError("evaluations must be positive")
    if (min_atoms is None) == (structures is None):
        raise ValueError("give exactly one of min_atoms or structures")
    with child_thread_environment(threads_per_worker), ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context(START_METHOD),
        initializer=_initialize_worker,
        initargs=(threads_per_worker, True, None, ()),
    ) as pool:
        return pool.submit(
            _measure_in_worker, dict(model), str(checkpoint), min_atoms, evaluations, structures
        ).result()


def probe_identity(
    model: Mapping[str, Any], *, min_atoms: int | None, threads_per_worker: int, evaluations: int,
    package_version: str, structure_file_sha256: str | None = None,
) -> dict[str, Any]:
    identity = {
        "probe_version": PROBE_VERSION,
        "canonical_name": model["canonical_name"],
        "weights_sha256": model["weights_sha256"],
        "package_name": model["package_name"],
        "package_version": package_version,
        "adapter_version": model["adapter_version"],
        "threads_per_worker": threads_per_worker,
        "min_atoms": min_atoms,
        "evaluations": evaluations,
    }
    if structure_file_sha256 is not None:
        identity["structure_file_sha256"] = structure_file_sha256
    return identity

