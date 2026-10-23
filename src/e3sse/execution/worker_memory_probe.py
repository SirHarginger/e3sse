"""Operational per-worker peak resident memory for one frozen model.

The probe is not a scientific measurement and not a smoke-test rerun: it loads one
SHA-verified checkpoint in a spawned worker with the planned thread configuration,
evaluates a synthetic Li2O supercell of at least ``min_atoms`` atoms (take this from
atom counts in the frozen input manifest, never from DFT labels), and reports the
process high-water mark (VmHWM).  ``plan_workers`` uses it to bound worker counts.
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
    model: Mapping[str, Any], checkpoint: str, min_atoms: int, evaluations: int
) -> dict[str, Any]:
    import numpy as np
    from ase.build import bulk

    from e3sse.protocol.adapters import build_calculator

    after_imports = peak_resident_bytes()
    calc = build_calculator(model, Path(checkpoint))
    after_load = peak_resident_bytes()
    repeats = probe_supercell_repeats(min_atoms)
    reference = bulk("Li2O", "fluorite", a=4.62, cubic=True).repeat((repeats,) * 3)
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
    min_atoms: int,
    threads_per_worker: int,
    evaluations: int = 3,
) -> dict[str, Any]:
    """Run the probe in one fresh spawned worker configured exactly like production."""

    if evaluations < 1:
        raise ValueError("evaluations must be positive")
    with child_thread_environment(threads_per_worker), ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context(START_METHOD),
        initializer=_initialize_worker,
        initargs=(threads_per_worker, True, None, ()),
    ) as pool:
        return pool.submit(
            _measure_in_worker, dict(model), str(checkpoint), min_atoms, evaluations
        ).result()


def probe_identity(
    model: Mapping[str, Any], *, min_atoms: int, threads_per_worker: int, evaluations: int,
    package_version: str,
) -> dict[str, Any]:
    return {
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

