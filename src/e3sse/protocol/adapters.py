"""Canonical ASE adapters for the frozen uMLIP checkpoints, plus a unit smoke test.

Every adapter returns an ASE calculator whose results are total configuration
energy in eV, forces in eV/angstrom (F = -dE/dx), and stress in eV/angstrom^3 (ASE
sign convention).  Package-specific defaults are overridden explicitly where they
differ (matgl reports GPa by default; CHGNet's model head predicts eV/atom and GPa,
which its ASE calculator converts).  Model packages are imported lazily so the
protocol package never requires torch.

The smoke test uses a synthetic Li2O cell only; it never reads LiTraj data or any
scientific outcome.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ADAPTER_VERSION = "e3sse-ase-adapter-v1"
_GPA_TO_EV_PER_A3 = 1.0 / 160.21766208
SMOKE_TEST_VERSION = "e3sse-unit-smoke-v3"


def checkpoint_digest(path: Path) -> str:
    """SHA-256 of a checkpoint file, or of the weights file of a matgl model directory."""

    target = path / "state.pt" if path.is_dir() else path
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_calculator(model: Mapping[str, Any], checkpoint: Path, *, device: str = "cpu") -> Any:
    """Construct the frozen calculator after verifying the checkpoint identity."""

    if model.get("adapter_version") != ADAPTER_VERSION:
        raise ValueError("model entry does not target this adapter version")
    actual = checkpoint_digest(checkpoint)
    if actual != model["weights_sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch for {model['canonical_name']}")
    # Directory checkpoints (matgl) carry architecture files too; verify every one.
    for name, expected in sorted(model.get("artifact_files", {}).items()):
        if not checkpoint.is_dir() or _file_sha256(checkpoint / name) != expected:
            raise ValueError(f"checkpoint artifact SHA-256 mismatch: {name}")
    options = dict(model.get("load_options", {}))
    name = model["adapter_name"]
    if name == "chgnet":
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(
            model=CHGNet.from_file(str(checkpoint)),
            use_device=device,
            stress_weight=_GPA_TO_EV_PER_A3,
        )
    if name == "matgl":
        import matgl
        from matgl.ext.ase import PESCalculator

        return PESCalculator(matgl.load_model(str(checkpoint)), stress_unit="eV/A3")
    if name == "mace":
        from mace.calculators import MACECalculator

        return MACECalculator(
            model_paths=str(checkpoint),
            device=device,
            default_dtype=options.get("default_dtype", "float64"),
        )
    if name == "sevennet":
        from sevenn.sevennet_calculator import SevenNetCalculator

        return SevenNetCalculator(model=str(checkpoint), device=device)
    raise ValueError(f"unknown adapter {name!r}")


def _energy(atoms: Any, calc: Any) -> float:
    copy = atoms.copy()
    copy.calc = calc
    return float(copy.get_potential_energy())


def smoke_test(calc: Any, *, force_step: float = 1e-2, strain_step: float = 2e-3) -> dict:
    """Check total-energy extensivity, F = -dE/dx and ASE eV/angstrom^3 stress.

    Tolerances are unit tests, not accuracy tests: a per-atom, GPa or kbar slip is off
    by at least 10x, while float32 central differences carry ~1e-3 absolute noise.
    """

    import numpy as np
    from ase.build import bulk

    atoms = bulk("Li2O", "fluorite", a=4.62)
    # Strain away from equilibrium (normal + shear) so every checked stress component is
    # large enough that a sign flip or a GPa/eV slip cannot hide under the tolerance.
    deformation = np.eye(3) + np.array([[0.03, 0.02, 0.0], [0.02, -0.02, 0.0], [0.0, 0.0, 0.01]])
    atoms.set_cell(atoms.cell @ deformation, scale_atoms=True)
    atoms.positions += np.random.default_rng(2026).normal(scale=0.05, size=atoms.positions.shape)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
    ratio = _energy(atoms.repeat((2, 1, 1)), calc) / energy

    force_pairs = []
    for index, axis in ((0, 0), (1, 2), (2, 1)):
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[index, axis] += force_step
        minus.positions[index, axis] -= force_step
        fd = -(_energy(plus, calc) - _energy(minus, calc)) / (2 * force_step)
        force_pairs.append([fd, float(forces[index, axis])])
    stress_pairs = []
    for i, j in ((0, 0), (1, 1), (0, 1)):
        strain = np.zeros((3, 3))
        strain[i, j] = strain[j, i] = strain_step / (2 if i != j else 1)
        energies = []
        for sign in (1, -1):
            deformed = atoms.copy()
            deformed.set_cell(atoms.cell @ (np.eye(3) + sign * strain), scale_atoms=True)
            energies.append(_energy(deformed, calc))
        fd = (energies[0] - energies[1]) / (2 * strain_step) / atoms.get_volume()
        stress_pairs.append([fd, float(stress[i, j])])

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-3 or abs(a - b) <= 0.05 * abs(b)

    def same_unit(a: float, b: float) -> bool:
        # A unit or sign slip changes the ratio by >= 10x or flips it; 0.5-2 is unit-safe.
        return abs(b) < 1e-2 or 0.5 <= a / b <= 2.0

    # Unit checks gate adapter acceptance; strict -dE/dx consistency is a model property
    # (e.g. non-smooth cutoffs) reported separately and registered where it fails.
    checks = {
        "total_energy_extensive": bool(abs(ratio - 2.0) < 1e-5),
        "force_unit_scale_eV_per_A": all(same_unit(a, b) for a, b in force_pairs),
        "stress_eV_per_A3_ase_convention": all(close(a, b) for a, b in stress_pairs),
        "stress_components_informative": all(abs(a) >= 1e-2 for a, _ in stress_pairs),
        "finite": bool(np.isfinite([energy, *forces.ravel(), *stress.ravel()]).all()),
    }
    return {
        "smoke_test_version": SMOKE_TEST_VERSION,
        "structure": (
            "synthetic Li2O fluorite a=4.62 A, strained (diag 0.03/-0.02/0.01, shear 0.02), "
            "seeded 0.05 A displacements"
        ),
        "energy_eV": energy,
        "supercell_energy_ratio": ratio,
        "fd_force_pairs_eV_per_A": force_pairs,
        "fd_stress_pairs_eV_per_A3": stress_pairs,
        "checks": checks,
        "passed": all(checks.values()),
        "force_energy_consistency": {
            "forces_are_negative_gradient_within_1e-3_or_5pct": all(
                close(a, b) for a, b in force_pairs
            ),
            "max_abs_fd_minus_analytic_eV_per_A": max(abs(a - b) for a, b in force_pairs),
        },
    }


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
