"""Read-only source-unit evidence for the static LiTraj quantities.

Trace: LiTraj DFT (VASP via ``ase.calculators.vasp.Vasp``, LiTraj
scripts/ase_run_vasp.py) -> ASE results in ASE units -> extended XYZ written by ASE
-> ``ase.io.read`` -> ``SinglePointCalculator`` results -> protocol quantity.  ASE
stores energies in eV, lengths in angstrom, forces in eV/angstrom and stress in
eV/angstrom^3 (https://wiki.fysik.dtu.dk/ase/ase/units.html).

Outcome safety: for every file only extxyz comment-line *keys* are parsed.  Numeric
values are read solely from published-train MPLiTrj frames that the provenance index
proves are endpoint-relaxation (non-path) frames, and only pass/fail booleans of
order-of-magnitude checks are recorded, never value ranges.  nebDFT2k trajectories
and barrier columns are never read numerically.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "e3sse.unit-evidence.v2"
ASE_UNITS_REFERENCE = "https://wiki.fysik.dtu.dk/ase/ase/units.html"
# Plausible magnitudes in ASE units for DFT-PBE inorganic Li compounds.
_ENERGY_PER_ATOM_EV = (-15.0, -0.5)
_MAX_ABS_FORCE_EV_PER_A = 50.0
_MAX_ABS_STRESS_EV_PER_A3 = 1.0
_KEY = re.compile(r"([A-Za-z_][A-Za-z0-9_\-]*)=")


def comment_keys(path: Path) -> dict[str, Any]:
    """Key names of the first frame's extxyz comment line; values are discarded."""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.readline()
        comment = handle.readline()
    keys = sorted(set(_KEY.findall(comment)))
    properties = re.search(r"Properties=(\S+)", comment)
    layout = properties.group(1).split(":") if properties else []
    return {
        "file": path.name,
        "comment_keys": keys,
        "properties_columns": layout[0::3],
        "has_lattice": "Lattice" in keys,
        "has_energy": "energy" in keys,
        "has_stress": "stress" in keys,
        "has_forces_column": "forces" in layout[0::3],
    }


def nonpath_magnitude_checks(path: Path, frame_indices: Iterable[int]) -> dict[str, Any]:
    """Unit-scale booleans from proven endpoint-relaxation published-train frames only."""

    import numpy as np
    from ase.io import read

    energy_ok, force_ok, stress_values, n = True, True, [], 0
    for index in frame_indices:
        atoms = read(str(path), index=int(index))
        energy = float(atoms.get_potential_energy()) / len(atoms)
        energy_ok &= _ENERGY_PER_ATOM_EV[0] < energy < _ENERGY_PER_ATOM_EV[1]
        force_ok &= float(np.abs(atoms.get_forces()).max()) < _MAX_ABS_FORCE_EV_PER_A
        if "stress" in (atoms.calc.results or {}):
            stress_values.append(float(np.abs(atoms.get_stress()).max()))
        n += 1
    if n == 0:
        raise ValueError("no proven non-path published-train frames were supplied")
    stress = (
        "absent"
        if not stress_values
        else bool(max(stress_values) < _MAX_ABS_STRESS_EV_PER_A3)
    )
    return {
        "frames_read": n,
        "frame_selection": "published-train MPLiTrj endpoint-relaxation frames (provenance)",
        "checks": {
            "energy_scale_is_eV": bool(energy_ok),
            "force_scale_is_eV_per_A": bool(force_ok),
            "stress_scale_is_eV_per_A3": stress,
        },
    }


def csv_identifier_columns(path: Path, wanted: tuple[str, ...]) -> tuple[list[str], list[tuple]]:
    """Header plus only the requested identifier columns; other values are skipped."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(io.StringIO(handle.readline())), [])
        positions = [header.index(name) for name in wanted]
        rows = [
            tuple(row[i] for i in positions)
            for row in csv.reader(handle)
            if len(row) > max(positions)
        ]
    return header, rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(
    *,
    mplitrj_files: dict[str, Path],
    train_nonpath_frame_indices: Iterable[int],
    nebdft2k_index: Path,
    nebdft2k_trajectories: Iterable[Path],
) -> dict[str, Any]:
    import ase

    if set(mplitrj_files) != {"train", "val", "test"}:
        raise ValueError("MPLiTrj evidence needs exactly the train/val/test files")
    header, _ = csv_identifier_columns(nebdft2k_index, ("edge_id",))
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "ase_version": ase.__version__,
        "ase_units_reference": ASE_UNITS_REFERENCE,
        "mplitrj": {
            split: {**comment_keys(path), "sha256": file_sha256(path)}
            for split, path in sorted(mplitrj_files.items())
        },
        "mplitrj_train_unit_scale": nonpath_magnitude_checks(
            mplitrj_files["train"], train_nonpath_frame_indices
        ),
        "nebdft2k": {
            "index_file": nebdft2k_index.name,
            "index_sha256": file_sha256(nebdft2k_index),
            "index_columns": header,
            "trajectory_keys": [comment_keys(path) for path in nebdft2k_trajectories],
        },
    }
