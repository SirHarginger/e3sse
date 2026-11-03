"""nebDFT2k hop files: geometry-only reads, and labelled reads after granted access.

Files are located exactly as Gate G0 pairs them (``dataset_audit.NEB_FILE_PATTERN``:
``<edge_id>_init.xyz`` is the BVSE-preconditioned initial path, ``<edge_id>_relaxed.xyz``
the converged DFT-NEB images).  ``read_geometry`` converts species, positions, cell
and pbc only (``extxyz_frames.parse_frame_geometry``); label columns are never
converted.  ``read_labelled`` converts DFT energies and forces and must only be called
after ``RunnerContext.require_outcome`` granted the read.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

INIT, RELAXED = "init", "relaxed"


def frame_texts(path: Path) -> Iterator[str]:
    """Split a multi-frame extended-XYZ file into single-frame texts (no parsing)."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    position = 0
    while position < len(lines):
        if not lines[position].strip():
            position += 1
            continue
        n_atoms = int(lines[position].split()[0])
        block = lines[position: position + 2 + n_atoms]
        if len(block) != 2 + n_atoms:
            raise ValueError(f"{path}: truncated frame at line {position + 1}")
        yield "\n".join(block) + "\n"
        position += 2 + n_atoms


def read_geometry(path: Path) -> list[dict[str, Any]]:
    """Every frame's numbers, positions, cell and pbc (labels untouched)."""

    from e3sse.softening.extxyz_frames import parse_frame_geometry

    frames = []
    for text in frame_texts(path):
        frame = parse_frame_geometry(text)
        frames.append({"numbers": [int(z) for z in frame["numbers"]],
                       "positions": [[float(x) for x in row] for row in frame["positions"]],
                       "cell": [[float(x) for x in row] for row in frame["cell"]],
                       "pbc": [bool(flag) for flag in frame["pbc"]]})
    return frames


def read_labelled(path: Path) -> list[dict[str, Any]]:
    """DFT total energy (eV) and forces (eV/A) per frame: an outcome read."""

    from e3sse.softening.extxyz_frames import parse_labelled_frame

    frames = []
    for text in frame_texts(path):
        frame = parse_labelled_frame(text)
        frames.append({"energy_eV": float(frame["dft_energy"]),
                       "forces": [[float(x) for x in row] for row in frame["dft_forces"]],
                       "numbers": [int(z) for z in frame["numbers"]]})
    return frames


def nebdft2k_files(raw_root: Path) -> dict[str, dict[str, Path]]:
    """``edge_id -> {"init": path, "relaxed": path}`` from the G0 dataset inventory."""

    from e3sse.data.dataset_audit import NEB_FILE_PATTERN, discover_datasets

    found: dict[str, dict[str, Path]] = {}
    for path in discover_datasets(raw_root).get("nebDFT2k", []):
        match = NEB_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        slot = found.setdefault(match["edge_id"], {})
        if match["kind"] in slot:
            raise ValueError(f"duplicate nebDFT2k file for {match['edge_id']} {match['kind']}")
        slot[match["kind"]] = path
    return found


def bvse_barriers(raw_root: Path, g0_inputs: Mapping[str, Any]) -> dict[str, float]:
    """nebBVSE122k BVSE barrier (covariate) per edge, via the Gate-1 audit's readers."""

    from e3sse.production.runners.gate1_tables import linkage_tables

    rows = linkage_tables(raw_root, g0_inputs)["nebBVSE122k"]
    barriers: dict[str, float] = {}
    for row in rows:
        try:
            barriers[row["edge_id"]] = float(row["bvse_migration_barrier_eV"])
        except ValueError:
            continue
    return barriers


def csv_header(path: Path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle), [])
