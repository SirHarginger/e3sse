"""Byte-offset access to frames of an immutable flattened MPLiTrj extended-XYZ file.

The frame index is built in one streamed pass over the source file.  The pass
recomputes the file's SHA-256, which must equal the frozen ``source_file_sha256``,
and records per frame its byte offset and length, its atom count, the header
``material_id`` and the cell volume from ``Lattice``.  These are structural
identifiers; no energy, force or stress value is parsed.  Workers then seek to a
frame directly instead of rescanning the multi-GB file.

Frames are parsed here rather than with ``ase.io.read`` because the frozen model
environments pin different ASE versions (3.22.1 to 3.29), and those versions place
extended-XYZ energies and forces differently.  The parser reads only the columns
named in ``Properties`` and fails closed on any layout it does not recognise.
:func:`parse_frame_geometry` reads species, positions and cell only; it never
converts a label column, so it is safe for operational probes before the outcome
seal is lifted.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FRAME_INDEX_SCHEMA = "e3sse.study1.mplitrj-frame-index.v1"
_PAIR = re.compile(r'([A-Za-z_][A-Za-z0-9_\-]*)=(?:"([^"]*)"|(\S+))')
_DTYPES = {"S": str, "R": float, "I": int, "L": str}


class FrameFormatError(ValueError):
    """A frame does not have the extended-XYZ layout this reader is certified for."""


def header_fields(header: str) -> dict[str, str]:
    """``key=value`` pairs of an extended-XYZ comment line (quoted values unquoted)."""

    fields: dict[str, str] = {}
    for key, quoted, bare in _PAIR.findall(header):
        if key in fields:
            raise FrameFormatError(f"duplicate header key {key!r}")
        fields[key] = quoted if bare == "" else bare
    return fields


def _lattice(fields: dict[str, str]) -> list[list[float]]:
    values = [float(item) for item in fields.get("Lattice", "").split()]
    if len(values) != 9:
        raise FrameFormatError("Lattice must have 9 components")
    return [values[0:3], values[3:6], values[6:9]]


def _pbc(fields: dict[str, str]) -> list[bool]:
    if "pbc" not in fields:
        return [True, True, True]  # ASE convention when a Lattice is present
    flags = fields["pbc"].split()
    if len(flags) != 3 or set(flags) - {"T", "F"}:
        raise FrameFormatError(f"invalid pbc {fields['pbc']!r}")
    return [flag == "T" for flag in flags]


def _columns(fields: dict[str, str]) -> dict[str, tuple[int, int, str]]:
    """Column name -> (first column, width, type) from ``Properties``."""

    parts = fields.get("Properties", "").split(":")
    if len(parts) % 3 or not parts[0]:
        raise FrameFormatError("Properties must be name:type:count triplets")
    layout: dict[str, tuple[int, int, str]] = {}
    column = 0
    for name, kind, count in zip(parts[0::3], parts[1::3], parts[2::3], strict=True):
        if kind not in _DTYPES or not count.isdigit() or name in layout:
            raise FrameFormatError(f"unsupported Properties entry {name}:{kind}:{count}")
        layout[name] = (column, int(count), kind)
        column += int(count)
    for name, width in (("species", 1), ("pos", 3)):
        if layout.get(name, (0, 0, ""))[1:] != (width, "S" if name == "species" else "R"):
            raise FrameFormatError(f"Properties lacks {name} with width {width}")
    return layout


def _cell_volume(cell: list[list[float]]) -> float:
    (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) = cell
    return abs(a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1))


@dataclass(frozen=True)
class FrameIndexEntry:
    frame_index: int
    byte_offset: int
    byte_length: int
    n_atoms: int
    material_id: str
    cell_volume: float


def scan_frame_index(path: Path) -> tuple[str, list[FrameIndexEntry]]:
    """One streamed pass: (exact file SHA-256, per-frame byte ranges and structure facts)."""

    digest = hashlib.sha256()
    entries: list[FrameIndexEntry] = []
    offset = 0
    with Path(path).open("rb") as handle:
        while True:
            start = offset
            line = handle.readline()
            if not line:
                break
            digest.update(line)
            offset += len(line)
            if not line.strip():
                raise FrameFormatError(f"blank line at byte {start}")
            try:
                n_atoms = int(line)
            except ValueError as exc:
                raise FrameFormatError(f"frame {len(entries)}: bad atom count") from exc
            header = handle.readline()
            if not header or n_atoms < 1:
                raise FrameFormatError(f"frame {len(entries)}: missing header or no atoms")
            digest.update(header)
            offset += len(header)
            for _ in range(n_atoms):
                atom_line = handle.readline()
                if not atom_line:
                    raise FrameFormatError(f"frame {len(entries)}: truncated")
                digest.update(atom_line)
                offset += len(atom_line)
            fields = header_fields(header.decode("utf-8"))
            layout = _columns(fields)
            if layout.get("forces", (0, 0, ""))[1:] != (3, "R") or "energy" not in fields:
                raise FrameFormatError(f"frame {len(entries)}: no forces:R:3 column or energy key")
            material = fields.get("material_id")
            if not material:
                raise FrameFormatError(f"frame {len(entries)}: header lacks material_id")
            entries.append(
                FrameIndexEntry(
                    frame_index=len(entries),
                    byte_offset=start,
                    byte_length=offset - start,
                    n_atoms=n_atoms,
                    material_id=material,
                    cell_volume=_cell_volume(_lattice(fields)),
                )
            )
    return digest.hexdigest(), entries


def read_frame_text(handle: Any, byte_offset: int, byte_length: int) -> str:
    handle.seek(byte_offset)
    payload = handle.read(byte_length)
    if len(payload) != byte_length:
        raise FrameFormatError("frame byte range extends past the end of the file")
    return payload.decode("utf-8")


def _split_frame(text: str) -> tuple[int, dict[str, str], list[list[str]]]:
    lines = text.split("\n")
    n_atoms = int(lines[0])
    fields = header_fields(lines[1])
    rows = [line.split() for line in lines[2 : 2 + n_atoms]]
    if len(rows) != n_atoms or any(not row for row in rows):
        raise FrameFormatError("frame has fewer atom lines than its count")
    if any(line.strip() for line in lines[2 + n_atoms :]):
        raise FrameFormatError("frame byte range contains more than one frame")
    return n_atoms, fields, rows


def _numeric(rows: list[list[str]], first: int, width: int) -> Any:
    import numpy as np

    # float() of a decimal string is correctly rounded, identical in every environment.
    return np.array([[float(value) for value in row[first : first + width]] for row in rows])


def parse_frame_geometry(text: str) -> dict[str, Any]:
    """Species, positions, cell and pbc only; label columns are never converted."""

    from ase.data import atomic_numbers

    n_atoms, fields, rows = _split_frame(text)
    layout = _columns(fields)
    width = sum(count for _, count, _ in layout.values())
    if any(len(row) != width for row in rows):
        raise FrameFormatError("atom line width differs from Properties")
    species = [row[layout["species"][0]] for row in rows]
    unknown = sorted({symbol for symbol in species if atomic_numbers.get(symbol, 0) <= 0})
    if unknown:
        raise FrameFormatError(f"unknown species {unknown}")
    return {
        "n_atoms": n_atoms,
        "material_id": fields.get("material_id"),
        "numbers": [atomic_numbers[symbol] for symbol in species],
        "positions": _numeric(rows, layout["pos"][0], 3),
        "cell": _lattice(fields),
        "pbc": _pbc(fields),
    }


def parse_labelled_frame(text: str) -> dict[str, Any]:
    """Geometry plus the DFT total energy (eV) and forces (eV/angstrom).

    Units are the verified production source units (identity conversion, frozen
    unit contract); reading them is an outcome access gated by the caller.
    """

    frame = parse_frame_geometry(text)
    _, fields, rows = _split_frame(text)
    layout = _columns(fields)
    forces = layout.get("forces")
    if forces is None or forces[1:] != (3, "R"):
        raise FrameFormatError("Properties lacks forces:R:3")
    if "energy" not in fields:
        raise FrameFormatError("header lacks the DFT energy")
    frame["dft_forces"] = _numeric(rows, forces[0], 3)
    frame["dft_energy"] = float(fields["energy"])
    if not math.isfinite(frame["dft_energy"]):
        raise FrameFormatError("DFT energy is not finite")
    return frame


def build_atoms(frame: dict[str, Any]) -> Any:
    from ase import Atoms

    return Atoms(
        numbers=frame["numbers"], positions=frame["positions"], cell=frame["cell"], pbc=frame["pbc"]
    )
