"""Pristine host structures per material, from one configured, G0-pinned source.

The protocol requires one pristine host structure per ``material_id`` from the same
source for development, calibration and pool units (never a nebDFT2k vacancy NEB
supercell), with the source confirmed by the server descriptor-availability audit.
Its file format has not been inspected locally, so nothing is guessed.  The
production config names the source explicitly (``runners.structures.source``):

* ``{"kind": "ase_per_material", "pattern": "<path with {material_id}>"}``: one
  ASE-readable file per material (e.g. CIF), read with ``ase.io.read``;
* ``{"kind": "optimade_jsonl", "path": "<file>", "id_field": "<dotted key>"}``:
  OPTIMADE structure entries (``lattice_vectors``, ``cartesian_site_positions``,
  ``species_at_sites``, ``species``).

Paths are relative to the data root.  Every file read must be in the pinned G0
input manifest (SHA-256 verified when G0 recorded one).  An unconfigured source is
missing evidence, never a fallback.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from e3sse.production.runners.context import MissingEvidence, RunnerValidationError


class StructureSource:
    def __init__(self, spec: Mapping[str, Any] | None, data_root: Path,
                 g0_inputs: Mapping[str, Any]) -> None:
        if not spec:
            raise MissingEvidence("pristine structure source not configured "
                                  "(runners.structures.source); the data-engineer server audit "
                                  "must pin it (open item descriptor_availability)")
        if spec.get("kind") not in ("ase_per_material", "optimade_jsonl"):
            raise RunnerValidationError(f"unknown structure source kind {spec.get('kind')!r}")
        self.spec, self.data_root, self.g0_inputs = dict(spec), Path(data_root), g0_inputs
        self._optimade: dict[str, dict[str, Any]] | None = None
        self.files_read: dict[str, str] = {}

    def identity(self) -> dict[str, Any]:
        return {"spec": self.spec, "files_read": dict(sorted(self.files_read.items()))}

    def _verified(self, path: Path) -> Path:
        from e3sse.production.runners.gate1_tables import linkage_module, verify_g0_file

        if str(path) not in self.files_read:
            try:
                digest = verify_g0_file(path, self.g0_inputs)
            except linkage_module().AuditError as exc:
                raise RunnerValidationError(str(exc)) from exc
            pinned = self.spec.get("sha256")
            if self.spec["kind"] == "optimade_jsonl" and pinned and pinned != digest:
                raise RunnerValidationError(f"{path}: SHA-256 differs from runners.structures pin")
            self.files_read[str(path)] = digest
        return path

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

    def load(self, material_id: str) -> Any:
        """ASE ``Atoms`` of the material's pristine structure (geometry only)."""

        if self.spec["kind"] == "ase_per_material":
            from ase.io import read

            path = self._resolve(self.spec["pattern"].format(material_id=material_id))
            if not path.is_file():
                raise MissingEvidence(f"no pristine structure file for {material_id}")
            frames = read(self._verified(path), index=":")
            if len(frames) != 1:
                raise RunnerValidationError(f"{path}: {len(frames)} structures, expected one")
            return _geometry_only(frames[0])
        return self._optimade_atoms(material_id)

    def _optimade_atoms(self, material_id: str) -> Any:
        from ase import Atoms

        if self._optimade is None:
            path = self._verified(self._resolve(self.spec["path"]))
            opener = gzip.open if path.suffix == ".gz" else open
            entries: dict[str, dict[str, Any]] = {}
            with opener(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("type") != "structures":
                        continue
                    key = record
                    for part in self.spec["id_field"].split("."):
                        key = key.get(part) if isinstance(key, Mapping) else None
                    if key is None:
                        continue
                    if str(key) in entries:
                        raise RunnerValidationError(f"two structures for material {key}")
                    entries[str(key)] = record["attributes"]
            self._optimade = entries
        attributes = self._optimade.get(material_id)
        if attributes is None:
            raise MissingEvidence(f"no pristine structure for {material_id} in the source")
        symbol_of = {item["name"]: item["chemical_symbols"] for item in attributes["species"]}
        symbols = []
        for name in attributes["species_at_sites"]:
            chemical = symbol_of[name]
            if len(chemical) != 1 or chemical[0] in ("X", "vacancy"):
                raise RunnerValidationError(f"{material_id}: disordered or vacant site {name}")
            symbols.append(chemical[0])
        return Atoms(symbols=symbols, positions=attributes["cartesian_site_positions"],
                     cell=attributes["lattice_vectors"],
                     pbc=[bool(d) for d in attributes.get("dimension_types", [1, 1, 1])])


def _geometry_only(atoms: Any) -> Any:
    """A fresh Atoms with numbers, positions, cell and pbc only (no labels, info, arrays)."""

    from ase import Atoms

    return Atoms(numbers=atoms.numbers.copy(), positions=atoms.positions.copy(),
                 cell=atoms.cell.array.copy(), pbc=atoms.pbc.copy())


def check_edge_sites(atoms: Any, edge: Mapping[str, Any]) -> None:
    """The edge_id's site indices must address Li sites of this structure.

    edge_id indices refer to LiTraj's site order; a pristine source in another order
    would silently give a wrong hop, so a non-Li or out-of-range site fails closed.
    """

    source, target = int(edge["source"]), int(edge["target"])
    if not (0 <= source < len(atoms) and 0 <= target < len(atoms)):
        raise RunnerValidationError("edge site index outside the structure")
    symbols = atoms.get_chemical_symbols()
    if symbols[source] != "Li" or symbols[target] != "Li":
        raise RunnerValidationError(f"edge sites {source}, {target} are not Li in this structure "
                                    "(site order differs from LiTraj's)")


def hop_length(atoms: Any, edge: Mapping[str, Any]) -> float:
    """|r_t + o.L - r_s| with the hop's own cell offset (not the minimum image)."""

    import numpy as np

    check_edge_sites(atoms, edge)
    source, target = int(edge["source"]), int(edge["target"])
    offset = np.asarray(edge["offset"], dtype=float)
    vector = atoms.positions[target] + offset @ np.asarray(atoms.cell) - atoms.positions[source]
    return float(np.linalg.norm(vector))
