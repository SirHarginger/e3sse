"""Pristine multi-shard structure lookup uses source IDs, never row order."""

from __future__ import annotations

import hashlib

import pytest
from ase import Atoms
from ase.io import write

from e3sse.production.runners.context import RunnerValidationError
from e3sse.production.runners.structures import StructureSource


def _source(tmp_path, ids, formulas=None):
    path = tmp_path / "BVEL13k_train.xyz"
    frames = []
    for i, material_id in enumerate(ids):
        formula = formulas[i] if formulas else "Li2O"
        atoms = Atoms(formula, positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                      cell=[5, 5, 5], pbc=True)
        atoms.info.update(material_id=material_id, E_3D=9.9)
        frames.append(atoms)
    write(path, frames, format="extxyz")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    g0 = {"BVEL13k": [{"path": str(path), "sha256": digest,
                        "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}]}
    spec = {"kind": "ase_extxyz_shards", "paths": [str(path)], "id_field": "material_id"}
    return StructureSource(spec, tmp_path, g0)


def test_extxyz_source_indexes_material_id_and_strips_metadata(tmp_path):
    source = _source(tmp_path, ["mp-2", "mp-1"])
    atoms = source.load("mp-1")
    assert atoms.get_chemical_formula() == "Li2O"
    assert atoms.info == {} and not atoms.arrays.keys() - {"numbers", "positions"}
    assert next(iter(source.identity()["files_read"].values()))


def test_extxyz_source_returns_the_requested_frame_not_the_first(tmp_path):
    source = _source(tmp_path, ["mp-1", "mp-2", "mp-3"], ["Li2O", "Li2S", "Li2Se"])
    assert source.load("mp-3").get_chemical_formula() == "Li2Se"
    assert source.load("mp-2").get_chemical_formula() == "Li2S"
    assert source.load("mp-1").get_chemical_formula() == "Li2O"


def test_extxyz_source_rejects_duplicate_ids(tmp_path):
    source = _source(tmp_path, ["mp-1", "mp-1"])
    with pytest.raises(RunnerValidationError, match="two pristine structures"):
        source.load("mp-1")
