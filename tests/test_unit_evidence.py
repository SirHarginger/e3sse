"""Unit-evidence collector on synthetic extxyz files (no LiTraj data)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from e3sse.data.unit_evidence import collect, comment_keys


def _frames(path, *, energy_per_atom=-4.5, n=3, stress=True):
    frames = []
    for index in range(n):
        atoms = bulk("Li2O", "fluorite", a=4.62)
        atoms.info["material_id"] = f"mp-{index}"
        results = {
            "energy": energy_per_atom * len(atoms),
            "forces": np.full((len(atoms), 3), 0.1),
        }
        if stress:
            results["stress"] = np.full(6, 0.01)
        atoms.calc = SinglePointCalculator(atoms, **results)
        frames.append(atoms)
    write(path, frames, format="extxyz")
    return path


def _unparseable(path):
    """An extxyz-looking file whose values would crash any numeric read."""

    path.write_text('3\nLattice="a b c d e f g h i" Properties=species:S:1:pos:R:3 energy=SEALED\n')
    return path


def test_values_are_read_only_from_proven_nonpath_train_frames(tmp_path):
    files = {
        "train": _frames(tmp_path / "MPLiTrj_train.xyz"),
        "val": _unparseable(tmp_path / "MPLiTrj_val.xyz"),
        "test": _unparseable(tmp_path / "MPLiTrj_test.xyz"),
    }
    index = tmp_path / "index.csv"
    index.write_text("edge_id,material_id,chemsys,_split,em_dft\nmp-0_0_1_0_0_0,mp-0,Li-O,train,x\n")
    trajectory = _unparseable(tmp_path / "mp-0_0_1_0_0_0_relaxed.xyz")
    evidence = collect(
        mplitrj_files=files,
        train_nonpath_frame_indices=[0, 2],
        nebdft2k_index=index,
        nebdft2k_trajectories=[trajectory],
    )
    scale = evidence["mplitrj_train_unit_scale"]
    assert scale["frames_read"] == 2
    assert scale["checks"] == {
        "energy_scale_is_eV": True,
        "force_scale_is_eV_per_A": True,
        "stress_scale_is_eV_per_A3": True,
    }
    assert "range" not in str(scale)
    assert evidence["mplitrj"]["test"]["has_energy"] is True
    assert evidence["nebdft2k"]["index_columns"][:4] == ["edge_id", "material_id", "chemsys", "_split"]
    assert evidence["nebdft2k"]["trajectory_keys"][0]["comment_keys"] == ["Lattice", "Properties", "energy"]


def test_unit_slips_and_missing_stress_are_reported(tmp_path):
    hartree = _frames(tmp_path / "MPLiTrj_train.xyz", energy_per_atom=-0.17, stress=False)
    files = {"train": hartree, "val": hartree, "test": hartree}
    index = tmp_path / "index.csv"
    index.write_text("edge_id,_split\n")
    evidence = collect(
        mplitrj_files=files,
        train_nonpath_frame_indices=[0],
        nebdft2k_index=index,
        nebdft2k_trajectories=[],
    )
    checks = evidence["mplitrj_train_unit_scale"]["checks"]
    assert checks["energy_scale_is_eV"] is False
    assert checks["stress_scale_is_eV_per_A3"] == "absent"
    with pytest.raises(ValueError, match="non-path"):
        collect(
            mplitrj_files=files,
            train_nonpath_frame_indices=[],
            nebdft2k_index=index,
            nebdft2k_trajectories=[],
        )
    with pytest.raises(ValueError, match="train/val/test"):
        collect(
            mplitrj_files={"train": hartree},
            train_nonpath_frame_indices=[0],
            nebdft2k_index=index,
            nebdft2k_trajectories=[],
        )
    assert comment_keys(hartree)["has_forces_column"] is True
