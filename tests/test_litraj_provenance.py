import hashlib
import zipfile

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from e3sse.data.litraj_provenance import (
    FrameGeometry,
    RawFrameRef,
    geometries_match,
    index_archive,
    inspect_hops,
    parse_edge_id,
    parse_member,
    select_sample_materials,
    summarize_archive,
    validate_mapping,
)


def _atoms(x: float, energy: float = -1.0) -> Atoms:
    atoms = Atoms("LiO", positions=[[x, 0, 0], [1, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((2, 3)))
    return atoms


def test_edge_ids_parse_deterministically_from_archive_paths():
    member = parse_member("MPLiTrj_raw/mp-1020015_0_14_1_0_0.neb/band_optim_step_12.traj")

    assert member.edge_id == "mp-1020015_0_14_1_0_0"
    assert member.kind == "band_optim_step"
    assert member.step == 12
    assert parse_edge_id(member.edge_id) == "mp-1020015"
    assert parse_member("MPLiTrj_raw/mp-1_0_1.neb/traj_init.xyz").kind == "traj_init"
    assert parse_member("MPLiTrj_raw/mp-1_0_1.neb/").kind == "directory"
    assert parse_member("MPLiTrj_raw/README.txt") is None
    assert parse_edge_id("not-an-edge") is None


def _archive(tmp_path):
    path = tmp_path / "MPLiTrj_raw.zip"
    step = tmp_path / "step.traj"
    write(step, [_atoms(0.0), _atoms(0.5), _atoms(1.5)], format="traj")
    with zipfile.ZipFile(path, "w") as archive:
        for edge in ("mp-2_0_1_1_0_0", "mp-1_0_1_1_0_0"):
            archive.write(step, f"MPLiTrj_raw/{edge}.neb/band_optim_step_0.traj")
            archive.write(step, f"MPLiTrj_raw/{edge}.neb/band_optim_step_2.traj")
            archive.writestr(f"MPLiTrj_raw/{edge}.neb/source.xyz", "")
        archive.writestr("MPLiTrj_raw/notes.txt", "x")
    return path


def test_archive_inspection_is_read_only_and_bounded(tmp_path):
    path = _archive(tmp_path)
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    listing = sorted(tmp_path.iterdir())

    with zipfile.ZipFile(path, mode="r") as archive:
        index = index_archive(archive)
        summary = summarize_archive(index, example_limit=1)
        contents = inspect_hops(archive, index, list(index.hops)[:1], example_limit=1)

    assert list(index.hops) == ["mp-1_0_1_1_0_0", "mp-2_0_1_1_0_0"]
    assert summary["hop_directories"] == 2
    assert summary["material_ids"]["count"] == 2
    assert summary["optimization_steps_per_hop"] == {"2": 2}
    assert summary["hops_with_non_contiguous_steps"]["count"] == 2
    assert summary["hops_with_non_contiguous_steps"]["examples"] == ["mp-1_0_1_1_0_0"]
    assert summary["endpoint_file_presence"] == {"source": 2, "target": 0, "traj_init": 0}
    assert summary["non_hop_members"]["examples"] == ["MPLiTrj_raw/notes.txt"]
    step = contents["file_kinds"]["band_optim_step"]
    assert step["frames_per_file"] == {"3": 2}
    assert step["calculator_result_keys"] == ["energy", "forces"]
    # The empty source.xyz is reported as a zero-frame file, not silently skipped.
    assert contents["file_kinds"]["source"]["frames_per_file"] == {"0": 1}
    assert contents["read_failures"]["count"] == 0
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before
    assert sorted(tmp_path.iterdir()) == listing


def test_geometry_match_is_modulo_lattice_and_rejects_displacement():
    base = FrameGeometry.from_atoms(_atoms(0.1))
    wrapped = FrameGeometry.from_atoms(_atoms(4.1))
    moved = FrameGeometry.from_atoms(_atoms(0.2))

    assert geometries_match(base, wrapped)
    assert not geometries_match(base, moved)


def test_mapping_separates_single_multi_and_unmatched_frames():
    shared = FrameGeometry.from_atoms(_atoms(0.0))
    raw = {
        "mp-1": [
            (RawFrameRef("mp-1_a", "band_optim_step", 0, 0), shared),
            (RawFrameRef("mp-1_a", "band_optim_step", 0, 1), FrameGeometry.from_atoms(_atoms(0.5))),
            (RawFrameRef("mp-1_b", "source", None, 0), shared),
        ]
    }
    flattened = [
        ("f.xyz:0", "mp-1", FrameGeometry.from_atoms(_atoms(0.5))),
        ("f.xyz:1", "mp-1", FrameGeometry.from_atoms(_atoms(0.0))),
        ("f.xyz:2", "mp-1", FrameGeometry.from_atoms(_atoms(0.9))),
    ]

    result = validate_mapping(flattened, raw, example_limit=5)

    assert result["status"] == "not_demonstrated"
    assert result["matched_single_hop"] == 1
    assert result["matched_multi_hop"]["examples"] == [
        {"frame": "f.xyz:1", "edge_ids": ["mp-1_a", "mp-1_b"]}
    ]
    assert result["unmatched"]["examples"] == ["f.xyz:2"]
    assert result["criteria"]["energy_used_for_matching"] is False


def test_negative_lattice_offsets_parse():
    assert parse_edge_id("mp-1234_0_1_-1_0_0") == "mp-1234"
    member = parse_member("MPLiTrj_raw/mp-1234_0_1_-1_0_0.neb/target.xyz")
    assert member.edge_id == "mp-1234_0_1_-1_0_0"
    assert member.kind == "target"


def test_incomplete_candidate_set_is_inconclusive():
    geometry = FrameGeometry.from_atoms(_atoms(0.0))
    raw = {"mp-1": [(RawFrameRef("mp-1_a", "band_optim_step", 0, 0), geometry)]}
    flattened = [("f.xyz:0", "mp-1", FrameGeometry.from_atoms(_atoms(0.0)))]

    clean = validate_mapping(flattened, raw, example_limit=2)
    failed_read = validate_mapping(flattened, raw, example_limit=2, raw_read_failures=1)
    unparsed = validate_mapping(flattened, raw, example_limit=2, unparseable_edge_ids=1)

    assert clean["status"] == "demonstrated_on_sample"
    assert failed_read["status"] == "inconclusive"
    assert unparsed["status"] == "inconclusive"


def test_sample_selection_skips_oversized_materials(tmp_path):
    path = _archive(tmp_path)
    with zipfile.ZipFile(path) as archive:
        index = index_archive(archive)

    selected, skipped = select_sample_materials(index, 1, max_members_per_material=2)
    both, none_skipped = select_sample_materials(index, 5, max_members_per_material=10)

    # Each fixture hop has two steps and one endpoint (three members).
    assert (selected, skipped) == ([], ["mp-1", "mp-2"])
    assert (both, none_skipped) == (["mp-1", "mp-2"], [])
