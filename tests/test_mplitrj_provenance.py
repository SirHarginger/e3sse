import hashlib
import io
import json
import multiprocessing
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, write
from test_dataset_audit import FIXTURE_DATA, _config

from e3sse.data import dataset_audit, mplitrj_provenance
from e3sse.data import litraj_provenance as lp
from e3sse.data.dataset_audit import G0ExecutionError, run_audit
from e3sse.data.litraj_provenance import FrameGeometry, RawFrameRef
from e3sse.data.mplitrj_provenance import (
    build_provenance_index,
    classify,
    evaluate_index,
    index_inputs,
    material_bucket,
    ordering_summary,
    read_frame_at,
    read_member,
    scan_flattened,
)

CELL = 6.0
BASE = np.random.default_rng(7).uniform(0.5, CELL - 0.5, size=(10, 3))
HOP_A = "mp-1_0_1_1_0_0"
HOP_B = "mp-1_0_2_-1_0_0"  # negative periodic-image offset
HOP_C = "mp-2_0_1_0_0_0"


def _atoms(axis: int, image: int, step: int, *, material: str | None = None) -> Atoms:
    positions = BASE.copy()
    positions[0, axis] += 0.3 * image  # the migrating Li
    positions += 0.01 * step  # optimisation-step relaxation of every atom
    positions[5, 2] += 0.05 * axis  # hops start from different Li configurations
    atoms = Atoms("LiO9", positions=positions, cell=[CELL] * 3, pbc=True)
    energy = -10.0 + 0.1 * image + 0.001 * step + 0.5 * axis
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((10, 3)))
    if material is not None:
        atoms.info["material_id"] = material
    return atoms


HOPS = {
    HOP_A: {"axis": 0, "steps": 2, "images": 3},
    HOP_B: {"axis": 1, "steps": 1, "images": 3},
    HOP_C: {"axis": 2, "steps": 1, "images": 2},
}


def _write_archive(path: Path, hops: dict[str, dict]) -> None:
    build = path.parent / "_build"
    build.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        for number, (edge, spec) in enumerate(hops.items()):
            for step in range(spec["steps"]):
                frames = spec.get("frames") or [
                    _atoms(spec["axis"], image, step) for image in range(spec["images"])
                ]
                traj = build / f"{edge}_{step}.traj"
                write(traj, frames, format="traj")
                method = zipfile.ZIP_DEFLATED if number % 2 else zipfile.ZIP_STORED
                name = f"MPLiTrj_raw/{edge}.neb/band_optim_step_{step}.traj"
                if spec.get("corrupt"):
                    archive.writestr(name, b"not an ASE trajectory")
                else:
                    archive.write(traj, name, method)
                if spec.get("endpoints") and step == 0:
                    for endpoint, endpoint_frames in (
                        ("source.xyz", frames[:1]),
                        ("target.xyz", frames[-1:]),
                        ("traj_init.xyz", frames),
                    ):
                        target = build / f"{edge}_{endpoint}"
                        write(target, endpoint_frames, format="extxyz")
                        archive.write(target, f"MPLiTrj_raw/{edge}.neb/{endpoint}", method)
            for extra, payload in spec.get("extra", {}).items():
                archive.writestr(f"MPLiTrj_raw/{edge}.neb/{extra}", payload)
    shutil.rmtree(build)


def _flattened(hops: dict[str, dict]) -> dict[str, list[Atoms]]:
    train = [
        _atoms(HOPS[edge]["axis"], image, step, material=lp.parse_edge_id(edge))
        for edge in (HOP_A, HOP_B)
        for step in range(HOPS[edge]["steps"])
        for image in range(HOPS[edge]["images"])
    ]
    # A lattice-translated (wrapped) copy of hop A, step 0, image 1.
    train[1].positions[0] += [CELL, 0.0, 0.0]
    return {
        "MPLiTrj_train.xyz": train,
        "MPLiTrj_val.xyz": [_atoms(2, 0, 0, material="mp-2")],
        "MPLiTrj_test.xyz": [_atoms(2, 1, 0, material="mp-2")],
    }


def _tree(tmp_path: Path, *, hops=HOPS, flattened=None) -> Path:
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DATA, data)
    mplitrj = data / "raw" / "MPLiTrj"
    shutil.rmtree(mplitrj)
    mplitrj.mkdir()
    for name, frames in (flattened or _flattened(hops)).items():
        write(mplitrj / name, frames, format="extxyz")
    _write_archive(data / "downloads" / "MPLiTrj_raw.zip", hops)
    return data


def _records(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    records = []
    for shard in manifest["shards"]:
        records.extend(json.loads((manifest_path.parent / shard["path"]).read_text())["records"])
    return sorted(records, key=lambda r: r["derived_frame_id"])


def _mpli(report: dict) -> dict:
    return next(item for item in report["datasets"] if item["dataset"] == "MPLiTrj")


def _normalize(value):
    # Shards carry created_at, so their SHA-256 differs between runs; records
    # are compared separately.
    runtime = {"created_at", "execution", "git_dirty", "sha256"}
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items() if k not in runtime}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _snapshot(root: Path) -> dict[Path, tuple[str, int]]:
    return {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --------------------------------------------------------------------------- primitives


def test_member_reader_matches_zipfile_and_checks_crc(tmp_path):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("stored.txt", b"x" * 100, compress_type=zipfile.ZIP_STORED)
        archive.writestr("deflated.txt", b"y" * 1000, compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(path) as archive:
        entries = {i.filename: mplitrj_provenance.member_entry(i) for i in archive.infolist()}
        expected = {name: archive.read(name) for name in entries}

    with path.open("rb") as handle:
        assert {name: read_member(handle, entry) for name, entry in entries.items()} == expected
        corrupted = list(entries["deflated.txt"])
        corrupted[5] ^= 1
        with pytest.raises(ValueError, match="CRC"):
            read_member(handle, corrupted)


def test_scan_streams_offsets_that_reproduce_each_frame(tmp_path):
    path = tmp_path / "MPLiTrj_train.xyz"
    frames = _flattened(HOPS)["MPLiTrj_train.xyz"]
    write(path, frames, format="extxyz")

    scan = scan_flattened(str(path))

    assert scan["frames"] == len(frames)
    assert scan["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sorted(scan["material_frames"]) == ["mp-1"]
    located = scan["material_frames"]["mp-1"]
    assert [index for index, _, _ in located] == list(range(len(frames)))
    with path.open("rb") as handle:
        for (index, offset, length), expected in zip(located, iread(path, index=":"), strict=True):
            atoms = read_frame_at(handle, offset, length)
            assert np.array_equal(atoms.positions, expected.positions), index
            assert atoms.info["material_id"] == "mp-1"


def test_stable_fingerprints():
    assert lp.composition_key(np.array([3, 8, 8])) == (
        "eebd9a3ca680c1948842b9e5ac133cd8105a6e9b3a121ac2f7cc0ce1f741507e"
    )
    assert material_bucket("mp-1020015") == "80"
    assert material_bucket(None) == "89"


def test_tolerance_boundary_and_wrapping():
    reference = FrameGeometry.from_atoms(_atoms(0, 1, 0))
    group = lp.RawGroup(
        refs=[RawFrameRef(HOP_A, "band_optim_step", 0, 1)],
        positions=reference.positions[None],
        cells=reference.cell[None],
        energies=np.array([np.nan]),
    )

    def query(shift: np.ndarray) -> np.ndarray:
        atoms = _atoms(0, 1, 0)
        atoms.positions[3] += shift
        return lp.matching_indices(
            group, FrameGeometry.from_atoms(atoms), position_atol=1e-4, cell_atol=1e-4
        )

    assert len(query(np.array([0.9e-4, 0, 0]))) == 1
    assert len(query(np.array([1.1e-4, 0, 0]))) == 0
    assert len(query(np.array([CELL, -CELL, 0]))) == 1  # lattice translation


def test_classification_never_picks_a_candidate():
    one = RawFrameRef(HOP_A, "band_optim_step", 0, 0)
    same_hop = RawFrameRef(HOP_A, "source", None, 0)
    other_hop = RawFrameRef(HOP_B, "band_optim_step", 0, 0)

    assert classify([]) == "unmapped"
    assert classify([one]) == "mapped_unique"
    assert classify([one, same_hop]) == "mapped_hop_unique"
    assert classify([one, other_hop]) == "ambiguous"


def test_ordering_detection():
    ordered = [("a", 0, 0), ("a", 0, 1), ("a", 1, 0), ("b", 0, 0), ("b", 0, 1)]
    shuffled = [("a", 0, 0), ("b", 0, 0), ("a", 0, 1)]
    backwards = [("a", 1, 0), ("a", 0, 0)]

    assert ordering_summary(ordered)["conclusion"].startswith("hop-contiguous with non-decreasing")
    assert ordering_summary(ordered)["hops_in_sorted_edge_id_order"]
    assert not ordering_summary(shuffled)["hop_contiguous"]
    assert ordering_summary(backwards)["step_image_order_violations"] == 1
    assert ordering_summary([(None, None, None)])["conclusion"].startswith("not determined")


# --------------------------------------------------------------------------- full build


def test_full_coverage_index_makes_same_hop_exclusion_true(tmp_path):
    data = _tree(tmp_path)
    before = _snapshot(data)
    config = _config(tmp_path, data)

    manifest, path = build_provenance_index(config, workers=1)
    records = _records(path)
    report, _ = run_audit(config, workers=1)
    mpli = _mpli(report)

    assert _snapshot(data) == before  # raw archive and sources untouched
    assert not list(data.rglob("*.tmp"))
    assert manifest["counts"]["record_count"] == manifest["counts"]["mapped_unique_count"] == 11
    assert manifest["complete_coverage"] and manifest["coverage_blockers"] == []
    wrapped = next(r for r in records if r["derived_frame_id"] == "MPLiTrj_train.xyz:1")
    assert (wrapped["edge_id"], wrapped["optimization_step"], wrapped["neb_image_index"]) == (
        HOP_A, 0, 1,
    )
    assert wrapped["raw_archive_path"] == f"MPLiTrj_raw/{HOP_A}.neb/band_optim_step_0.traj"
    assert {r["edge_id"] for r in records} == {HOP_A, HOP_B, HOP_C}
    assert all(r["energy_consistent"] for r in records)
    assert {s["path"].split("/")[0] for s in manifest["shards"]} == {"train", "val", "test"}
    assert manifest["ordering"]["MPLiTrj_train.xyz"]["hop_contiguous"]
    assert manifest["nebdft2k_crosscheck"]["intersection"]["examples"] == [HOP_A]
    assert manifest["nebdft2k_crosscheck"]["mplitrj_only"]["count"] == 2
    assert manifest["source_archive"]["sha256"] == hashlib.sha256(
        (data / "downloads" / "MPLiTrj_raw.zip").read_bytes()
    ).hexdigest()

    same_hop = mpli["capabilities"]["same_hop_exclusion_supported"]
    assert same_hop["available"] is True
    assert "11/11 G0 frames mapped" in same_hop["detail"]
    assert mpli["identifiers"]["hop"]["status"] == "satisfied"
    assert mpli["identifiers"]["hop"]["availability"] == "derived"
    assert mpli["status"] == "pass"
    assert report["mplitrj_provenance_index"]["status"] == "valid"
    assert report["linkage_summary"]["nebDFT2k_to_MPLiTrj"]["hop"]["intersection_count"] == 1


def test_ambiguous_and_unmapped_frames_keep_capability_false(tmp_path):
    hops = {**HOPS}
    # Hop B additionally stores a structure identical to hop A, step 0, image 0.
    hop_b = [_atoms(1, image, 0) for image in range(3)] + [_atoms(0, 0, 0)]
    hops[HOP_B] = {**HOPS[HOP_B], "frames": hop_b}
    flattened = _flattened(HOPS)
    flattened["MPLiTrj_test.xyz"][0].positions[4] += [0.05, 0, 0]  # beyond tolerance
    data = _tree(tmp_path, hops=hops, flattened=flattened)
    config = _config(tmp_path, data)

    manifest, path = build_provenance_index(config, workers=1)
    records = {r["derived_frame_id"]: r for r in _records(path)}
    mpli = _mpli(run_audit(config, workers=1)[0])

    shared = records["MPLiTrj_train.xyz:0"]  # hop A image 0 == hop B image 0
    assert shared["mapping_status"] == "ambiguous"
    assert shared["edge_id"] is None
    assert shared["candidate_edge_ids"] == sorted([HOP_A, HOP_B])
    assert records["MPLiTrj_test.xyz:0"]["mapping_status"] == "unmapped"
    counts = manifest["counts"]
    assert (counts["ambiguous_count"], counts["unmapped_count"]) == (1, 1)
    assert manifest["examples"]["ambiguous"]["examples"][0]["frame"] == "MPLiTrj_train.xyz:0"
    same_hop = mpli["capabilities"]["same_hop_exclusion_supported"]
    assert same_hop["available"] is False
    assert "ambiguous=1, unmapped=1" in same_hop["detail"]
    assert mpli["identifiers"]["hop"]["missing"]["count"] == 2


def test_serial_and_parallel_builds_are_identical(tmp_path):
    config = _config(tmp_path, _tree(tmp_path))

    serial, path = build_provenance_index(config, workers=1)
    serial_records = _records(path)
    shutil.rmtree(path.parent)
    parallel, parallel_path = build_provenance_index(config, workers=2)

    assert parallel_path == path
    assert _normalize(serial) == _normalize(parallel)
    assert _records(parallel_path) == serial_records


# --------------------------------------------------------------------------- resume and staleness


def test_resume_reuses_verified_units_and_redoes_damaged_ones(tmp_path):
    config = _config(tmp_path, _tree(tmp_path))
    first, path = build_provenance_index(config, workers=1)
    messages: list[str] = []

    again, _ = build_provenance_index(config, workers=1, resume=True, progress=messages.append)
    assert again == first
    assert any("reused complete index" in m for m in messages)

    path.unlink()
    damaged = path.parent / first["shards"][0]["path"]
    damaged.write_text("{}")
    messages.clear()
    rebuilt, _ = build_provenance_index(config, workers=1, resume=True, progress=messages.append)

    bucket = damaged.stem.removeprefix("material-bucket-")
    assert f"G0 provenance/material-bucket-{bucket}: starting" in messages
    assert sum(m.endswith(": starting") for m in messages) == 1
    assert _normalize(rebuilt) == _normalize(first)


@pytest.fixture
def fork_workers(monkeypatch):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method unavailable")
    monkeypatch.setattr(dataset_audit, "_START_METHOD", "fork")


def test_killed_build_resumes_from_completed_units(tmp_path, monkeypatch, fork_workers):
    config = _config(tmp_path, _tree(tmp_path))
    original = mplitrj_provenance.map_bucket
    failing = material_bucket("mp-2")

    def crash(bucket, *args, **kwargs):
        if bucket == failing:
            raise RuntimeError("synthetic crash")
        return original(bucket, *args, **kwargs)

    monkeypatch.setattr(mplitrj_provenance, "map_bucket", crash)
    with pytest.raises(G0ExecutionError) as excinfo:
        build_provenance_index(config, workers=2)
    root = excinfo.value.failures_path.parent
    assert list(excinfo.value.failures) == [f"provenance/material-bucket-{failing}"]
    assert not (root / "manifest.json").exists()

    monkeypatch.undo()
    messages: list[str] = []
    manifest, _ = build_provenance_index(config, workers=2, resume=True, progress=messages.append)
    started = [m for m in messages if m.endswith(": starting")]
    assert started == [f"G0 provenance/material-bucket-{failing}: starting"]
    assert manifest["complete_coverage"]
    assert not (root / "failures.json").exists()


def test_stale_or_modified_index_is_rejected_by_g0(tmp_path):
    data = _tree(tmp_path)
    config = _config(tmp_path, data)
    manifest, path = build_provenance_index(config, workers=1)
    settings = {**dataset_audit.DEFAULT_SETTINGS}
    output_root = tmp_path / "outputs"
    sources = sorted((data / "raw" / "MPLiTrj").glob("*.xyz"))
    archive = data / "downloads" / "MPLiTrj_raw.zip"

    def state():
        return evaluate_index(output_root, settings, index_inputs(sources, archive, data), data)

    assert state()["status"] == "valid"
    shard = path.parent / manifest["shards"][0]["path"]
    original = shard.read_bytes()
    shard.write_bytes(original.replace(b'"mapped_unique"', b'"mapped_hop_unique"', 1))
    assert state()["status"] == "invalid"
    shard.write_bytes(original)

    source = data / "raw" / "MPLiTrj" / "MPLiTrj_val.xyz"
    source.write_bytes(source.read_bytes())  # new mtime: a different input identity
    stale = state()
    assert stale["status"] == "stale"
    assert stale["other_index_ids"] == [manifest["index_id"]]

    mpli = _mpli(run_audit(config, workers=1)[0])
    assert not mpli["capabilities"]["same_hop_exclusion_supported"]["available"]
    assert "provenance index stale" in mpli["capabilities"]["same_hop_exclusion_supported"]["detail"]


def test_g0_resume_picks_up_a_newly_built_index(tmp_path):
    config = _config(tmp_path, _tree(tmp_path))
    before, _ = run_audit(config, workers=1)
    assert not _mpli(before)["capabilities"]["same_hop_exclusion_supported"]["available"]

    build_provenance_index(config, workers=1)
    messages: list[str] = []
    after, _ = run_audit(config, workers=1, resume=True, progress=messages.append)

    assert _mpli(after)["capabilities"]["same_hop_exclusion_supported"]["available"]
    assert "G0 MPLiTrj/MPLiTrj_train: reused checkpoint" in messages  # no reparse
    assert not any("reused complete report" in m for m in messages)


# --------------------------------------------------------------------------- review regressions


def _build(tmp_path, hops, flattened=None, **audit):
    config = _config(tmp_path, _tree(tmp_path, hops=hops, flattened=flattened), **audit)
    manifest, path = build_provenance_index(config, workers=1)
    return config, manifest, {r["derived_frame_id"]: r for r in _records(path)}


def test_endpoint_members_give_hop_unique_mapping(tmp_path):
    hops = {**HOPS, HOP_C: {**HOPS[HOP_C], "endpoints": True}}

    _, manifest, records = _build(tmp_path, hops)

    # val = hop C image 0: in the step file, source.xyz and traj_init.xyz.
    endpoint = records["MPLiTrj_val.xyz:0"]
    assert endpoint["mapping_status"] == "mapped_hop_unique"
    assert endpoint["edge_id"] == HOP_C
    assert endpoint["candidate_count"] == 3
    assert endpoint["optimization_step"] is None and endpoint["raw_archive_path"] is None
    assert endpoint["raw_hop_directory"] == f"MPLiTrj_raw/{HOP_C}.neb/"
    assert manifest["complete_coverage"]


def test_unrecognised_hop_files_block_coverage(tmp_path):
    hops = {**HOPS, HOP_B: {**HOPS[HOP_B], "extra": {"band_optim_extra.traj": b"x"}}}

    _, manifest, _ = _build(tmp_path, hops)

    assert not manifest["complete_coverage"]
    assert manifest["source_archive"]["hops_with_unrecognised_files"]["examples"] == [HOP_B]


def test_unreadable_member_marks_whole_material_error(tmp_path):
    hops = {**HOPS, HOP_B: {**HOPS[HOP_B], "corrupt": True}}

    _, manifest, records = _build(tmp_path, hops)

    mp1 = [r for r in records.values() if r["material_id"] == "mp-1"]
    assert {r["mapping_status"] for r in mp1} == {"error"}
    assert all("raw archive member unreadable" in r["reason"] for r in mp1)
    assert manifest["counts"]["error_count"] == len(mp1) == 9


def test_material_memory_cap_reports_error(tmp_path):
    _, manifest, records = _build(tmp_path, HOPS, provenance_max_material_raw_bytes=1)

    assert {r["mapping_status"] for r in records.values()} == {"error"}
    assert "max_material_raw_bytes=1" in records["MPLiTrj_val.xyz:0"]["reason"]
    assert not manifest["complete_coverage"]


def test_scan_treats_empty_material_id_as_missing(tmp_path):
    path = tmp_path / "MPLiTrj_train.xyz"
    path.write_text('1\nProperties=species:S:1:pos:R:3 material_id="" pbc="F F F"\nLi 0 0 0\n')

    scan = scan_flattened(str(path))

    assert scan["error"] is None
    assert list(scan["material_frames"]) == [mplitrj_provenance.MISSING_MATERIAL]


def test_changed_tolerance_or_content_or_manifest_invalidates_index(tmp_path):
    data = _tree(tmp_path)
    config = _config(tmp_path, data)
    manifest, path = build_provenance_index(config, workers=1)
    output_root = tmp_path / "outputs"
    sources = sorted((data / "raw" / "MPLiTrj").glob("*.xyz"))
    archive = data / "downloads" / "MPLiTrj_raw.zip"
    settings = {**dataset_audit.DEFAULT_SETTINGS}

    def state(**overrides):
        inputs = index_inputs(sources, archive, data)
        return evaluate_index(output_root, {**settings, **overrides}, inputs, data)

    assert state()["status"] == "valid"
    assert state(provenance_position_atol=2e-4)["status"] == "stale"

    source = data / "raw" / "MPLiTrj" / "MPLiTrj_val.xyz"
    original, stat = source.read_bytes(), source.stat()
    position = original.index(b"energy=") + len(b"energy=-")
    digit = original[position : position + 1]
    assert digit.isdigit()
    tampered = original[:position] + (b"1" if digit != b"1" else b"2") + original[position + 1 :]
    assert len(tampered) == len(original) and tampered != original
    source.write_bytes(tampered)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same size and mtime
    changed = state()
    assert changed["status"] == "invalid" and "MPLiTrj_val.xyz" in changed["reason"]
    source.write_bytes(original)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    edited = dict(manifest)
    del edited["counts"]
    path.write_text(json.dumps(edited))
    assert state()["status"] == "invalid"


def test_member_reader_handles_data_descriptors():
    class Unseekable(io.RawIOBase):
        def __init__(self):
            self.buffer = bytearray()

        def writable(self):
            return True

        def write(self, data):
            self.buffer.extend(data)
            return len(data)

    stream = Unseekable()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a.txt", b"payload" * 50)
    raw = bytes(stream.buffer)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.getinfo("a.txt")
        assert info.flag_bits & 0x08  # sizes live in a trailing data descriptor
        entry = mplitrj_provenance.member_entry(info)

    assert read_member(io.BytesIO(raw), entry) == b"payload" * 50
