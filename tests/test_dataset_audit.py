import gzip
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from e3sse.data import dataset_audit
from e3sse.data.dataset_audit import (
    DATASETS,
    audit_dataset,
    concise_summary,
    discover_datasets,
    inspect_optimade_jsonl,
    locate_quantities,
    run_audit,
    split_from_filename,
)

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "g0" / "data"
EDGE = "mp-1_0_1_1_0_0"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict:
    return {"path": str(path), "identifier": path.name, "bytes": 0, "sha256": _digest(path)}


def _write_raw_archive(path: Path, frames_by_edge: dict[str, list[Atoms]]) -> None:
    """Minimal MPLiTrj_raw.zip: one optimisation step plus endpoints per hop."""

    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / "_build"
    scratch.mkdir()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("MPLiTrj_raw/", "")
        for edge, frames in frames_by_edge.items():
            base = f"MPLiTrj_raw/{edge}.neb"
            archive.writestr(f"{base}/", "")
            step = scratch / f"{edge}.traj"
            write(step, frames, format="traj")
            archive.write(step, f"{base}/band_optim_step_0.traj")
            for name, atoms in (("source.xyz", frames[0]), ("target.xyz", frames[-1])):
                target = scratch / name
                write(target, atoms, format="extxyz")
                archive.write(target, f"{base}/{name}")
    shutil.rmtree(scratch)


def _data_tree(tmp_path: Path, *, archive_frames: dict[str, list[Atoms]] | None = None) -> Path:
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DATA, data)
    raw = data / "raw"
    if archive_frames is None:
        archive_frames = {
            EDGE: read(raw / "MPLiTrj" / "MPLiTrj_train.xyz", index=":"),
        }
    _write_raw_archive(data / "downloads" / "MPLiTrj_raw.zip", archive_frames)
    with zipfile.ZipFile(raw / "FPMD_screening.aiida", "w") as archive:
        archive.writestr("metadata.json", json.dumps({"export_version": "main_0001"}))
        archive.writestr("db.sqlite3", b"")
    with gzip.open(raw / "FPMD_optimade.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"x-optimade": {"meta": {"api_version": "1.2.0"}}}) + "\n")
        handle.write(json.dumps({"type": "structures", "id": "s-1", "attributes": {}}) + "\n")
    return data


def _config(tmp_path: Path, data: Path, **audit) -> Path:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "project": {"name": "e3sse-test"},
                "paths": {
                    "data_root": str(data),
                    "outputs_root": str(tmp_path / "outputs"),
                    "logs_root": str(tmp_path / "logs"),
                    "scratch_root": str(tmp_path / "scratch"),
                },
                "runtime": {"seed": 7},
                "audit": {"hash_max_bytes": 1_000_000, **audit},
            }
        ),
        encoding="utf-8",
    )
    return path


def _neb_dataset(tmp_path: Path, *, files: dict[str, str], rows: list[str]) -> list[Path]:
    root = tmp_path / "nebDFT2k"
    root.mkdir()
    source = FIXTURE_DATA / "raw" / "nebDFT2k"
    (root / "index.csv").write_text(
        "material_id,edge_id,chemsys,has_specific_TM,em_bvse,em_dft,_split\n"
        + "".join(f"{row}\n" for row in rows),
        encoding="utf-8",
    )
    for name, template in files.items():
        shutil.copy(source / template, root / name)
    return sorted(root.iterdir())


# --------------------------------------------------------------------------- full run


def test_full_audit_role_statuses_and_linkage(tmp_path):
    data = _data_tree(tmp_path)
    report, report_path = run_audit(_config(tmp_path, data))
    by_name = {item["dataset"]: item for item in report["datasets"]}

    assert set(discover_datasets(data / "raw")) == set(DATASETS)
    assert by_name["nebDFT2k"]["status"] == "pass"
    assert by_name["BVEL13k"]["status"] == "pass"
    assert by_name["nebBVSE122k"]["status"] == "pass"
    # MPLiTrj stays partial solely because same-hop exclusion is not yet supported.
    assert by_name["MPLiTrj"]["status"] == "partial"
    assert by_name["MPLiTrj"]["blockers"] == [
        f"same_hop_exclusion_supported: {by_name['MPLiTrj']['capabilities']['same_hop_exclusion_supported']['detail']}"
    ]
    assert by_name["FPMD"]["status"] == "partial"
    assert by_name["FPMD"]["blockers"] == [
        "aiida_query_available: AiiDA query capability unavailable in current environment"
    ]
    assert report["status"] == "partial"
    linkage = report["linkage_summary"]["nebDFT2k_to_MPLiTrj"]
    assert linkage["material"]["intersection_count"] == 1
    assert linkage["hop_vs_raw_archive"]["intersection_examples"] == [EDGE]
    assert report["implementation"]["litraj_provenance"]
    assert report_path.is_relative_to(tmp_path / "outputs")
    assert (report_path.parent / "inputs.json").exists()
    assert "Gate G0: partial" in concise_summary(report, report_path)


def test_audit_does_not_modify_raw_data_and_resume_reuses_report(tmp_path):
    data = _data_tree(tmp_path)
    source_files = sorted(path for path in data.rglob("*") if path.is_file())
    before = {path: (_digest(path), path.stat().st_mode) for path in source_files}
    config = _config(tmp_path, data)

    first, path = run_audit(config)
    first_mtime = path.stat().st_mtime_ns
    second, second_path = run_audit(config, resume=True)

    assert second == first
    assert second_path == path
    assert path.stat().st_mtime_ns == first_mtime
    after_files = sorted(path for path in data.rglob("*") if path.is_file())
    assert after_files == source_files  # nothing extracted next to the archive
    assert before == {path: (_digest(path), path.stat().st_mode) for path in source_files}


def test_resume_does_not_reuse_errored_final_report(tmp_path):
    config = _config(tmp_path, _data_tree(tmp_path))
    first, path = run_audit(config)
    poisoned = dict(first)
    poisoned["errors"] = ["transient parser failure"]
    path.write_text(json.dumps(poisoned), encoding="utf-8")

    resumed, resumed_path = run_audit(config, resume=True)

    assert resumed_path == path
    assert "transient parser failure" not in resumed["errors"]


def test_malformed_extxyz_fails_without_fabricating_schema():
    path = Path(__file__).parent / "fixtures" / "malformed.xyz"

    report = audit_dataset("MPLiTrj", [path], [_identity(path)])

    assert report["status"] == "fail"
    assert report["errors"]
    assert report["record_counts"]["frames"] == 0
    assert report["units"]["observed_unit_metadata"] == {}


def test_discovery_prefers_full_mplitrj_over_overlapping_subsample(tmp_path):
    raw = tmp_path / "raw"
    full = raw / "MPLiTrj" / "full.xyz"
    sample = raw / "MPLiTrj_subsample" / "sample.xyz"
    full.parent.mkdir(parents=True)
    sample.parent.mkdir(parents=True)
    full.write_text("", encoding="utf-8")
    sample.write_text("", encoding="utf-8")

    assert discover_datasets(raw)["MPLiTrj"] == [full]


def test_audit_is_deterministic():
    files = discover_datasets(FIXTURE_DATA / "raw")["MPLiTrj"]
    identities = [_identity(path) for path in files]

    first = audit_dataset("MPLiTrj", files, identities)
    second = audit_dataset("MPLiTrj", files, identities)

    for key in ("record_counts", "identifiers", "capabilities", "schema_summary", "split_summary"):
        assert first[key] == second[key]


# --------------------------------------------------------------------------- nebDFT2k


def test_nebdft2k_recognises_init_suffix_and_index_identifiers():
    files = discover_datasets(FIXTURE_DATA / "raw")["nebDFT2k"]

    report = audit_dataset("nebDFT2k", files, [_identity(p) for p in files])
    summary = report["dataset_summary"]
    ids = report["identifiers"]

    assert report["status"] == "pass"
    assert summary["indexed_hops"] == summary["paired_hops"] == 1
    assert summary["missing_init_files"]["count"] == 0
    assert summary["frame_count_distribution"] == {"init": {"2": 1}, "relaxed": {"2": 1}}
    assert ids["material"]["value_examples"] == ["mp-1"]
    assert ids["hop"]["value_examples"] == [EDGE]
    assert ids["chemical_system"]["value_examples"] == ["Li-O"]
    assert ids["published_split"]["value_sources"] == {"_split": 1}
    for dimension in ("material", "hop", "chemical_system", "published_split"):
        assert ids[dimension]["availability"] == "source_provided"
    assert ids["frame"]["availability"] == "derived"
    assert ids["neb_path"]["status"] == "not_applicable"
    assert ids["neb_path"]["missing"] is None
    assert report["capabilities"]["dft_forces_readable"]["available"]
    assert report["capabilities"]["dft_stress_readable"]["available"]


def test_nebdft2k_exact_pairing_reports_missing_and_unindexed(tmp_path):
    other = "mp-2_0_1_1_0_0"
    files = _neb_dataset(
        tmp_path,
        files={
            f"{EDGE}_init.xyz": f"{EDGE}_init.xyz",
            f"{EDGE}_relaxed.xyz": f"{EDGE}_relaxed.xyz",
            f"{other}_relaxed.xyz": f"{EDGE}_relaxed.xyz",
            f"{EDGE}_initial.xyz": f"{EDGE}_init.xyz",
            "mp-9_0_0_0_0_0_init.xyz": f"{EDGE}_init.xyz",
        },
        rows=[
            f"mp-1,{EDGE},Li-O,False,0.4,0.42,train",
            f"mp-2,{other},Li-O,False,0.5,0.55,test",
        ],
    )

    report = audit_dataset("nebDFT2k", files, [_identity(p) for p in files])
    summary = report["dataset_summary"]

    assert report["status"] == "partial"
    assert summary["paired_hops"] == 1
    assert summary["missing_init_files"]["examples"] == [f"{other}_init.xyz"]
    assert summary["missing_relaxed_files"]["count"] == 0
    assert summary["unexpected_unindexed_files"]["examples"] == [
        f"{EDGE}_initial.xyz",
        "mp-9_0_0_0_0_0_init.xyz",
    ]
    assert any(b.startswith("all_indexed_hops_have_init") for b in report["blockers"])
    assert summary["chemical_systems_spanning_splits"]["count"] == 1


# --------------------------------------------------------------------------- ASE fields


def test_scientific_fields_read_from_single_point_calculator(tmp_path):
    atoms = Atoms("LiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1.0, forces=np.zeros((2, 3)), stress=np.zeros(6)
    )
    path = tmp_path / "frame.xyz"
    write(path, atoms, format="extxyz")
    loaded = read(path)

    assert loaded.info == {}
    assert type(loaded.calc).__name__ == "SinglePointCalculator"
    assert locate_quantities(loaded) == {
        "energy": "calc.results:energy",
        "forces": "calc.results:forces",
        "stress": "calc.results:stress",
    }
    bare = Atoms("Li", positions=[[0, 0, 0]])
    assert locate_quantities(bare) == {"energy": None, "forces": None, "stress": None}


# --------------------------------------------------------------------------- MPLiTrj


def test_split_from_filename():
    assert split_from_filename("MPLiTrj_train.xyz") == "train"
    assert split_from_filename("MPLiTrj_val.xyz") == "val"
    assert split_from_filename("MPLiTrj_test.xyz") == "test"
    assert split_from_filename("MPLiTrj_raw.xyz") is None


def test_mplitrj_split_and_frame_ids_are_derived_not_source():
    files = discover_datasets(FIXTURE_DATA / "raw")["MPLiTrj"]

    report = audit_dataset("MPLiTrj", files, [_identity(p) for p in files])
    ids = report["identifiers"]
    caps = report["capabilities"]

    assert report["split_summary"]["counts"] == {"test": 1, "train": 2, "val": 1}
    assert ids["published_split"]["availability"] == "derived"
    assert ids["published_split"]["value_sources"] == {"source_filename": 4}
    assert ids["published_split"]["missing"]["count"] == 0
    assert ids["frame"]["availability"] == "derived"
    assert ids["frame"]["missing"]["count"] == 0
    assert ids["material"]["availability"] == "source_provided"
    assert ids["chemical_system"]["availability"] == "derived"
    assert ids["hop"]["availability"] == "unavailable"
    assert caps["published_split_supported"]["available"]
    assert caps["published_split_supported"]["value_source"] == "source_filename"
    assert caps["material_exclusion_supported"]["available"]
    assert not caps["same_hop_exclusion_supported"]["available"]
    assert caps["dft_energy_readable"]["available"]
    assert caps["dft_forces_readable"]["available"]
    assert caps["dft_stress_readable"]["available"]
    per_file = {item["file"]: item for item in report["dataset_summary"]["files"]}
    assert per_file["MPLiTrj_train.xyz"]["published_split_value_source"] == "source_filename"


def test_mplitrj_frame_ids_are_deterministic(tmp_path):
    files = discover_datasets(FIXTURE_DATA / "raw")["MPLiTrj"]
    reports = [
        audit_dataset("MPLiTrj", files, [_identity(p) for p in files], settings={"example_limit": 1})
        for _ in range(2)
    ]
    # Frames lacking stress would be listed by derived ID; strip stress to observe them.
    path = tmp_path / "MPLiTrj_train.xyz"
    frames = read(files[1], index=":")
    for atoms in frames:
        atoms.calc = SinglePointCalculator(atoms, energy=atoms.calc.results["energy"])
    write(path, frames, format="extxyz")
    report = audit_dataset("MPLiTrj", [path], [_identity(path)])

    assert reports[0]["identifiers"] == reports[1]["identifiers"]
    missing = report["schema_summary"][-1]["quantities"]["forces"]["missing"]
    assert missing["examples"] == ["MPLiTrj_train.xyz:0", "MPLiTrj_train.xyz:1"]


def test_same_hop_capability_reflects_mapping_evidence(tmp_path):
    data = _data_tree(tmp_path)
    report, _ = run_audit(_config(tmp_path, data))
    mpli = next(item for item in report["datasets"] if item["dataset"] == "MPLiTrj")
    mapping = mpli["dataset_summary"]["raw_archive"]["mapping_validation"]

    assert mapping["status"] == "demonstrated_on_sample"
    assert mapping["flattened_frames_sampled"] == 2
    assert mapping["matched_single_hop"] == 2
    assert mapping["energy_comparison"] == {
        "compared": 2,
        "agree": 2,
        "disagree": mapping["energy_comparison"]["disagree"],
    }
    same_hop = mpli["capabilities"]["same_hop_exclusion_supported"]
    # A sample demonstration does not by itself provide per-frame hop provenance.
    assert same_hop["available"] is False
    assert same_hop["requirement"] == "required"
    assert "full frame-to-hop provenance index has not been built" in same_hop["detail"]


def test_same_hop_mapping_not_demonstrated_for_unrelated_archive(tmp_path):
    decoy = Atoms("LiO", positions=[[1.5, 1.5, 1.5], [3.0, 0, 0]], cell=[4, 4, 4], pbc=True)
    data = _data_tree(tmp_path, archive_frames={EDGE: [decoy]})
    report, _ = run_audit(_config(tmp_path, data))
    mpli = next(item for item in report["datasets"] if item["dataset"] == "MPLiTrj")
    mapping = mpli["dataset_summary"]["raw_archive"]["mapping_validation"]

    assert mapping["status"] == "not_demonstrated"
    assert mapping["unmatched"]["examples"] == ["MPLiTrj_train.xyz:0", "MPLiTrj_train.xyz:1"]
    assert not mpli["capabilities"]["same_hop_exclusion_supported"]["available"]


# --------------------------------------------------------------------------- role semantics


def test_not_applicable_dimensions_are_never_missing():
    raw = FIXTURE_DATA / "raw"
    files = discover_datasets(raw)
    bvel = audit_dataset("BVEL13k", files["BVEL13k"], [])
    fpmd = audit_dataset("FPMD", files["FPMD"], [])

    assert bvel["identifiers"]["frame"]["status"] == "not_applicable"
    assert bvel["identifiers"]["frame"]["missing"] is None
    assert bvel["status"] == "pass"
    for dimension in ("hop", "neb_path", "frame", "published_split"):
        assert fpmd["identifiers"][dimension]["status"] == "not_applicable"
        assert fpmd["identifiers"][dimension]["missing"] is None
    assert not any("identifier" in blocker for blocker in fpmd["blockers"])


def test_fpmd_status_depends_only_on_transport_capabilities(tmp_path, monkeypatch):
    data = _data_tree(tmp_path)
    files = discover_datasets(data / "raw")["FPMD"]

    unavailable = audit_dataset("FPMD", files, [])
    monkeypatch.setattr(dataset_audit, "aiida_query_available", lambda: True)
    available = audit_dataset("FPMD", files, [])

    assert unavailable["status"] == "partial"
    assert unavailable["blockers"] == [
        "aiida_query_available: AiiDA query capability unavailable in current environment"
    ]
    caps = available["capabilities"]
    assert available["status"] == "pass"
    for name in (
        "archive_present",
        "archive_container_readable",
        "documentation_present",
        "trajectory_archive_present",
        "diffusion_reference_documented",
        "msd_reference_documented",
    ):
        assert caps[name]["available"], name
    aiida = next(s for s in available["schema_summary"] if s["format"] == "aiida")
    assert aiida["export_version"] == "main_0001"


def test_required_identifier_gap_makes_pool_partial(tmp_path):
    root = tmp_path / "nebBVSE122k"
    root.mkdir()
    (root / "index.csv").write_text("material_id,chemsys\nmp-1,Li-O\n", encoding="utf-8")

    report = audit_dataset("nebBVSE122k", [root / "index.csv"], [])

    assert report["status"] == "partial"
    assert report["blockers"] == [
        "required identifier 'hop' is unavailable (1 records missing, 0 alias conflicts)"
    ]


# --------------------------------------------------------------------------- compaction


def test_diagnostics_keep_exact_counts_with_bounded_examples(tmp_path):
    path = tmp_path / "MPLiTrj_train.xyz"
    frames = [Atoms("Li", positions=[[0.1 * i, 0, 0]], cell=[4, 4, 4], pbc=True) for i in range(25)]
    write(path, frames, format="extxyz")

    report = audit_dataset("MPLiTrj", [path], [_identity(path)], settings={"example_limit": 3})
    missing = report["identifiers"]["material"]["missing"]

    assert missing == {
        "count": 25,
        "example_limit": 3,
        "examples": ["MPLiTrj_train.xyz:0", "MPLiTrj_train.xyz:1", "MPLiTrj_train.xyz:2"],
        "provenance": "records without a value",
    }
    assert "value_examples" not in report["identifiers"]["frame"]
    assert len(json.dumps(report)) < 20_000


# --------------------------------------------------------------------------- OPTIMADE


def test_optimade_streaming_is_bounded_and_skips_header(tmp_path):
    path = tmp_path / "optimade.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"x-optimade": {"meta": {"api_version": "1.2.0"}}}) + "\n")
        handle.write(json.dumps({"type": "info", "id": "structures"}) + "\n")
        for index in range(50):
            handle.write(
                json.dumps(
                    {"type": "structures", "id": f"s-{index}", "attributes": {"chemical_formula_reduced": "LiO"}}
                )
                + "\n"
            )

    result = inspect_optimade_jsonl(path, line_limit=5, example_limit=2)

    assert result["header_present"]
    assert result["header_keys"] == ["meta"]
    assert result["lines_inspected"] == 5
    assert result["truncated"]
    assert result["record_types"] == {"info": 1, "structures": 3, "x-optimade header": 1}
    assert result["structure_ids"] == {
        "count": 3,
        "example_limit": 2,
        "examples": ["s-0", "s-1"],
        "provenance": "OPTIMADE entry id",
    }
    assert result["identifier_like_attribute_keys"] == ["chemical_formula_reduced"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_missing_dataset_fails(dataset):
    report = audit_dataset(dataset, [], [])

    assert report["status"] == "fail"


# --------------------------------------------------------------------------- review regressions


def test_non_finite_em_dft_is_not_a_barrier_reference(tmp_path):
    files = _neb_dataset(
        tmp_path,
        files={f"{EDGE}_init.xyz": f"{EDGE}_init.xyz", f"{EDGE}_relaxed.xyz": f"{EDGE}_relaxed.xyz"},
        rows=[f"mp-1,{EDGE},Li-O,False,inf,nan,train"],
    )

    report = audit_dataset("nebDFT2k", files, [_identity(p) for p in files])

    assert report["status"] == "partial"
    assert not report["capabilities"]["barrier_reference_available"]["available"]
    assert not report["capabilities"]["bvse_reference_available"]["available"]


def test_free_energy_alone_does_not_count_as_energy(tmp_path):
    atoms = Atoms("Li", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, free_energy=-1.0, forces=np.zeros((1, 3)))

    assert locate_quantities(atoms)["energy"] is None


def test_source_split_disagreeing_with_filename_is_a_conflict(tmp_path):
    path = tmp_path / "MPLiTrj_train.xyz"
    path.write_text(
        '1\nLattice="4 0 0 0 4 0 0 0 4" Properties=species:S:1:pos:R:3:forces:R:3 '
        'material_id=mp-1 split=test energy=-1.0 pbc="T T T"\nLi 0 0 0 0 0 0\n',
        encoding="utf-8",
    )

    report = audit_dataset("MPLiTrj", [path], [_identity(path)])
    split = report["identifiers"]["published_split"]

    assert split["alias_conflicts"]["examples"] == ["MPLiTrj_train.xyz:0"]
    assert split["status"] == "incomplete"
    assert not report["capabilities"]["published_split_supported"]["available"]


def test_full_identifier_sets_stay_out_of_reports(tmp_path):
    frames = read(FIXTURE_DATA / "raw" / "MPLiTrj" / "MPLiTrj_train.xyz", index=":")
    edges = {f"mp-1_0_{i}_1_0_0": frames for i in range(30)}
    data = _data_tree(tmp_path, archive_frames=edges)

    report, path = run_audit(_config(tmp_path, data, example_limit=3))
    text = path.read_text(encoding="utf-8")
    sidecar = json.loads((path.parent / "linkage" / "MPLiTrj.json").read_text(encoding="utf-8"))
    checkpoint = (path.parent / "datasets" / "MPLiTrj.json").read_text(encoding="utf-8")

    assert len(sidecar["values"]["raw_archive_hop"]) == 30
    assert "mp-1_0_29_1_0_0" not in text
    assert "mp-1_0_29_1_0_0" not in checkpoint
    mpli = next(item for item in report["datasets"] if item["dataset"] == "MPLiTrj")
    assert mpli["linkage_values"]["counts"]["raw_archive_hop"] == 30
    resumed, _ = run_audit(_config(tmp_path, data, example_limit=3), resume=True)
    assert resumed["linkage_summary"] == report["linkage_summary"]
