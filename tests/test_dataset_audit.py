import hashlib
import json
from pathlib import Path

from e3sse.data.dataset_audit import (
    DATASETS,
    audit_dataset,
    concise_summary,
    discover_datasets,
    run_audit,
)


FIXTURE_DATA = Path(__file__).parent / "fixtures" / "g0" / "data"


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "project": {"name": "e3sse-test"},
                "paths": {
                    "data_root": str(FIXTURE_DATA),
                    "outputs_root": str(tmp_path / "outputs"),
                    "logs_root": str(tmp_path / "logs"),
                    "scratch_root": str(tmp_path / "scratch"),
                },
                "runtime": {"seed": 7},
                "audit": {"hash_max_bytes": 1_000_000},
            }
        ),
        encoding="utf-8",
    )
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_discovery_and_schema_preserve_leakage_identifiers(tmp_path):
    report, report_path = run_audit(_config(tmp_path))
    by_name = {item["dataset"]: item for item in report["datasets"]}

    assert set(discover_datasets(FIXTURE_DATA / "raw")) == set(DATASETS)
    assert by_name["nebDFT2k"]["record_counts"]["hops"] == 1
    assert by_name["MPLiTrj"]["record_counts"]["frames"] == 1
    assert by_name["MPLiTrj"]["split_summary"]["counts"] == {"train": 1}
    assert by_name["MPLiTrj"]["linkage_summary"]["material"]["distinct_values"] == ["mat-1"]
    assert by_name["MPLiTrj"]["linkage_summary"]["hop"]["distinct_values"] == ["hop-1"]
    assert by_name["MPLiTrj"]["linkage_summary"]["neb_path"]["distinct_values"] == ["path-1"]
    assert by_name["MPLiTrj"]["units"]["energy"]["values"] == ["eV"]
    assert by_name["MPLiTrj"]["units"]["unclassified"] == {"length_unit": ["Angstrom"]}
    assert by_name["nebDFT2k"]["dataset_summary"]["indexed_hops"] == 1
    assert by_name["nebDFT2k"]["dataset_summary"]["complete_energy_hops"] == 1
    assert report["linkage_summary"]["nebDFT2k_to_MPLiTrj"]["material"][
        "intersection_count"
    ] == 1
    assert report["implementation"]["dataset_audit"]
    assert report_path.is_relative_to(tmp_path / "outputs")
    assert "Gate G0:" in concise_summary(report, report_path)


def test_audit_does_not_modify_raw_data_and_resume_reuses_report(tmp_path):
    raw_files = sorted(path for path in FIXTURE_DATA.rglob("*") if path.is_file())
    before = {path: (_digest(path), path.stat().st_mode) for path in raw_files}
    config = _config(tmp_path)

    first, path = run_audit(config)
    first_mtime = path.stat().st_mtime_ns
    second, second_path = run_audit(config, resume=True)

    assert second == first
    assert second_path == path
    assert path.stat().st_mtime_ns == first_mtime
    assert before == {path: (_digest(path), path.stat().st_mode) for path in raw_files}


def test_malformed_extxyz_is_reported_without_fabricating_schema():
    path = Path(__file__).parent / "fixtures" / "malformed.xyz"
    identity = {"path": str(path), "identifier": path.name, "sha256": _digest(path)}

    report = audit_dataset("MPLiTrj", [path], [identity])

    assert report["status"] == "error"
    assert report["errors"]
    assert report["record_counts"]["frames"] == 0
    assert report["units"]["energy"]["values"] == []


def test_missing_source_frame_id_remains_unresolved(tmp_path):
    path = tmp_path / "no-frame-id.extxyz"
    path.write_text(
        "1\nProperties=species:S:1:pos:R:3 material_id=mat-1 split=train\nLi 0 0 0\n",
        encoding="utf-8",
    )
    identity = {"path": str(path), "identifier": path.name, "sha256": _digest(path)}

    report = audit_dataset("MPLiTrj", [path], [identity])

    assert report["linkage_summary"]["frame"]["distinct_count"] == 0
    assert report["linkage_summary"]["frame"]["missing_records"] == 1
    assert not report["validation_checks"]["leakage_identifier_dimensions_complete"]


def test_discovery_prefers_full_mplitrj_over_overlapping_subsample(tmp_path):
    raw = tmp_path / "raw"
    full = raw / "MPLiTrj" / "full.xyz"
    sample = raw / "MPLiTrj_subsample" / "sample.xyz"
    full.parent.mkdir(parents=True)
    sample.parent.mkdir(parents=True)
    full.write_text("", encoding="utf-8")
    sample.write_text("", encoding="utf-8")

    files = discover_datasets(raw)["MPLiTrj"]

    assert files == [full]


def test_resume_does_not_reuse_errored_final_report(tmp_path):
    config = _config(tmp_path)
    first, path = run_audit(config)
    poisoned = dict(first)
    poisoned["errors"] = ["transient parser failure"]
    path.write_text(json.dumps(poisoned), encoding="utf-8")

    resumed, resumed_path = run_audit(config, resume=True)

    assert resumed_path == path
    assert "transient parser failure" not in resumed["errors"]


def test_audit_schema_summary_is_deterministic():
    files = discover_datasets(FIXTURE_DATA / "raw")["MPLiTrj"]
    identities = [
        {"path": str(path), "identifier": path.name, "sha256": _digest(path)}
        for path in files
    ]

    first = audit_dataset("MPLiTrj", files, identities)
    second = audit_dataset("MPLiTrj", files, identities)

    for key in ("record_counts", "schema_summary", "units", "split_summary", "linkage_summary"):
        assert first[key] == second[key]
