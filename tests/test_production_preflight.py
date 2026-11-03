"""Preflight evidence checks: synthetic evidence only, never an outcome."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from e3sse.production import preflight

REPOSITORY = Path(__file__).resolve().parents[1]
ACTIVE = json.loads((REPOSITORY / "configs" /
                     "reliability_gated_internal_freeze_protocol.json").read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_git_identity_requires_clean_expected_commit():
    git = {"commit": "a" * 40, "tree": "b" * 40, "branch": "x", "dirty": True,
           "dirty_paths": ["uv.lock"]}
    assert preflight.check_git_identity(git, require_clean=True, expected_commit=None).blocking
    clean = {**git, "dirty": False}
    assert not preflight.check_git_identity(clean, require_clean=True, expected_commit=None).blocking
    wrong = preflight.check_git_identity(clean, require_clean=True, expected_commit="c" * 40)
    assert wrong.status == "failed"


def test_output_root_inside_data_or_on_tmpfs_is_refused(tmp_path, monkeypatch):
    from e3sse.softening import study1_inputs

    data = tmp_path / "data"
    data.mkdir()
    assert preflight.check_output_root(data / "out", [data]).status == "failed"
    monkeypatch.setattr(study1_inputs, "refuse_tmpfs", lambda path: None)
    assert preflight.check_output_root(tmp_path / "outputs", [data]).status == "passed"

    def on_tmpfs(path):
        raise study1_inputs.Study1InputError(f"output {path} is on tmpfs")

    monkeypatch.setattr(study1_inputs, "refuse_tmpfs", on_tmpfs)
    assert preflight.check_output_root(tmp_path / "outputs", [data]).status == "failed"


def test_pinned_inputs_detect_absent_and_changed_files(tmp_path):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text("{}")
    bad.write_text("{}")
    config = {"inputs": {"a": {"path": str(good), "sha256": _sha(good)},
                         "nested": {"b": {"path": str(bad), "sha256": "0" * 64}},
                         "c": {"path": "absent.json", "sha256": "1" * 64},
                         "unpinned": None}}
    checks = {c.check_id: c.status for c in
              preflight.check_freeze_input_pins(config, config_dir=tmp_path)}
    assert checks == {"pinned_input:a": "passed", "pinned_input:nested.b": "failed",
                      "pinned_input:c": "missing_evidence"}
    listed = {"inputs": {"locks": [{"path": str(good), "sha256": _sha(good)}]}}
    assert [c.check_id for c in preflight.check_freeze_input_pins(listed, config_dir=tmp_path)] \
        == ["pinned_input:locks[0]"]
    assert preflight.check_freeze_input_pins({"inputs": {}}, config_dir=tmp_path)[0].status \
        == "failed"


def test_dataset_hashes_check_size_and_rehash_when_requested(tmp_path):
    data, big = tmp_path / "x.csv", tmp_path / "archive.zip"
    data.write_text("material_id\n")
    big.write_text("large file, no G0 hash")
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({
        "inputs": {"BVEL13k": [{"path": str(data), "sha256": _sha(data),
                                "bytes": data.stat().st_size}]},
        "mplitrj_raw_archive": [{"path": str(big), "sha256": None,
                                 "bytes": big.stat().st_size}]}))
    passed = preflight.check_dataset_hashes(manifest, rehash=True)
    assert passed.status == "passed" and passed.evidence["files"] == 2
    assert passed.evidence["size_verified_without_sha256"] == 1
    data.write_text("material_iX\n")  # same size, different bytes
    assert preflight.check_dataset_hashes(manifest, rehash=False).status == "passed"
    assert preflight.check_dataset_hashes(manifest, rehash=True).status == "failed"
    big.write_text("truncated")
    assert preflight.check_dataset_hashes(manifest, rehash=False).evidence["size_changed"]
    manifest.write_text(json.dumps({"inputs": {}}))
    assert preflight.check_dataset_hashes(manifest, rehash=True).status == "failed"


def _linkage(**overrides):
    report = {"outcome_columns_read": False,
              "calibration_side": {"nebDFT2k_edges": 1681, "nebDFT2k_edges_in_nebBVSE122k": 1681,
                                   "by_dimension": {"1": {}, "2": {}, "3": {"eligible": 473}}},
              "edge_quality": {"ambiguous_materials": 0, "chemsys_inconsistent_materials": 0},
              "pool": {"by_dimension": {"3": {}}},
              "inputs": {name: {"sha256": name, "short_rows_skipped": 0}
                         for name in ("BVEL13k", "nebBVSE122k", "nebDFT2k")}}
    report.update(overrides)
    return report


def test_gate1_linkage_invariants(tmp_path):
    path = tmp_path / "audit.json"
    assert preflight.check_gate1_linkage(path).status == "missing_evidence"
    path.write_text(json.dumps(_linkage()))
    assert preflight.check_gate1_linkage(path).status == "passed"
    skipped = _linkage()
    skipped["inputs"]["nebBVSE122k"]["short_rows_skipped"] = 3
    path.write_text(json.dumps(skipped))
    assert "skipped" in preflight.check_gate1_linkage(path).detail
    absent = _linkage()
    del absent["calibration_side"]["nebDFT2k_edges"]
    del absent["calibration_side"]["nebDFT2k_edges_in_nebBVSE122k"]
    path.write_text(json.dumps(absent))
    assert preflight.check_gate1_linkage(path).status == "failed"
    path.write_text(json.dumps(_linkage(outcome_columns_read=True)))
    assert preflight.check_gate1_linkage(path).status == "failed"
    broken = _linkage()
    broken["calibration_side"]["nebDFT2k_edges_in_nebBVSE122k"] = 1680
    path.write_text(json.dumps(broken))
    assert "not every" in preflight.check_gate1_linkage(path).detail


@pytest.mark.parametrize(("count", "status"), [(0, "failed"), (5, "failed"), (6, "passed")])
def test_gate1_population_feasibility_is_checked_before_outcomes(tmp_path, count, status):
    path = tmp_path / "populations.json"
    path.write_text(json.dumps({
        "outcome_columns_read": False, "dimension": 3,
        "counts": {"development_structural_upper_bound_systems": count},
        "inputs": {},
    }))
    check = preflight.check_gate1_structural_populations(path, tmp_path / "absent.json")
    assert check.status == status
    if status == "failed":
        assert check.evidence == {
            "path": str(path), "development_structural_upper_bound_systems": count,
            "support_k": 5, "required_minimum": 6,
            "study5_resolution": "defer or owner-approved pre-outcome redesign"}


def _report(name: str, *, permutation: bool = True) -> dict:
    checks = {"atom_level": True, "finite": True, "deterministic": True,
              "rotation_invariant": True, "translation_invariant": True, "aba_consistent": True,
              "environment_sensitive": True, "permutation_consistent": permutation}
    return {"schema": "e3sse.embedding-capability", "canonical_name": name,
            "status": "pinned_module", "package_version_matches_registry": True,
            "structure_checks": [checks]}


def test_embedding_pins_are_rehashed_and_must_agree_with_content(tmp_path):
    active = copy.deepcopy(ACTIVE)
    pins = active["amendments"]["representation"]["embedding_capability_evidence"]["reports"]
    for name, pin in pins.items():
        path = tmp_path / f"{name}.json"
        rejected = pin["latent_support_status"] != "validated"
        path.write_text(json.dumps(_report(name, permutation=not rejected)))
        pin.update(path=str(path), sha256=_sha(path))
    checks = preflight.check_embedding_evidence(active, pins)
    assert [c.status for c in checks] == ["passed"] * 6
    # A byte change, or a pin that disagrees with the report, blocks the freeze.
    first = min(pins)
    Path(pins[first]["path"]).write_text(json.dumps(_report(first, permutation=False)))
    statuses = {c.check_id: c.status for c in preflight.check_embedding_evidence(active, pins)}
    assert statuses[f"embedding_evidence:{first}"] == "failed"
    pins[first]["sha256"] = _sha(Path(pins[first]["path"]))
    statuses = {c.check_id: c.detail for c in preflight.check_embedding_evidence(active, pins)}
    assert "disagrees" in statuses[f"embedding_evidence:{first}"]


def _audit(directory: Path, environment: str, ase: str, code, settings, barrier=0.76, passed=True):
    directory.mkdir(exist_ok=True)
    (directory / f"{environment}-0123456789abcdef.json").write_text(json.dumps({
        "identity": {"ase_version": ase, "code_sha256": code}, "settings_sha256": settings,
        "passed": passed, "checks": {"reference_neb": {"barrier_eV": barrier}}}))


def test_ml_neb_runtime_audits_require_one_ase_version(tmp_path):
    code, settings = {"f": "1"}, "s"
    _audit(tmp_path, "env-a", "3.22.1", code, settings)
    _audit(tmp_path, "env-b", "3.24.0", code, settings, barrier=0.7600001)
    checks = {c.check_id: c for c in preflight.check_ml_neb_runtime_audits(
        tmp_path, ["env-a", "env-b", "env-c"], expected_code_sha256=code,
        expected_settings_sha256=settings)}
    assert checks["ml_neb_runtime:env-a"].status == "passed"
    assert checks["ml_neb_runtime:env-c"].status == "missing_evidence"
    equivalence = checks["ml_neb_cross_runtime_equivalence"]
    assert equivalence.status == "failed" and equivalence.blocks == "freeze"  # 1e-7 eV apart
    assert equivalence.evidence["ase_versions"] == {"env-a": "3.22.1", "env-b": "3.24.0"}
    stale = preflight.check_ml_neb_runtime_audits(tmp_path, ["env-a"],
                                                  expected_code_sha256={"f": "2"},
                                                  expected_settings_sha256=settings)
    assert stale[0].status == "failed" and "stale" in stale[0].detail
    # A newer audit with the current identity is chosen over the stale one.
    (tmp_path / "env-a-fedcba9876543210.json").write_text(json.dumps({
        "identity": {"ase_version": "3.22.1", "code_sha256": {"f": "2"}},
        "settings_sha256": settings, "passed": True,
        "checks": {"reference_neb": {"barrier_eV": 0.76}}}))
    fresh = preflight.check_ml_neb_runtime_audits(tmp_path, ["env-a"],
                                                  expected_code_sha256={"f": "2"},
                                                  expected_settings_sha256=settings)
    assert fresh[0].status == "passed"


def test_protocol_open_items_block_the_freeze_until_resolved():
    # Evidence-verified resolutions block until this run's named checks pass.
    checks = preflight.check_protocol_open_items(ACTIVE)
    ids = {c.check_id for c in checks}
    assert "protocol_open_item:bvse_reproduction" in ids
    assert "protocol_open_item:embedding_probe_server_run" not in ids
    assert all(c.blocks == "freeze" and c.blocking for c in checks)
    required = {name for item in ACTIVE["open_items"]
                for name in (item.get("resolution") or {}).get("verified_by_preflight_checks", [])}
    evidence = [preflight.PreflightCheck(name, "passed", "freeze", "ok") for name in required]
    assert [c.status for c in preflight.check_protocol_open_items(ACTIVE, evidence)] == ["passed"]
    failed = [preflight.PreflightCheck(n, "failed" if n == "bvse_ions_pin_confirmed" else "passed",
                                       "freeze", "x") for n in required]
    ids = {c.check_id for c in preflight.check_protocol_open_items(ACTIVE, failed)}
    assert ids == {"protocol_open_item:bvse_reproduction",
                   "protocol_open_item:ions_bvse_version_pin"}
    blocking = copy.deepcopy(ACTIVE)
    blocking["open_items"][0]["blocking"] = True
    assert any(c.status == "missing_evidence"
               for c in preflight.check_protocol_open_items(blocking, evidence))


def test_development_manifest_pin_and_bvse_pin_checks(tmp_path):
    manifest = tmp_path / "development_manifest.json"
    pinned = ACTIVE["amendments"]["prospective"]["development_manifest_sha256"]
    manifest.write_text(json.dumps({"manifest_sha256": pinned, "hop_ids": ["a"]}))
    outputs = {"development_manifest": {"path": str(manifest), "sha256": "x"}}
    assert preflight.check_development_manifest_pin(ACTIVE, outputs).status == "passed"
    manifest.write_text(json.dumps({"manifest_sha256": "0" * 64}))
    assert preflight.check_development_manifest_pin(ACTIVE, outputs).status == "failed"
    assert preflight.check_development_manifest_pin(ACTIVE, None).status == "missing_evidence"
    report = tmp_path / "bvse.json"
    bvse = {"bvse_reproduction_report": {"path": str(report), "sha256": "x"}}
    for rate, status in ((0.9, "passed"), (0.5, "failed"), (None, "failed")):
        report.write_text(json.dumps({"calibration_reproduction_rate": rate}))
        assert preflight.check_bvse_pin(bvse).status == status
    assert preflight.check_run_stage_evidence("bvse_reproduction", None).status == "missing_evidence"


def test_scope_keeps_transport_deferred():
    assert preflight.check_scope(ACTIVE).status == "passed"
    revived = copy.deepcopy(ACTIVE)
    revived["amendments"]["scope"]["study_3"] = "active"
    assert preflight.check_scope(revived).status == "failed"


def test_outcome_access_declaration_detects_prior_study1_records(tmp_path):
    study1 = tmp_path / "study1.json"
    root = tmp_path / "study1_outputs"
    study1.write_text(json.dumps({"output_root": str(root)}))
    assert preflight.check_outcome_access_declaration(study1, freeze_bound=False).status == "passed"
    record = root / "abc" / "inference" / "CHGNet" / "study1_runs" / "x.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}")
    assert preflight.check_outcome_access_declaration(study1, freeze_bound=False).status == "failed"
    assert preflight.check_outcome_access_declaration(study1, freeze_bound=True).status == "passed"


def test_summary_separates_freeze_and_compute_blockers():
    summary = preflight.summarize([
        preflight.PreflightCheck("a", "passed", "freeze", "ok"),
        preflight.PreflightCheck("b", "missing_evidence", "production_compute", "profile"),
    ])
    assert summary["freeze_permitted"] is True
    assert summary["production_compute_permitted"] is False
    assert summary["blocking"] == ["b"]


def test_embedding_report_must_match_frozen_weights_and_capture(tmp_path):
    active = copy.deepcopy(ACTIVE)
    pins = active["amendments"]["representation"]["embedding_capability_evidence"]["reports"]
    name = "CHGNet-0.3.0"
    report = {**_report(name), "checkpoint_sha256": "f" * 64, "capture_mode": "documented_api"}
    path = tmp_path / "chgnet.json"
    path.write_text(json.dumps(report))
    pins[name].update(path=str(path), sha256=_sha(path))
    models = {name: {"weights_sha256": "a" * 64}}
    detail = preflight.check_embedding_evidence(active, models)[0].detail
    assert "frozen weights" in detail
    models[name]["weights_sha256"] = "f" * 64
    assert preflight.check_embedding_evidence(active, models)[0].status == "passed"


def test_freeze_waits_for_outcome_runners_unless_deviation_declared():
    statuses = {"estimate": "implemented", "transfer": "missing_runner"}
    blocked = preflight.check_outcome_runners_implemented(statuses, required=True)
    assert blocked.blocking and blocked.blocks == "freeze"
    assert blocked.evidence["not_yet_implemented"] == ["transfer"]
    declared = preflight.check_outcome_runners_implemented(statuses, required=False)
    assert not declared.blocking and "declared deviation" in declared.detail


def test_ambiguous_resource_profiles_are_reported_not_guessed():
    checks = preflight.check_resource_profiles({}, [("m", "ml_neb_band")],
                                               ambiguous=["m/ml_neb_band"])
    assert [c.status for c in checks] == ["failed"] and "several profiles" in checks[0].detail
