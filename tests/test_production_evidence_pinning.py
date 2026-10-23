"""Immutable production evidence is pinned, re-hashed and bound to the frozen models."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_protocol_configuration import _production_config
from test_protocol_safeguards import _clean_repository, _production_fixture

from e3sse.protocol.configuration import assess_production_configuration
from e3sse.protocol.freeze import ProtocolFreezeError, materialize_protocol
from e3sse.protocol.registration import registration_status


def _blockers(config: dict) -> str:
    return "\n".join(assess_production_configuration(config).freeze_blockers)


def test_complete_evidence_clears_and_any_gap_blocks():
    config = _production_config()
    assert _blockers(config) == ""
    missing = copy.deepcopy(config)
    del missing["inputs"]["production_evidence"]["model_smoke_reports"]["model-B2-0"]
    assert "missing ['model-B2-0']" in _blockers(missing)
    unpinned = copy.deepcopy(config)
    unpinned["inputs"]["production_evidence"]["unit_evidence"]["sha256"] = None
    assert "unit_evidence.sha256" in _blockers(unpinned)
    wrong_version = copy.deepcopy(config)
    for environment in wrong_version["inputs"]["production_evidence"][
        "runtime_environments"
    ].values():
        environment["pinned_packages"]["test-package"] = "2.0"
    assert "registry requires 1.0" in _blockers(wrong_version)
    lock = copy.deepcopy(config)
    for environment in lock["inputs"]["production_evidence"]["runtime_environments"].values():
        environment["environment_lock"]["sha256"] = "not-a-digest"
    assert "environment_lock.sha256" in _blockers(lock)
    duplicated = copy.deepcopy(config)
    environments = duplicated["inputs"]["production_evidence"]["runtime_environments"]
    (only,) = environments.values()
    environments["second"] = copy.deepcopy(only)
    assert "exactly one production runtime environment" in _blockers(duplicated)


def test_checked_in_registry_evidence_binds_every_frozen_model():
    config = json.loads(Path("configs/protocol_freeze.json").read_text(encoding="utf-8"))
    evidence = config["inputs"]["production_evidence"]
    reports = evidence["model_smoke_reports"]
    assert len(reports) == 6 and len(evidence["runtime_environments"]) == 5
    assert reports["SevenNet-omat"]["runtime_environment"] == "e3sse-sevennet-0.11.0"
    assert reports["SevenNet-0-11July2024"]["runtime_environment"] == "e3sse-sevennet-0.10.1"
    matgl = evidence["runtime_environments"]["e3sse-matgl-1.2.7"]
    assert matgl["environment_variables"] == {"DGLBACKEND": "pytorch"}
    assert matgl["pinned_packages"]["torch"] == "2.1.1"
    assert evidence["unit_evidence"]["sha256"] == (
        "4b7c4b9c558e5e749cc0065062d9acc63aeccde41a4fca0d451856aa28fa330d"
    )
    # Execution settings are never protocol inputs.
    assert "workers" not in json.dumps(config)


def test_frozen_protocol_records_evidence_and_stays_registration_pending(tmp_path, monkeypatch):
    _clean_repository(monkeypatch)
    protocol, _path = materialize_protocol(_production_fixture(tmp_path, complete=True))
    assert protocol["freeze_status"] == "frozen"
    assert set(protocol["production_evidence"]["model_smoke_reports"]) == {
        "model-A-0", "model-A-1", "model-A-2", "model-A-3", "model-B1-0", "model-B2-0"
    }
    assert registration_status(protocol, None) == "protocol_frozen_registration_pending"
    assert "worker" not in json.dumps(protocol).lower()


def test_tampered_or_mismatched_smoke_report_stops_materialization(tmp_path, monkeypatch):
    _clean_repository(monkeypatch)
    config_path = _production_fixture(tmp_path, complete=True)
    config = json.loads(config_path.read_text())
    record = config["inputs"]["production_evidence"]["model_smoke_reports"]["model-A-0"]
    report_path = Path(record["path"])
    report = json.loads(report_path.read_text())
    report["result"]["passed"] = False
    report_path.write_text(json.dumps(report))
    with pytest.raises(ProtocolFreezeError, match="smoke report model-A-0 SHA-256 mismatch"):
        materialize_protocol(config_path)
    import hashlib

    record["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    with pytest.raises(ProtocolFreezeError, match=r"does not bind its model: \['passed'\]"):
        materialize_protocol(config_path)


def test_smoke_report_must_have_run_in_its_pinned_environment(tmp_path, monkeypatch):
    import hashlib

    _clean_repository(monkeypatch)
    config_path = _production_fixture(tmp_path, complete=True)
    config = json.loads(config_path.read_text())
    record = config["inputs"]["production_evidence"]["model_smoke_reports"]["model-A-0"]
    report_path = Path(record["path"])
    report = json.loads(report_path.read_text())
    report["python"] = "3.12.1 (main)"
    report["fingerprint"]["dependency_versions"] = {"numpy": "9.9"}
    report_path.write_text(json.dumps(report))
    record["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    with pytest.raises(
        ProtocolFreezeError, match=r"\['dependency_versions', 'python_version'\]"
    ):
        materialize_protocol(config_path)


def _rewrite_unit_evidence(config_path: Path, mutate) -> None:
    import hashlib

    config = json.loads(config_path.read_text())
    record = config["inputs"]["production_evidence"]["unit_evidence"]
    path = Path(record["path"])
    report = json.loads(path.read_text())
    mutate(report)
    path.write_text(json.dumps(report))
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))


def test_unit_evidence_must_come_from_the_pinned_files(tmp_path, monkeypatch):
    _clean_repository(monkeypatch)
    config_path = _production_fixture(tmp_path, complete=True)
    _rewrite_unit_evidence(
        config_path, lambda r: r["mplitrj"]["train"].update({"sha256": "0" * 64})
    )
    with pytest.raises(ProtocolFreezeError, match="pinned MPLiTrj files"):
        materialize_protocol(config_path)


def test_a_convergence_flag_column_contradicts_the_registered_resolution(tmp_path, monkeypatch):
    _clean_repository(monkeypatch)
    config_path = _production_fixture(tmp_path, complete=True)
    _rewrite_unit_evidence(
        config_path, lambda r: r["nebdft2k"]["index_columns"].append("neb_converged")
    )
    with pytest.raises(ProtocolFreezeError, match="contradict the flag resolution"):
        materialize_protocol(config_path)


def test_unverifiable_runtime_pins_must_be_declared_lock_only():
    config = _production_config()
    environments = config["inputs"]["production_evidence"]["runtime_environments"]
    next(iter(environments.values()))["pinned_packages"]["torchdata"] = "0.7.1"
    assert "lock_recorded_pins" in _blockers(config)


@pytest.mark.parametrize(
    ("field", "digest_key", "label"),
    [
        ("g0_audit", "sha256", "G0 audit"),
        ("g0_input_manifest", "sha256", "G0 input manifest"),
        ("mplitrj_provenance", "manifest_sha256", "provenance manifest"),
    ],
)
def test_server_trust_anchors_fail_closed(tmp_path, monkeypatch, field, digest_key, label):
    _clean_repository(monkeypatch)
    config_path = _production_fixture(tmp_path, complete=True)
    config = json.loads(config_path.read_text())
    spec = config["inputs"][field]
    assert len(spec[digest_key]) == 64
    wrong = copy.deepcopy(config)
    wrong["inputs"][field][digest_key] = "0" * 64
    config_path.write_text(json.dumps(wrong))
    with pytest.raises(ProtocolFreezeError, match=f"{label} SHA-256 mismatch"):
        materialize_protocol(config_path)
    target = Path(spec["root"]) / "manifest.json" if "root" in spec else Path(spec["path"])
    target.rename(target.with_name(target.name + ".moved"))
    config_path.write_text(json.dumps(config))
    with pytest.raises(ProtocolFreezeError, match=f"missing {label}"):
        materialize_protocol(config_path)
