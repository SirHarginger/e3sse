"""The model smoke script pins package version and Git state and rejects stale reports."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path

import pytest

from e3sse.protocol.adapters import ADAPTER_VERSION, checkpoint_digest


def _smoke_module():
    spec = importlib.util.spec_from_file_location(
        "smoke_test_models", Path("scripts/smoke_test_models.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(tmp_path, monkeypatch, *, package_version=None, git_status=""):
    from ase.calculators.lj import LennardJones

    module = _smoke_module()
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"synthetic weights")
    model = {
        "canonical_name": "synthetic-model",
        "adapter_name": "synthetic",
        "adapter_version": ADAPTER_VERSION,
        "weights_sha256": checkpoint_digest(checkpoint),
        # Any installed distribution stands in for the frozen model package.
        "package_name": "numpy",
        "package_version": package_version or importlib.metadata.version("numpy"),
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"protocol": {"model_groups": {"A": [model]}}}))
    outputs = {("rev-parse", "HEAD"): "a" * 40 + "\n", ("status", "--porcelain"): git_status}
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda cmd, text: outputs[tuple(cmd[3:])]
    )
    loaded = []

    def fake_build(entry, path):
        loaded.append(path)
        return LennardJones(sigma=2.0, epsilon=0.1, rc=6.0, smooth=True)

    monkeypatch.setattr(module, "build_calculator", fake_build)
    argv = [
        "--config", str(config), "--model", "synthetic-model",
        "--checkpoint", str(checkpoint), "--output", str(tmp_path / "smoke"),
    ]
    return module, argv, checkpoint, loaded


def test_report_is_written_once_and_resume_rejects_stale_fingerprints(tmp_path, monkeypatch):
    module, argv, checkpoint, loaded = _setup(tmp_path, monkeypatch)
    assert module.main(argv) == 0
    report_path = tmp_path / "smoke" / "synthetic-model.json"
    report = json.loads(report_path.read_text())
    fingerprint = report["fingerprint"]
    assert fingerprint["git_sha"] == "a" * 40 and fingerprint["git_dirty"] is False
    assert fingerprint["adapter_version"] == ADAPTER_VERSION
    assert fingerprint["checkpoint_sha256"] == checkpoint_digest(checkpoint)
    assert fingerprint["dependency_versions"]["numpy"] == importlib.metadata.version("numpy")
    assert report["result"]["passed"] is True
    assert [p.name for p in report_path.parent.iterdir()] == [report_path.name]

    with pytest.raises(SystemExit, match="use --resume"):
        module.main(argv)
    assert module.main([*argv, "--resume"]) == 0
    assert len(loaded) == 1  # resume verifies the existing report without reloading

    installed_version = module._installed_version
    monkeypatch.setattr(module, "_installed_version", lambda name: "0.0.0")
    with pytest.raises(SystemExit, match="stale"):
        module.main([*argv, "--resume"])
    monkeypatch.setattr(module, "_installed_version", installed_version)
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(SystemExit, match="stale"):
        module.main([*argv, "--resume"])


def test_package_version_mismatch_is_rejected_before_model_loading(tmp_path, monkeypatch):
    module, argv, _, loaded = _setup(tmp_path, monkeypatch, package_version="0.0.0")
    with pytest.raises(SystemExit, match="package version"):
        module.main(argv)
    assert loaded == []


def test_dirty_tree_is_rejected_unless_explicitly_allowed(tmp_path, monkeypatch):
    module, argv, _, _ = _setup(tmp_path, monkeypatch, git_status=" M file.py\n")
    with pytest.raises(SystemExit, match="clean reviewed commit"):
        module.main(argv)
    assert module.main([*argv, "--allow-dirty"]) == 0
    report = json.loads((tmp_path / "smoke" / "synthetic-model.json").read_text())
    assert report["fingerprint"]["git_dirty"] is True
