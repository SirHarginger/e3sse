"""Verified checkpoint discovery is reproducible and never trusts a hand path map."""

from __future__ import annotations

import hashlib
import json

import pytest

from e3sse.production.checkpoints import (
    CheckpointInventoryError,
    build_inventory,
    load_inventory,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_inventory_binds_smoke_registry_and_checkpoint_bytes(tmp_path):
    checkpoint = tmp_path / "cache" / "weights.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"frozen")
    model = {"canonical_name": "model", "checkpoint_id": "weights.bin",
             "weights_sha256": _sha(b"frozen"), "adapter_version": "adapter",
             "package_version": "1"}
    smoke = {"fingerprint": {"model": model, "checkpoint_sha256": _sha(b"frozen"),
                              "adapter_version": "adapter", "package_version": "1",
                              "git_dirty": False}, "fingerprint_sha256": "fingerprint",
             "result": {"passed": True}}
    report = tmp_path / "smoke.json"
    report.write_text(json.dumps(smoke))
    inventory = build_inventory(models={"model": model},
                                smoke_pins={"model": {"path": str(report),
                                                       "sha256": _sha(report.read_bytes())}},
                                search_roots=[checkpoint.parent])
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    assert load_inventory(path, {"model": model}) == {"model": checkpoint}
    checkpoint.write_bytes(b"changed")
    with pytest.raises(CheckpointInventoryError, match="changed"):
        load_inventory(path, {"model": model})


def test_inventory_refuses_unbound_smoke_or_missing_checkpoint(tmp_path):
    model = {"canonical_name": "model", "checkpoint_id": "absent.bin",
             "weights_sha256": _sha(b"frozen"), "adapter_version": "adapter",
             "package_version": "1"}
    report = tmp_path / "smoke.json"
    report.write_text(json.dumps({"fingerprint": {"model": model,
                                                   "checkpoint_sha256": "0" * 64,
                                                   "adapter_version": "adapter",
                                                   "package_version": "1",
                                                   "git_dirty": False},
                                  "result": {"passed": True}}))
    with pytest.raises(CheckpointInventoryError, match="binding failed"):
        build_inventory(models={"model": model},
                        smoke_pins={"model": {"path": str(report),
                                               "sha256": _sha(report.read_bytes())}},
                        search_roots=[tmp_path])
