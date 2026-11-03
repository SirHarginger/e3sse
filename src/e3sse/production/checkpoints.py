"""Create and consume a verified operational inventory of frozen checkpoints.

Checkpoint paths are host-local and are not scientific identity.  The frozen model
registry and pinned smoke reports establish byte identity; this module discovers a
usable copy, re-hashes it (including MatGL side artifacts), and publishes a
create-once inventory consumed by every production path.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "e3sse.production.checkpoint-inventory"


class CheckpointInventoryError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path, model: Mapping[str, Any]) -> bool:
    from e3sse.protocol.adapters import checkpoint_digest

    try:
        if checkpoint_digest(path) != model["weights_sha256"]:
            return False
        return all((path / name).is_file() and _sha256(path / name) == digest
                   for name, digest in model.get("artifact_files", {}).items())
    except (OSError, KeyError):
        return False


def _verify_smoke(name: str, model: Mapping[str, Any], pin: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(pin["path"])
    if not path.is_file() or _sha256(path) != pin["sha256"]:
        raise CheckpointInventoryError(f"{name}: pinned model-smoke report is absent or changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = report.get("fingerprint") or {}
    frozen = fingerprint.get("model") or {}
    checks = {
        "canonical_name": frozen.get("canonical_name") == name,
        "checkpoint_id": frozen.get("checkpoint_id") == model.get("checkpoint_id"),
        "weights_sha256": fingerprint.get("checkpoint_sha256") == model.get("weights_sha256"),
        "adapter_version": fingerprint.get("adapter_version") == model.get("adapter_version"),
        "package_version": fingerprint.get("package_version") == model.get("package_version"),
        "clean_tree": fingerprint.get("git_dirty") is False,
        "passed": (report.get("result") or {}).get("passed") is True,
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise CheckpointInventoryError(f"{name}: model-smoke binding failed {failed}")
    return {"path": str(path), "sha256": pin["sha256"],
            "fingerprint_sha256": report.get("fingerprint_sha256")}


def _candidates(model: Mapping[str, Any], roots: Iterable[Path]) -> list[Path]:
    checkpoint_id = Path(model["checkpoint_id"])
    basename = checkpoint_id.name
    found: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        for direct in (root / checkpoint_id, root / basename):
            if direct.exists():
                found.add(direct.resolve())
        if not root.is_dir():
            continue
        for directory, dirs, files in os.walk(root):
            base = Path(directory)
            if basename in dirs:
                found.add((base / basename).resolve())
            if basename in files:
                found.add((base / basename).resolve())
    return sorted(found, key=str)


def build_inventory(*, models: Mapping[str, Mapping[str, Any]],
                    smoke_pins: Mapping[str, Mapping[str, Any]],
                    search_roots: Iterable[Path]) -> dict[str, Any]:
    if set(smoke_pins) != set(models):
        raise CheckpointInventoryError("model-smoke report set differs from frozen model set")
    roots = [Path(root).expanduser().resolve() for root in search_roots]
    records = {}
    for name, model in sorted(models.items()):
        smoke = _verify_smoke(name, model, smoke_pins[name])
        valid = [path for path in _candidates(model, roots) if _verify_checkpoint(path, model)]
        if not valid:
            raise CheckpointInventoryError(f"{name}: no SHA-verified checkpoint below search roots")
        records[name] = {
            "path": str(valid[0]), "checkpoint_id": model["checkpoint_id"],
            "weights_sha256": model["weights_sha256"],
            "artifact_files": dict(sorted(model.get("artifact_files", {}).items())),
            "verified_copies": [str(path) for path in valid], "model_smoke": smoke,
        }
    return {"schema_version": SCHEMA, "models": records,
            "search_roots": [str(path) for path in roots]}


def load_inventory(path: Path, models: Mapping[str, Mapping[str, Any]]) -> dict[str, Path]:
    try:
        inventory = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointInventoryError(f"checkpoint inventory {path} is absent or invalid") from exc
    if inventory.get("schema_version") != SCHEMA or set(inventory.get("models", {})) != set(models):
        raise CheckpointInventoryError("checkpoint inventory schema or model set differs")
    resolved = {}
    for name, model in sorted(models.items()):
        record = inventory["models"][name]
        if (record.get("checkpoint_id") != model.get("checkpoint_id")
                or record.get("weights_sha256") != model.get("weights_sha256")
                or record.get("artifact_files", {}) != model.get("artifact_files", {})):
            raise CheckpointInventoryError(f"{name}: inventory identity differs from registry")
        checkpoint = Path(record["path"])
        if not _verify_checkpoint(checkpoint, model):
            raise CheckpointInventoryError(f"{name}: inventoried checkpoint changed")
        resolved[name] = checkpoint
    return resolved
