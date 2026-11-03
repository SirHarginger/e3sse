"""The Gate-1 identifier/covariate tables, read only through the executed linkage audit.

``scripts/audit_rate_limiting_hop_linkage.py`` owns the readers: exactly one CSV per
dataset with the required columns, its G0 identity (path and SHA-256) verified, the
column allowlist (``REQUIRED``) and the outcome-column denylist.  This module loads
that script's functions instead of copying them, so production and the audit can
never read different columns.  Rows shorter than the header stop the read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def linkage_module() -> Any:
    path = REPOSITORY / "scripts" / "audit_rate_limiting_hop_linkage.py"
    spec = importlib.util.spec_from_file_location("audit_rate_limiting_hop_linkage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def pinned_g0_inputs(freeze_config: Mapping[str, Any]) -> dict[str, Any]:
    """The G0 input manifest, verified against its pinned SHA-256."""

    manifest = freeze_config["inputs"]["g0_input_manifest"]
    raw = Path(manifest["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("G0 input manifest SHA-256 differs from the pinned value")
    return json.loads(raw).get("inputs", {})


def _g0_identity(path: Path, g0_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [item for items in g0_inputs.values() if isinstance(items, list)
               for item in items if isinstance(item, Mapping)
               and Path(item.get("path", "")).resolve() == Path(path).resolve()]
    if len(matches) != 1:
        raise linkage_module().AuditError(f"{path}: not exactly one identity in the G0 manifest")
    return matches[0]


def verify_g0_file(path: Path, g0_inputs: Mapping[str, Any]) -> str:
    """SHA-256 of a source file whose G0 identity it matches; G0 recorded no SHA-256
    above its hash limit, so such files must match the recorded size and mtime."""

    linkage = linkage_module()
    identity = _g0_identity(path, g0_inputs)
    digest = linkage._sha256(Path(path))
    if identity.get("sha256") is not None:
        if identity["sha256"] != digest:
            raise linkage.AuditError(f"{path}: SHA-256 differs from the G0 input manifest")
    else:
        stat = Path(path).stat()
        if (identity.get("bytes"), identity.get("mtime_ns")) != (stat.st_size, stat.st_mtime_ns):
            raise linkage.AuditError(f"{path}: size or mtime differs from the G0 input manifest "
                                     "(G0 recorded no SHA-256 for this file)")
    return digest


def _refuse_ragged_rows(path: Path) -> None:
    """Rows longer or shorter than the header shift positional columns: refuse them."""

    import csv

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        width = len(next(reader, []))
        bad = sum(1 for row in reader if len(row) != width)
    if bad:
        raise linkage_module().AuditError(f"{path}: {bad} row(s) differ in width from the header")


def linkage_tables(raw_root: Path, g0_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """``{dataset: rows}`` plus ``"_provenance"``; refuses short rows and outcome columns."""

    from e3sse.data.dataset_audit import discover_datasets

    linkage = linkage_module()
    inventory = discover_datasets(raw_root)
    tables: dict[str, Any] = {"_provenance": {}}
    for dataset, columns in linkage.REQUIRED.items():
        source = linkage.select_index(dataset, inventory.get(dataset, []))
        digest = verify_g0_file(source, g0_inputs)
        _refuse_ragged_rows(source)
        rows, short = linkage.read_columns(source, columns)
        if short:
            raise linkage.AuditError(f"{dataset}: {short} row(s) shorter than the header")
        tables[dataset] = linkage.adapt_nebbvse_rows(rows) if dataset == "nebBVSE122k" else rows
        tables["_provenance"][dataset] = {"path": str(source), "sha256": digest,
                                          "rows": len(rows), "columns_read": list(columns)}
    return tables
