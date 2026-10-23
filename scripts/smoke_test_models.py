#!/usr/bin/env python3
"""Run the non-outcome unit smoke test for one frozen model with its canonical adapter.

Each model runs in its own pinned environment.  The report is written atomically to
``<output>/<canonical_name>.json`` and binds the checkpoint SHA-256, adapter and
smoke-test versions, package and key dependency versions, and Git SHA.  ``--resume``
reuses an existing report only when that fingerprint is unchanged and rejects a stale
one.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from e3sse.protocol.adapters import (
    ADAPTER_VERSION,
    SMOKE_TEST_VERSION,
    build_calculator,
    canonical_json_sha256,
    checkpoint_digest,
    smoke_test,
)

# Transitive packages whose versions change model numerics (e.g. e3nn CG conventions).
RECORDED_DEPENDENCIES = ("ase", "dgl", "e3nn", "numpy", "torch", "torch_geometric")


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _model_entry(config: dict, name: str) -> dict:
    for group in config["protocol"]["model_groups"].values():
        for model in group:
            if model.get("canonical_name") == name:
                return model
    raise SystemExit(f"model {name!r} is not in the frozen registry")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="canonical_name in the registry")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="local development only; recorded in report"
    )
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = _model_entry(config, args.model)
    repo = Path(__file__).resolve().parents[1]
    git_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True)
    dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True))
    if dirty and not args.allow_dirty:
        raise SystemExit("smoke reports for registration require a clean reviewed commit")
    fingerprint = {
        "checkpoint_sha256": checkpoint_digest(args.checkpoint),
        "git_dirty": dirty,
        "adapter_version": ADAPTER_VERSION,
        "smoke_test_version": SMOKE_TEST_VERSION,
        "model": model,
        "package_version": importlib.metadata.version(model["package_name"]),
        "dependency_versions": {name: _installed_version(name) for name in RECORDED_DEPENDENCIES},
        "git_sha": git_sha.strip(),
    }
    fingerprint_sha = canonical_json_sha256(fingerprint)
    path = args.output / f"{args.model}.json"
    if path.exists():
        if not args.resume:
            raise SystemExit(f"{path} exists; use --resume")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint_sha256") != fingerprint_sha:
            raise SystemExit("stale smoke-test report: fingerprint changed")
        print(f"verified existing report {path}")
        return 0 if existing["result"]["passed"] else 1

    if fingerprint["package_version"] != model["package_version"]:
        raise SystemExit("installed package version differs from the frozen registry")
    calc = build_calculator(model, args.checkpoint)
    result = smoke_test(calc)
    report = {
        "schema_version": "e3sse.model.smoke.v1",
        "fingerprint": fingerprint,
        "fingerprint_sha256": fingerprint_sha,
        "python": sys.version,
        "result": result,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=args.output, prefix=f".{args.model}.")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.link(temporary, path)
    os.unlink(temporary)
    print(json.dumps({"model": args.model, "passed": result["passed"], **result["checks"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
