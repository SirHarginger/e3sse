#!/usr/bin/env python3
"""Collect read-only production source-unit evidence for the static protocol freeze.

Files are resolved through the G0-pinned input inventory named in the protocol
configuration.  Only extxyz comment-line keys are parsed for every file; numeric
unit-scale checks use published-train MPLiTrj frames that the provenance index proves
are endpoint-relaxation (non-path) frames, and only booleans are recorded.  The report
is written atomically under ``<output>/<fingerprint>/evidence.json``; ``--resume``
verifies an existing report and rejects a stale one.  A dirty worktree is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from e3sse.data.unit_evidence import EVIDENCE_SCHEMA, collect, csv_identifier_columns


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _nonpath_train_indices(root: Path, limit: int) -> list[int]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    indices = []
    for descriptor in manifest["shards"]:
        shard = json.loads((root / descriptor["path"]).read_text(encoding="utf-8"))
        if shard.get("split") != "train":
            continue
        for record in shard["records"]:
            if (
                record.get("flattened_file") == "MPLiTrj_train.xyz"
                and record.get("mapping_status") == "mapped_unique"
                and record.get("raw_file_kind") in {"source", "target"}
            ):
                indices.append(int(record["flattened_frame_index"]))
    return sorted(indices)[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="protocol-freeze JSON configuration")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    inputs = config["inputs"]
    server = json.loads(_resolve(base, inputs["server_config"]["path"]).read_text())
    data_root = Path(server["paths"]["data_root"]).resolve()
    output = args.output.resolve()
    if output.is_relative_to(data_root):
        raise SystemExit("unit-evidence output must not lie under the data root")
    inventory_path = _resolve(base, inputs["g0_input_manifest"]["path"])
    inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    pinned = inputs["g0_input_manifest"].get("sha256")
    if pinned is not None and pinned != inventory_sha:
        raise SystemExit("G0 input manifest SHA-256 differs from the configured pin")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))["inputs"]

    def data_file(identifier: str) -> Path:
        path = (data_root / identifier).resolve()
        path.relative_to(data_root)
        return path

    mplitrj = {
        split: data_file(item["identifier"])
        for item in inventory["MPLiTrj"]
        for split in ("train", "val", "test")
        if Path(item["identifier"]).name == f"MPLiTrj_{split}.xyz"
    }
    neb = [data_file(item["identifier"]) for item in inventory["nebDFT2k"]]
    indices = []
    for path in neb:
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig") as handle:
                if "edge_id" in handle.readline().split(","):
                    indices.append(path)
    if len(indices) != 1:
        raise SystemExit("expected exactly one nebDFT2k CSV index with edge_id")
    _, rows = csv_identifier_columns(indices[0], ("edge_id", "_split"))
    train_edges = sorted(edge for edge, split in rows if split == "train")
    by_name = {path.name: path for path in neb}
    trajectories = [
        by_name[f"{edge}_relaxed.xyz"] for edge in train_edges if f"{edge}_relaxed.xyz" in by_name
    ][: args.trajectories]
    provenance_root = _resolve(base, inputs["mplitrj_provenance"]["root"])
    frame_indices = _nonpath_train_indices(provenance_root, args.frames)

    repo = Path(__file__).resolve().parents[1]
    git_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True)
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True):
        raise SystemExit("unit evidence requires a clean worktree at the reviewed commit")
    fingerprint = {
        "schema": EVIDENCE_SCHEMA,
        "git_sha": git_sha.strip(),
        "inventory_sha256": inventory_sha,
        "files": sorted(str(p) for p in [*mplitrj.values(), indices[0], *trajectories]),
        "train_nonpath_frame_indices": frame_indices,
    }
    run_dir = output / _sha(fingerprint)[:16]
    report_path = run_dir / "evidence.json"
    if report_path.exists():
        if not args.resume:
            raise SystemExit(f"{report_path} exists; use --resume")
        if json.loads(report_path.read_text(encoding="utf-8")).get("fingerprint") != fingerprint:
            raise SystemExit("stale unit-evidence report")
        print(report_path)
        return 0
    evidence = collect(
        mplitrj_files=mplitrj,
        train_nonpath_frame_indices=frame_indices,
        nebdft2k_index=indices[0],
        nebdft2k_trajectories=trajectories,
    )
    report = {"fingerprint": fingerprint, "python": sys.version, "evidence": evidence}
    run_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=run_dir, prefix=".evidence.")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.link(temporary, report_path)
    os.unlink(temporary)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
