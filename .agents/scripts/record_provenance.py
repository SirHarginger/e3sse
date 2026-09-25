#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--command")
    p.add_argument("--seed")
    args = p.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = Path(args.config) if args.config else None

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "command": args.command,
        "seed": args.seed,
        "config": str(config) if config else None,
        "config_sha256": sha256(config) if config else None,
    }

    target = output / "run_metadata.json"
    target.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(target)


if __name__ == "__main__":
    main()
