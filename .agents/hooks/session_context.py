#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from pathlib import Path


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


root = Path(
    git(["rev-parse", "--show-toplevel"])
    if git(["rev-parse", "--show-toplevel"]) != "unknown"
    else "."
).resolve()

branch = git(["branch", "--show-current"])
sha = git(["rev-parse", "--short", "HEAD"])

print(
    f"""E3-SSE session context:
repository={root}
branch={branch}
commit={sha}

Critical project rules:
- Read AGENTS.md before substantial work.
- Never destructively modify data/ or /srv/ben/e3sse/data.
- Treat data/raw as immutable.
- Preserve scientific leakage controls.
- Never fabricate missing scientific values.
- Run .agents/scripts/validate_repo.sh before declaring implementation complete.
"""
)
