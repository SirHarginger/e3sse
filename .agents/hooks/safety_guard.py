#!/usr/bin/env python3
"""Block destructive commands that could damage E3-SSE data or Git history."""

from __future__ import annotations

import json
import re
import sys
from typing import Any


def load_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def tool_command(payload: dict[str, Any]) -> str:
    tool_name = (
        payload.get("tool_name")
        or payload.get("toolName")
        or payload.get("tool")
        or ""
    )

    args: Any = (
        payload.get("tool_input")
        or payload.get("toolArgs")
        or payload.get("tool_args")
        or {}
    )

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            if tool_name.lower() in {"bash", "shell", "powershell"}:
                return args
            return ""

    if not isinstance(args, dict):
        return ""

    command = (
        args.get("command")
        or args.get("cmd")
        or args.get("script")
        or ""
    )

    return str(command)


def blocked_reason(command: str) -> str | None:
    if not command.strip():
        return None

    c = command.lower()

    destructive_git = [
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\s+-[a-z]*f[a-z]*\b",
        r"\bgit\s+push\b[^\n;]*\s(--force|-f)(\s|$)",
        r"\bgit\s+checkout\s+--\s+\.",
        r"\bgit\s+restore\s+--source=[^\s]+\s+--worktree\s+--staged\s+\.",
    ]

    destructive_system = [
        r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+/\s*(?:$|[;&])",
        r"\brm\s+-[a-z]*f[a-z]*r[a-z]*\s+/\s*(?:$|[;&])",
        r"\bchmod\s+-r\s+777\s+/",
    ]

    data_target = (
        r"(?:/srv/ben/e3sse/data"
        r"|(?:^|[\s\"'])\.?/?data(?:/|[\s\"']))"
    )

    destructive_data = [
        rf"\brm\b[^\n;]*{data_target}",
        rf"\bmv\b[^\n;]*{data_target}",
        rf"\btruncate\b[^\n;]*{data_target}",
        rf"\bfind\b[^\n;]*{data_target}[^\n;]*-delete\b",
        rf"\bdd\b[^\n;]*\bof=[^\s]*{data_target}",
        rf"\brsync\b[^\n;]*--delete[^\n;]*{data_target}",
    ]

    for pattern in destructive_git:
        if re.search(pattern, c):
            return "Blocked by E3-SSE policy: destructive Git history operation."

    for pattern in destructive_system:
        if re.search(pattern, c):
            return "Blocked by E3-SSE policy: destructive filesystem operation."

    for pattern in destructive_data:
        if re.search(pattern, c):
            return (
                "Blocked by E3-SSE policy: research data are immutable. "
                "Write generated material to outputs/ or scratch/ instead."
            )

    return None


def main() -> int:
    payload = load_payload()
    command = tool_command(payload)

    reason = blocked_reason(command)

    if reason:
        print(reason, file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
