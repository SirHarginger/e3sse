#!/usr/bin/env python3
"""Flag newly introduced numeric version labels in first-party E3-SSE architecture names.

Convention: our own protocols, schemas, modules, configs and reports are named for
what they do.  Their identity is a content hash, never a sequence number such as
``protocol_v4.json`` or ``e3sse.selection.v2``.

The check is deliberately narrow:

* scope: first-party files under src/, scripts/, configs/, tests/, docs/ and .agents/
  (tracked or untracked-but-not-ignored); binary documents are skipped;
* historical identifiers are exempt: every flagged token that already existed at
  the pinned baseline commit (before the convention) is grandfathered, so immutable
  schemas such as ``e3sse.protocol.freeze.v1`` and frozen hash domains stay exact;
* external versions are not matched: dotted releases (``v0.3.0``,
  ``PBE-v2025.1``), ``pkg==0.11.0``, DOIs and environment names contain no bare
  ``vN`` token of the forms below.

Flagged forms: path components with a ``vN`` segment or ``versionN``; schema IDs
``e3sse.<...>.vN``; identifiers ``name_vN`` / ``vN_name``; standalone ``vN`` in prose
(``v4``) unless followed by a dotted release number.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

BASELINE_COMMIT = "0fab51e5ba3c905118475062d29bc414bd380af2"
SELF_FILES = {".agents/scripts/check_semantic_names.py", "tests/test_semantic_naming_check.py"}
SCOPES = ("src", "scripts", "configs", "tests", "docs", ".agents", ".claude", ".github")
ROOT_FILES = ("AGENTS.md", "CLAUDE.md", "README.md", "README_data.md", "pyproject.toml")
TEXT_SUFFIXES = {".py", ".json", ".md", ".sh", ".toml", ".txt", ".cfg"}
PATH_PATTERN = re.compile(
    r"(?:^|[_.\-])v\d+(?:[_.\-]|$)|version[_-]?\d|[a-z]V\d+(?:[_.\-]|$)", re.IGNORECASE
)
TOKEN_PATTERNS = (
    re.compile(r"\be3sse[.-][A-Za-z0-9_.-]*?[._-]v\d+\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)*[_-]v\d+\b(?![.]\d)", re.IGNORECASE),
    re.compile(r"\b[a-z][a-z0-9]*V\d+\b(?![.]\d)"),
    re.compile(r"\bv\d+[_-][A-Za-z]\w*", re.IGNORECASE),
    re.compile(r"(?<![\w.-])v\d+\b(?![.]\d)", re.IGNORECASE),
)
# External release labels written as "<name> vN" are not our architecture names.
EXTERNAL_CONTEXT = re.compile(r"(?i)\b(?:arxiv|cuda|python|pep|http|tls|ipv|unicode)\s+$")
# Frozen historical identifiers (schemas, rule ids, hash domains) stay exact everywhere.
HISTORICAL_IDENTIFIER = re.compile(r"(?i)^(?:e3sse[.-].*|.*[_-]v\d+)$")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def _tokens(text: str) -> set[str]:
    found = set()
    for pattern in TOKEN_PATTERNS:
        for match in pattern.finditer(text):
            if EXTERNAL_CONTEXT.search(text[max(0, match.start() - 12):match.start()]):
                continue
            found.add(match.group(0))
    return found


def _in_scope(name: str) -> bool:
    return name in ROOT_FILES or name.startswith(tuple(f"{scope}/" for scope in SCOPES))


def baseline_tokens() -> tuple[set[str], dict[str, set[str]], set[str]]:
    """Baseline paths, per-file baseline tokens, and global historical identifiers.

    Bare ``vN`` prose is exempt only in the same file where it already existed.
    Structured identifiers (``e3sse.<...>.vN``, ``rule-id-vN``, ``name_vN``) that
    existed anywhere at baseline are frozen historical identifiers, exempt everywhere.
    """

    paths = set(_git("ls-tree", "-r", "--name-only", BASELINE_COMMIT).splitlines())
    per_file: dict[str, set[str]] = {}
    historical: set[str] = set()
    for path in paths:
        if Path(path).suffix in TEXT_SUFFIXES and _in_scope(path):
            try:
                tokens = _tokens(_git("show", f"{BASELINE_COMMIT}:{path}"))
            except subprocess.CalledProcessError:
                continue
            per_file[path] = tokens
            historical |= {t for t in tokens if HISTORICAL_IDENTIFIER.match(t)}
    return paths, per_file, historical


def violations(root: Path) -> list[str]:
    old_paths, per_file, historical = baseline_tokens()
    files = _git("ls-files", "-co", "--exclude-standard").splitlines()
    problems = []
    for name in sorted({f for f in files if _in_scope(f)}):
        path = root / name
        # The checker and its own test necessarily spell out the forbidden examples.
        if name in SELF_FILES or not path.is_file():
            continue
        if name not in old_paths and any(PATH_PATTERN.search(part) for part in Path(name).parts):
            problems.append(f"{name}: numeric version label in path")
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        allowed = per_file.get(name, set()) | historical
        for token in sorted(_tokens(text) - allowed):
            problems.append(f"{name}: numeric version label {token!r}")
    return problems


def main() -> int:
    root = Path(_git("rev-parse", "--show-toplevel").strip())
    problems = violations(root)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print("Use descriptive names; identity comes from content hashes.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
