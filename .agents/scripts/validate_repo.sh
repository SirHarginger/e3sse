#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "==> Python syntax"
python -m compileall -q src scripts

if command -v ruff >/dev/null 2>&1; then
    echo "==> Ruff"
    ruff check .
else
    echo "==> Ruff not installed; skipping"
fi

if command -v pytest >/dev/null 2>&1 && \
   find tests -type f -name 'test_*.py' -print -quit | grep -q .; then
    echo "==> Pytest"
    pytest -q
else
    echo "==> No runnable pytest suite yet; skipping"
fi

echo "==> Git data-safety check"

if git ls-files | grep -E '^(data|outputs|logs|scratch)/' >/dev/null 2>&1; then
    echo "ERROR: generated/server data are tracked by Git:"
    git ls-files | grep -E '^(data|outputs|logs|scratch)/'
    exit 1
fi

echo "==> E3-SSE validation passed"
