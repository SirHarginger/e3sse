# Coding Conventions

## Core Sections (Required)

### 1) Naming Rules

| Item | Rule | Example | Evidence |
|---|---|---|---|
| Files | snake_case | `dataset_audit.py` | `src/e3sse/data/` |
| Functions | snake_case | `run_audit` | `dataset_audit.py` |
| Types | PascalCase | `ProtocolFreezeError` | `protocol/freeze.py` |
| Constants | UPPER_SNAKE_CASE | `SCHEMA_VERSION` | `dataset_audit.py` |

### 2) Formatting and Linting

- Formatter: no separately configured formatter.
- Linter: Ruff in `pyproject.toml`.
- Relevant rules: default Ruff set plus explicit broad-exception exemptions for audit boundaries.
- Command: `ruff check .`.

### 3) Import and Module Conventions

- Standard library, third-party, then local imports.
- Absolute `e3sse` imports across packages; relative imports inside `protocol`.
- Package exports are intentionally small.

### 4) Error and Logging Conventions

- Boundary validation raises explicit exceptions; CLIs convert expected failures to nonzero exits.
- Long jobs accept progress callbacks or print concise progress.
- Credentials must never enter tracked configuration (`AGENTS.md`).

### 5) Testing Conventions

- Tests live in `tests/test_*.py` and use tiny synthetic fixtures.
- Filesystem boundaries use `tmp_path`; production datasets are not copied.
- Coverage threshold: [TODO] no threshold is configured.

### 6) Evidence

- `pyproject.toml`
- `tests/test_dataset_audit.py`
- `scripts/build_protocol_freeze.py`
