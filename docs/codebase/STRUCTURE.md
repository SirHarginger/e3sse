# Codebase Structure

## Core Sections (Required)

### 1) Top-Level Map

| Path | Purpose | Evidence |
|---|---|---|
| `src/e3sse/` | reusable scientific implementation | `AGENTS.md` |
| `scripts/` | command-line entry points | `scripts/check_g0.py` |
| `configs/` | JSON runtime configuration; legacy YAML | `src/e3sse/config.py` |
| `tests/` | synthetic fixtures and pytest suite | `pyproject.toml` |
| `docs/research/` | canonical protocol sources | `E3-SSE_Research_Design.docx` |
| `.agents/` | skills, hooks, validation helpers | `.agents/README.md` |

### 2) Entry Points

- Main runtime entries: scripts under `scripts/`.
- Protocol entry: `scripts/build_protocol_freeze.py`.
- Entry selection: direct Python invocation with explicit JSON configuration.

### 3) Module Boundaries

| Boundary | Belongs here | Must not be here |
|---|---|---|
| `data` | schema/provenance audits | scientific outcome tuning |
| `protocol` | manifests, partitions, freeze contracts | MLIP inference execution |
| `barriers`, `transport`, `selection`, `screening` | named scientific stages | raw-data mutation |

### 4) Naming and Organization Rules

- Python files/functions use snake_case; classes use PascalCase.
- Modules are organized by scientific domain.
- Imports use the installed `e3sse` package rooted at `src/`.

### 5) Evidence

- `src/e3sse/`
- `scripts/`
- `pyproject.toml`
