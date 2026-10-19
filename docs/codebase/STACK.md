# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area | Value | Evidence |
|---|---|---|
| Primary language | Python | `pyproject.toml` |
| Runtime + version | Python >=3.10 | `pyproject.toml` |
| Package manager | pip-compatible Python packaging | `pyproject.toml`, `README.md` |
| Module/build system | setuptools, `src/` layout | `pyproject.toml` |

### 2) Production Frameworks and Dependencies

| Dependency | Version | Role | Evidence |
|---|---|---|---|
| ASE | unconstrained | atomistic structures and I/O | `pyproject.toml`, `src/e3sse/data/` |
| NumPy | unconstrained | numerical arrays | `pyproject.toml` |
| pandas | unconstrained | tabular dataset inspection | `pyproject.toml` |
| PyYAML | unconstrained | legacy configuration support | `pyproject.toml`, `src/e3sse/config.py` |

### 3) Development Toolchain

| Tool | Purpose | Evidence |
|---|---|---|
| pytest | tests | `pyproject.toml` |
| Ruff | linting | `pyproject.toml` |

### 4) Key Commands

```bash
pip install -e '.[dev]'
python -m pytest -q
ruff check .
bash .agents/scripts/validate_repo.sh
```

### 5) Environment and Config

- Config sources: `configs/*.json`; YAML remains legacy-compatible.
- Required environment variables: none identified.
- Production data and computation live outside Git under `/srv/ben/e3sse`.

### 6) Evidence

- `pyproject.toml`
- `src/e3sse/config.py`
- `.agents/scripts/validate_repo.sh`
