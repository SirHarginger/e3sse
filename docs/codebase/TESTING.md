# Testing Patterns

## Core Sections (Required)

### 1) Test Stack and Commands

- Primary framework: pytest.
- Assertions/mocking: pytest assertions, fixtures, monkeypatch.
- Commands:

```bash
python -m pytest -q
bash .agents/scripts/validate_repo.sh
ruff check .
```

### 2) Test Layout

- Tests are under `tests/test_*.py`.
- Small scientific files live under `tests/fixtures/`.
- No global setup file is present.

### 3) Test Scope Matrix

| Scope | Covered? | Target | Notes |
|---|---|---|---|
| Unit | yes | parsers, hashes, contracts | synthetic |
| Integration | yes | G0/provenance/protocol builders | temporary trees |
| E2E production | no | server datasets | intentionally server-only |

### 4) Mocking and Isolation Strategy

- Temporary directories isolate all writes.
- Synthetic ASE structures exercise scientific schemas.
- Production paths, models, and held-out outcomes are not used locally.

### 5) Coverage and Quality Signals

- Coverage threshold: [TODO] not configured.
- Current reported coverage: [TODO] not measured by validation script.
- Known gap: full-data scalability is validated only on the server.

### 6) Evidence

- `pyproject.toml`
- `tests/test_protocol_freeze.py`
- `.agents/scripts/validate_repo.sh`
