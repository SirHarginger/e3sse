# Architecture

## Core Sections (Required)

### 1) Architectural Style

- Primary style: scientific pipeline with domain modules and thin CLI adapters.
- Evidence: scripts delegate to reusable `src/e3sse` functions.
- Constraints: immutable raw data, explicit provenance, and sealed held-out outcomes.

### 2) System Flow

```text
JSON config -> read-only source artifacts -> validation/domain logic -> atomic derived JSON
```

Gate G0 inventories source data. The provenance index maps flattened MPLiTrj frames to hops. Protocol materialization consumes those artifacts, validates identities, constructs grouped partitions, and writes content-addressed outputs. Later scientific stages consume the frozen IDs.

### 3) Layer/Module Responsibilities

| Module | Owns | Must not own | Evidence |
|---|---|---|---|
| `e3sse.config` | safe path resolution | scientific defaults | `src/e3sse/config.py` |
| `e3sse.data` | audit and provenance | source mutation | `src/e3sse/data/` |
| `e3sse.protocol` | freeze-time contracts, classified decision register (`decisions.py`) | held-out analysis | `src/e3sse/protocol/` |
| `scripts` | CLI parsing/reporting | core scientific logic | `scripts/` |

### 4) Reused Patterns

| Pattern | Where | Purpose |
|---|---|---|
| content hashes | G0, provenance, protocol | deterministic identity/staleness |
| atomic JSON replacement | data/protocol modules | avoid partial canonical artifacts |
| fail-closed validation | provenance/protocol | preserve leakage and provenance |

### 5) Known Architectural Risks

- Production schemas are large and must be streamed rather than copied into Git.
- Source units remain blocked until authoritative evidence is recorded.

### 6) Evidence

- `src/e3sse/data/dataset_audit.py`
- `src/e3sse/data/mplitrj_provenance.py`
- `src/e3sse/protocol/freeze.py`
