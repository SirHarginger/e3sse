# External Integrations

## Core Sections (Required)

### 1) Integration Inventory

| System | Type | Purpose | Auth | Criticality | Evidence |
|---|---|---|---|---|---|
| filesystem datasets | files | scientific inputs | OS permissions | high | `configs/server.json` |
| Git | CLI | code/provenance identity | local checkout | high | `.agents/scripts/record_provenance.py` |
| AiiDA archives | read-only archive | FPMD inventory | none | high for Study 3 | `fpmd_aiida_inventory.py` |
| OSF | manual future registration | preregistration | user-managed | future | Research Design §11 |

### 2) Data Stores

| Store | Role | Access layer | Risk | Evidence |
|---|---|---|---|---|
| `data/raw` | immutable research data | `e3sse.data` | accidental mutation | `AGENTS.md` |
| `outputs` | derived artifacts | atomic JSON writers | stale mixing | `docs/workflow.md` |
| `scratch` | temporary derived state | configured jobs | disk pressure | `configs/server.json` |

### 3) Secrets and Credentials Handling

- No application credentials were identified.
- `.env*` is ignored; tracked secrets are prohibited by `AGENTS.md`.
- Rotation/lifecycle: not applicable to current file-based workflow.

### 4) Reliability and Failure Behavior

- Resume validates hashes/fingerprints before reuse.
- Network retry policy: [TODO] acquisition script-specific.
- Protocol construction requires explicit paths and does not discover “latest” production runs.

### 5) Observability for Integrations

- Commands record Git/config/input identities and explicit failures.
- No metrics/tracing service is configured.
- Full production job monitoring remains [TODO].

### 6) Evidence

- `configs/server.json`
- `src/e3sse/data/fpmd_aiida_inventory.py`
- `src/e3sse/protocol/freeze.py`
