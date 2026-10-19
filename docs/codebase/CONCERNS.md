# Codebase Concerns

## Core Sections (Required)

### 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|---|---|---|---|---|
| high | Source-side units remain unverified | G0 dataset reports | invalid numerical comparisons | resolve during server materialization |
| high | TM/Hubbard-U predicate is not machine-frozen | Research Design §5.3 | primary-set drift | register authoritative rule before Study 1 |
| high | v1 MPLiTrj provenance omits chemical system | provenance record schema | grouping requires a derived fact | protocol builder derives a read-only, source-hashed species linkage |
| high | Production trust hashes and preregistration choices are unresolved in the checked-in template | `configs/protocol_freeze.json` | server runs yield a sealed `candidate_unfrozen` protocol listing `freeze_blockers`; no freeze until all are resolved | populate only from authoritative server/source evidence |
| medium | Large audit modules mix responsibilities | scan metrics | fragile changes | make narrow additions with focused tests |

### 2) Technical Debt

| Debt | Why | Where | Risk | Suggested fix |
|---|---|---|---|---|
| unconstrained runtime dependencies | early research repository | `pyproject.toml` | environment drift | add lock/spec before production |
| legacy JSON/YAML duplication | compatibility | `configs/` | divergent settings | make JSON canonical |

### 3) Security Concerns

| Risk | Category | Evidence | Mitigation | Gap |
|---|---|---|---|---|
| unsafe paths | N/A | config/provenance readers | resolution checks and hooks | keep tests current |
| credentials in files | N/A | `AGENTS.md` | ignored env files | no automated secret scanner |

### 4) Performance and Scaling Concerns

| Concern | Evidence | Symptom | Risk | Improvement |
|---|---|---|---|---|
| 929k-frame species linkage | MPLiTrj design | long one-pass scan | restart cost | material-level resumable index if needed |
| large JSON provenance | shard schema | many small reads | server I/O | manifest-driven streaming |

### 5) Fragile/High-Churn Areas

| Area | Why | Churn | Safe strategy |
|---|---|---|---|
| `dataset_audit.py` | central G0 schema | high | preserve schema and add regression tests |
| `docs/workflow.md` | operational source | high | document exact validated commands |

### 6) `[ASK USER]` Questions

1. [ASK USER] Which authoritative MPtrj transition-metal/Hubbard-U predicate should be registered?
2. [ASK USER] Should stress be retained downstream; if so, which canonical stress unit and source evidence apply?
3. [ASK USER] Which exact B1/B2 checkpoints should be frozen before Study 1 inference?
4. [ASK USER] What mixed-anion family precedence should be registered?
5. [ASK USER] What configurational-energy reference population and decile rule should be registered?
6. [ASK USER] What label-budget repetition/stability rule and Holm family should be registered?
7. [ASK USER] Do Study 1 robustness repeats use the full repeated training/calibration side or only published-train frames within that side?
8. [ASK USER] Does “all evaluation chemical systems” for Study 5 include the Study 3 transport panel in addition to published and repeated LiTraj test systems?
9. [ASK USER] Should repeated same-hop endpoint matches be verified by a new provenance index, a targeted read-only archive check, or excluded conservatively?

### 7) Evidence

- `pyproject.toml`
- `src/e3sse/data/dataset_audit.py`
- `docs/research/E3-SSE_Research_Design.docx`
