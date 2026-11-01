# E3-SSE

**Reliability-gated, training-free recalibration of foundation interatomic potentials for Li-ion migration and controlled materials screening**

E3-SSE asks whether a small number of independent DFT force labels can diagnose systematic softening in a frozen foundation interatomic potential. It then tests three things:

- whether a scalar correction is trustworthy;
- whether that correction transfers to unseen Li-ion migration barriers without retraining;
- whether it improves candidate selection while controlling false discoveries.

The six pretrained potentials stay frozen. No neural-network weights are trained or fine-tuned.

## Method

```text
ESTIMATE              softening scale s from leakage-free DFT force labels
   ↓
DIAGNOSE              is one scalar an adequate local model of the error?
   ↓
CORRECT OR ABSTAIN    SRT barrier = raw barrier / s where supported; abstain otherwise
   ↓
SELECT                weighted conformal selection with false-discovery-rate control
   ↓
DISCOVER              frozen prospective screen of an independent candidate pool
```

| Stage | Internal name (code, provenance) |
|---|---|
| Calibration and applicability | Study 1 |
| Unseen-kinetics transfer | Study 2a |
| Risk-controlled selection | Study 4 |
| Prospective discovery | Study 5 |

Transport validation against FPMD (Study 3) is formally deferred.

## Protocol status

- **Registration-gated historical protocol.** Immutable; frozen at commit `971f4c42` as protocol `5f9aa711…`.
- **Reliability-gated internal-freeze protocol** (`configs/reliability_gated_internal_freeze_protocol.json`).
  - Layered on the historical protocol by exact protocol ID and file SHA-256.
  - No external registration is required, and pre-registration is not claimed.
  - **Not yet frozen.** Every local engineering item is resolved. Server-evidence items remain: embedding capability, real BVEL13k/nebBVSE122k linkage, `ions`/BVSE and ASE/FIRE/NEB reproduction, descriptor availability, and the development manifest. The freeze builder refuses to freeze while any remain.

No scientific outcome is read until the internal-freeze protocol is materialized.

## Repository layout

| Path | Contents |
|---|---|
| `src/e3sse/protocol/` | Historical protocol freeze, outcome seals, internal-freeze protocol |
| `src/e3sse/softening/` | Softening estimation (Study 1), applicability diagnostics |
| `src/e3sse/barriers/` | Fixed-geometry raw and SRT barriers, paired metrics |
| `src/e3sse/selection/` | Risk–coverage/AURC, conformal and weighted conformal selection, density-ratio weights |
| `src/e3sse/screening/` | Prospective support gate, Gate-1 rate-limiting hop, population funnel, model selection |
| `src/e3sse/representation/` | Embedding-capability probe |
| `src/e3sse/execution/` | Deterministic, resumable, memory-aware parallel execution |
| `src/e3sse/figures/` | Publication style and figure manifests |
| `configs/` | Protocol configurations, figure/table registries, descriptor manifest |
| `docs/research/` | Research design, design report, novelty boundary |
| `docs/workflow.md` | Server workflow and production commands |

## Development

```bash
source .venv/bin/activate
bash .agents/scripts/validate_repo.sh   # syntax, ruff, tests, naming and data-safety checks
```

Read `AGENTS.md` before substantial work. Raw data are immutable; generated results go under `outputs/`. New first-party names describe what they do and never carry numeric version labels; identity comes from content hashes.

## Locations

- GitHub: `github.com/SirHarginger/e3sse`
- Production server: `/srv/ben/e3sse` (host `aiserver`)

## Server handoff

Production runs only from an exact reviewed commit:

```bash
cd /srv/ben/e3sse
git fetch origin
git checkout <reviewed_commit_sha>
```

Then:

1. Run the four server audits: embedding capability, rate-limiting-hop linkage, `ions`/BVSE reproduction, ASE/FIRE/NEB pins.
2. Resolve the server-evidence items.
3. Materialize the internal-freeze protocol.
4. Start production: calibration → transfer → reliability → selection → prospective discovery.

Study 1 production commands are in `docs/workflow.md`.
