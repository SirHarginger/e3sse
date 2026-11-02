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
  - **Not yet frozen.** Every local engineering item is resolved, and the six authoritative embedding-capability reports are pinned by SHA-256 (`amendments.representation.embedding_capability_evidence`). Seven items still block the freeze, all server evidence. `ions_bvse_version_pin`, `bvse_reproduction`, `rate_limiting_hop_algorithm`, `prospective_gate1_rule`, `descriptor_availability`, `development_manifest` and `path_representation_mapping`. The owner decisions of 2026-11-02 are recorded as protocol rules: P is published-train systems only; weight-fit slices are fixed from chemical-system identity before any ML-NEB (fraction 0.2); per-runtime ASE versions are retained with explicit NEB settings and cross-runtime equivalence.

    The freeze builder refuses to freeze while any blocking item remains.

No scientific outcome is read until the internal-freeze protocol is materialized.

## Repository layout

| Path | Contents |
|---|---|
| `src/e3sse/protocol/` | Historical protocol freeze, outcome seals, internal-freeze protocol |
| `src/e3sse/softening/` | Softening estimation (Study 1), applicability diagnostics |
| `src/e3sse/barriers/` | Fixed-geometry raw and SRT barriers, paired metrics; frozen ML-NEB engine |
| `src/e3sse/selection/` | Risk–coverage/AURC, conformal and weighted conformal selection, density-ratio weights |
| `src/e3sse/screening/` | Prospective support gate, Gate-1 rate-limiting hop, population funnel, model selection |
| `src/e3sse/representation/` | Embedding-capability probe |
| `src/e3sse/execution/` | Deterministic, resumable, memory-aware parallel execution; workload benchmark |
| `src/e3sse/production/` | Production stage graph, scheduler, runtime routing, preflight, run identity, orchestrator |
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

## Production run

One resumable command drives the whole pipeline. It is a thin orchestrator (`src/e3sse/production/`) over the tested stage runners and never reimplements a scientific method:

```bash
cd /srv/ben/e3sse
.venv/bin/python scripts/run_e3sse_production.py \
  --config configs/production_execution.json --stage all --resume
```

### Stage graph

Stages and dependencies are declared in `configs/production_execution.json`. The code (`e3sse.production.stage_graph`) fixes what a configuration cannot change:

- **Outcome access.** Each stage's permitted outcome kinds (`PERMITTED_OUTCOME_ACCESS`) can only be narrowed. Every child command receives its stage's set as `E3SSE_STAGE_OUTCOME_ACCESS`, and `contracts.require_outcome_access` refuses any other outcome kind.
- **Freeze boundary.** Every pre-freeze evidence stage is an ancestor of `freeze` and reads no outcome. Every other stage descends from `freeze`, and runs only after the orchestrator has verified the frozen artifact. Verification covers its content hash, parent, frozen status, and equality with the active protocol, so a protocol edit made after the freeze is refused.
- **Freeze order.** The protocol's ordering is enforced. No ancestor of `model_selection` reads a label of the conformal-calibration systems. Calibration ML-NEB, pool ML-NEB and density-ratio weights all come before weighted selection.

| Stage | Public name | Reads | Status |
|---|---|---|---|
| `runtime_audit`, `resource_profile`, `gate1_population_audit` | PREFLIGHT evidence | nothing | implemented |
| `bvse_reproduction`, `descriptor_availability`, `development_population`, `development_ml_neb`, `path_representation` | PREFLIGHT evidence | nothing | missing runner (each names its next task) |
| `preflight` | PREFLIGHT | nothing | implemented |
| `freeze` | INTERNAL SCIENTIFIC FREEZE | nothing | implemented |
| `estimate` | ESTIMATE / CALIBRATION (Study 1) | published-train forces | implemented |
| `family_scale` | prospective family scale | training-side forces without nebDFT2k hops | missing runner |
| `transfer`, `reliability` | TRANSFER, RELIABILITY | evaluation barriers, test-side non-path forces | missing runner |
| `weight_slice_ml_neb`, `model_selection` | MODEL SELECTION | development barriers only | missing runner |
| `deployment_calibration`, `discovery`, `density_ratio` | calibration and pool ML-NEB, weights | nothing (draw → run → drop) | missing runner |
| `linear_recalibration_comparator` | B5 comparator | calibration-system barriers (after selection) | missing runner |
| `controlled_selection` | RISK-CONTROLLED SELECTION (Study 4) | repeated-partition barriers | missing runner |
| `discovery_selection` | PROSPECTIVE DISCOVERY (weighted conformal) | calibration barriers | missing runner |
| `publication` | PUBLICATION BUILD | derived tables only | missing runner |

`--stage X` runs X plus every dependency without a completion record. A blocked or failed stage stops only its dependents. By default (`freeze.require_outcome_runners_implemented: true`), the freeze also waits until every outcome-reading stage has a tested runner, so no analysis code is written after outcomes are seen. Setting it to `false` is a declared deviation, recorded with the run.

### Environments and checkpoints

- Project environment: `/srv/ben/e3sse/.venv` (Python 3.14), `pip install -e ".[dev,figures]"`. It holds no MLIP package.
- Frozen model runtimes: `~/.venvs/<env>` (Python 3.11.16), one per package version, rebuilt only from the lock files pinned in `configs/protocol_freeze.json` (table in `docs/workflow.md`). Model work runs with each runtime's own interpreter, `PYTHONPATH=src` and `PYTHONNOUSERSITE=1`. Production never installs or upgrades a package. A missing interpreter is reported as missing evidence.
- Checkpoints: `/srv/ben/e3sse/outputs/model_checkpoints/checkpoint_map.json` (`{"<canonical_name>": "<path>"}`). Preflight re-hashes every checkpoint against the registry `weights_sha256`, and every model load verifies it again.

### Server sequence

```bash
cd /srv/ben/e3sse && git fetch origin && git checkout <reviewed_commit_sha>
source .venv/bin/activate && bash .agents/scripts/validate_repo.sh
export TMPDIR=/srv/ben/e3sse/scratch        # /tmp is RAM-backed tmpfs
PROD="python scripts/run_e3sse_production.py --config configs/production_execution.json"

$PROD --stage all --dry-run > scratch/production_dry_run.json       # 1. plan; writes nothing
$PROD --stage runtime_audit --resume                                 # 2. ASE/FIRE/NEB audits
$PROD --stage resource_profile --resume                              # 3. memory + throughput
$PROD --stage gate1_population_audit --resume                        # 4. structural populations
$PROD --stage preflight --resume      # 5. runs every pre-freeze evidence stage, then checks all
#   implement the missing pre-freeze runners, record the owner decisions and server evidence
#   by reviewed protocol edits and commits, then repeat 1-5 until preflight passes
$PROD --stage freeze --resume                                        # 6. internal freeze
$PROD --stage all --resume                                           # 7. full production
```

The dry run prints the resolved DAG, inputs, model runtimes, predicted allocation (or the missing resource profiles), output paths, blocking evidence and the exact commands. It reads no outcome and launches nothing. Before full inference, ESTIMATE runs the frozen Study-1 pilot gate for each model (8 frames, finite-difference force check). That is the production smoke test, and it never tunes anything. The synthetic end-to-end tests run the whole orchestrator without a model. `--max-cpus` and `--max-memory-gib` can only narrow the observed resources. The threads per worker chosen at a stage's first start stay fixed for that run.

### Resources

The central scheduler (`e3sse.production.resource_scheduler`) reads the schedulable CPUs from the affinity mask and the memory from `MemAvailable`. Swap is never counted. It keeps a reserve of max(2.5 GiB, 12 % of RAM) and plans each worker at 1.35 × its measured peak RSS, so that `workers × planned RSS ≤ MemAvailable − reserve` and `workers × threads ≤ CPUs`. Among measured (processes, threads) points that fit, the highest aggregate throughput wins. Without a benchmark the PyTorch bound `threads = ⌊CPUs / workers⌋` applies, and the allocation is labelled unbenchmarked. No worker count is hard-coded. Each worker is a fresh `spawn` interpreter: OpenMP/BLAS thread variables are set before start, then `torch.set_num_threads(T)` and `set_num_interop_threads(1)` run before any model work. While `MemAvailable` sits inside the reserve, no new shard or command is launched. Threads per worker can change CPU reductions, and so a borderline ML-NEB convergence. They are therefore pinned once per run for each (model, workload) in `numerical_threads/`, and every later stage and resume uses the same count. Only the worker count follows the host. Worker count and placement never change results.

### Outputs and provenance

```text
outputs/production/<run_id>/
  run_identity.json                 scientific identity (defines run_id)
  <stage>/stage_completion.json     outputs + SHA-256, dependency records, allocation
  numerical_threads/*.json          run-level pinned threads per model/workload
  preflight/attempts/*.json         every preflight report
  freeze/internal_freeze_binding.json
  estimate/study1_softening_atlas_internal_freeze_config.json
  logs/<stage>/*.log                combined output of every child command
  invocations/*.json                host, resources, command, per-stage outcome
```

`run_id` hashes the Git commit and tree, every bound configuration, the model registry, the environment locks, the G0 manifest, the embedding evidence pins, the frozen ML-NEB settings and the project environment's installed distributions. Resource settings, paths and stage prose are recorded in `invocations/` but excluded from `run_id`. Pre-freeze evidence files are named by their content identity (`<name>-<fingerprint>.json`), so a code change writes new evidence and never collides with old, create-once evidence. Study-1 artifacts stay in their canonical root (`outputs/study1_softening_atlas/`, so the freeze builder's outcome-access scan still sees them), and the estimate stage records their manifests and SHA-256. Evidence produced before the freeze lives under `outputs/{ml_neb_runtime_audit,production_resource_profiles,gate1_linkage}/`.

### Resume, reruns and verification

- After an interruption, rerun the same command with `--resume`. Completed stages are reused only after their run ID, stage definition, dependency records and every output SHA-256 verify. Inside a stage, completed shards are reused through `e3sse.execution`. A stale or corrupt record is refused, never overwritten. Without `--resume`, an existing run directory is refused.
- To rerun one stage, use `--stage <name> --resume`. A changed commit or configuration gives a new `run_id`, so old results are never mixed in.
- Verify Study-1 artifacts with `python scripts/run_study1_softening_atlas.py --config outputs/production/<run_id>/estimate/study1_softening_atlas_internal_freeze_config.json verify --rehash-source`. Figures are never overwritten. To rebuild them, use `scripts/plot_study1_figures.py --analysis-dir <analysis>` into a fresh analysis directory.

### Exit codes and failure classes

| Exit | Meaning |
|---|---|
| 0 | every requested stage completed or was verified |
| 2 | blocked: `blocked_missing_evidence`, `blocked_missing_runner`, `deferred` or `blocked_by_dependency` |
| 3 | `validation_failure`: evidence contradicts a requirement (e.g. a hash mismatch, two ASE versions, a refused freeze build, or a protocol edit after the freeze) |
| 4 | `infrastructure_failure`: a child command failed, a worker died or memory stayed inside the reserve; rerun the same command with `--resume` |
| 1 | usage, configuration or stale-record refusal |

Unit-level scientific failures are recorded statuses inside results, never stage failures, and are never retried: `optimizer_non_convergence`, `calculator_exception`, `non_finite_value` and `geometry_validity` for ML-NEB. Support abstention (a converged band outside support) is a separate status.

### ML-NEB

`e3sse.barriers.ml_neb` reads every setting from `amendments.prospective.ml_neb.frozen_numeric_settings`. Each setting is passed explicitly: `method="improvedtangent"`, climbing image, k = 5.0, FIRE dt/maxstep/dtmax/Nmin/finc/fdec/astart/fa/a/downhill_check, fmax 0.1, 100 steps, fixed cell, one calculator per image, no image parallelism. ASE's NEB docstring marks the default change to `improvedtangent` as `versionchanged:: 3.27.1`, but no 3.27.1 release exists; 3.27.0 still defaults to `aseneb` and 3.28.0 is the first release with the new default. No ASE default may define the method. Per-runtime ASE versions (3.22.1, 3.24.0, ...) are retained by owner decision; preflight requires every runtime to solve the EMT reference identically. Equivalence evidence: on ASE 3.22.1, 3.24.0 and 3.29.0 the engine takes 23 band steps on the EMT reference, and the barriers agree to 1.2×10⁻¹³ eV. Every record carries a provenance block (model, weights, environment lock, dtype, device, threads). A PyTorch allocator out-of-memory error is re-raised as infrastructure, never recorded as a model failure. A malformed input path is an `input_path` (initialisation) failure, identical for every model. Success requires max |force component| < 0.1 eV/Å on the projected band and both endpoints, finite values, and the frozen geometry checks. The barrier is max − min of the band energies.
