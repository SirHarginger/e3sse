# E3-SSE Development Workflow

The project uses the same repository name everywhere:

- Local VS Code workspace: `e3sse`
- GitHub: `github.com/SirHarginger/e3sse`
- Server: `/srv/ben/e3sse`

## Workflow

1. Develop and test locally in VS Code.
2. Commit work to a feature branch.
3. Push to GitHub.
4. Merge approved work into `main`.
5. Pull `main` on the server.
6. Run full-data calculations on the server.
7. Keep raw data, logs, scratch files and generated outputs outside Git.

## Gate G0 on the server

After pushing and merging the validated commit, run:

```bash
cd /srv/ben/e3sse
git status
git pull --ff-only
git rev-parse HEAD
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src python scripts/check_g0.py --config configs/server.json --workers 8 --resume
```

`--workers` (default: `audit.workers` in the JSON config; `1` = serial) sets a
bounded process pool. The work units are each non-MPLiTrj dataset, each MPLiTrj
source file (streamed; one file per worker), and then the MPLiTrj provenance
sample. Only the parent writes checkpoints (`datasets/`, `checkpoints/MPLiTrj/`,
`linkage/`) and the canonical `audit.json`, each atomically. The worker count is
operational: it does not change the run ID, the checkpoints, or any scientific
output, so serial and parallel runs are interchangeable under `--resume`. If a
unit crashes, completed checkpoints are kept, `failures.json` is written, any
earlier `audit.json` for the run is moved to `audit.previous.json`, and the
command exits non-zero. The server has 80 cores
but about 12 GiB of free RAM: start at 8 workers, and try 12 only if memory stays
stable. MPLiTrj runtime is bounded by its largest file
(`execution.longest_unit` in the report).

The command reads `/srv/ben/e3sse/data/raw` and, read-only,
`/srv/ben/e3sse/data/downloads/MPLiTrj_raw.zip`, and writes only below the
configured `outputs_root`. It requires `ase` in the environment. Re-running with `--resume` reuses completed
per-dataset checkpoints only when their input fingerprint matches.

## MPLiTrj frame-to-hop provenance index

G0 reports `same_hop_exclusion_supported = true` only against a valid, complete
index built by:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src python scripts/build_mplitrj_provenance.py \
  --config configs/server.json --workers 8 --resume
```

The build reads `data/raw/MPLiTrj/*.xyz` and `data/downloads/MPLiTrj_raw.zip`
read-only and writes under `outputs/provenance/mplitrj/<index_id>/`:

- `manifest.json`: identities, hashes, counts, ordering evidence, and the nebDFT2k cross-check;
- `edges.json`: frame counts per mapped edge;
- `<split>/material-bucket-XX.json`: one record per flattened frame. `XX` is the first two hex digits of `sha256(material_id)`.

Matching is per material. It requires identical atomic numbers in order, a cell
within 1e-4 Å, and every atom within 1e-4 Å after minimum-image wrapping.
Frame statuses are `mapped_unique`, `mapped_hop_unique`, `ambiguous` (candidates
in more than one hop, so no hop is asserted), `unmapped`, and `error`.

`index_id` covers the input identities, the mapping code and the tolerances.
Changed inputs, code or tolerances make an index stale, and G0 then reverts to
false. G0 also re-verifies every shard's SHA-256.

## Isolated FPMD AiiDA inventory

AiiDA is intentionally absent from the ordinary G0 environment. Create a
dedicated, locked Python 3.12 environment and query the archives first:

```bash
uv venv --python 3.12 ~/.venvs/e3sse-aiida
uv pip sync --python ~/.venvs/e3sse-aiida/bin/python requirements/aiida-archive.lock
source ~/.venvs/e3sse-aiida/bin/activate
PYTHONPATH=src python scripts/audit_fpmd_aiida.py --config configs/server.json --resume
deactivate
```

The command confines `AIIDA_PATH` to the configured scratch root and opens
each source archive read-only. If the legacy schema cannot be opened directly,
only a provenance-recorded migrated copy below scratch is queried. It never
imports into or modifies a user AiiDA profile. The inventory is written below
`outputs/fpmd_aiida/`; ordinary G0 independently re-hashes every configured
source archive before consuming it.

Resume works per unit: per-file scans, the archive hash, and per-bucket mapping.
Checkpoints are keyed by the Git SHA, so a new commit rebuilds the units in place.

Safeguards. These keep `same_hop_exclusion_supported` false rather than
guessing:

- a hop directory containing files other than step, source, target or traj_init files;
- an unparseable edge ID;
- a duplicate archive member;
- a flattened scan error;
- an unreadable raw member (the whole material is marked `error`);
- a material whose raw members exceed `audit.provenance_max_material_raw_bytes` (default 1 GiB, marked `error`).

G0 re-hashes the indexed inputs (`audit.provenance_verify_input_sha256`) on every
run. Content changes that preserve size and mtime therefore still invalidate the
index.

## Protocol freeze: local validation and server materialization

Protocol-freeze code is developed and tested locally with synthetic JSON only.
The canonical artifacts are materialized on the server from the existing G0
audit and MPLiTrj provenance index; neither artifact is regenerated or copied
into Git.

The builder has two production outcomes, both sealed for outcome access:

- `candidate_unfrozen`: every production input was validated and the dataset
  manifest, chemical-system partitions, zero-overlap proofs, and Study 5
  exclusion set were materialized, but `freeze_blockers` lists what is still
  missing (unpinned hashes, unevidenced source units, unregistered scientific
  resolutions, unresolved model checkpoints, open `unresolved_items`).
- `frozen`: zero freeze blockers, a clean worktree at the pinned Git SHA, and
  pinned SHA-256 for the G0 audit, G0 `inputs.json`, server configuration, and
  MPLiTrj provenance manifest.

Outcome access has two seals, mirroring the Research Design (registration, then
a training-only pilot, then test analysis).  Seal 1: every outcome, training
included, requires a validated `frozen` protocol, and training access is
checked against the frozen partition membership rather than a declared role.
Seal 2: non-training outcomes also require a hashed pilot amendment
(`build_pilot_amendment`) that records the training-only thresholds and MDE
analysis and is bound to the `protocol_id`.  Tuning and threshold selection are
training-only.  FPMD outcomes additionally require a hashed transport addendum
(`build_transport_addendum`) declaring the transport branch `ready`; it is kept
outside the protocol hash so resolving the FPMD blockers never changes the
identity of Studies 1, 2a and 4.

Partitions and seeds do not depend on the open items, so a candidate and its
later frozen protocol share identical partition memberships.  Different
content always yields a different content-addressed `protocol_id` directory.

Every consequential rule is listed in the protocol's `decision_register` as
`paper_specified`, `repository_existing`, `engineering_choice`, or
`production_derived`.  Each item in `configs/protocol_freeze.json`
`unresolved_items` carries a `resolution_category` fixed in code
(`PROTOCOL_ITEM_CATEGORIES`): `resolve_from_production_artifact` and
`requires_preanalysis_scientific_decision` block the static freeze;
`resolve_from_paper` items must carry a Research Design citation; `study3_only`
and `optional_later` never block it.  Canonical analysis units are frozen in the
protocol; production source units must still be evidenced in Phase B before any
conversion.  Resolve items only with authoritative values; never edit a checklist
entry to make a freeze pass.

The primary softening eligibility rule (`research_design_3.6_v1`) is
paper-specified: a target hop's NEB-path frames (`band_optim_step`, `traj_init`)
are prohibited and oracle-only, its endpoint-relaxation frames (`source`,
`target`) are admissible, other hops of the same structure are admissible (on
the test side only through non-path frames, section 5.4), and unidentified
provenance fails closed.  The 60/40 repeated-partition fractions count
chemical-system groups.

The v1 provenance index does not carry chemical systems, so the builder performs
one read-only pass over the G0-hashed flattened MPLiTrj sources unless an
explicit hashed `e3sse.material-chemical-system.v1` linkage artifact is supplied.
The builder derives the nebDFT2k partition linkage from only the identifier
columns (`edge_id`, `material_id`, `chemsys`, `_split`) of the unique G0-pinned
CSV index and records the index's column names as convergence-flag evidence.
Outcome columns are neither copied nor used.

After checking out the reviewed commit on the server (clean worktree), run
exactly, substituting that commit's full SHA (a commit cannot contain its own
SHA, so it is pinned on the command line and recorded in `resolved_config.json`):

```bash
cd /srv/ben/e3sse
git checkout <REVIEWED_COMMIT_SHA>
PYTHONPATH=src python scripts/build_protocol_freeze.py \
  --config configs/protocol_freeze.json --git-sha <REVIEWED_COMMIT_SHA> --resume
```

Required checklist items are fixed in code (`REQUIRED_RESOLUTION_ITEMS`): deleting
an entry or setting `blocking: false` without a resolved record and evidence
never clears its blocker.

The builder validates the G0/provenance schemas, internal hashes, current server
configuration, complete frame-to-hop coverage, chemical-system atomicity, and
cross-dataset split propagation.  It writes only to the configured
`outputs/protocol_freeze/<protocol-id>/` directory and refuses to overwrite an
existing run without a valid `--resume`; resume rejects a changed Git SHA, code,
configuration, input hash, or schema.  It does not run G0, MLIP inference, NEB,
MD, softening estimation, or held-out outcome analysis.

### Static freeze: production evidence and model smoke tests

The pre-analysis decisions, model registry and canonical adapters are frozen in
`configs/protocol_freeze.json` and `src/e3sse/protocol/{chemistry,adapters}.py`.
The production evidence below is complete and pinned by SHA-256 in
`inputs.production_evidence`.  Do not regenerate it.  The materializer re-hashes
every pinned file and binds each smoke report to its model's checkpoint SHA-256,
package version and adapter version, a passed result, and a clean tree.

1. Source-unit evidence (`scripts/collect_unit_evidence.py`, read-only):
   `outputs/unit_evidence/c0b395303df2f352/evidence.json`.  It establishes
   positions and cell in angstrom, DFT energy in eV, forces in eV/angstrom, stress
   in eV/angstrom^3, and barrier in eV (max - min of DFT image energies, LiTraj
   `litraj.utils.get_barrier`).  The nebDFT2k index has no convergence flag, so
   `exclude_and_count_when_flag_provided` excludes no hop by flag.
2. Model smoke reports (`scripts/smoke_test_models.py`, synthetic Li2O only):
   `outputs/model_smoke/<canonical_name>.json`, one per frozen model, all passed.
   Each model ran in its own isolated environment pinned to the registry
   `package_version`; environments are never shared across package versions.  The
   lock files in `outputs/model_smoke/environment_locks/` are the complete
   dependency records.  The table lists only the compatibility pins found during
   reconstruction.  The materializer checks each pin against the versions recorded
   in the smoke report, except `torchdata`, which reports do not record; it is held
   in `lock_recorded_pins` and evidenced by the lock file.

   | Environment (`~/.venvs/<name>`) | Python | Pinned packages | Environment | Lock | Frozen models |
   |---|---|---|---|---|---|
   | `e3sse-chgnet-0.3.0` | 3.11.16 | `chgnet==0.3.0`, `ase==3.22.1`, `numpy==1.26.4` | | `chgnet-0.3.0.txt` | `CHGNet-0.3.0` |
   | `e3sse-matgl-1.2.7` | 3.11.16 | `matgl==1.2.7`, `dgl==2.1.0`, `torchdata==0.7.1`, `torch==2.1.1`, `ase==3.24.0`, `numpy==1.26.4` | `DGLBACKEND=pytorch` | `matgl-1.2.7.txt` | `M3GNet-MP-2021.2.8-PES`, `M3GNet-MatPES-PBE-v2025.1-PES` |
   | `e3sse-mace-0.3.6` | 3.11.16 | `mace-torch==0.3.6`, `torch==2.5.1` | | `mace-0.3.6.txt` | `MACE-MP-0-small-2023-12-10` |
   | `e3sse-sevennet-0.10.1` | 3.11.16 | `sevenn==0.10.1` | | `sevennet-0.10.1.txt` | `SevenNet-0-11July2024` |
   | `e3sse-sevennet-0.11.0` | 3.11.16 | `sevenn==0.11.0` | | `sevennet-0.11.0.txt` | `SevenNet-omat` |

   The two SevenNet releases need separate environments.  SevenNet-0 keeps the
   LiTraj benchmark release, while `7net-omat` first ships in sevenn 0.11.0, which
   also changes tensor-product ordering and requires `e3nn>=0.5.0`.  To rebuild an
   environment, install from its lock file rather than from the pins above.
   Production hardware currently has no GPU, so every environment runs on CPU.
   This is an operational fact about the current server, not a scientific
   assumption: the adapters take a `device` argument.

3. The server-only input hashes are pinned: `inputs.g0_audit.sha256`,
   `inputs.g0_input_manifest.sha256` and `inputs.mplitrj_provenance.manifest_sha256`
   were read with `sha256sum` from the existing artifacts, without rerunning
   anything.  `inputs.server_config.sha256` comes from the tracked
   `configs/server.json`.  Materialization re-hashes all four and fails closed on
   any mismatch or missing file.  The reviewed commit is the only trust anchor that
   is not tracked: a commit cannot contain its own SHA, so it is supplied by
   `--git-sha`.

   Registration is not a freeze input: materialize the frozen protocol first,
   submit that exact `protocol.json` to OSF, then record the registration with
   `scripts/record_registration.py`.  The registration addendum binds the frozen
   protocol ID, `protocol.json` SHA-256 and Git SHA, leaves the protocol ID
   unchanged, and is required before any outcome is unsealed, training included
   (`protocol_frozen_registration_pending` -> `protocol_registered`).  After the
   training-only pilot, deposit the pilot amendment as an OSF update and record it
   with `--pilot-amendment`; test-side outcomes require that registered update.

## Production parallel execution

Studies 1, 2a and 4 run independent work units (frames or hops per model,
partitions, bootstrap and label-budget replicates, conformal Monte Carlo draws)
through `src/e3sse/execution/` (skill: `parallel-execution`).  Execution settings
are operational metadata: they are written to per-invocation JSON manifests under
`<run>/execution_manifests/` and never enter the protocol, shard fingerprints or
seeds.  The one exception is the per-process numerical thread configuration.  CPU
floating-point reductions can depend on it, so it enters shard fingerprints, and a
production run fixes it.

- Workers: `--workers N` or `--workers auto`.  There is no project default above 1,
  because the right count differs by model and workload.  `auto` and explicit counts
  both satisfy `workers x threads_per_worker <= usable CPUs` (affinity mask) and
  `workers x measured per-worker peak <= MemAvailable - reserve` (default reserve
  2 GiB, `--memory-reserve-gib`).  Swap is never counted as capacity.  `auto`
  refuses to run without a measured per-worker memory instead of guessing.  Workers
  are capped at the number of units.
- Per-worker memory: measure once per model, thread count and largest frame size,
  in the model's environment.  This is operational and not a smoke-test rerun:

  ```bash
  TMPDIR=/srv/ben/e3sse/scratch PYTHONPATH=src ~/.venvs/<env>/bin/python \
    scripts/probe_model_worker_memory.py --config configs/protocol_freeze.json \
    --model <canonical_name> --checkpoint <path> --min-atoms <largest frame atoms> \
    --threads-per-worker <t> --output outputs/execution_resources \
    --protected-root /srv/ben/e3sse/data --resume
  ```

  Take `--min-atoms` from atom counts in the frozen input manifest, never from DFT
  labels.  Planning uses 1.5x the measured VmHWM.
- Nested threading: the parent sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to
  `threads_per_worker` before starting a `spawn` pool, because OpenMP/BLAS
  runtimes, including the one DGL uses, read them at load time.  The worker
  initializer then fixes PyTorch inter-op threads to 1 (`set_num_interop_threads`
  works only once and before any inter-op work), sets intra-op threads to
  `threads_per_worker`, and verifies both before loading the model once per worker.
  `fork` is never used with initialized torch or OpenMP state.
- MLIP versus statistics: MLIP inference is RAM-bound on the current server
  (80 CPUs, about 19 GiB RAM).  Expect few workers, each with one or a few threads,
  sized by the probe.  Bootstrap and Monte Carlo workloads use one thread per worker
  and a small footprint, so they can use more CPUs under their own policy.
- NUMA: no binding.  The kernel's default local allocation, with fallback to the
  nearer node, keeps workers working when one roughly 9.5 GiB node fills.  Hard
  binding could kill a worker for lack of memory on one node while the other node
  still has free memory.  The topology is recorded in each manifest.
- Granularity: a shard is a whole scientific unit.  One frozen RNG stream (for
  example one bootstrap analysis) never spans shards, and a shard never spans
  partitions, chemical systems, or the calibration and test sides.  Reducers order
  results by scientific keys, never by shard ID.
- Determinism: shard IDs are content hashes of canonical identifiers.  Seeds come
  only from `analysis_seed` over canonical identifiers and replicate indices.
  Results are merged in shard-ID order, so worker count, completion order, PID and
  scheduling cannot change any checkpoint byte or the merged SHA-256.
- Resume: the run directory binds the run identity and the shard set, and one
  orchestrator holds it at a time (`.run.lock`).  Verified shards are reused.  Missing shards are rerun.  A stale shard
  fingerprint, a changed run identity (code, config, model, environment lock,
  inputs) or a corrupt checkpoint stops the run.  Only the parent publishes, each
  file once (create-new hard link), under the output root.  Outputs may not overlap
  the protected data roots.  A failed or killed worker never produces a result;
  completed siblings are kept and the run exits non-zero without a merged digest.
- Server notes: set `TMPDIR` to the ext4 output disk (`/tmp` is about 9.6 GiB of
  RAM-backed tmpfs).  The parent holds one checkpoint file open at a time, well
  under the 1024 open-file limit.  There is no scheduler, so other load can change
  `MemAvailable` during a run: re-plan before each production invocation.

Boundary: Study-specific runners (frame selection, inference record schema,
`e3sse.mlip.inference.run.v1` manifest, batching) belong to each study's
implementation.  They supply `scientific_identity` (Git SHA and code hash, config
SHA-256, model identity, environment-lock SHA-256, input manifest SHA-256), shards
and a task.  This layer supplies bounded, resumable, deterministic execution and
the operational record.

Design sources: Python `concurrent.futures` and `multiprocessing` (start methods,
`BrokenProcessPool`; `Executor.map` only bounds its buffer from 3.14, so submission
uses an explicit sliding window); PyTorch multiprocessing notes (per-process
threads at most floor(N/M); avoid fork after runtime initialization) and
`torch.set_num_interop_threads`; NumPy parallel random generation (seeds keyed by
unique, deterministically created IDs, never `seed + worker_id`) and NumPy BLAS
thread control; DGL CPU best practices (OpenMP threads); Linux NUMA memory policy
(local allocation with distance-ordered fallback); `/proc/meminfo` MemAvailable and
`/proc/<pid>/status` VmHWM.
