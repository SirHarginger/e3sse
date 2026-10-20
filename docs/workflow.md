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
Before the final materialization on the server, at the reviewed commit:

1. Collect read-only source-unit evidence (headers for every split; numeric scale
   checks on a few published-train frames only):

   ```bash
   PYTHONPATH=src python scripts/collect_unit_evidence.py \
     --config configs/protocol_freeze.json --output outputs/unit_evidence --resume
   ```

   Record the report path and SHA-256, and the observed extxyz keys, in
   `unit_contract.<quantity>.production_source_unit_evidence` for position, cell,
   dft_energy, dft_force, stress and barrier (barrier = max - min of DFT image
   energies, LiTraj `litraj.utils.get_barrier`), and resolve
   `nebdft2k_convergence_flag_availability` from `index_columns`.
2. Regenerate the model smoke reports with each pinned model environment
   (`chgnet==0.3.0`; `matgl==1.2.7` + `dgl==2.1.0` + `torchdata==0.7.1`;
   `mace-torch==0.3.6`; `sevenn==0.10.1`) and checkpoints matching the registry
   SHA-256 values:

   ```bash
   PYTHONPATH=src <env>/bin/python scripts/smoke_test_models.py \
     --config configs/protocol_freeze.json --model <canonical_name> \
     --checkpoint <path> --output outputs/model_smoke --resume
   ```

3. Pin `inputs.*.sha256` from the production artifacts.  Registration is not a
   freeze input: materialize the frozen protocol first, submit that exact
   `protocol.json` to OSF, then record the registration with
   `scripts/record_registration.py`.  The registration addendum binds the frozen
   protocol ID, `protocol.json` SHA-256 and Git SHA, leaves the protocol ID
   unchanged, and is required before any outcome is unsealed, training included
   (`protocol_frozen_registration_pending` -> `protocol_registered`).  After the
   training-only pilot, deposit the pilot amendment as an OSF update and record it
   with `--pilot-amendment`; test-side outcomes require that registered update.
