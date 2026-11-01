---
name: parallel-execution
description: "Run or implement E3-SSE production work over independent units (models, frames, hops, partitions, bootstrap or Monte Carlo replicates) with bounded, memory-aware, deterministic and resumable parallelism via e3sse.execution."
---

# Parallel Execution

Use `src/e3sse/execution/` for any production path over independent units. Do not
write new multiprocessing boilerplate. G0 and the MPLiTrj provenance build keep their
own already-executed pools because their run identities cover their code.

## Contract

1. Shards come from `make_shard(namespace, identity, payload)`. The identity contains
   only canonical scientific identifiers: never worker IDs, chunk numbers or queue
   positions. Shard membership and any batch size come from the protocol/config, never
   from the worker count.
2. `ShardedRun.scientific_identity` binds code, config, model (`weights_sha256`,
   package version, adapter version), environment-lock SHA-256 and input hashes. If
   any of these changes, existing checkpoints become stale and are refused.
3. Tasks are top-level functions `task(payload, context)` that return strict JSON. A
   model is loaded once per worker with `worker_setup`.
4. Randomness comes from `e3sse.protocol.partitions.analysis_seed` over canonical
   identifiers and replicate indices. Never use the PID, worker, order or time.
5. Scientific failures (non-finite prediction, model error on a frame) are record
   statuses inside the result. Infrastructure failures (exceptions, dead workers) are
   never saved as results. A run with any missing shard has no merged digest.
6. Pass immutable data roots as `protected_roots`. Outputs go under `outputs/` only.
7. A shard is a whole scientific unit. Never split one frozen RNG stream (for
   example the resamples of one bootstrap analysis) across shards. Never let a shard
   span partitions, chemical systems, or the calibration and test sides.
8. Results come back in shard-ID (hash) order. Reducers that sum or stack results
   order them by scientific keys, as the protocol specifies.
9. The run directory binds the run identity and the shard set. Only one
   orchestrator may hold a run directory at a time (`.run.lock`).

## Choosing workers

- MLIP inference is memory-bound on the production host. Measure first with
  `scripts/probe_model_worker_memory.py` (synthetic, operational). Then use
  `--workers auto --worker-memory-report <report>`, or an explicit `--workers N`,
  which is refused if it exceeds the CPU or memory envelope.
- Statistical workloads (bootstrap, conformal Monte Carlo) use their own policy:
  1 thread per worker and a small memory footprint, so they can run more workers.
- Keep `threads_per_worker` fixed for a production run. It enters shard
  fingerprints because CPU reductions depend on it. Worker count does not.

## Before a production run

- Clean worktree at the reviewed commit.
- Set `TMPDIR` to the ext4 output disk, because `/tmp` is RAM-backed tmpfs.
- Keep `--workers` well below the 1024 open-file limit.
- Record the execution manifest path with the run's provenance (`experiment-run`).
