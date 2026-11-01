"""Top-level (picklable) synthetic tasks for the parallel execution tests; no outcomes."""

from __future__ import annotations

import os
import time

import numpy as np

from e3sse.execution.thread_control import THREAD_ENVIRONMENT_VARIABLES
from e3sse.protocol.partitions import analysis_seed

BASE_SEED = 2026


def bootstrap_mean_task(payload: dict, context: object) -> dict:
    """Synthetic cluster bootstrap whose draws depend only on canonical identifiers."""

    seed = analysis_seed(BASE_SEED, "cluster_bootstrap", payload["unit_id"], payload["replicate"])
    rng = np.random.Generator(np.random.PCG64DXSM(seed))  # the frozen generator
    values = np.arange(1.0, payload["n_values"] + 1.0)
    draws = rng.choice(values, size=(payload["resamples"], values.size), replace=True)
    # Scramble completion order so canonical merge is exercised, not assumed.
    time.sleep(((seed % 7) + 1) * 0.002)
    return {"unit_id": payload["unit_id"], "replicate": payload["replicate"],
            "seed": seed, "bootstrap_means": draws.mean(axis=1).round(12).tolist()}


def failing_bootstrap_task(payload: dict, context: object) -> dict:
    if payload["unit_id"] == "unit-02":
        raise RuntimeError("synthetic shard failure")
    return bootstrap_mean_task(payload, context)


def crashing_bootstrap_task(payload: dict, context: object) -> dict:
    if payload["unit_id"] == "unit-02":
        os._exit(1)  # simulate an out-of-memory kill of the worker process
    return bootstrap_mean_task(payload, context)


def thread_environment_task(payload: dict, context: object) -> dict:
    observed = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES}
    try:
        import torch
    except ImportError:
        torch_threads = None
    else:
        torch_threads = [torch.get_num_threads(), torch.get_num_interop_threads()]
    return {"environment": observed, "torch_threads": torch_threads}


def load_synthetic_model(scale: float) -> dict:
    """Stands in for loading a model once per worker."""

    return {"scale": scale, "loaded_in_pid": os.getpid()}


def scaled_energy_task(payload: dict, context: dict) -> dict:
    return {"frame_id": payload["frame_id"], "energy": payload["n_atoms"] * context["scale"]}


def global_random_task(payload: dict, context: object) -> dict:
    """Discouraged global-RNG use; must still be independent of scheduling."""

    return {"unit_id": payload["unit_id"], "draw": float(np.random.random())}
