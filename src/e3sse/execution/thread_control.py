"""Explicit per-process numerical threading so workers x threads never oversubscribes.

OpenMP/BLAS runtimes (OpenBLAS, MKL, the OpenMP runtime DGL uses) read their thread
counts when first loaded, so the variables are placed in the parent environment
before a spawn pool starts and are inherited by each fresh interpreter.  PyTorch's
intra-op pool is set with ``torch.set_num_threads``; its inter-op pool can be set
only once and before any inter-op work, so it is fixed to 1 in the worker
initializer before any model loads.  This module imports neither numpy nor torch at
import time.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
TORCH_INTEROP_THREADS = 1


class ThreadControlError(RuntimeError):
    """The requested thread configuration could not be enforced."""


def numerical_thread_identity(threads_per_worker: int, *, configure_torch: bool) -> dict[str, Any]:
    """Thread settings that can change CPU floating-point reductions.

    This enters shard fingerprints: results computed with a different per-process
    thread configuration are not reused or merged.  Worker count does not enter.
    """

    return {
        "environment": {name: str(threads_per_worker) for name in THREAD_ENVIRONMENT_VARIABLES},
        "torch_intra_op_threads": threads_per_worker if configure_torch else None,
        "torch_inter_op_threads": TORCH_INTEROP_THREADS if configure_torch else None,
    }


@contextmanager
def child_thread_environment(threads_per_worker: int) -> Iterator[dict[str, str]]:
    """Set thread variables for processes started inside the block, then restore."""

    previous = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES}
    values = {name: str(threads_per_worker) for name in THREAD_ENVIRONMENT_VARIABLES}
    os.environ.update(values)
    try:
        yield values
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def configure_process_threads(threads_per_worker: int, *, configure_torch: bool) -> dict[str, Any]:
    """Apply and verify the thread configuration inside the current process."""

    observed: dict[str, Any] = {
        "environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES}
    }
    expected = str(threads_per_worker)
    if any(value != expected for value in observed["environment"].values()):
        raise ThreadControlError(
            "thread environment was not set before process start: "
            f"{observed['environment']} (expected {expected})"
        )
    if configure_torch:
        import torch

        try:
            torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
        except RuntimeError:
            # Already fixed (e.g. serial in-process reuse); accept only the same value.
            pass
        torch.set_num_threads(threads_per_worker)
        intra, inter = torch.get_num_threads(), torch.get_num_interop_threads()
        if intra != threads_per_worker or inter != TORCH_INTEROP_THREADS:
            raise ThreadControlError(
                f"torch threads intra={intra} inter={inter}; expected "
                f"{threads_per_worker}/{TORCH_INTEROP_THREADS}"
            )
        observed["torch_intra_op_threads"] = intra
        observed["torch_inter_op_threads"] = inter
    return observed
