"""Deterministic, resumable, resource-bounded execution of independent work units.

Execution settings (worker count, in-flight bound, memory reserve, NUMA layout) are
operational and never enter a scientific fingerprint, seed, or protocol.  The only
execution setting that does enter shard identity is the per-process numerical thread
configuration, because CPU floating-point reductions can depend on it.
"""

from .resource_planning import (
    ExecutionPlan,
    HostResources,
    InsufficientResourcesError,
    WorkloadPolicy,
    observe_host_resources,
    plan_workers,
)
from .sharded_execution import (
    ShardedExecutionError,
    ShardedRun,
    StaleCheckpointError,
    WorkShard,
    execute_shards,
    make_shard,
)

__all__ = [
    "ExecutionPlan",
    "HostResources",
    "InsufficientResourcesError",
    "ShardedExecutionError",
    "ShardedRun",
    "StaleCheckpointError",
    "WorkShard",
    "WorkloadPolicy",
    "execute_shards",
    "make_shard",
    "observe_host_resources",
    "plan_workers",
]
