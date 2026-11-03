"""Explicit production stage graph with code-enforced leakage and ordering rules.

Stages and their dependencies come from ``configs/production_execution.json``.  The
code fixes what a configuration may not change:

* ``PERMITTED_OUTCOME_ACCESS``: the largest outcome set each stage may read.  A
  configuration can only narrow it.  The orchestrator hands the stage's set to every
  child command (``E3SSE_STAGE_OUTCOME_ACCESS``), and
  ``e3sse.protocol.contracts.require_outcome_access`` refuses any other outcome kind;
* ``PRE_FREEZE_STAGES``: outcome-free evidence stages, each an ancestor of
  ``freeze``.  Every other stage descends from ``freeze``;
* ``REQUIRED_PRECEDENCE``: orderings from the protocol's ``freeze_order``, e.g. model
  selection before any conformal-calibration label is read, and calibration and
  pool ML-NEB before weighted selection;
* no ancestor of ``model_selection`` reads a label of the conformal-calibration
  systems (``CALIBRATION_LABEL_KINDS``), and every reader descends from it.

Stage status is declared honestly: ``implemented`` (an adapter drives tested
runners), ``missing_runner`` (blocks with a precise ``next_task``; dependents are
skipped), ``deferred`` (excluded by the active protocol; never run).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FREEZE_STAGE = "freeze"
MODEL_SELECTION_STAGE = "model_selection"
ALL_STAGES = "all"
IMPLEMENTATION_STATUSES = ("implemented", "missing_runner", "deferred")

TRAIN_FORCES = "mplitrj_published_train_dft_forces_energies"
TEST_SIDE_NONPATH_FORCES = "mplitrj_test_side_nonpath_forces"
VALIDATION_FORCES = "mplitrj_validation_frame_forces"  # validation systems are Study-4 units
OWN_PATH_FORCES_ORACLE = "mplitrj_own_path_forces_oracle"
FAMILY_SCALE_FORCES = "mplitrj_train_systems_without_nebdft2k_forces"
EVALUATION_BARRIERS = "nebdft2k_fixed_geometry_evaluation_barriers"
DEVELOPMENT_BARRIERS = "nebdft2k_development_barriers"
CALIBRATION_BARRIERS = "nebdft2k_conformal_calibration_barriers"
REPEATED_PARTITION_BARRIERS = "nebdft2k_repeated_partition_barriers"
CALIBRATION_LABEL_KINDS = frozenset({CALIBRATION_BARRIERS, REPEATED_PARTITION_BARRIERS})

PRE_FREEZE_STAGES = frozenset({
    "runtime_audit", "resource_profile", "gate1_population_audit", "bvse_reproduction",
    "descriptor_availability", "development_population", "development_ml_neb",
    "path_representation", "preflight",
})

# The frozen leakage map.  The prospective pool has no label at all; development
# barriers belong to model selection (and are Study-2a published-test hops, a
# declared reuse); conformal-calibration labels are read only after model selection.
PERMITTED_OUTCOME_ACCESS: dict[str, frozenset[str]] = {
    **{name: frozenset() for name in PRE_FREEZE_STAGES},
    "freeze": frozenset(),
    "estimate": frozenset({TRAIN_FORCES}),
    # Seal 2: training-only pilot over Study-1 force-derived estimates (no barrier).
    "pilot": frozenset({TRAIN_FORCES}),
    "family_scale": frozenset({FAMILY_SCALE_FORCES}),
    "transfer": frozenset({TRAIN_FORCES, TEST_SIDE_NONPATH_FORCES, OWN_PATH_FORCES_ORACLE,
                           EVALUATION_BARRIERS}),
    "reliability": frozenset({EVALUATION_BARRIERS}),
    "weight_slice_ml_neb": frozenset(),
    "model_selection": frozenset({DEVELOPMENT_BARRIERS}),
    "deployment_calibration": frozenset(),
    "linear_recalibration_comparator": frozenset({CALIBRATION_BARRIERS, EVALUATION_BARRIERS}),
    "controlled_selection": frozenset({TRAIN_FORCES, TEST_SIDE_NONPATH_FORCES,
                                       VALIDATION_FORCES, REPEATED_PARTITION_BARRIERS}),
    "density_ratio": frozenset(),
    "discovery": frozenset(),
    "discovery_selection": frozenset({CALIBRATION_BARRIERS}),
    "publication": frozenset(),
}

# (earlier, later): when both stages exist, earlier must be an ancestor of later.
REQUIRED_PRECEDENCE: tuple[tuple[str, str], ...] = (
    ("estimate", "pilot"),
    ("pilot", "transfer"),
    ("estimate", "transfer"),
    ("transfer", "reliability"),
    ("transfer", "model_selection"),
    ("reliability", "model_selection"),
    ("family_scale", "model_selection"),
    ("weight_slice_ml_neb", "model_selection"),
    ("model_selection", "deployment_calibration"),
    ("model_selection", "discovery"),
    ("model_selection", "controlled_selection"),
    ("model_selection", "linear_recalibration_comparator"),
    ("deployment_calibration", "density_ratio"),
    ("discovery", "density_ratio"),
    ("weight_slice_ml_neb", "density_ratio"),
    ("deployment_calibration", "discovery_selection"),
    ("discovery", "discovery_selection"),
    ("density_ratio", "discovery_selection"),
    ("controlled_selection", "discovery_selection"),
)


class StageGraphError(ValueError):
    """The stage graph is malformed or would weaken a leakage or ordering boundary."""


@dataclass(frozen=True)
class ProductionStage:
    name: str
    public_name: str
    depends_on: tuple[str, ...]
    adapter: str
    implementation_status: str
    outcome_access: tuple[str, ...]
    workloads: tuple[tuple[str, str], ...]
    next_task: str | None
    description: str

    @property
    def phase(self) -> str:
        return ("pre_freeze" if self.name in PRE_FREEZE_STAGES
                else "freeze" if self.name == FREEZE_STAGE else "post_freeze")

    def as_json(self) -> dict[str, Any]:
        return {"name": self.name, "public_name": self.public_name, "phase": self.phase,
                "depends_on": list(self.depends_on), "adapter": self.adapter,
                "implementation_status": self.implementation_status,
                "outcome_access": list(self.outcome_access),
                "workloads": [list(item) for item in self.workloads],
                "next_task": self.next_task, "description": self.description}

    def definition(self) -> dict[str, Any]:
        """What the stage does; prose (names, descriptions, next tasks) is excluded."""

        return {"name": self.name, "depends_on": sorted(self.depends_on),
                "adapter": self.adapter, "implementation_status": self.implementation_status,
                "outcome_access": list(self.outcome_access),
                "workloads": [list(item) for item in self.workloads]}


@dataclass(frozen=True)
class StageGraph:
    stages: dict[str, ProductionStage]
    order: tuple[str, ...]

    def resolve(self, requested: str) -> list[str]:
        """``all`` or one stage plus its transitive dependencies, in topological order."""

        if requested == ALL_STAGES:
            return list(self.order)
        if requested not in self.stages:
            raise StageGraphError(f"unknown stage {requested!r}; choose one of "
                                  f"{[ALL_STAGES, *self.order]}")
        needed = self.ancestors(requested) | {requested}
        return [name for name in self.order if name in needed]

    def ancestors(self, name: str) -> set[str]:
        found: set[str] = set()
        frontier = list(self.stages[name].depends_on)
        while frontier:
            current = frontier.pop()
            if current not in found:
                found.add(current)
                frontier.extend(self.stages[current].depends_on)
        return found


def _topological(stages: Mapping[str, ProductionStage], declared: Sequence[str]) -> tuple[str, ...]:
    """Kahn's algorithm; ties keep declaration order, so the order is deterministic."""

    remaining = {name: set(stage.depends_on) for name, stage in stages.items()}
    order: list[str] = []
    while remaining:
        ready = [name for name in declared if name in remaining and not remaining[name]]
        if not ready:
            raise StageGraphError(f"stage dependency cycle among {sorted(remaining)}")
        for name in ready:
            order.append(name)
            del remaining[name]
        for deps in remaining.values():
            deps.difference_update(ready)
    return tuple(order)


def _validate_order(graph: StageGraph) -> None:
    stages = graph.stages
    freeze_ancestors = graph.ancestors(FREEZE_STAGE)
    for name, stage in stages.items():
        if name in PRE_FREEZE_STAGES:
            if name not in freeze_ancestors:
                raise StageGraphError(f"pre-freeze stage {name!r} must be an ancestor of freeze")
            if stage.outcome_access:
                raise StageGraphError(f"pre-freeze stage {name!r} may not read outcomes")
        elif name != FREEZE_STAGE and FREEZE_STAGE not in graph.ancestors(name):
            raise StageGraphError(f"stage {name!r} must depend on the freeze")
    if stages[FREEZE_STAGE].outcome_access:
        raise StageGraphError("the freeze reads no outcome")
    for earlier, later in REQUIRED_PRECEDENCE:
        if earlier in stages and later in stages and earlier not in graph.ancestors(later):
            raise StageGraphError(f"{earlier!r} must precede {later!r} (protocol freeze order)")
    readers = {n for n, s in stages.items() if CALIBRATION_LABEL_KINDS & set(s.outcome_access)}
    if readers:
        if MODEL_SELECTION_STAGE not in stages:
            raise StageGraphError("calibration labels need a model_selection stage before them")
        selection_ancestors = graph.ancestors(MODEL_SELECTION_STAGE) | {MODEL_SELECTION_STAGE}
        for name in readers:
            if name in selection_ancestors or MODEL_SELECTION_STAGE not in graph.ancestors(name):
                raise StageGraphError(f"stage {name!r} reads calibration labels and must follow "
                                      "model_selection")


def build_stage_graph(entries: Iterable[Mapping[str, Any]], *,
                      known_adapters: Iterable[str]) -> StageGraph:
    adapters = set(known_adapters)
    stages: dict[str, ProductionStage] = {}
    declared: list[str] = []
    for entry in entries:
        name = entry["name"]
        if name in stages:
            raise StageGraphError(f"duplicate stage {name!r}")
        if name not in PERMITTED_OUTCOME_ACCESS:
            raise StageGraphError(f"stage {name!r} has no frozen outcome-access boundary")
        status = entry["implementation_status"]
        if status not in IMPLEMENTATION_STATUSES:
            raise StageGraphError(f"stage {name!r}: unknown status {status!r}")
        if entry["adapter"] not in adapters:
            raise StageGraphError(f"stage {name!r}: unknown adapter {entry['adapter']!r}")
        if status != "implemented" and not entry.get("next_task"):
            raise StageGraphError(f"stage {name!r} is {status} and must state its next_task")
        access = tuple(sorted(entry.get("outcome_access", [])))
        widened = set(access) - PERMITTED_OUTCOME_ACCESS[name]
        if widened:
            raise StageGraphError(f"stage {name!r} may not read {sorted(widened)}")
        stages[name] = ProductionStage(
            name=name, public_name=entry["public_name"],
            depends_on=tuple(entry.get("depends_on", [])), adapter=entry["adapter"],
            implementation_status=status, outcome_access=access,
            workloads=tuple((str(model), str(workload))
                            for model, workload in entry.get("workloads", [])),
            next_task=entry.get("next_task"), description=entry.get("description", ""),
        )
        declared.append(name)
    for stage in stages.values():
        unknown = [dep for dep in stage.depends_on if dep not in stages]
        if unknown:
            raise StageGraphError(f"stage {stage.name!r} depends on unknown {unknown}")
    if FREEZE_STAGE not in stages:
        raise StageGraphError("the production graph needs a freeze stage")
    graph = StageGraph(stages=stages, order=_topological(stages, declared))
    _validate_order(graph)
    return graph
