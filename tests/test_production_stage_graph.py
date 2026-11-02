"""Stage graph: resolution, cycles, and the enforced outcome-access boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from e3sse.production.model_runtimes import (
    RuntimeRoutingError,
    assert_no_package_installation,
    model_runtimes,
    require_present,
    runtime_command,
    runtime_environment,
)
from e3sse.production.orchestrator import DEFAULT_ADAPTERS
from e3sse.production.stage_graph import StageGraphError, build_stage_graph

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPOSITORY / "configs" / "production_execution.json").read_text())
FREEZE = json.loads((REPOSITORY / "configs" / "protocol_freeze.json").read_text())


def _entry(name, depends_on=(), outcome_access=(), status="implemented", adapter="preflight"):
    return {"name": name, "public_name": name, "adapter": adapter,
            "implementation_status": status, "depends_on": list(depends_on),
            "outcome_access": list(outcome_access), "next_task": None if status ==
            "implemented" else "task"}


def test_checked_in_graph_resolves_with_freeze_before_every_outcome():
    graph = build_stage_graph(CONFIG["stages"], known_adapters=DEFAULT_ADAPTERS)
    order = graph.order
    assert order.index("preflight") < order.index("freeze") < order.index("estimate")
    for name, stage in graph.stages.items():
        if stage.outcome_access:
            assert "freeze" in graph.ancestors(name)
    assert graph.resolve("estimate")[-1] == "estimate"
    assert "transfer" not in graph.resolve("estimate")
    assert graph.resolve("all") == list(order)
    pre_freeze = graph.ancestors("freeze")
    assert {"runtime_audit", "resource_profile", "gate1_population_audit"} <= pre_freeze
    missing = {n for n, s in graph.stages.items() if s.implementation_status == "missing_runner"}
    assert all(graph.stages[n].next_task for n in missing)


def test_invalid_stage_name():
    graph = build_stage_graph(CONFIG["stages"], known_adapters=DEFAULT_ADAPTERS)
    with pytest.raises(StageGraphError, match="unknown stage"):
        graph.resolve("study_1")


def test_cycle_and_unknown_dependency_are_refused():
    with pytest.raises(StageGraphError, match="cycle"):
        build_stage_graph([_entry("preflight", ["freeze"]), _entry("freeze", ["preflight"],
                                                                     adapter="freeze")],
                          known_adapters=DEFAULT_ADAPTERS)
    with pytest.raises(StageGraphError, match="unknown"):
        build_stage_graph([_entry("freeze", ["nonexistent"], adapter="freeze")],
                          known_adapters=DEFAULT_ADAPTERS)


def test_outcome_access_cannot_be_widened_by_config():
    entries = [_entry("freeze", adapter="freeze"),
               _entry("discovery", ["freeze"], ["nebdft2k_development_barriers"],
                      status="missing_runner", adapter="missing_runner")]
    with pytest.raises(StageGraphError, match="may not read"):
        build_stage_graph(entries, known_adapters=DEFAULT_ADAPTERS)


def test_outcome_stage_must_follow_the_freeze():
    entries = [_entry("freeze", adapter="freeze"),
               _entry("estimate", [], ["mplitrj_published_train_dft_forces_energies"],
                      adapter="estimate")]
    with pytest.raises(StageGraphError, match="must depend on the freeze"):
        build_stage_graph(entries, known_adapters=DEFAULT_ADAPTERS)


def test_missing_runner_must_name_its_next_task():
    entry = _entry("transfer", status="missing_runner", adapter="missing_runner")
    entry["next_task"] = None
    with pytest.raises(StageGraphError, match="next_task"):
        build_stage_graph([_entry("freeze", adapter="freeze"), entry],
                          known_adapters=DEFAULT_ADAPTERS)


def test_model_runtime_routing_uses_each_frozen_interpreter(tmp_path):
    runtimes = model_runtimes(FREEZE, tmp_path)
    assert len(runtimes) == 6
    assert runtimes["M3GNet-MP-2021.2.8-PES"].environment == "e3sse-matgl-1.2.7"
    assert runtimes["M3GNet-MatPES-PBE-v2025.1-PES"].environment == "e3sse-matgl-1.2.7"
    assert runtimes["SevenNet-omat"].python == tmp_path / "e3sse-sevennet-0.11.0" / "bin" / "python"
    assert runtimes["SevenNet-0-11July2024"].environment == "e3sse-sevennet-0.10.1"
    with pytest.raises(RuntimeRoutingError, match="never installs"):
        require_present(runtimes["CHGNet-0.3.0"])
    env = runtime_environment(runtimes["M3GNet-MP-2021.2.8-PES"], REPOSITORY, base={})
    assert env["DGLBACKEND"] == "pytorch" and env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"] == str(REPOSITORY / "src")
    command = runtime_command(runtimes["CHGNet-0.3.0"], REPOSITORY / "scripts" / "x.py", ["--a"],
                              project_python=Path("/usr/bin/python3"))
    assert command[0].endswith("e3sse-chgnet-0.3.0/bin/python")


@pytest.mark.parametrize("command", [["pip", "install", "ase"], ["/usr/bin/python", "-m", "pip",
                                                                  "install", "x"],
                                     ["uv", "pip", "sync"], ["/opt/conda/bin/conda", "install"]])
def test_no_package_installation_command_is_ever_run(command):
    with pytest.raises(RuntimeRoutingError, match="package manager"):
        assert_no_package_installation(command)


def test_production_source_never_invokes_a_package_manager():
    for path in (REPOSITORY / "src" / "e3sse" / "production").glob("*.py"):
        text = path.read_text()
        assert "pip install" not in text and "uv sync" not in text, path


def test_pre_freeze_evidence_must_precede_the_freeze():
    entries = [_entry("freeze", adapter="freeze"),
               _entry("bvse_reproduction", ["freeze"], status="missing_runner",
                      adapter="missing_runner")]
    with pytest.raises(StageGraphError, match="ancestor of freeze"):
        build_stage_graph(entries, known_adapters=DEFAULT_ADAPTERS)


def test_protocol_freeze_order_cannot_be_dropped_by_config():
    stages = json.loads(json.dumps(CONFIG["stages"]))
    discovery = next(s for s in stages if s["name"] == "discovery_selection")
    discovery["depends_on"].remove("density_ratio")
    with pytest.raises(StageGraphError, match="'density_ratio' must precede"):
        build_stage_graph(stages, known_adapters=DEFAULT_ADAPTERS)


def test_calibration_labels_are_read_only_after_model_selection():
    stages = json.loads(json.dumps(CONFIG["stages"]))
    b5 = next(s for s in stages if s["name"] == "linear_recalibration_comparator")
    b5["depends_on"] = ["transfer"]
    with pytest.raises(StageGraphError, match="must precede|must follow model_selection"):
        build_stage_graph(stages, known_adapters=DEFAULT_ADAPTERS)


def test_checked_in_graph_reads_no_calibration_label_before_selection():
    graph = build_stage_graph(CONFIG["stages"], known_adapters=DEFAULT_ADAPTERS)
    before = graph.ancestors("model_selection")
    assert not any(set(graph.stages[n].outcome_access) & {
        "nebdft2k_conformal_calibration_barriers", "nebdft2k_repeated_partition_barriers"}
        for n in before)
    assert graph.stages["deployment_calibration"].outcome_access == ()
    assert graph.stages["discovery"].outcome_access == ()
    assert "nebdft2k_development_barriers" not in graph.stages["discovery_selection"].outcome_access
