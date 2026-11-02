"""Synthetic production layout and fake stage adapters; no model and no outcome."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from e3sse.production.orchestrator import (
    COMPLETED,
    DEFAULT_ADAPTERS,
    StageAdapter,
    StageResult,
    output_record,
)

REPOSITORY = Path(__file__).resolve().parents[1]
COPIED_CONFIGS = ("reliability_gated_internal_freeze_protocol.json", "protocol_freeze.json",
                  "study1_softening_atlas.json", "figure_registry.json", "table_registry.json",
                  "descriptor_fallback_manifest.json")


class WriteResultAdapter(StageAdapter):
    """Runs a child interpreter that writes one JSON result into the stage directory."""

    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code

    def _command(self, ctx, stage):
        target = ctx.stage_dir(stage.name) / "result.json"
        code = ("import json, sys; open(sys.argv[1], 'x').write(json.dumps({'stage': sys.argv[2]}));"
                f" sys.exit({self.exit_code})")
        return [sys.executable, "-c", code, str(target), stage.name], target

    def plan(self, ctx, stage):
        return [" ".join(self._command(ctx, stage)[0])]

    def run(self, ctx, stage):
        command, target = self._command(ctx, stage)
        target.parent.mkdir(parents=True, exist_ok=True)
        ctx.run_command(stage.name, command, ctx.child_environment(None, 1))
        return StageResult(stage.name, COMPLETED, "synthetic stage",
                           outputs={"result": output_record(target)}, commands=[command])


class FakeFreezeAdapter(StageAdapter):
    """Writes a binding file; tests replace the freeze validator (tested elsewhere)."""

    def run(self, ctx, stage):
        from e3sse.production.orchestrator import FREEZE_BINDING_FILE

        path = ctx.stage_dir(stage.name) / FREEZE_BINDING_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"protocol_id": "f" * 64, "file_sha256": "0" * 64,
                                    "directory": str(path.parent)}))
        return StageResult(stage.name, COMPLETED, "synthetic freeze",
                           outputs={"internal_freeze_binding": output_record(path)})


def adapters(**overrides: StageAdapter) -> dict[str, StageAdapter]:
    table = dict(DEFAULT_ADAPTERS)
    table.update({"write_result": WriteResultAdapter(), "failing_child": WriteResultAdapter(7),
                  "fake_freeze": FakeFreezeAdapter()})
    table.update(overrides)
    return table


def stage(name, depends_on=(), adapter="write_result", status="implemented",
          outcome_access=(), workloads=(), next_task=None):
    return {"name": name, "public_name": name.upper(), "adapter": adapter,
            "implementation_status": status, "depends_on": list(depends_on),
            "outcome_access": list(outcome_access), "workloads": [list(w) for w in workloads],
            "next_task": next_task, "description": f"synthetic {name}"}


def synthetic_stages() -> list[dict]:
    return [
        stage("runtime_audit"),
        stage("resource_profile"),
        stage("preflight", ["runtime_audit", "resource_profile"]),
        stage("freeze", ["preflight"], adapter="fake_freeze"),
        stage("estimate", ["freeze"], outcome_access=["mplitrj_published_train_dft_forces_energies"]),
        stage("transfer", ["estimate"], adapter="missing_runner", status="missing_runner",
              next_task="implement the transfer runner"),
    ]


def make_layout(root: Path, stages: list[dict] | None = None, **runtime) -> Path:
    """Config directory with copied protocol configs and a synthetic server layout."""

    configs = root / "configs"
    configs.mkdir(parents=True)
    for name in COPIED_CONFIGS:
        shutil.copy(REPOSITORY / "configs" / name, configs / name)
    for name in ("data", "outputs", "logs", "scratch"):
        (root / name).mkdir(exist_ok=True)
    server = {"project": {"name": "e3sse"},
              "paths": {"relative_to": "..", "data_root": "data", "outputs_root": "outputs",
                        "logs_root": "logs", "scratch_root": "scratch"},
              "runtime": {"seed": 2026}}
    (configs / "server.json").write_text(json.dumps(server))
    production = json.loads((REPOSITORY / "configs" / "production_execution.json").read_text())
    production["runtime"].update({"model_runtime_root": str(root / "venvs"),
                                  "checkpoint_map": str(root / "checkpoint_map.json"),
                                  "require_clean_worktree": False, "tmpdir": None, **runtime})
    production["resource_policy"].update({"memory_reserve_gib": 0.05,
                                          "memory_reserve_fraction": 0.0,
                                          "memory_pressure_wait_seconds": 0})
    production["stages"] = synthetic_stages() if stages is None else stages
    path = configs / "production_execution.json"
    path.write_text(json.dumps(production, indent=2))
    return path
