"""A synthetic stage runner for RunnerAdapter integration tests (no model, no outcome).

Run by the real ``scripts/run_production_stage.py`` in a child interpreter.  The mode
(environment ``E3SSE_SYNTHETIC_RUNNER_MODE``) selects success or one failure class.
"""

from __future__ import annotations

import json
import os
from typing import Any

MODE = "E3SSE_SYNTHETIC_RUNNER_MODE"
CODE_FILES = ("tests/synthetic_runner_module.py",)


def square_task(payload: dict[str, Any], context: Any) -> dict[str, Any]:
    return {"n": payload["n"], "square": payload["n"] ** 2}


def run(ctx) -> dict[str, Any]:
    from e3sse.production.runners.context import MissingEvidence, RunnerValidationError
    from e3sse.protocol.contracts import STAGE_OUTCOME_ACCESS_ENV, _require_stage_outcome_scope

    mode = os.environ.get(MODE, "ok")
    scope = json.loads(os.environ[STAGE_OUTCOME_ACCESS_ENV])
    if mode == "missing":
        raise MissingEvidence("synthetic: server evidence absent")
    if mode == "invalid":
        raise RunnerValidationError("synthetic: evidence contradicts a frozen rule")
    if mode == "crash":
        raise KeyError("synthetic infrastructure crash")
    if mode == "forbidden_outcome":
        _require_stage_outcome_scope("nebdft2k_conformal_calibration_barriers")
    _require_stage_outcome_scope("mplitrj_published_train_dft_forces_energies")
    results = ctx.execute(namespace="synthetic.squares",
                          units=[({"n": n}, {"n": n}) for n in range(6)], task=square_task,
                          identity={"code_sha256": ctx.code_sha256(CODE_FILES)})
    merged = sorted(results.values(), key=lambda r: r["n"])
    path = ctx.publish("synthetic_result.json", {
        "squares": merged, "outcome_scope": scope,
        "interpreter": os.environ.get("E3SSE_SYNTHETIC_INTERPRETER")})
    return {"outputs": {"synthetic_result": path}, "summary": {"units": len(merged)}}
