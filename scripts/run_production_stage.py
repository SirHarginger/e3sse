#!/usr/bin/env python3
"""Run one phase of one production stage runner (called by the production orchestrator).

    run_production_stage.py --config C --run-dir D --stage S --phase {prepare,model,reduce}
        [--model M] --workers W --threads T --memory-reserve-gib R

``prepare`` and ``reduce`` run in the project environment (or the BVSE runtime for
``bvse_reproduction``); ``model`` runs in the model's frozen runtime.  Workers and
threads come from the central scheduler; this script never chooses them.  Every phase
is resumable (create-once artifacts, verified shard checkpoints).

Exit codes: 0 done (the last line is ``{"manifest": path}``); 2 missing evidence; 3
validation failure (evidence contradicts a frozen requirement or an outcome read was
refused); anything else is an infrastructure failure.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PHASES = {"prepare": "prepare", "model": "run_model", "reduce": "run"}
MANIFEST_NAMES = {"prepare": "prepare_manifest.json", "model": "model_manifest.json",
                  "reduce": "stage_manifest.json"}


def main(argv: list[str] | None = None) -> int:
    from e3sse.production.runners import RUNNER_MODULES
    from e3sse.production.runners.context import (
        EXIT_MISSING_EVIDENCE,
        EXIT_VALIDATION,
        MissingEvidence,
        RunnerContext,
        RunnerValidationError,
        file_sha256,
    )
    from e3sse.protocol.contracts import OutcomeAccessDenied

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=sorted(k for k, v in RUNNER_MODULES.items() if v))
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-reserve-gib", type=float, default=None)
    args = parser.parse_args(argv)
    if (args.phase == "model") != (args.model is not None):
        parser.error("--model is required for, and only for, the model phase")
    module = importlib.import_module(RUNNER_MODULES[args.stage])
    function = getattr(module, PHASES[args.phase], None)
    if function is None:
        parser.error(f"{args.stage} has no {args.phase} phase")
    ctx = RunnerContext(
        repository=REPOSITORY, config_path=args.config.resolve(), run_dir=args.run_dir.resolve(),
        stage=args.stage, model=args.model, workers=args.workers, threads=args.threads,
        memory_reserve_bytes=None if args.memory_reserve_gib is None
        else int(args.memory_reserve_gib * 1024 ** 3))
    try:
        result = function(ctx)
        outputs = {name: Path(path) for name, path in result["outputs"].items()}
        record = {"schema_version": "e3sse.production.stage-manifest", "stage": args.stage,
                  "phase": args.phase, "model": args.model,
                  "outputs": {name: {"path": str(path), "sha256": file_sha256(path)}
                              for name, path in sorted(outputs.items())},
                  "summary": result.get("summary", {})}
        manifest = ctx.publish(MANIFEST_NAMES[args.phase], json.loads(json.dumps(record, default=str)))
    except MissingEvidence as exc:
        print(f"missing evidence: {exc}", file=sys.stderr)
        return EXIT_MISSING_EVIDENCE
    except (RunnerValidationError, OutcomeAccessDenied) as exc:
        print(f"validation failure: {exc}", file=sys.stderr)
        return EXIT_VALIDATION
    print(json.dumps({"manifest": str(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
