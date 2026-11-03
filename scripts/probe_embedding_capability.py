#!/usr/bin/env python3
"""Probe atom-level embedding capability of one frozen model in its pinned runtime.

Run on the server with the model's own environment (as for the smoke tests), e.g.

    TMPDIR=/srv/ben/e3sse/scratch PYTHONPATH=src ~/.venvs/<env>/bin/python \\
      scripts/probe_embedding_capability.py --config configs/protocol_freeze.json \\
      --model <canonical_name> --checkpoint <path> --output outputs/embedding_capability

then merge the per-model reports with ``--merge`` from the project environment.
Inputs are synthetic Li2O structures only; no dataset, label or outcome is read.
``--pinned-module`` verifies one capture chosen by the mlip-specialist for a model
without a documented API; without it such models report candidates only.  In pinned
mode only that module is hooked and only that tensor is kept (no candidate
enumeration), so memory stays near a plain forward pass.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from e3sse.representation.atom_embeddings import documented_embedding, torch_module
from e3sse.representation.embedding_capability import (
    CAPABILITY_SCHEMA,
    DOCUMENTED_APIS,
    RECORDED_DEPENDENCIES,
    STRUCTURE_POOLING_RULE,
    array_checks,
    capability_status,
    capture_module_outputs,
    capture_pinned_output,
    checks_pass,
    failed_checks,
    hook_candidates,
    merge_manifest,
    parse_pinned_capture,
    permutation,
    permuted,
    probe_structures,
    rotated,
    translated_wrapped,
)


def _model_entry(config: dict, name: str) -> dict:
    for group in config["protocol"]["model_groups"].values():
        for model in group:
            if model.get("canonical_name") == name:
                return model
    raise SystemExit(f"model {name!r} is not in the frozen registry")


# Shared with production path-representation extraction (one implementation).
_torch_module = torch_module
_documented = documented_embedding


def _installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe(model: dict, checkpoint: Path, pinned_module: str | None) -> dict:
    import time

    import torch

    from e3sse.execution.worker_memory_probe import peak_resident_bytes
    from e3sse.protocol.adapters import build_calculator, checkpoint_digest

    adapter = model["adapter_name"]
    installed = _installed(model["package_name"])
    report = {
        "schema": CAPABILITY_SCHEMA,
        "canonical_name": model["canonical_name"],
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        "adapter_name": adapter,
        "package": f"{model['package_name']}=={installed}",
        "package_version_matches_registry": installed == model["package_version"],
        "dependency_versions": {name: _installed(name) for name in RECORDED_DEPENDENCIES},
        "python": platform.python_version(),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "batch_size": 1,
        "structure_pooling_rule": STRUCTURE_POOLING_RULE,
        "inputs": "synthetic antifluorite Li2O cell and rattled 2x2x2 supercell (seed 2026); "
                  "each also rotated, translated+wrapped and permuted; A-B-A state check",
        "capture_mode": "documented_api" if adapter in DOCUMENTED_APIS
                        else "pinned_module" if pinned_module is not None else "discovery",
        "memory_note": "peak RSS includes all-module hook storage; it overstates production"
                       if adapter not in DOCUMENTED_APIS and pinned_module is None
                       else "peak RSS of documented-API or single-module pinned capture",
    }
    error, location, pinned_ok = None, None, None
    structure_checks, candidates = [], []
    try:
        if installed != model["package_version"]:
            raise RuntimeError(f"installed {installed} != registry {model['package_version']}")
        if pinned_module is not None and adapter not in DOCUMENTED_APIS:
            parse_pinned_capture(pinned_module)  # fail closed before loading the model
        calc = build_calculator(model, checkpoint, device="cpu")
        module = _torch_module(calc, adapter)
        parameters = list(module.parameters()) if hasattr(module, "parameters") else []
        report["parameter_dtype"] = str(parameters[0].dtype) if parameters else None
        report["module_training_flags"] = sorted({bool(m.training) for m in module.modules()})
        structures = probe_structures()

        def representation(structure):
            if adapter in DOCUMENTED_APIS:
                return {"documented": _documented(calc, adapter, structure)}

            def call():
                calc.reset()  # bypass ASE result caching so the forward really runs
                copy = structure.copy()
                copy.calc = calc
                return copy.get_potential_energy()
            if pinned_module is not None:
                return capture_pinned_output(module, pinned_module, call)
            return capture_module_outputs(module, call)

        for index, atoms in enumerate(structures):
            other = structures[1 - index]
            n, started = len(atoms), time.perf_counter()
            order = permutation(n)
            li_rows = [k for k, s in enumerate(atoms.get_chemical_symbols()) if s == "Li"]
            first, second = representation(atoms), representation(atoms)
            representation(other)
            after_other = representation(atoms)
            turned, moved = representation(rotated(atoms)), representation(translated_wrapped(atoms))
            shuffled = representation(permuted(atoms, order))
            key = "documented" if adapter in DOCUMENTED_APIS else pinned_module
            if adapter in DOCUMENTED_APIS:
                location = DOCUMENTED_APIS[adapter]
            elif pinned_module is None:
                candidates = hook_candidates(first, n)
            else:
                location = f"captured tensor {pinned_module!r}"
                if pinned_module not in first:
                    raise RuntimeError(f"pinned capture {pinned_module!r} produced no output")
                module_name = pinned_module.split(":")[0]
                if first["__calls__"].get(module_name) != 1:
                    raise RuntimeError(f"pinned module {module_name!r} must run exactly once")
            checks = {}
            if key is not None:
                checks = array_checks(
                    first[key], second.get(key), n, turned.get(key),
                    translated_output=moved.get(key), permuted_output=shuffled.get(key),
                    order=order, after_other_structure=after_other.get(key),
                    li_rows=li_rows if index == 1 else None)
            structure_checks.append({"n_atoms": n, **checks,
                                     "wall_seconds": round(time.perf_counter() - started, 3)})
        if pinned_module is not None:
            pinned_ok = all(checks_pass(c) for c in structure_checks)
    except Exception as exc:  # noqa: BLE001 - recorded as the capability outcome
        error = f"{type(exc).__name__}: {exc}"
    failed = sorted({name for c in structure_checks for name in failed_checks(c)}) \
        if structure_checks and structure_checks[0].get("shape") else []
    documented_ok = adapter in DOCUMENTED_APIS and error is None and not failed
    if error is None and failed and (adapter in DOCUMENTED_APIS or pinned_module is not None):
        error = f"checks_failed: {', '.join(failed)}"
    report.update({
        "status": capability_status(documented=documented_ok, pinned_ok=pinned_ok,
                                    n_candidates=len(candidates), error=error),
        "location": location,
        "structure_checks": structure_checks,
        "hook_candidates": candidates,
        "error": error,
        "peak_resident_bytes": peak_resident_bytes(),
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pinned-module", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", action="store_true",
                        help="merge <output>/*.json into embedding_capability_manifest.json")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        reports = [json.loads(p.read_text()) for p in sorted(args.output.glob("*.json"))
                   if p.name != "embedding_capability_manifest.json"]
        path = args.output / "embedding_capability_manifest.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(merge_manifest(reports), handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(path)
        return 0
    if not (args.config and args.model and args.checkpoint):
        raise SystemExit("--config, --model and --checkpoint are required unless --merge")
    model = _model_entry(json.loads(args.config.read_text(encoding="utf-8")), args.model)
    report = probe(model, args.checkpoint, args.pinned_module)
    path = args.output / f"{args.model}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"{args.model}: {report['status']}")
    return 0 if report["status"] != "unavailable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
