#!/usr/bin/env python3
"""ASE / FIRE / NEB identity audit of one frozen model runtime (no data, no model).

Run with the runtime's own interpreter (the production orchestrator's
``runtime_audit`` stage does this for every frozen environment).  It records the
evidence that the protocol's runtime requirements ask for:

* the interpreter, pinned packages and frozen environment variables match the
  registry, and the environment lock matches its pinned SHA-256;
* the installed ASE version and where NEB lives (``ase.neb`` or ``ase.mep``);
* NEB accepts the frozen constructor explicitly (method, climb, k, no shared
  calculator, no image parallelism) and leaves the DyNEB options inert;
* FIRE accepts every frozen parameter explicitly and stores them unchanged;
* ``run(fmax, steps)`` on a fresh optimizer stops after exactly ``steps`` steps
  (ASE 3.23 changed ``max_steps`` from absolute to relative);
* a reference Cu-vacancy NEB under ASE's EMT, run twice with the frozen settings,
  gives bit-identical records.  Its barrier is compared across runtimes by
  preflight, as equivalence evidence between ASE releases.

The report goes once to ``<output>/<environment>-<fingerprint>.json`` and its path is
printed as JSON.  ``--resume`` verifies an existing report.  Exit codes: 0 passed, 3
a check failed (report written), anything else a crash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME_AUDIT_SCHEMA = "e3sse.production.ml-neb-runtime-audit"
# study1_inference.py holds the lock/pin verification, so a fix there re-runs the audit.
CODE_FILES = ("scripts/audit_ml_neb_runtime.py", "src/e3sse/barriers/ml_neb.py",
              "src/e3sse/screening/prospective_contract.py",
              "src/e3sse/softening/study1_inference.py")
CHECK_FAILED_EXIT = 3  # the audit ran and recorded a failed check (1 = crashed)
# ASE's NEB method-name -> implementation class (identical in ASE 3.22 to 3.29).
NEB_METHOD_CLASSES = {"aseneb": "ASENEBMethod", "improvedtangent": "ImprovedTangentMethod",
                      "eb": "FullSpringMethod", "spline": "SplineMethod", "string": "StringMethod"}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_checks(freeze_config: dict, environment: str) -> dict:
    from e3sse.softening.study1_inference import verify_runtime

    record = freeze_config["inputs"]["production_evidence"]["runtime_environments"][environment]
    lock = Path(record["environment_lock"]["path"])
    lock_ok = lock.is_file() and _sha256_file(lock) == record["environment_lock"]["sha256"]
    try:
        verify_runtime(record, None, lock if lock_ok else None)
        verified, error = True, None
    except RuntimeError as exc:
        verified, error = False, str(exc)
    return {"environment_lock_path": str(lock), "environment_lock_sha256_matches": lock_ok,
            "registry_pins_verified": verified and lock_ok, "error": error,
            "pinned_packages": record["pinned_packages"]}


def audit(active_protocol: dict) -> dict:
    import ase
    from ase.calculators.emt import EMT

    from e3sse.barriers.ml_neb import (
        build_neb,
        fire_optimizer,
        neb_class,
        reference_emt_vacancy_band,
        run_ml_neb,
        settings_from_protocol,
    )

    settings = settings_from_protocol(active_protocol)
    neb_type, module, compatibility = neb_class()
    checks: dict = {"ase_version": ase.__version__, "neb_module": module,
                    "neb_import_compatibility": compatibility,
                    "neb_signature": str(inspect.signature(neb_type.__init__))}
    band = reference_emt_vacancy_band()
    for image in band:
        image.calc = EMT()
    neb = build_neb(band, settings)
    checks["neb_constructor"] = {
        "method": getattr(neb, "neb_method", None) and type(neb.neb_method).__name__,
        "requested_method": settings.method, "climb": neb.climb,
        "parallel": neb.parallel, "allow_shared_calculator": neb.allow_shared_calculator,
        "dynamic_relaxation": getattr(neb, "dynamic_relaxation", None),
        "scale_fmax": getattr(neb, "scale_fmax", None),
        "spring_constants": [float(k) for k in neb.k],
    }
    optimizer = fire_optimizer(neb, settings)
    stored = {name: getattr(optimizer, name, None) for name in
              ("dt", "maxstep", "dtmax", "Nmin", "finc", "fdec", "astart", "fa", "a",
               "downhill_check")}
    checks["fire_parameters_stored"] = stored
    checks["fire_parameters_match"] = all(stored[k] == v for k, v in
                                          settings.fire_parameters.items())
    optimizer.run(fmax=1e-12, steps=3)
    checks["fresh_optimizer_steps_for_steps_3"] = int(optimizer.get_number_of_steps())
    provenance = {"model": "ase-emt-reference", "weights_sha256": None,
                  "environment_lock_sha256": None, "dtype": "float64", "device": "cpu",
                  "torch_intra_op_threads": None, "torch_inter_op_threads": None}

    def reference(calculator):
        return run_ml_neb(reference_emt_vacancy_band(), calculator, settings,
                          unit_id="emt-reference", provenance=provenance,
                          migrating_species="Cu")

    records = [reference(EMT) for _ in range(2)]
    checks["reference_neb"] = {
        "status": records[0]["status"], "barrier_eV": records[0]["barrier_eV"],
        "band_optimizer_steps": records[0].get("band_optimizer_steps"),
        "record_sha256": _canonical(records[0]),
        "deterministic_repeat": _canonical(records[0]) == _canonical(records[1]),
    }
    import numpy as np
    from ase.calculators.calculator import Calculator

    class Raising(Calculator):
        implemented_properties = ("energy", "free_energy", "forces")

        def calculate(self, atoms=None, properties=("energy",), system_changes=None):
            raise RuntimeError("synthetic model error")

    class NonFinite(EMT):
        implemented_properties = ("energy", "free_energy", "forces")

        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results["forces"] = np.full_like(self.results["forces"], np.nan)

    # Classification must hold in this ASE: e.g. ASE 3.22 evaluates the model while
    # constructing the optimizer.
    checks["failure_classification"] = {
        "raising_calculator": reference(Raising)["failure_category"],
        "non_finite_calculator": reference(NonFinite)["failure_category"],
    }
    checks["passed"] = bool(
        checks["failure_classification"] == {"raising_calculator": "calculator_exception",
                                             "non_finite_calculator": "non_finite_value"}
        and checks["fire_parameters_match"]
        and checks["neb_import_compatibility"] in
            ("native", "scipy_cumulative_trapezoid_alias")
        and checks["neb_constructor"]["method"] == NEB_METHOD_CLASSES.get(settings.method)
        and all(k == settings.k for k in checks["neb_constructor"]["spring_constants"])
        and checks["fresh_optimizer_steps_for_steps_3"] == 3
        and checks["neb_constructor"]["climb"] is settings.climb
        and checks["neb_constructor"]["parallel"] is False
        and checks["neb_constructor"]["allow_shared_calculator"] is False
        and checks["neb_constructor"]["scale_fmax"] in (None, 0.0)
        and not checks["neb_constructor"]["dynamic_relaxation"]
        and checks["reference_neb"]["deterministic_repeat"]
        and checks["reference_neb"]["status"] == "converged"
    )
    return {"settings": settings.as_json(), "settings_sha256": settings.settings_sha256,
            "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol-freeze-config", type=Path, required=True)
    parser.add_argument("--active-protocol", type=Path, required=True)
    parser.add_argument("--environment", required=True, help="frozen runtime environment name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protected-root", type=Path, action="append", required=True)
    parser.add_argument("--synthetic-test-runtime", action="store_true",
                        help=argparse.SUPPRESS)  # tests: audit the current interpreter
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    from e3sse.execution.sharded_execution import ensure_safe_output_dir, publish_json_once

    freeze_config = json.loads(args.protocol_freeze_config.read_text(encoding="utf-8"))
    active = json.loads(args.active_protocol.read_text(encoding="utf-8"))
    environments = freeze_config["inputs"]["production_evidence"]["runtime_environments"]
    if args.environment not in environments and not args.synthetic_test_runtime:
        raise SystemExit(f"{args.environment!r} is not a frozen runtime environment")
    identity = {
        "environment": args.environment, "python": sys.version.split()[0],
        "executable": sys.executable,
        "ase_version": _version("ase"),
        "frozen_ml_neb_settings_sha256": _canonical(
            active["amendments"]["prospective"]["ml_neb"]["frozen_numeric_settings"]),
        "code_sha256": {name: _sha256_file(REPOSITORY / name) for name in CODE_FILES},
    }
    fingerprint = _canonical(identity)
    output = ensure_safe_output_dir(args.output, args.protected_root)
    # One report per identity: a code or settings change writes a new report and never
    # collides with (or overwrites) an old one.
    path = output / f"{args.environment}-{fingerprint[:16]}.json"
    if path.exists():
        if not args.resume:
            raise SystemExit(f"{path} exists; use --resume")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("fingerprint_sha256") != fingerprint:
            raise SystemExit(f"runtime audit {path} does not match its name; inspect it")
        print(json.dumps({"environment": args.environment, "passed": stored["passed"],
                          "report": str(path), "verified_existing": True}))
        return 0 if stored["passed"] else CHECK_FAILED_EXIT
    runtime = (None if args.synthetic_test_runtime
               else _runtime_checks(freeze_config, args.environment))
    result = audit(active)
    passed = result["checks"]["passed"] and (runtime is None or runtime["registry_pins_verified"])
    report = {
        "schema_version": RUNTIME_AUDIT_SCHEMA,
        "classification": "runtime_identity_evidence_no_data_no_model",
        "identity": identity, "fingerprint_sha256": fingerprint,
        "runtime": runtime, **result,
        "installed_versions": {name: _version(name) for name in ("ase", "numpy", "torch",
                                                                 "scipy")},
        "installed_ions": _version("ions"),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "torch_threads": _torch_threads(),
        "device": "cpu",
        "passed": passed,
    }
    publish_json_once(path, report, root=output)
    print(json.dumps({"environment": args.environment, "ase": identity["ase_version"],
                      "passed": passed, "report": str(path)}))
    return 0 if passed else CHECK_FAILED_EXIT


def _torch_threads() -> dict | None:
    try:
        import torch
    except ImportError:
        return None
    return {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}


if __name__ == "__main__":
    raise SystemExit(main())
