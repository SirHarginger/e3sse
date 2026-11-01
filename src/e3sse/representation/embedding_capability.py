"""Embedding-capability probe for frozen foundation models (no scientific outcomes).

For each frozen model this records whether a *stable* atom-level representation can
be extracted without changing the model:

* ``documented_api``: an official extraction API exists in the pinned package.
  CHGNet: ``CHGNet.predict_structure(..., return_atom_feas=True)``.  MACE:
  ``MACECalculator.get_descriptors(invariants_only=True)``.
* ``pinned_module``: no documented API, but the mlip-specialist pinned one module
  whose forward output is verified here (shape, determinism, finiteness);
* ``candidates_only``: no documented API and nothing pinned.  Candidate modules
  (float outputs with one row per atom) are listed for review and are not selected
  automatically;
* ``unavailable``: extraction failed.

Inputs are synthetic Li-containing structures (antifluorite Li2O, plus a
deterministically rattled supercell), so no dataset and no outcome is read.  The
probe does not modify, retrain or wrap the model; forward hooks only observe outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

CAPABILITY_SCHEMA = "e3sse.embedding-capability"
MANIFEST_SCHEMA = "e3sse.embedding-capability-manifest"
STATUSES = ("documented_api", "pinned_module", "candidates_only", "unavailable")
RECORDED_DEPENDENCIES = ("ase", "dgl", "e3nn", "numpy", "torch", "torch_geometric")
DOCUMENTED_APIS = {
    "chgnet": "chgnet.model.CHGNet.predict_structure(return_atom_feas=True)['atom_fea']",
    "mace": "mace.calculators.MACECalculator.get_descriptors(invariants_only=True)",
}
STRUCTURE_POOLING_RULE = "mean over atoms (structure level); migrating-Li row (hop level)"


def probe_structures() -> list[Any]:
    """Antifluorite Li2O conventional cell and a 2x2x2 supercell rattled with a fixed seed."""

    from ase.build import bulk

    cell = bulk("OLi2", "fluorite", a=4.61, cubic=True)
    rattled = cell.repeat((2, 2, 2))
    rattled.rattle(stdev=0.05, seed=2026)
    return [cell, rattled]


def rotation_matrix(seed: int = 2026) -> Any:
    """A fixed proper rotation (QR of a seeded Gaussian matrix, det = +1)."""

    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotated(atoms: Any, seed: int = 2026) -> Any:
    """Rigidly rotate positions and cell together (same atom order)."""

    copy = atoms.copy()
    matrix = rotation_matrix(seed)
    copy.set_cell(copy.cell.array @ matrix.T, scale_atoms=False)
    copy.positions = copy.positions @ matrix.T
    return copy


def translated_wrapped(atoms: Any, vector: Sequence[float] = (0.37, -0.21, 0.53)) -> Any:
    """Rigid translation by a non-lattice vector, then wrapping into the cell."""

    copy = atoms.copy()
    copy.positions = copy.positions + np.asarray(vector)
    copy.wrap()
    return copy


def permutation(n_atoms: int, seed: int = 2026) -> np.ndarray:
    return np.random.Generator(np.random.PCG64DXSM(seed)).permutation(n_atoms)


def permuted(atoms: Any, order: np.ndarray) -> Any:
    return atoms[order]


def array_checks(first: Any, second: Any, n_atoms: int, rotated_output: Any = None, *,
                 translated_output: Any = None, permuted_output: Any = None,
                 order: np.ndarray | None = None, after_other_structure: Any = None,
                 li_rows: Sequence[int] | None = None,
                 tolerance: float = 1e-4) -> dict[str, Any]:
    """Shape, finiteness, determinism, invariances, row-atom correspondence and state.

    * rotation and translation (with wrapping): outputs unchanged within tolerance;
    * permutation: the output rows follow the atoms (permuted output equals first[order]);
    * A-B-A: evaluating another structure in between leaves A unchanged (no carried state);
    * environment sensitivity (when ``li_rows`` holds inequivalent Li sites): Li rows differ,
      which rules out a pure species embedding.
    """

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64) if second is not None else None
    same = b is not None and a.shape == b.shape
    scale = max(1.0, float(np.abs(a).max())) if a.size else 1.0

    def close(other: Any) -> bool | None:
        if other is None:
            return None
        o = np.asarray(other, dtype=np.float64)
        return bool(o.shape == a.shape and np.abs(o - a).max() <= tolerance * scale)

    checks = {
        "shape": list(a.shape),
        "dtype": str(np.asarray(first).dtype),
        "atom_level": a.ndim == 2 and a.shape[0] == n_atoms and a.shape[1] != 3,
        "dimension": int(a.shape[1]) if a.ndim == 2 else None,
        "finite": bool(np.isfinite(a).all()),
        "deterministic": bool(same and np.array_equal(a, b)),
        "max_abs_run_difference": float(np.abs(a - b).max()) if same else None,
        "rotation_invariant": close(rotated_output),
        "translation_invariant": close(translated_output),
        "aba_consistent": close(after_other_structure),
        "permutation_consistent": None,
        "environment_sensitive": None,
    }
    if permuted_output is not None and order is not None and a.ndim == 2:
        o = np.asarray(permuted_output, dtype=np.float64)
        checks["permutation_consistent"] = bool(
            o.shape == a.shape and np.abs(o - a[order]).max() <= tolerance * scale)
    if li_rows is not None and len(li_rows) >= 2 and a.ndim == 2:
        rows = a[list(li_rows)]
        checks["environment_sensitive"] = bool(np.abs(rows - rows[0]).max() > tolerance * scale)
    return checks


REQUIRED_CHECKS = ("atom_level", "finite", "deterministic", "rotation_invariant",
                   "translation_invariant", "permutation_consistent", "aba_consistent")


def checks_pass(checks: Mapping[str, Any]) -> bool:
    """Every required check is True; environment sensitivity is required where evaluated."""

    if not all(checks.get(name) is True for name in REQUIRED_CHECKS):
        return False
    return checks.get("environment_sensitive") is not False


def failed_checks(checks: Mapping[str, Any]) -> list[str]:
    failed = [name for name in REQUIRED_CHECKS if checks.get(name) is not True]
    if checks.get("environment_sensitive") is False:
        failed.append("environment_sensitive")
    return failed


def hook_candidates(outputs: Mapping[str, Any], n_atoms: int) -> list[dict[str, Any]]:
    """Captured arrays that are floating, 2-D, one row per atom, and not (n, 3) vectors."""

    candidates = []
    calls = outputs.get("__calls__", {})
    for name, value in sorted(outputs.items()):
        if name == "__calls__" or calls.get(name.split(":")[0], 1) != 1:
            continue  # ambiguous: the module ran more than once
        array = np.asarray(value)
        if (array.ndim == 2 and array.shape[0] == n_atoms and array.shape[1] != 3
                and np.issubdtype(array.dtype, np.floating)):
            candidates.append({"capture": name, "dimension": int(array.shape[1])})
    return candidates


def _tensors(value: Any, prefix: str) -> dict[str, Any]:
    """Cloned copies of the torch tensors in a value: tensors, (nested) tuples and lists,
    and every tensor entry of dict-like graph data (for example SevenNet AtomGraphData)."""

    import torch

    found: dict[str, Any] = {}
    if isinstance(value, torch.Tensor):
        found[prefix] = value.detach().cpu().clone().numpy()
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            found.update(_tensors(item, f"{prefix}[{index}]"))
    elif hasattr(value, "keys") and callable(value.keys):
        for key in sorted(map(str, value.keys())):
            try:
                found.update(_tensors(value[key], f"{prefix}[{key!r}]"))
            except (KeyError, TypeError, IndexError):
                continue
    return found


def capture_module_outputs(torch_module: Any, run: Callable[[], Any]) -> dict[str, Any]:
    """Run once with read-only hooks: pre-hooks record inputs, forward hooks outputs.

    Inputs are recorded before the module runs, so in-place mutation of graph data
    cannot disguise an output as an input.  ``calls`` counts invocations per module; a
    capture from a module that ran more than once is ambiguous and is never pinned.
    The caller must reset the calculator so the forward pass really runs.
    """

    outputs: dict[str, Any] = {}
    calls: dict[str, int] = {}
    handles = []

    def pre_hook(name: str):
        def hook(_module, inputs):
            outputs.update(_tensors(tuple(inputs), f"{name}:input"))
        return hook

    def post_hook(name: str):
        def hook(_module, _inputs, output):
            calls[name] = calls.get(name, 0) + 1
            outputs.update(_tensors(output, f"{name}:output"))
        return hook

    for name, module in torch_module.named_modules():
        if name:
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(post_hook(name)))
    try:
        run()
    finally:
        for handle in handles:
            handle.remove()
    outputs["__calls__"] = calls
    return outputs


def capability_status(*, documented: bool, pinned_ok: bool | None, n_candidates: int,
                      error: str | None) -> str:
    if error is not None:
        return "unavailable"
    if documented:
        return "documented_api"
    if pinned_ok:
        return "pinned_module"
    return "candidates_only" if n_candidates else "unavailable"


def merge_manifest(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One manifest over per-model reports; usable means documented_api or pinned_module."""

    names = [r["canonical_name"] for r in reports]
    if len(names) != len(set(names)):
        raise ValueError("duplicate model in capability reports")
    for report in reports:
        if report.get("schema") != CAPABILITY_SCHEMA or report.get("status") not in STATUSES:
            raise ValueError(f"invalid capability report for {report.get('canonical_name')}")
    return {
        "schema": MANIFEST_SCHEMA,
        "scientific_outcomes_read": False,
        "models": sorted(reports, key=lambda r: r["canonical_name"]),
        "usable_for_latent_support": sorted(
            r["canonical_name"] for r in reports
            if r["status"] in ("documented_api", "pinned_module")
            and r.get("structure_checks")
            and all(checks_pass(c) for c in r["structure_checks"])
            and r.get("package_version_matches_registry") is True
        ),
    }
