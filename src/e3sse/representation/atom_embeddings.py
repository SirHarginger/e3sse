"""Atom-level embeddings of a frozen model through its verified capture.

The capture is the one pinned for the model in
``amendments.representation.embedding_capability_evidence`` (its report's
``capture_mode`` and ``location``): a documented API (CHGNet ``atom_fea``, MACE
invariant descriptors) or one pinned torch module observed with a single forward
hook (``embedding_capability.capture_pinned_output``).  Shared by the capability
probe and the production path-representation extraction, so both read exactly the
same tensor.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from e3sse.representation.embedding_capability import capture_pinned_output


def torch_module(calc: Any, adapter: str) -> Any:
    """The underlying torch module, located by documented attributes only."""

    for attribute in ("model", "models", "potential"):
        module = getattr(calc, attribute, None)
        if isinstance(module, list) and module:
            module = module[0]
        if module is not None and hasattr(module, "named_modules"):
            return module
    raise RuntimeError(f"no torch module found on the {adapter} calculator")


def documented_embedding(calc: Any, adapter: str, atoms: Any) -> Any:
    if adapter == "mace":
        return calc.get_descriptors(atoms, invariants_only=True)
    if adapter == "chgnet":
        from pymatgen.io.ase import AseAtomsAdaptor

        structure = AseAtomsAdaptor.get_structure(atoms)
        return calc.model.predict_structure(structure, return_atom_feas=True)["atom_fea"]
    raise KeyError(adapter)


def embedding_extractor(model: Mapping[str, Any], calc: Any, *, capture_mode: str,
                        location: str) -> Callable[[Any], Any]:
    """``atoms -> (n_atoms, d)`` array of the pinned representation (atom order kept)."""

    import numpy as np

    adapter = model["adapter_name"]
    if capture_mode == "documented_api":
        return lambda atoms: np.asarray(documented_embedding(calc, adapter, atoms), dtype=float)
    if capture_mode != "pinned_module":
        raise ValueError(f"no production capture for mode {capture_mode!r}")
    module = torch_module(calc, adapter)

    def extract(atoms: Any) -> Any:
        copy = atoms.copy()
        copy.calc = calc
        calc.reset()  # the forward pass must really run under the hook
        captured = capture_pinned_output(module, location, copy.get_potential_energy)
        return np.asarray(captured[location], dtype=float)

    return extract
