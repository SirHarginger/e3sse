"""Embedding-capability probe logic (model-free parts; models run only on the server)."""

from __future__ import annotations

import numpy as np
import pytest

from e3sse.representation.embedding_capability import (
    array_checks,
    capability_status,
    hook_candidates,
    merge_manifest,
    probe_structures,
    rotated,
    rotation_matrix,
)


def test_rotation_is_proper_and_preserves_distances():
    atoms = probe_structures()[0]
    turned = rotated(atoms)
    assert np.linalg.det(rotation_matrix()) == pytest.approx(1.0)
    assert np.allclose(atoms.get_all_distances(mic=True), turned.get_all_distances(mic=True))


def test_probe_structures_are_synthetic_deterministic_and_contain_li():
    first, second = probe_structures(), probe_structures()
    assert [len(a) for a in first] == [12, 96]
    assert "Li" in first[0].get_chemical_symbols()
    assert np.array_equal(first[1].positions, second[1].positions)


def test_checks_detect_atom_level_nondeterminism_and_nonfinite_values():
    a = np.ones((4, 8))
    ok = array_checks(a, a.copy(), 4)
    assert ok["atom_level"] and ok["deterministic"] and ok["finite"] and ok["dimension"] == 8
    rng = np.random.default_rng(0)
    features = rng.normal(size=(4, 8))
    assert array_checks(features, features, 4, features)["rotation_invariant"]
    assert not array_checks(features, features, 4, features + 0.1)["rotation_invariant"]
    assert not array_checks(np.ones((4, 3)), np.ones((4, 3)), 4)["atom_level"]  # vectors
    noisy = array_checks(a, a + 1e-9, 4)
    assert not noisy["deterministic"]
    bad = a.copy()
    bad[0, 0] = np.nan
    assert not array_checks(bad, bad, 4)["finite"]
    assert not array_checks(np.ones(4), np.ones(4), 4)["atom_level"]


def test_hook_candidates_are_listed_never_selected():
    outputs = {"layer.1": np.zeros((5, 64)), "readout": np.zeros((1, 1)),
               "edges": np.zeros((30, 16)), "ints": np.zeros((5, 3), dtype=int)}
    outputs["forces"] = np.zeros((5, 3))
    assert hook_candidates(outputs, 5) == [{"capture": "layer.1", "dimension": 64}]
    assert capability_status(documented=False, pinned_ok=None, n_candidates=1,
                             error=None) == "candidates_only"
    assert capability_status(documented=False, pinned_ok=True, n_candidates=1,
                             error=None) == "pinned_module"
    assert capability_status(documented=True, pinned_ok=None, n_candidates=0,
                             error="RuntimeError: x") == "unavailable"


def test_manifest_lists_only_verified_usable_models():
    base = {"schema": "e3sse.embedding-capability"}
    good = {"atom_level": True, "finite": True, "deterministic": True, "rotation_invariant": True,
            "translation_invariant": True, "permutation_consistent": True, "aba_consistent": True}
    reports = [
        {**base, "canonical_name": "A", "status": "documented_api",
         "structure_checks": [good, good], "package_version_matches_registry": True},
        {**base, "canonical_name": "B", "status": "candidates_only", "structure_checks": []},
        {**base, "canonical_name": "C", "status": "documented_api",
         "structure_checks": [good, {**good, "rotation_invariant": False}],
         "package_version_matches_registry": True},
        {**base, "canonical_name": "D", "status": "documented_api",
         "structure_checks": [good], "package_version_matches_registry": False},
    ]
    manifest = merge_manifest(reports)
    assert manifest["usable_for_latent_support"] == ["A"]
    assert manifest["scientific_outcomes_read"] is False
    with pytest.raises(ValueError):
        merge_manifest([reports[0], reports[0]])


def test_permutation_translation_state_and_environment_checks():
    from e3sse.representation.embedding_capability import (
        checks_pass,
        failed_checks,
        permutation,
        permuted,
        translated_wrapped,
    )

    atoms = probe_structures()[1]
    order = permutation(len(atoms))
    assert sorted(order.tolist()) == list(range(len(atoms)))
    assert permuted(atoms, order).get_chemical_symbols() == \
        [atoms.get_chemical_symbols()[i] for i in order]
    moved = translated_wrapped(atoms)
    assert np.allclose(np.sort(atoms.get_all_distances(mic=True).ravel()),
                       np.sort(moved.get_all_distances(mic=True).ravel()))
    rng = np.random.default_rng(3)
    features = rng.normal(size=(len(atoms), 8))
    li = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Li"]
    good = array_checks(features, features, len(atoms), features, translated_output=features,
                        permuted_output=features[order], order=order,
                        after_other_structure=features, li_rows=li)
    assert checks_pass(good) and good["environment_sensitive"]
    rows_not_following_atoms = array_checks(features, features, len(atoms), features,
                                            translated_output=features, permuted_output=features,
                                            order=order, after_other_structure=features)
    assert "permutation_consistent" in failed_checks(rows_not_following_atoms)
    carried_state = array_checks(features, features, len(atoms), features,
                                 translated_output=features, permuted_output=features[order],
                                 order=order, after_other_structure=features + 0.1)
    assert failed_checks(carried_state) == ["aba_consistent"]
    species_only = np.zeros((len(atoms), 8))
    species_only[li] = 1.0
    flat = array_checks(species_only, species_only, len(atoms), species_only,
                        translated_output=species_only, permuted_output=species_only[order],
                        order=order, after_other_structure=species_only, li_rows=li)
    assert not checks_pass(flat) and "environment_sensitive" in failed_checks(flat)


def test_capture_hooks_clone_inputs_outputs_and_count_calls():
    torch = pytest.importorskip("torch")
    from e3sse.representation.embedding_capability import capture_module_outputs

    class Block(torch.nn.Module):
        def forward(self, x):
            return x * 2.0, {"x": x + 1.0}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

        def forward(self, x):
            out, _ = self.block(x)
            out.add_(1.0)  # in-place change after the hook must not alter the capture
            return out

    model, x = Model(), torch.ones(4, 5)
    captured = capture_module_outputs(model, lambda: model(x))
    assert captured["__calls__"] == {"block": 1}
    assert np.allclose(captured["block:output[0]"], 2.0)
    assert np.allclose(captured["block:output[1]['x']"], 2.0)
    assert np.allclose(captured["block:input[0]"], 1.0)
