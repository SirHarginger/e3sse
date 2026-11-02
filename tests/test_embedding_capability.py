"""Embedding-capability probe logic (model-free parts; models run only on the server)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from e3sse.representation.embedding_capability import (
    array_checks,
    capability_status,
    failed_checks,
    hook_candidates,
    merge_manifest,
    probe_structures,
    rotated,
    rotation_matrix,
)
from e3sse.screening.applicability_support import applicability_source


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


# --- synthetic Li2O fixture -------------------------------------------------------------

def test_li2o_fixture_is_explicit_periodic_stoichiometric_and_deterministic(monkeypatch):
    import ase.build

    def broken_builder(*_args, **_kwargs):  # the ase 3.22.1 (CHGNet runtime) failure
        raise RuntimeError("Cannot create cubic cell for fluorite structure")

    monkeypatch.setattr(ase.build, "bulk", broken_builder)
    cell, supercell = probe_structures()
    again = probe_structures()[1]
    for atoms, n in ((cell, 12), (supercell, 96)):
        symbols = atoms.get_chemical_symbols()
        assert len(atoms) == n and set(symbols) == {"Li", "O"}
        assert symbols.count("Li") == 2 * symbols.count("O")
        assert atoms.pbc.all() and atoms.cell.volume > 0 and atoms.cell.rank == 3
        assert np.isfinite(atoms.positions).all() and np.isfinite(atoms.cell.array).all()
    assert np.allclose(cell.cell.array, 4.61 * np.eye(3))
    assert np.allclose(supercell.cell.array, 2 * cell.cell.array)
    # Antifluorite geometry: nearest Li-O distance a*sqrt(3)/4, no overlapping atoms.
    distances = cell.get_all_distances(mic=True) + np.diag([np.inf] * len(cell))
    assert distances.min() == pytest.approx(4.61 * np.sqrt(3) / 4)
    # Deterministic rattle (seed 2026) of the 2x2x2 repeat.
    assert np.array_equal(supercell.positions, again.positions)
    assert supercell.get_chemical_symbols() == again.get_chemical_symbols()
    displacement = supercell.positions - cell.repeat((2, 2, 2)).positions
    assert 0 < np.abs(displacement).max() < 0.5


def test_li2o_fixture_reproduces_the_previous_builder_where_it_works():
    from ase.build import bulk

    try:
        previous = bulk("OLi2", "fluorite", a=4.61, cubic=True)
    except RuntimeError:
        pytest.skip("this ASE version cannot build the cubic fluorite cell")
    cell = probe_structures()[0]
    assert cell.get_chemical_symbols() == previous.get_chemical_symbols()
    assert np.array_equal(cell.positions, previous.positions)
    assert np.array_equal(cell.cell.array, previous.cell.array)


# --- pinned capture grammar (no torch needed) -------------------------------------------

def test_pinned_capture_grammar_matches_discovery_names_and_fails_closed():
    from e3sse.representation.embedding_capability import parse_pinned_capture

    assert parse_pinned_capture("model.graph_layers.2:output[1]") == \
        ("model.graph_layers.2", "output", (1,))
    assert parse_pinned_capture("block:output[1]['x']") == ("block", "output", (1, "x"))
    assert parse_pinned_capture("block:input[0]") == ("block", "input", (0,))
    assert parse_pinned_capture("block:output") == ("block", "output", ())
    for bad in ("block", ":output", "block:forward", "block:output[", "block:output[]",
                "block:output[1]x", "block:output[01]", 'block:output["x"]',
                "block:output['\\x']", "", None):
        with pytest.raises(ValueError):
            parse_pinned_capture(bad)


# --- toy network and calculator for hook tests ------------------------------------------

def _toy_calculator():
    """An ASE calculator around a tiny invariant torch network (no checkpoint, no data)."""

    torch = pytest.importorskip("torch")
    from ase.calculators.calculator import Calculator, all_changes

    class Embed(torch.nn.Module):
        def forward(self, species, distances):
            weights = torch.exp(-distances) * (1.0 - torch.eye(len(species), dtype=torch.float64))
            environment = weights @ species
            return torch.cat([species, environment, environment**2], dim=1), {"w": weights}

    class ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(0)  # local: global RNG untouched
            self.embed = Embed()
            self.interaction = torch.nn.Linear(6, 8, dtype=torch.float64)
            self.act = torch.nn.Tanh()  # runs twice: never a valid pinned capture
            self.readout = torch.nn.Linear(8, 1, dtype=torch.float64)
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                                dtype=torch.float64))

        def forward(self, species, distances):
            features, _ = self.embed(species, distances)
            hidden = self.act(self.act(self.interaction(features)))
            return self.readout(hidden).sum()

    class ToyCalculator(Calculator):
        implemented_properties = ("energy",)

        def __init__(self):
            super().__init__()
            self.model = ToyNet().eval()
            self.hooked_modules: list[set[str]] = []

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.hooked_modules.append({
                name for name, m in self.model.named_modules()
                if name and (m._forward_hooks or m._forward_pre_hooks)})
            symbols = self.atoms.get_chemical_symbols()
            species = torch.tensor([[s == "Li", s == "O"] for s in symbols], dtype=torch.float64)
            distances = torch.tensor(self.atoms.get_all_distances(mic=True))
            with torch.no_grad():
                self.results["energy"] = float(self.model(species, distances))

    return ToyCalculator()


def _run(calc, atoms):
    def call():
        calc.reset()
        copy = atoms.copy()
        copy.calc = calc
        return copy.get_potential_energy()
    return call


def test_pinned_capture_hooks_only_the_requested_module_and_matches_discovery():
    from e3sse.representation.embedding_capability import (
        capture_module_outputs,
        capture_pinned_output,
    )

    calc = _toy_calculator()
    atoms = probe_structures()[1]
    for capture, module in (("interaction:output", "interaction"),
                            ("embed:output[0]", "embed"), ("embed:input[1]", "embed")):
        pinned = capture_pinned_output(calc.model, capture, _run(calc, atoms))
        assert calc.hooked_modules[-1] == {module}
        assert set(pinned) == {capture, "__calls__"} and pinned["__calls__"] == {module: 1}
        everything = capture_module_outputs(calc.model, _run(calc, atoms))
        assert calc.hooked_modules[-1] == {"embed", "interaction", "act", "readout"}
        assert np.array_equal(pinned[capture], everything[capture])
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in calc.model.modules())


def test_invalid_pinned_captures_fail_closed():
    from e3sse.representation.embedding_capability import capture_pinned_output

    calc = _toy_calculator()
    atoms = probe_structures()[0]
    cases = {
        "missing:output": "not found in the model",
        "act:output": "must run exactly once",
        "embed:output[1]": "is not a tensor",
        "embed:output[1]['absent']": "not found",
        "embed:output[5]": "not found",
        "interaction:output[0]": "not found",
    }
    for capture, message in cases.items():
        with pytest.raises(RuntimeError, match=message):
            capture_pinned_output(calc.model, capture, _run(calc, atoms))
    with pytest.raises(RuntimeError, match="exactly once"):  # module never runs
        capture_pinned_output(calc.model, "interaction:output", lambda: None)
    with pytest.raises(ValueError):
        capture_pinned_output(calc.model, "interaction", _run(calc, atoms))
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in calc.model.modules())


# --- end-to-end probe on the toy calculator ---------------------------------------------

_OPENED: list[str] | None = None
_AUDIT_INSTALLED = False


def _audit(event, args):
    if _OPENED is not None and event in ("open", "os.listdir", "os.scandir") and args:
        path = args[0]
        _OPENED.append(os.path.abspath(os.fsdecode(path)) if isinstance(path, (str, bytes))
                       else repr(path))


@pytest.fixture
def toy_probe(tmp_path, monkeypatch):
    """The real probe() with build_calculator replaced by the toy (SHA still checked)."""

    import importlib.metadata
    import importlib.util
    import sys
    from pathlib import Path

    from e3sse.protocol import adapters

    pytest.importorskip("torch")
    repository = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "probe_embedding_capability_under_test",
        repository / "scripts" / "probe_embedding_capability.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    checkpoint = tmp_path / "toy.ckpt"
    checkpoint.write_bytes(b"toy weights")
    built: list = []

    def build(model, path, *, device):
        assert device == "cpu"
        if adapters.checkpoint_digest(path) != model["weights_sha256"]:
            raise ValueError("checkpoint SHA-256 mismatch")
        built.append(_toy_calculator())
        return built[-1]

    monkeypatch.setattr(adapters, "build_calculator", build)
    model = {"canonical_name": "Toy", "adapter_name": "toy", "package_name": "numpy",
             "package_version": importlib.metadata.version("numpy"),
             "weights_sha256": adapters.checkpoint_digest(checkpoint)}
    global _AUDIT_INSTALLED
    if not _AUDIT_INSTALLED:  # audit hooks cannot be removed; it records only inside run()
        sys.addaudithook(_audit)
        _AUDIT_INSTALLED = True

    def run(pinned, entry=None):
        global _OPENED
        _OPENED = []
        try:
            report = script.probe(entry or model, checkpoint, pinned)
            return report, list(_OPENED)
        finally:
            _OPENED = None

    run.script, run.model, run.built, run.repository = script, model, built, repository
    return run


def _without_timing(checks):
    return [{k: v for k, v in c.items() if k != "wall_seconds"} for c in checks]


def test_probe_pinned_mode_agrees_with_all_module_capture_and_reads_no_dataset(
        toy_probe, monkeypatch):
    from e3sse.representation.embedding_capability import capture_module_outputs

    report, opened = toy_probe("interaction:output")
    assert report["status"] == "pinned_module" and report["error"] is None
    assert report["capture_mode"] == "pinned_module" and report["hook_candidates"] == []
    assert report["location"] == "captured tensor 'interaction:output'"
    assert report["device"] == "cpu" and report["batch_size"] == 1
    assert report["package_version_matches_registry"] is True
    assert report["checkpoint_sha256"] == toy_probe.model["weights_sha256"]
    assert [c["n_atoms"] for c in report["structure_checks"]] == [12, 96]
    assert all(c["environment_sensitive"] for c in report["structure_checks"][1:])
    assert all(hooked == {"interaction"} for hooked in toy_probe.built[-1].hooked_modules)
    # Synthetic inputs only: nothing under a data root is opened or listed.
    data_roots = (str(toy_probe.repository / "data"), "/srv/ben/e3sse/data")
    assert not [p for p in opened if p.startswith(data_roots)]

    # Previous implementation: all-module capture, then the same pinned key and checks.
    monkeypatch.setattr(toy_probe.script, "capture_pinned_output",
                        lambda module, _capture, run: capture_module_outputs(module, run))
    previous, _ = toy_probe("interaction:output")
    assert previous["status"] == report["status"]
    assert _without_timing(previous["structure_checks"]) == \
        _without_timing(report["structure_checks"])
    assert len(toy_probe.built[-1].hooked_modules[-1]) == 4


def test_probe_discovery_mode_is_unchanged(toy_probe):
    report, _ = toy_probe(None)
    assert report["status"] == "candidates_only" and report["error"] is None
    assert report["capture_mode"] == "discovery"
    captures = {c["capture"] for c in report["hook_candidates"]}
    assert {"interaction:output", "embed:output[0]"} <= captures
    assert not any(c.startswith("act:") for c in captures)  # ran twice: ambiguous
    assert all(c.keys() == {"n_atoms", "wall_seconds"} for c in report["structure_checks"])
    assert all(len(h) == 4 for h in toy_probe.built[-1].hooked_modules)


def test_probe_invalid_pinned_capture_or_identity_fails_closed(toy_probe):
    for capture in ("missing:output", "act:output", "embed:output[1]"):
        report, _ = toy_probe(capture)
        assert report["status"] == "unavailable" and report["error"].startswith("RuntimeError")
    built = len(toy_probe.built)
    report, _ = toy_probe("interaction")  # malformed: rejected before the model loads
    assert report["status"] == "unavailable" and report["error"].startswith("ValueError")
    assert len(toy_probe.built) == built
    report, _ = toy_probe("interaction:output", {**toy_probe.model, "package_version": "0.0.0"})
    assert report["status"] == "unavailable" and "!= registry" in report["error"]
    report, _ = toy_probe("interaction:output", {**toy_probe.model, "weights_sha256": "0" * 64})
    assert report["status"] == "unavailable" and "SHA-256 mismatch" in report["error"]


def test_m3gnet_mp_disposition_is_recorded_as_permutation_rejection():
    import json
    from pathlib import Path

    config = json.loads((Path(__file__).resolve().parents[1] / "configs"
                         / "reliability_gated_internal_freeze_protocol.json").read_text())
    item = next(i for i in config["open_items"] if i["id"] == "embedding_extraction_verification")
    record = item["recorded_model_dispositions"]["M3GNet-MP-2021.2.8-PES"]
    assert record["capture"] == "model.graph_layers.2:output[1]"
    assert record["latent_support_status"] == "rejected_permutation_consistency"
    assert record["applicability_branch"].startswith("descriptor_space")
    assert item["blocking"] is True  # the other models are still pending
    # A synthetic report with exactly that failure can never enter latent support.
    passed = {"atom_level": True, "finite": True, "deterministic": True,
              "rotation_invariant": True, "translation_invariant": True,
              "aba_consistent": True, "environment_sensitive": True}
    report = {"schema": "e3sse.embedding-capability", "canonical_name": "M3GNet-MP-2021.2.8-PES",
              "status": "unavailable", "package_version_matches_registry": True,
              "structure_checks": [{**passed, "permutation_consistent": False}]}
    assert merge_manifest([report])["usable_for_latent_support"] == []
    assert failed_checks(report["structure_checks"][0]) == ["permutation_consistent"]
    assert applicability_source(representation_verified=False,
                                descriptors_available=True) == "descriptor_space"
