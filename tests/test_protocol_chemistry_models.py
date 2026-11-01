"""Composition rules, frozen model registry and adapter identity checks (no torch)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from e3sse.protocol.adapters import ADAPTER_VERSION, build_calculator, checkpoint_digest
from e3sse.protocol.chemistry import (
    MP2020_HUBBARD_U_ELEMENTS,
    anion_family,
    primary_exclusion_elements,
)
from e3sse.protocol.partitions import classify_primary_population


def test_tm_rule_matches_pymatgen_set_and_contains_every_hubbard_u_element():
    excluded = set(primary_exclusion_elements())
    assert set(MP2020_HUBBARD_U_ELEMENTS) <= excluded
    assert {"Sc", "Ti", "Zn", "Y", "Cd", "La", "Hf", "Hg", "Ac"} <= excluded
    assert {"Li", "O", "S", "P", "Al", "Ga", "Ce", "Lu", "B"}.isdisjoint(excluded)
    assert len(excluded) == 10 + 10 + 1 + 9 + 1 + 9
    population = classify_primary_population(["Li-O-P", "Fe-Li-O-P", "Ce-Li-O"], excluded)
    assert population == {
        "Ce-Li-O": "primary",
        "Fe-Li-O-P": "tm_hubbard_u_secondary",
        "Li-O-P": "primary",
    }


@pytest.mark.parametrize(
    ("composition", "family"),
    [
        ({"Li": 6, "P": 1, "S": 5, "Cl": 1}, "sulfide"),
        ({"Li": 3, "Y": 1, "Cl": 6}, "halide"),
        ({"Li": 3, "O": 1, "Cl": 1}, "other"),
        ({"Li": 2, "S": 1, "O": 4}, "oxide"),
        ({"Li": 3, "N": 1}, "nitride"),
        ({"Li": 2, "Se": 1}, "other"),
        ({"Li": 7, "La": 3, "Zr": 2, "O": 12}, "oxide"),
        ({"Li": 1, "Nb": 1, "O": 1, "Cl": 4}, "halide"),
    ],
)
def test_anion_family_is_composition_only_and_deterministic(composition, family):
    assert anion_family(composition) == family
    assert anion_family(dict(reversed(list(composition.items())))) == family


def test_anion_family_rejects_invalid_compositions():
    with pytest.raises(ValueError):
        anion_family({"Xx": 1})
    with pytest.raises(ValueError):
        anion_family({"O": -1})


def test_frozen_registry_is_complete_and_pins_every_checkpoint():
    config = json.loads(Path("configs/protocol_freeze.json").read_text(encoding="utf-8"))
    groups = config["protocol"]["model_groups"]
    names = {group: [m.get("canonical_name") for m in models] for group, models in groups.items()}
    assert names["A"] == [
        "CHGNet-0.3.0",
        "M3GNet-MP-2021.2.8-PES",
        "MACE-MP-0-small-2023-12-10",
        "SevenNet-0-11July2024",
    ]
    assert names["B1"] == ["M3GNet-MatPES-PBE-v2025.1-PES"]
    assert names["B2"] == ["SevenNet-omat"]
    assert groups["C"][0]["resolution_status"] == "optional_unresolved"
    for group in ("A", "B1", "B2"):
        for model in groups[group]:
            assert len(model["weights_sha256"]) == 64
            assert model["adapter_version"] == ADAPTER_VERSION
            assert "latest" not in json.dumps(model).lower()
            assert model["license_identifier"].split()[0] in {
                "MIT",
                "BSD-3-Clause",
                "BSD-3-Clause-LBNL",
                "GPL-3.0",
            }
    matgl = [m for g in groups.values() for m in g if m.get("adapter_name") == "matgl"]
    assert all(m["load_options"]["stress_unit"] == "eV/A3" for m in matgl)


def test_sevennet_releases_are_pinned_independently_with_historical_licenses():
    config = json.loads(Path("configs/protocol_freeze.json").read_text(encoding="utf-8"))
    groups = config["protocol"]["model_groups"]
    sevennet_0 = next(m for m in groups["A"] if m["canonical_name"] == "SevenNet-0-11July2024")
    (sevennet_omat,) = groups["B2"]
    # SevenNet-0 keeps the LiTraj benchmark release; 7net-omat first ships in sevenn 0.11.0.
    assert (sevennet_0["package_name"], sevennet_0["package_version"]) == ("sevenn", "0.10.1")
    assert (sevennet_omat["package_name"], sevennet_omat["package_version"]) == ("sevenn", "0.11.0")
    assert sevennet_0["weights_sha256"] == (
        "7052cb42b7b3be42b40b97fa0d21077a48c54b5548948fc4dcf346629f813c36"
    )
    assert sevennet_omat["weights_sha256"] == (
        "52d7111451a39e6d66b1f4c85cb7cb240dbda48e7bb7341b834ffa0cd8bdc29a"
    )
    assert sevennet_omat["checkpoint_url"].endswith("/v0.11.0.cp/checkpoint_sevennet_omat.pth")
    # Both frozen releases predate the upstream MIT relicensing (first released in v0.12.0).
    for model in (sevennet_0, sevennet_omat):
        assert model["license_identifier"].split()[0] == "GPL-3.0"
        assert model["checkpoint_license"].startswith("GPL-3.0")
        assert "v0.12.0" in model["license_evidence"]


def test_adapter_verifies_checkpoint_identity_before_any_model_import(tmp_path):
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"synthetic weights")
    model = {
        "canonical_name": "synthetic",
        "adapter_name": "mace",
        "adapter_version": ADAPTER_VERSION,
        "weights_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_calculator(model, checkpoint)
    with pytest.raises(ValueError, match="adapter version"):
        build_calculator({**model, "adapter_version": "other"}, checkpoint)
    directory = tmp_path / "matgl_model"
    directory.mkdir()
    (directory / "state.pt").write_bytes(b"state")
    assert checkpoint_digest(directory) == hashlib.sha256(b"state").hexdigest()


def test_snapshot_reports_population_and_family_counts_without_outcomes(tmp_path):
    from test_protocol_freeze import _phase_b_fixture, _sha, _write

    from e3sse.protocol.freeze import materialize_protocol

    config_path, _ = _phase_b_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    linkage_path = Path(config["inputs"]["material_chemical_system_linkage"]["path"])
    linkage = json.loads(linkage_path.read_text())
    compositions = {"Li-O": {"Li": 2, "O": 1}, "Li-S": {"Li": 2, "S": 1},
                    "Cl-Li": {"Li": 1, "Cl": 1}, "Li-N": {"Li": 3, "N": 1},
                    "F-Li": {"Li": 1, "F": 1}}
    for record in linkage["materials"]:
        record["composition"] = compositions[record["chemical_system_id"]]
    _write(linkage_path, linkage)
    config["inputs"]["material_chemical_system_linkage"]["sha256"] = _sha(linkage_path)
    _write(config_path, config)
    protocol, _ = materialize_protocol(config_path)
    partition = protocol["partition_definition"]
    families = partition["anion_families"]
    assert families["material_counts"] == {
        "halide": 2, "nitride": 1, "oxide": 1, "sulfide": 1
    }
    assert families["materials_without_composition"] == []
    population = partition["primary_population"]
    assert population["status"] == "resolved"
    assert set(population["nebdft2k_hop_counts_by_published_split"]) == {"train", "val", "test"}
    exclusions = protocol["study5_exclusions"]
    assert exclusions["rule"] == "entire-partition-universe-v1"
    assert exclusions["count"] == partition["partition_universe"]["counts"]["chemical_systems"]


class _Scaled:
    """Wrap a calculator and corrupt its stress, to prove the smoke test catches slips."""

    def __new__(cls, factor):
        from ase.calculators.lj import LennardJones

        class Broken(LennardJones):
            def calculate(self, *args, **kwargs):
                super().calculate(*args, **kwargs)
                self.results["stress"] = self.results["stress"] * factor

        return Broken(sigma=2.0, epsilon=0.1, rc=6.0, smooth=True)


def test_smoke_test_passes_a_consistent_calculator_and_catches_stress_slips():
    from ase.calculators.lj import LennardJones

    from e3sse.protocol.adapters import smoke_test

    good = smoke_test(LennardJones(sigma=2.0, epsilon=0.1, rc=6.0, smooth=True))
    assert good["passed"], good["checks"]
    assert good["force_energy_consistency"]["forces_are_negative_gradient_within_1e-3_or_5pct"]
    for factor in (-1.0, 1 / 160.21766208, 160.21766208):
        broken = smoke_test(_Scaled(factor))
        assert broken["checks"]["stress_eV_per_A3_ase_convention"] is False
        assert broken["passed"] is False


def test_matgl_directory_checkpoints_verify_every_artifact(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    for name, content in (("state.pt", b"state"), ("model.json", b"{}"), ("model.pt", b"pt")):
        (directory / name).write_bytes(content)
    model = {
        "canonical_name": "synthetic",
        "adapter_name": "matgl",
        "adapter_version": ADAPTER_VERSION,
        "weights_sha256": hashlib.sha256(b"state").hexdigest(),
        "artifact_files": {"model.json": hashlib.sha256(b"tampered").hexdigest()},
    }
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch: model.json"):
        build_calculator(model, directory)


def test_missing_compositions_block_the_freeze_instead_of_reporting_families(tmp_path):
    from test_protocol_freeze import _phase_b_fixture

    from e3sse.protocol.freeze import materialize_protocol

    config_path, _ = _phase_b_fixture(tmp_path)
    protocol, _ = materialize_protocol(config_path)
    assert protocol["partition_definition"]["anion_families"]["status"] == "incomplete_composition"
    assert any("anion family undetermined" in item for item in protocol["freeze_blockers"])


def test_smoke_test_catches_force_unit_and_sign_slips():
    from ase.calculators.lj import LennardJones

    from e3sse.protocol.adapters import smoke_test

    for factor in (-1.0, 1 / 27.211386245988, 10.0):

        class Broken(LennardJones):
            scale = factor

            def calculate(self, *args, **kwargs):
                super().calculate(*args, **kwargs)
                self.results["forces"] = self.results["forces"] * self.scale

        report = smoke_test(Broken(sigma=2.0, epsilon=0.1, rc=6.0, smooth=True))
        assert report["checks"]["force_unit_scale_eV_per_A"] is False
        assert report["passed"] is False
