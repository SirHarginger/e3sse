"""Frozen ML-NEB engine: explicit settings, determinism and classified failures (EMT only)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT

from e3sse.barriers.ml_neb import (
    MLNEBSettingsError,
    build_neb,
    reference_emt_vacancy_band,
    run_ml_neb,
    settings_from_protocol,
)
from e3sse.screening.prospective_contract import FAILURE_CATEGORIES

REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((REPOSITORY / "configs" /
                       "reliability_gated_internal_freeze_protocol.json").read_text())
SETTINGS = settings_from_protocol(PROTOCOL)
PROVENANCE = {"model": "emt-test", "weights_sha256": None, "environment_lock_sha256": None,
              "dtype": "float64", "device": "cpu", "torch_intra_op_threads": None,
              "torch_inter_op_threads": None}


class CountingEMT(EMT):
    calls: int = 0

    def calculate(self, *args, **kwargs):
        CountingEMT.calls += 1
        super().calculate(*args, **kwargs)


class RaisingCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        raise RuntimeError("model blew up")


class NaNCalculator(EMT):
    def calculate(self, *args, **kwargs):
        super().calculate(*args, **kwargs)
        self.results["forces"] = np.full_like(self.results["forces"], np.nan)


def test_settings_are_read_explicitly_from_the_protocol():
    assert SETTINGS.method == "improvedtangent" and SETTINGS.climb is True and SETTINGS.k == 5.0
    assert SETTINGS.fire_parameters["maxstep"] == 0.2 and SETTINGS.endpoint_steps == 100
    assert SETTINGS.convergence_max_force_component == 0.1
    broken = copy.deepcopy(PROTOCOL)
    del broken["amendments"]["prospective"]["ml_neb"]["frozen_numeric_settings"]["fire"]["dt"]
    with pytest.raises(MLNEBSettingsError, match="fire lacks"):
        settings_from_protocol(broken)
    implicit = copy.deepcopy(PROTOCOL)
    del implicit["amendments"]["prospective"]["ml_neb"]["frozen_numeric_settings"][
        "neb_constructor"]["method"]
    with pytest.raises(MLNEBSettingsError, match="neb_constructor keys"):
        settings_from_protocol(implicit)


def test_neb_is_built_with_the_frozen_method_not_the_ase_default():
    band = reference_emt_vacancy_band()
    for image in band:
        image.calc = EMT()
    neb = build_neb(band, SETTINGS)
    assert type(neb.neb_method).__name__ == "ImprovedTangentMethod"
    assert neb.climb is True and neb.parallel is False and not neb.allow_shared_calculator


def test_legacy_ase_scipy_alias_is_narrow_and_idempotent(monkeypatch):
    from scipy import integrate

    from e3sse.barriers.ml_neb import _legacy_ase_scipy_compatibility

    replacement = integrate.cumulative_trapezoid
    monkeypatch.setattr(integrate, "cumtrapz", None, raising=False)  # undone after the test
    assert _legacy_ase_scipy_compatibility() == "scipy_cumulative_trapezoid_alias"
    assert integrate.cumtrapz is replacement
    assert _legacy_ase_scipy_compatibility() == "native"


def test_converged_band_is_deterministic_and_reports_a_barrier():
    first = run_ml_neb(reference_emt_vacancy_band(), EMT, SETTINGS, unit_id="u", provenance=PROVENANCE,
                       migrating_species="Cu")
    second = run_ml_neb(reference_emt_vacancy_band(), EMT, SETTINGS, unit_id="u", provenance=PROVENANCE,
                        migrating_species="Cu")
    assert first == second
    assert first["status"] == "converged" and first["failure_category"] is None
    energies = first["energies_eV"]
    assert math.isclose(first["barrier_eV"], max(energies) - min(energies))
    assert first["neb_method"] == "improvedtangent" and first["retry_policy"] == "none"
    assert first["max_force_component_eV_per_A"]["band"] < 0.1
    assert len(first["band_geometry"]["positions"]) == 7
    assert first["provenance"] == PROVENANCE and "highest_energy_image_index" in first
    json.dumps(first, allow_nan=False)


def test_step_budget_exhaustion_is_non_convergence_without_retry():
    tight = SETTINGS.__class__(**{**SETTINGS.__dict__, "band_steps": 2})
    CountingEMT.calls = 0
    record = run_ml_neb(reference_emt_vacancy_band(), CountingEMT, tight, unit_id="u", provenance=PROVENANCE,
                        migrating_species="Cu")
    assert record["failure_category"] == "optimizer_non_convergence"
    assert record["band_optimizer_steps"] == 2 and record["barrier_eV"] is None
    calls = CountingEMT.calls
    run_ml_neb(reference_emt_vacancy_band(), CountingEMT, tight, unit_id="u", provenance=PROVENANCE,
               migrating_species="Cu")
    assert CountingEMT.calls == 2 * calls  # the same work again, never an adaptive retry


def test_calculator_exception_and_non_finite_values_are_classified():
    raised = run_ml_neb(reference_emt_vacancy_band(), RaisingCalculator, SETTINGS, unit_id="u", provenance=PROVENANCE,
                        migrating_species="Cu")
    assert raised["failure_category"] == "calculator_exception"
    assert "model blew up" in raised["failure_detail"]
    nan = run_ml_neb(reference_emt_vacancy_band(), NaNCalculator, SETTINGS, unit_id="u", provenance=PROVENANCE,
                     migrating_species="Cu")
    assert nan["failure_category"] in ("non_finite_value", "calculator_exception")
    assert {raised["failure_category"], nan["failure_category"]} <= set(FAILURE_CATEGORIES)


def test_malformed_geometry_is_its_own_failure_category():
    band = reference_emt_vacancy_band()
    band[2].positions[0] = band[2].positions[1] + 0.1  # two atoms collapse
    record = run_ml_neb(band, EMT, SETTINGS, unit_id="u", provenance=PROVENANCE, migrating_species="Cu")
    assert record["failure_category"] == "geometry_validity" and record["failure_stage"] == "input"
    assert record["failure_origin"] == "input_path"  # model-independent initialisation failure
    reordered = reference_emt_vacancy_band()
    reordered[3].numbers[0] = 29 if reordered[3].numbers[0] != 29 else 13
    record = run_ml_neb(reordered, EMT, SETTINGS, unit_id="u", provenance=PROVENANCE, migrating_species="Cu")
    assert record["failure_category"] == "geometry_validity"
    no_li = run_ml_neb(reference_emt_vacancy_band(), EMT, SETTINGS, unit_id="u", provenance=PROVENANCE)
    assert no_li["failure_category"] == "geometry_validity" and "no Li" in no_li["failure_detail"]


def test_infrastructure_errors_are_raised_not_recorded():
    class OOM(EMT):
        def calculate(self, *args, **kwargs):
            raise MemoryError("worker out of memory")

    with pytest.raises(MemoryError):
        run_ml_neb(reference_emt_vacancy_band(), OOM, SETTINGS, unit_id="u", provenance=PROVENANCE,
                   migrating_species="Cu")


def test_torch_allocator_oom_is_infrastructure_not_a_model_failure():
    class TorchOOM(EMT):
        def calculate(self, *args, **kwargs):
            raise RuntimeError("[enforce fail at alloc_cpu.cpp:83] DefaultCPUAllocator: can't "
                               "allocate memory: you tried to allocate 123456789 bytes")

    with pytest.raises(MemoryError, match="allocator out of memory"):
        run_ml_neb(reference_emt_vacancy_band(), TorchOOM, SETTINGS, unit_id="u",
                   provenance=PROVENANCE, migrating_species="Cu")


def test_nan_then_crash_is_classified_non_finite():
    class NaNThenRaise(EMT):
        calls: int = 0

        def calculate(self, atoms=None, *args, **kwargs):
            if not np.isfinite(atoms.positions).all():
                raise ValueError("graph builder received non-finite positions")
            super().calculate(atoms, *args, **kwargs)
            self.results["forces"] = np.full_like(self.results["forces"], np.nan)

    record = run_ml_neb(reference_emt_vacancy_band(), NaNThenRaise, SETTINGS, unit_id="u",
                        provenance=PROVENANCE, migrating_species="Cu")
    assert record["failure_category"] == "non_finite_value"
    assert record["failure_origin"] == "model_run"


def test_provenance_is_required():
    with pytest.raises(ValueError, match="provenance lacks"):
        run_ml_neb(reference_emt_vacancy_band(), EMT, SETTINGS, unit_id="u", provenance={},
                   migrating_species="Cu")


def test_every_triggered_final_check_is_recorded():
    tight = SETTINGS.__class__(**{**SETTINGS.__dict__, "band_steps": 1,
                                  "site_collapse_A": 100.0})
    record = run_ml_neb(reference_emt_vacancy_band(), EMT, tight, unit_id="u",
                        provenance=PROVENANCE, migrating_species="Cu")
    categories = [check["category"] for check in record["failure_checks"]]
    assert categories == ["geometry_validity", "optimizer_non_convergence"]
    assert record["failure_category"] == "geometry_validity"
