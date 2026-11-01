"""Label-free prospective support: hierarchy, development-only fitting, fail-closed gaps."""

from __future__ import annotations

import numpy as np
import pytest

from e3sse.screening.applicability_support import (
    DESCRIPTOR_NAMES,
    applicability_source,
    descriptor_vector,
    fit_reference,
    support_decisions,
    support_distances,
)


def test_fallback_hierarchy_is_deterministic_and_never_combines():
    assert applicability_source(representation_verified=True, descriptors_available=True) == \
        "frozen_model_representation"
    assert applicability_source(representation_verified=False, descriptors_available=True) == \
        "descriptor_space"
    assert applicability_source(representation_verified=False, descriptors_available=False) == \
        "unavailable"


def test_descriptors_are_label_free_and_complete_or_missing():
    row = descriptor_vector(["Li", "Li", "O"], cell_volume_A3=30.0, hop_length_A=2.5,
                            em_bvse_eV=0.4)
    assert len(row) == len(DESCRIPTOR_NAMES)
    assert row[0] == pytest.approx(2 / 3)
    assert row[DESCRIPTOR_NAMES.index("family_oxide")] == 1.0
    missing = descriptor_vector(["Li", "S"], cell_volume_A3=20.0, hop_length_A=None,
                                em_bvse_eV=0.3)
    assert missing[DESCRIPTOR_NAMES.index("hop_length_A")] is None
    with pytest.raises(ValueError):
        descriptor_vector([], cell_volume_A3=1.0, hop_length_A=1.0, em_bvse_eV=0.1)


def test_reference_is_fitted_on_development_only_and_gate_is_fixed():
    rng = np.random.default_rng(0)
    development = rng.normal(size=(60, 3)).tolist()
    systems = [f"S{i // 3}" for i in range(60)]
    reference = fit_reference(development, chemical_systems=systems, k=5, quantile_level=0.8)
    near = [[0.0, 0.0, 0.0]]
    far = [[8.0, 8.0, 8.0]]
    decisions = support_decisions(reference, support_distances(reference, near + far + [[1.0, None, 0.0]]))
    assert [d["supported"] for d in decisions] == [True, False, False]
    assert [d["reason"] for d in decisions] == [None, "outside_support", "missing_feature"]
    # About 80% of the development units fall within their own LOO threshold.
    again = fit_reference(development, chemical_systems=systems, k=5, quantile_level=0.8)
    assert again == reference
    # Calibration or prospective units never change the frozen gate.
    calibration = rng.normal(loc=3.0, size=(40, 3)).tolist()
    support_distances(reference, calibration)
    assert fit_reference(development, chemical_systems=systems, k=5,
                         quantile_level=0.8)["tau_d"] == reference["tau_d"]


def test_threshold_leaves_out_whole_chemical_systems():
    # Tight clusters of sibling hops: unit-level LOO would give a tiny threshold.
    rng = np.random.default_rng(1)
    centres = rng.normal(scale=5.0, size=(20, 2))
    rows = [(c + rng.normal(scale=0.01, size=2)).tolist() for c in centres for _ in range(4)]
    systems = [f"S{i}" for i in range(20) for _ in range(4)]
    by_system = fit_reference(rows, chemical_systems=systems, k=3, quantile_level=0.5)
    by_unit = fit_reference(rows, chemical_systems=[str(i) for i in range(80)], k=3,
                            quantile_level=0.5)
    assert by_system["tau_d"] > 10 * by_unit["tau_d"]


def test_reference_rejects_too_small_or_invalid_populations():
    with pytest.raises(ValueError):
        fit_reference([[0.0], [1.0]], chemical_systems=["a", "b"], k=5, quantile_level=0.8)
    with pytest.raises(ValueError):
        fit_reference([[0.0]] * 10, chemical_systems=list("abcdefghij"), k=2,
                      quantile_level=1.0)
    with pytest.raises(ValueError, match="one chemical system"):
        fit_reference([[0.0]] * 10, chemical_systems=["a"], k=2, quantile_level=0.5)
    constant = fit_reference([[1.0, float(i)] for i in range(10)],
                             chemical_systems=list("abcdefghij"), k=2, quantile_level=0.5)
    assert constant["sd"][0] == 1.0


def test_family_one_hots_keep_unit_weight_whatever_their_frequency():
    from e3sse.screening.applicability_support import FAMILY_COLUMNS

    rows = [[float(i % 7), 1.0, 0.0] for i in range(40)] + [[3.0, 0.0, 1.0]]  # rare family
    systems = [f"S{i}" for i in range(41)]
    scaled = fit_reference(rows, chemical_systems=systems, k=3, quantile_level=0.8)
    fixed = fit_reference(rows, chemical_systems=systems, k=3, quantile_level=0.8,
                          unscaled_columns=(1, 2))
    assert fixed["sd"][1:] == [1.0, 1.0] and fixed["mean"][1:] == [0.0, 0.0]
    assert scaled["sd"][2] < 1.0  # z-scoring would inflate the rare family's weight
    assert len(FAMILY_COLUMNS) == 5
