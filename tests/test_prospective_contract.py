"""Frozen prospective contract: denominators, hybrid labels, family record, model selection."""

from __future__ import annotations

import pytest

from e3sse.screening.prospective_contract import (
    anion_family_record,
    funnel,
    prediction_source,
    select_prospective_model,
)


def _unit(attempted=True, failure=None, supported=True, selected=False):
    return {"attempted": attempted, "failure": failure, "supported": supported,
            "selected": selected}


def test_funnel_uses_explicit_denominators_and_separates_failure_from_abstention():
    units = [_unit(selected=True), _unit(), _unit(supported=False),
             {**_unit(), "weight_fit_holdout": True},
             _unit(failure="optimizer_non_convergence", supported=False),
             _unit(failure="timeout", supported=False), _unit(attempted=False, supported=False)]
    result = funnel(units)
    assert result["counts"] == {"gate1": 7, "ml_neb_attempted": 6, "ml_neb_converged": 4,
                                "applicability_supported": 3, "conformal_predictor_eligible": 4,
                                "weight_fit_holdout": 1, "wcs_run": 3,
                                "selected": 1, "failures_by_category": {
                                    "optimizer_non_convergence": 1, "calculator_exception": 0,
                                    "non_finite_value": 0, "geometry_validity": 0, "timeout": 1,
                                    "other_deterministic": 0}}
    assert result["rates"]["ml_neb_failure_rate"] == pytest.approx(2 / 6)
    assert result["rates"]["applicability_coverage"] == pytest.approx(3 / 4)
    assert result["rates"]["predictor_coverage"] == pytest.approx(4 / 7)
    assert result["rates"]["selection_rate"] == pytest.approx(1 / 3)
    assert result["fdr_population"] == "wcs_run"
    with pytest.raises(ValueError, match="not attempted"):
        funnel([_unit(attempted=False, failure="timeout")])
    with pytest.raises(ValueError, match="WCS-run"):
        funnel([{**_unit(selected=True), "weight_fit_holdout": True}])
    with pytest.raises(ValueError, match="WCS-run"):
        funnel([_unit(failure="timeout", selected=True)])
    with pytest.raises(ValueError, match="unknown failure"):
        funnel([_unit(failure="bad_luck")])


def test_hybrid_fallback_is_labelled_never_called_family_srt():
    assert prediction_source(supported=True, family_scale_available=True) == "srt_family"
    assert prediction_source(supported=True, family_scale_available=False) == \
        "raw_family_scale_unavailable"
    assert prediction_source(supported=False, family_scale_available=True) == \
        "raw_outside_support"


def test_family_record_keeps_frozen_family_and_flags_mixed_anions():
    assert anion_family_record({"Li": 2, "O": 1}) == {
        "family": "oxide", "mixed_anion": False, "anion_groups_present": ["oxide"]}
    mixed = anion_family_record({"Li": 3, "O": 4, "Cl": 1})
    assert mixed["family"] == "oxide" and mixed["mixed_anion"]
    tie = anion_family_record({"Li": 2, "S": 1, "O": 1})
    assert tie["family"] == "other" and tie["mixed_anion"]
    assert anion_family_record({"Li": 1, "P": 1})["family"] == "other"


def _model(name, conv, cov, mae, rmse):
    return {"model": name, "convergence_rate": conv, "independent_support_coverage": cov,
            "mae_common_eV": mae, "rmse_common_eV": rmse}


def test_model_selection_is_lexicographic_with_relative_feasibility():
    records = [_model("A", 0.9, 0.8, 0.10, 0.15), _model("B", 0.3, 0.8, 0.05, 0.07),
               _model("C", 0.85, 0.75, 0.10, 0.14)]
    result = select_prospective_model(records)
    assert result["excluded_infeasible"] == ["B"]  # 0.3 < 0.5 x 0.9
    assert result["selected_model"] == "C"  # MAE tie broken by lower RMSE
    tied = [_model("Z", 0.9, 0.8, 0.1, 0.1), _model("Y", 0.9, 0.8, 0.1, 0.1)]
    assert select_prospective_model(tied)["selected_model"] == "Y"
    with pytest.raises(ValueError):
        select_prospective_model([])
    with pytest.raises(ValueError, match="duplicate"):
        select_prospective_model([records[0], records[0]])
    with pytest.raises(ValueError, match="finite"):
        select_prospective_model([_model("A", 0.9, 0.8, float("nan"), 0.1)])
