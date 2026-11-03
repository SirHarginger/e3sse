"""Production stage runners on synthetic, outcome-free inputs (no dataset is read)."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from e3sse.production.runners import deployed, inference_stats, population, representation
from e3sse.production.runners.bvse_paths import compare_paths
from e3sse.production.runners.context import (
    MissingEvidence,
    RunnerContext,
    RunnerValidationError,
)
from e3sse.production.runners.frame_scales import FrameStats, hop_diagnostics, scope_frames
from e3sse.production.runners.ml_neb_units import compare_reruns, reproducibility_subset
from e3sse.production.runners.pilot import mde_simulation
from e3sse.production.runners.recalibration import fit_linear
from e3sse.protocol.chemistry import primary_exclusion_elements

REPOSITORY = Path(__file__).resolve().parents[1]
SUPPORT = {"full_estimate_min_frames": 10, "full_estimate_min_informative_components": 100,
           "informative_force_eV_per_A": 0.1, "pooled_min_materials": 3}


# ----------------------------------------------------------------------- frame statistics

def synthetic_statistics(n_frames: int, *, scale: float = 0.8, seed: int = 3):
    """Study-1-format arrays: two elements (Li, O) per frame, informative forces."""

    rng = np.random.default_rng(seed)
    rows, z, sxy, sxx, syy, info = [], [], [], [], [], []
    for frame in range(n_frames):
        for element in (3, 8):
            dft = rng.normal(0, 1, size=30)
            ml = scale * dft + rng.normal(0, 0.05, size=30)
            rows.append(frame)
            z.append(element)
            sxy.append(float(dft @ ml))
            sxx.append(float(dft @ dft))
            syy.append(float(ml @ ml))
            info.append(int((np.abs(dft) >= 0.1).sum()))
    model = {"frame_index": np.arange(n_frames), "element_frame_row": np.asarray(rows),
             "element_Z": np.asarray(z), "element_sxy": np.asarray(sxy),
             "element_sxx": np.asarray(sxx), "element_syy": np.asarray(syy),
             "element_n_informative": np.asarray(info), "success": np.ones(n_frames, bool)}
    reference = {"frame_index": np.arange(n_frames), "n_atoms": np.full(n_frames, 20),
                 "dft_energy_eV": rng.normal(-100, 1, size=n_frames)}
    return model, reference


def test_frame_stats_recover_the_scale_with_the_frozen_support_rule():
    model, reference = synthetic_statistics(40, scale=0.8)
    stats = FrameStats([f"f{i}" for i in range(40)], model, reference)
    estimate = stats.estimate(stats.rows([f"f{i}" for i in range(40)]), "all",
                              materials=["m"], pooled=False, parameters=SUPPORT)
    assert estimate["supported"] and abs(estimate["s_hat"] - 0.8) < 0.01
    small = stats.estimate(stats.rows(["f0", "f1"]), "all", materials=["m"], pooled=False,
                           parameters=SUPPORT)
    assert small["s_hat"] is None and small["unsupported_reasons"]
    first = stats.bootstrap(stats.rows([f"f{i}" for i in range(40)]), "all", seed=11, resamples=200)
    assert first == stats.bootstrap(stats.rows([f"f{i}" for i in range(40)]), "all", seed=11,
                                    resamples=200)
    diag = hop_diagnostics(stats, stats.rows([f"f{i}" for i in range(40)]), material="m",
                           seed=5, resamples=300, parameters=SUPPORT)
    assert diag["rho_F"] is not None and 0 <= diag["rho_F"] < 0.2
    assert diag["scale_uncertainty"] is not None and diag["li_host_disagreement"] is not None


def test_failed_model_frames_make_the_estimate_unsupported():
    model, reference = synthetic_statistics(20)
    model["success"][3] = False
    stats = FrameStats([f"f{i}" for i in range(20)], model, reference)
    estimate = stats.estimate(stats.rows([f"f{i}" for i in range(20)]), "all", materials=["m"],
                              pooled=False, parameters=SUPPORT)
    assert "model_frame_failure" in estimate["unsupported_reasons"]


def test_s_scope_excludes_every_path_frame_and_the_oracle_is_own_path_only(tmp_path):
    sys.path.insert(0, str(REPOSITORY / "tests"))
    import study1_fixtures as fx

    universe = fx.build_universe(tmp_path)
    membership, protocol = universe["membership"], universe["protocol"]
    available = {r["frame_id"] for r in membership if r.get("record_kind") == "mplitrj_frame"}
    hops = [r for r in membership if r.get("record_kind") == "nebdft2k_hop"
            and r["hop_id"] == fx.CONTAMINATED_HOP]
    kinds = {r["frame_id"]: (r["raw_file_kind"], r["hop_id"]) for r in membership
             if r.get("record_kind") == "mplitrj_frame"}
    primary = scope_frames(protocol=protocol, membership=membership, hops=hops,
                           available=available, partition="published",
                           side="training_or_calibration")[fx.CONTAMINATED_HOP]
    assert primary and all(kinds[f][0] in ("source", "target") for f in primary)
    oracle = scope_frames(protocol=protocol, membership=membership, hops=hops, available=available,
                          partition="published", side="training_or_calibration",
                          estimator_kind="oracle")[fx.CONTAMINATED_HOP]
    assert oracle and all(kinds[f] == ("band_optim_step", fx.CONTAMINATED_HOP) for f in oracle)
    assert not set(primary) & set(oracle)


# ----------------------------------------------------------------------- populations

def test_weight_fit_slice_is_identity_only_and_near_the_frozen_fraction():
    systems = [f"Li-X{i}" for i in range(4000)]
    chosen = [s for s in systems if population.weight_fit_slice("p" * 64, s, 0.2)]
    assert 0.17 < len(chosen) / len(systems) < 0.23
    assert chosen == [s for s in systems if population.weight_fit_slice("p" * 64, s, 0.2)]
    assert chosen != [s for s in systems if population.weight_fit_slice("q" * 64, s, 0.2)]


def _vector(family="oxide", hop=2.5):
    one_hot = [1.0 if f == family else 0.0 for f in ("oxide", "sulfide", "halide", "nitride", "other")]
    return [0.3, 8.0, 3.0, 12.0, 0.03, hop, 0.4, *one_hot]


def _unit(side, system, material):
    edge = f"{material}_0_1_0_0_0"
    return {"side": side, "chemical_system": system, "unit_edge_id": edge, "material_id": material,
            "rate_limiting_edge_ids": [edge], "unit_edge_ids": [edge],
            "unit_hop_role": "rate_limiting_exact", "critical_bvse_barrier_eV": 0.4,
            "tm_hubbard_u": False}


def test_population_roles_gate0_and_development_manifest_hash(monkeypatch):
    from e3sse.protocol.contracts import _verify_development_manifest

    # Li-P-S is a P weight-fit-slice system (a Gate-0 reference unit); Li-O-Cl is a P
    # calibration unit.  Calibration units never shape the Gate-0 envelope.
    slice_systems = {"Li-P-S"}
    monkeypatch.setattr(population, "weight_fit_slice",
                        lambda parent, system, fraction: system in slice_systems)
    units = [_unit("development", "Li-O", "mp-1"), _unit("P", "Li-P-S", "mp-2"),
             _unit("P", "Cl-Li-O", "mp-3"), _unit("P", "Li-O-P", "mp-4"),
             _unit("Q", "Li-O-S", "mp-7"), _unit("Q", "Ge-Li", "mp-8"),
             _unit("Q", "Li-O-P-S", "mp-9"), _unit("Q", "Li-S", "mp-10")]
    descriptors = {
        "development:Li-O": {"vector": _vector(hop=2.0), "chemical_system": "Li-O"},
        "P:Li-P-S": {"vector": _vector("sulfide", 3.0), "chemical_system": "Li-P-S"},
        # wide calibration hop and an extra element: must not widen the envelope
        "P:Cl-Li-O": {"vector": _vector(hop=9.0), "chemical_system": "Cl-Li-O"},
        "P:Li-O-P": {"vector": _vector(hop=2.5), "chemical_system": "Li-O-P"},
        "Q:Li-O-S": {"vector": _vector(hop=2.5), "chemical_system": "Li-O-S"},
        "Q:Ge-Li": {"vector": _vector(hop=2.5), "chemical_system": "Ge-Li"},
        "Q:Li-O-P-S": {"vector": _vector(hop=8.0), "chemical_system": "Li-O-P-S"},
        "Q:Li-S": {"vector": None, "chemical_system": "Li-S"}}
    built, manifest = population.build_population(units, descriptors, parent_protocol_id="p" * 64,
                                                  fraction=0.2)
    roles = {u["unit_key"]: u["role"] for u in built["units"]}
    assert roles["development:Li-O"] == "development"
    assert roles["P:Li-P-S"] == "calibration_weight_slice"
    assert roles["P:Cl-Li-O"] == "excluded_gate0"  # Gate 0 applies to P calibration units too
    assert roles["P:Li-O-P"] == "calibration"
    assert roles["Q:Ge-Li"] == "excluded_gate0"  # Ge outside the reference elements
    assert roles["Q:Li-O-P-S"] == "excluded_gate0"  # 8.0 A only inside a calibration unit's range
    assert roles["Q:Li-S"] == "excluded_missing_covariate"  # missing evidence, not a Gate-0 failure
    assert roles["Q:Li-O-S"] == "pool"
    assert built["gate0"]["hop_length_range_A"] == [2.0, 3.0]
    assert built["gate0"]["element_support"] == ["Li", "O", "P", "S"]
    internal = {"amendments": {"prospective": {"development_manifest_sha256": manifest["manifest_sha256"]}}}
    _verify_development_manifest(manifest, internal, ["mp-1_0_1_0_0_0"])  # contract-compatible hash
    with pytest.raises(ValueError):
        _verify_development_manifest(manifest, internal, ["mp-2_0_1_0_0_0"])  # not development


def test_weight_fit_slice_is_an_exact_identity_only_rule():
    import hashlib
    from fractions import Fraction

    for system in ("Li-O", "Li-P-S", "Cl-Li", "Ge-Li-S"):
        digest = hashlib.sha256(b"e3sse.weight-holdout\0" + b"pid\0" + system.encode()).hexdigest()
        assert population.weight_fit_slice("pid", system, 0.2) == (
            Fraction(int(digest, 16), 2 ** 256) < Fraction(1, 5))


# ----------------------------------------------------------------------- representation/support

def test_path_vector_orients_by_ml_energy_and_picks_the_interior_saddle():
    embeddings = [[float(i), 0.0] for i in range(5)]
    forward = representation.path_vector(embeddings, [0.0, 0.3, 0.9, 0.2, 0.1])
    assert forward[:2] == [0.0, 0.0] and forward[2:4] == [2.0, 0.0] and forward[4:6] == [4.0, 0.0]
    reversed_ = representation.path_vector(embeddings, [0.5, 0.3, 0.9, 0.2, 0.1])
    assert reversed_[:2] == [4.0, 0.0]  # lower-energy endpoint first


def test_degeneracy_check_and_deployed_predictor():
    from e3sse.screening.applicability_support import fit_reference

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(40, 4)).tolist()
    reference = fit_reference(rows, chemical_systems=[f"s{i}" for i in range(40)], k=5,
                              quantile_level=0.8) | {"unit_keys": [f"u{i}" for i in range(40)],
                                                     "source": "frozen_model_representation"}
    check = representation.degeneracy(reference, seed=1)
    assert check["locso_cv"] > 0 and math.isfinite(check["tau_bootstrap_cv"])
    branch = {"branch": "latent_support", "reference": reference}
    own = deployed.support_decisions(branch, {"u0": rows[0]}, development=True)
    assert own["u0"]["distance"] == reference["loo_distances"][0]
    record = {"status": "converged", "barrier_eV": 0.4}
    scales = {"family": {"oxide": {"s_hat": 0.8}}}
    srt = deployed.predict(record, family="oxide", scales=scales, supported=True, g1_passes=True)
    assert srt["source"] == "srt_family" and abs(srt["prediction_eV"] - 0.5) < 1e-12
    assert deployed.predict(record, family="halide", scales=scales, supported=True,
                            g1_passes=True)["source"] == "raw_family_scale_unavailable"
    assert deployed.predict(record, family="oxide", scales=scales, supported=False,
                            g1_passes=True)["source"] == "raw_outside_support"
    assert deployed.predict(record, family="oxide", scales=scales, supported=True,
                            g1_passes=False)["source"] == "raw_g1_contingency"
    assert deployed.predict({"status": "failed"}, family="oxide", scales=scales, supported=True,
                            g1_passes=True) is None  # ML-NEB failure is not abstention


# ----------------------------------------------------------------------- misc runners

def test_bvse_reproduction_tolerance():
    path = [{"numbers": [3, 8], "positions": [[0, 0, 0], [1, 1, 1]], "cell": np.eye(3).tolist() * 1,
             "pbc": [True] * 3}]
    path[0]["cell"] = (np.eye(3) * 5).tolist()
    near = [{**path[0], "positions": [[0.005, 0, 0], [1, 1, 1]]}]
    far = [{**path[0], "positions": [[0.05, 0, 0], [1, 1, 1]]}]
    settings = {"cell_tolerance_A": 1e-4, "position_tolerance_A": 0.01}
    assert compare_paths(path, near, settings)["matches"]
    assert compare_paths(path, far, settings)["reason"] == "geometry_outside_tolerance"
    assert compare_paths(path, path * 2, settings)["reason"] == "image_count_differs"


def test_pilot_mde_is_a_power_analysis():
    barriers = [0.3, 0.5, 0.7, 0.4] * 10
    clusters = [f"c{i // 2}" for i in range(len(barriers))]
    args = ([0.6, 0.7, 0.8], 0.02, barriers, clusters)
    first = mde_simulation(*args, noise_eV=0.02, alpha=0.025, seed=4, simulations=60,
                           resamples=100)
    assert first == mde_simulation(*args, noise_eV=0.02, alpha=0.025, seed=4, simulations=60,
                                   resamples=100)
    curve = first["power_curve"]
    assert curve[0]["lambda"] == 0.0 and abs(curve[0]["true_rho_srt"]) < 1e-12  # no correction
    assert curve[0]["power"] <= 0.1  # the test holds its level at no effect
    assert curve[-1]["true_rho_srt"] > 0.5 and curve[-1]["power"] >= 0.8
    assert first["mde_rho_srt"] is not None and 0 < first["mde_rho_srt"] <= curve[-1]["true_rho_srt"]


def test_holm_and_cluster_bootstrap():
    # an undefined member stays in the family (m = 3) and enters as p = 1
    adjusted = inference_stats.holm({"a": 0.01, "b": 0.04, "c": None})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.08, "c": 1.0})
    assert inference_stats.p_resolution([1.0, 2.0, None])["p_reported_as"] == "< 0.5"
    clusters = ["x", "x", "y", "z", "z"]
    values = inference_stats.cluster_bootstrap(clusters, lambda idx: float(len(idx)), seed=1,
                                               resamples=50)
    assert values == inference_stats.cluster_bootstrap(clusters, lambda idx: float(len(idx)),
                                                       seed=1, resamples=50)
    assert inference_stats.one_sided_p([-1.0, 1.0, 2.0, None]) == pytest.approx(1 / 3)


def test_linear_recalibration_and_reruns():
    a, b = fit_linear([0.1, 0.2, 0.3], [0.25, 0.45, 0.65])
    assert a == pytest.approx(2.0) and b == pytest.approx(0.05)
    subset = reproducibility_subset([f"mp-{i}_0_1_0_0_0" for i in range(50)])
    assert len(subset) == 20 and subset == sorted(subset, key=lambda h: (
        hashlib.sha256(h.encode()).hexdigest(), h))
    same = [{"unit_key": "u", "status": "converged", "barrier_eV": 0.5}]
    assert compare_reruns(same, same)["all_pass"]
    assert not compare_reruns(same, [{**same[0], "barrier_eV": 0.6}])["all_pass"]


def test_family_scale_frames_exclude_hop_systems_and_secondary_population():
    from e3sse.production.runners.family_scale import scale_frames

    membership = [
        {"record_kind": "nebdft2k_hop", "chemical_system_id": "Li-O"},
        {"record_kind": "mplitrj_frame", "chemical_system_id": "Li-O", "published_split": "train",
         "frame_id": "a"},
        {"record_kind": "mplitrj_frame", "chemical_system_id": "Li-S", "published_split": "train",
         "frame_id": "b"},
        {"record_kind": "mplitrj_frame", "chemical_system_id": "Fe-Li", "published_split": "train",
         "frame_id": "c"},
        {"record_kind": "mplitrj_frame", "chemical_system_id": "Li-S", "published_split": "test",
         "frame_id": "d"}]
    frames = scale_frames(membership, {"Li-O": "primary", "Li-S": "primary",
                                       "Fe-Li": "tm_hubbard_u_secondary"})
    assert [f["frame_id"] for f in frames] == ["b"]
    assert [f["frame_id"] for f in scale_frames(membership, {"Li-S": "primary"},
                                                training_systems={"Li-S"})] == ["b", "d"]


def test_reliability_reports_curves_and_the_p4_endpoint():
    from e3sse.production.runners.reliability import model_reliability

    rng = np.random.default_rng(2)
    rows = []
    for i in range(30):
        rho = float(rng.uniform(0.01, 0.2))
        dft = float(rng.uniform(0.1, 0.8))
        rows.append({"hop_id": f"h{i}", "chemical_system": f"s{i % 10}", "raw_barrier_eV": dft * 0.8,
                     "dft_barrier_eV": dft, "srt_material_eV": dft + rho * rng.normal(),
                     "gate_passes": rho < 0.15,
                     "scores": {"rho_F": rho, "scale_uncertainty": rho / 2,
                                "li_host_disagreement": None, "energy_dependence": rho}})
    result = model_reliability(rows, latent_distance={}, seed=3, resamples=100)
    assert "P4" in result and result["P4"]["p_one_sided"] is not None
    curve = result["scores"]["rho_F"]["selective_primary_curve"]
    assert len(curve["coverage"]) == 30 and curve["coverage"][-1] == 1.0


def test_funnel_keeps_failure_and_abstention_apart():
    from e3sse.production.runners.discovery_selection import funnel_units
    from e3sse.screening.prospective_contract import funnel

    units = {"units": [{"unit_key": "Q:a", "role": "pool"}, {"unit_key": "Q:b", "role": "pool"},
                       {"unit_key": "Q:c", "role": "pool"}, {"unit_key": "Q:d", "role": "pool"},
                       {"unit_key": "Q:e", "role": "pool_weight_slice"}]}
    pool = [{"unit_key": "Q:a", "ml_neb_status": "converged", "failure_category": None,
             "supported": True},
            {"unit_key": "Q:b", "ml_neb_status": "converged", "failure_category": None,
             "supported": False},
            {"unit_key": "Q:c", "ml_neb_status": "failed",
             "failure_category": "optimizer_non_convergence", "supported": None}]
    slices = [{"unit_key": "Q:e", "status": "converged", "failure_category": None}]
    counts = funnel(funnel_units(units, pool, slices, {"Q:d": {"status": "decorator_failure"}}))
    assert counts["counts"]["gate1"] == 5 and counts["counts"]["ml_neb_attempted"] == 4
    assert counts["counts"]["failures_by_category"]["optimizer_non_convergence"] == 1
    assert counts["counts"]["applicability_supported"] == 1
    assert counts["counts"]["weight_fit_holdout"] == 1


# ----------------------------------------------------------------------- runner CLI and context

def test_runner_cli_publishes_a_manifest_and_maps_failures(tmp_path, monkeypatch):
    import types

    from e3sse.production import runners

    module = types.ModuleType("synthetic_runner")
    state = {"mode": "ok"}

    def run(ctx):
        if state["mode"] == "missing":
            raise MissingEvidence("server evidence absent")
        if state["mode"] == "invalid":
            raise RunnerValidationError("contradiction")
        return {"outputs": {"result": ctx.publish("result.json", {"value": 1})},
                "summary": {"ok": True}}

    module.run = run
    monkeypatch.setitem(sys.modules, "synthetic_runner", module)
    monkeypatch.setitem(runners.RUNNER_MODULES, "publication", "synthetic_runner")
    sys.path.insert(0, str(REPOSITORY / "scripts"))
    import run_production_stage

    config = tmp_path / "production.json"
    config.write_text("{}")
    argv = ["--config", str(config), "--run-dir", str(tmp_path / "run"), "--stage", "publication",
            "--phase", "reduce"]
    assert run_production_stage.main(argv) == 0
    manifest = json.loads((tmp_path / "run" / "publication" / "stage_manifest.json").read_text())
    assert set(manifest["outputs"]) == {"result"}
    assert run_production_stage.main(argv) == 0  # resume: identical, create-once
    state["mode"] = "missing"
    assert run_production_stage.main(argv) == 2
    state["mode"] = "invalid"
    assert run_production_stage.main(argv) == 3


def test_non_training_outcomes_need_the_pilot_of_this_run(tmp_path, monkeypatch):
    ctx = RunnerContext(repository=REPOSITORY, config_path=tmp_path / "c.json",
                        run_dir=tmp_path / "run", stage="transfer")
    monkeypatch.setattr(RunnerContext, "internal_freeze", lambda self: {"protocol_id": "x"})
    with pytest.raises(MissingEvidence, match="pilot amendment"):
        ctx.require_outcome(role="test", outcome="x", outcome_kind="nebdft2k_fixed_geometry_"
                            "evaluation_barriers", unit_ids=["h"])


def test_gate0_element_list_is_the_frozen_code_rule():
    assert "Fe" in primary_exclusion_elements() and "Li" not in primary_exclusion_elements()


# ----------------------------------------------------------------------- study 4

def test_study4_family_fallback_is_symmetric_and_per_family(monkeypatch):
    from e3sse.production.runners import family_scale, study4

    class FakePooled:
        def estimate(self, frame_ids, *, materials, parameters):
            frame_ids = list(frame_ids)
            return {"s_hat": 0.8 if frame_ids else None, "frames": frame_ids,
                    "unsupported_reasons": [] if frame_ids else ["no_frames"]}

    # the training-side stratum has only oxide frames: sulfide needs the fallback
    monkeypatch.setattr(family_scale, "scale_frames", lambda membership, population_of, *,
                        training_systems: [{"frame_id": "t1", "material_id": "m-ox"}])
    family_of = {"m-ox": "oxide", "m-s1": "sulfide", "m-s2": "sulfide", "m-ox2": "oxide"}
    unit_frames = {"h1": [("c1", "m-s1")], "h2": [("c2", "m-s2")], "h3": [("c3", "m-ox2")]}
    scales = study4.repeat_scales(FakePooled(), [], {}, family_of, set(), unit_frames, {})
    assert scales["family"]["oxide"]["source"] == "training_side_systems_without_nebdft2k_hops"
    sulfide = scales["family"]["sulfide"]
    assert sulfide["source"] == "symmetric_fallback_family_units_forces"
    assert sulfide["frames"] == ["c1", "c2"]  # forces of that family's units only, both sides


def test_study4_h5b_is_paired_by_draw_and_resamples_repeats():
    from e3sse.production.runners import study4

    records = []
    for repeat in range(4):
        for draw in range(3):
            for predictor, power in (("raw", 0.4), ("SRT_global", 0.6), ("SRT_family", 0.4),
                                     ("SRT_gated_correct_or_raw", None)):
                records.append({"repeat": repeat, "draw": draw, "predictor": predictor,
                                "c_eV": 0.5, "alpha": 0.2, "method": "BH", "power": power})
    contract = {"threshold_eV": [0.5], "alpha": [0.2]}
    h5b = study4.h5b_power_differences(records, contract, seed=3, resamples=200)
    glob = h5b["SRT_global-raw|c=0.5|alpha=0.2|BH"]
    assert glob["mean_power_difference"] == pytest.approx(0.2) and glob["repeats"] == 4
    assert glob["supported"] is True
    assert h5b["SRT_family-raw|c=0.5|alpha=0.2|BH"]["supported"] is False
    assert h5b["SRT_gated_correct_or_raw-raw|c=0.5|alpha=0.2|BH"]["repeats"] == 0
    assert study4._repeat_se([{"repeat": r, "power": p} for r, p in
                              ((0, 0.2), (1, 0.4), (2, None))], "power") == pytest.approx(0.1)


# ----------------------------------------------------------------------- publication

def test_every_registered_figure_is_built_or_declared_exactly_once():
    import json

    from e3sse.production.runners import publication

    registry = json.loads((REPOSITORY / "configs" / "figure_registry.json").read_text())
    registered = {entry["figure_id"] for entry in registry["figures"]}
    built = set(publication.figure_builders())
    declared = set(publication.NOT_BUILT) | set(publication.ALSO_FROM_STUDY1)
    assert not built & declared
    assert registered <= built | declared | {"fig2_panel_D_label_budget_downstream"}
    assert set(publication.PARTIAL) <= built


def test_empty_derived_table_reads_as_empty_frame_and_tables_are_create_once(tmp_path):
    from e3sse.production.runners import publication

    path = publication.write_table(tmp_path / "derived" / "empty.csv", [], root=tmp_path)
    assert publication._frame(path).empty
    rows = [{"a": 1.0, "b": None}]
    table = publication.write_table(tmp_path / "derived" / "t.csv", rows, root=tmp_path)
    assert publication.write_table(table, rows, root=tmp_path) == table  # identical resume
    with pytest.raises(publication.RunnerValidationError):
        publication.write_table(table, [{"a": 2.0}], root=tmp_path)


def test_runner_constants_equal_the_structured_protocol_values():
    from e3sse.production.runners import bvse_paths, ml_neb_units, transfer

    protocol = json.loads((REPOSITORY / "configs" /
                           "reliability_gated_internal_freeze_protocol.json").read_text())
    amendments = protocol["amendments"]
    check = amendments["representation"]["path_representation"]["degeneracy_check"]
    assert (representation.SUPPORT_K, representation.LOCSO_CV_MIN,
            representation.TAU_BOOTSTRAP_CV_MAX, representation.TAU_BOOTSTRAP_RESAMPLES) == (
        check["support_k"], check["locso_cv_min"], check["tau_bootstrap_cv_max"],
        check["tau_bootstrap_resamples"])
    budget = amendments["label_budget_downstream"]
    assert (transfer.LABEL_BUDGET_MIN_FRAMES, transfer.LABEL_BUDGET_MIN_HOPS) == (
        budget["minimum_frames"], budget["minimum_hops_count"])
    ml_neb = amendments["prospective"]["ml_neb"]
    assert ml_neb_units.REPRODUCIBILITY_SUBSET == \
        ml_neb["frozen_numeric_settings"]["reproducibility_subset"]["size_hops"]
    recipe = bvse_paths.recipe(amendments)
    assert recipe["bvse_neb_options"] == {"k": 2.0, "default": True, "distort": True, "gm": False}
    assert (recipe["superedge_cutoff_A"], recipe["n_images"], recipe["fire_fmax"],
            recipe["fire_steps"]) == (8.0, 5, 0.1, 100)


def test_collapsed_embeddings_fail_the_degeneracy_check_without_infinities():
    reference = {"loo_distances": [0.0] * 6, "reference_chemical_systems": list("aabbcc"),
                 "quantile_level": 0.8}
    check = representation.degeneracy(reference, seed=1)
    assert check["passes"] is False and check["tau_bootstrap_cv"] is None
    json.dumps(check, allow_nan=False)


def test_population_units_must_be_one_hop_and_included_in_the_right_split():
    unit = _unit("P", "Li-O", "mp-1")
    hop = {"record_kind": "nebdft2k_hop", "hop_id": unit["unit_edge_id"],
           "eligibility_status": "included", "published_split": "train"}
    both = ["mp-1_0_1_0_0_0", "mp-1_1_0_0_0_0"]  # both directions of one hop
    population.validate_units([{**unit, "rate_limiting_edge_ids": both, "unit_edge_ids": both}],
                              [hop, {**hop, "hop_id": both[1]}])
    two = ["mp-1_0_1_0_0_0", "mp-1_0_2_0_0_0"]
    with pytest.raises(RunnerValidationError, match="more than one physical hop"):
        population.validate_units([{**unit, "rate_limiting_edge_ids": two, "unit_edge_ids": two}],
                                  [hop])
    with pytest.raises(RunnerValidationError, match="exact unit is not its h"):
        population.validate_units([{**unit, "rate_limiting_edge_ids": ["mp-1_0_2_0_0_0"]}], [hop])
    proxy = {**unit, "rate_limiting_edge_ids": ["mp-1_0_2_0_0_0"],
             "unit_hop_role": "bottleneck_proxy"}
    population.validate_units([proxy], [hop])
    with pytest.raises(RunnerValidationError, match="published split"):
        population.validate_units([unit], [{**hop, "published_split": "val"}])
    # development: test or train hops (spanning systems), never validation
    population.validate_units([{**unit, "side": "development"}], [hop])
    population.validate_units([{**unit, "side": "development"}],
                              [{**hop, "published_split": "test"}])
    with pytest.raises(RunnerValidationError, match="published split"):
        population.validate_units([{**unit, "side": "development"}],
                                  [{**hop, "published_split": "val"}])


def test_upstream_path_verifies_prepare_model_and_stage_manifests(tmp_path):
    run = tmp_path / "run"
    ctx = RunnerContext(repository=REPOSITORY, config_path=tmp_path / "c.json", run_dir=run,
                        stage="study4")
    artifact = run / "transfer" / "test_frames" / "dft_reference" / "m.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    record = {"outputs": {"test_frame_statistics:dft_reference": {"path": str(artifact),
                                                                  "sha256": digest}}}
    (run / "transfer" / "prepare_manifest.json").write_text(json.dumps(record))
    assert ctx.upstream_path("transfer", "test_frame_statistics:dft_reference",
                             phase="prepare") == artifact
    with pytest.raises(MissingEvidence):  # the reduce manifest does not exist yet
        ctx.upstream_path("transfer", "test_frame_statistics:dft_reference")
    artifact.write_text('{"tampered": true}\n')
    with pytest.raises(RunnerValidationError):
        ctx.upstream_path("transfer", "test_frame_statistics:dft_reference", phase="prepare")


def test_bvse_runtime_lock_is_required_and_parsed(tmp_path):
    from e3sse.production.runners import bvse_paths

    lock = tmp_path / "requirements.lock"
    lock.write_text("# comment\nions==0.4.1 \\\n    --hash=sha256:abc\nase==3.29.0\n"
                    "PyYAML==6.0.3 ; python_version >= '3.11'\n")
    assert bvse_paths._lock_versions(lock) == {"ions": "0.4.1", "ase": "3.29.0", "pyyaml": "6.0.3"}
    with pytest.raises(MissingEvidence, match="lock not pinned"):
        bvse_paths.runtime_versions({"ions_version": "0.4.1", "environment_lock": {
            "path": str(lock), "sha256": "server_evidence_pending"}})
    with pytest.raises(RunnerValidationError, match="SHA-256"):
        bvse_paths.runtime_versions({"ions_version": "0.4.1", "environment_lock": {
            "path": str(lock), "sha256": "0" * 64}})


def test_every_runner_seed_uses_a_frozen_random_operation_domain():
    """analysis_seed refuses unknown domains at run time; catch any call site statically."""

    import ast

    frozen = {"study4_hop_draw", "cluster_bootstrap", "study1_label_budget"}
    for path in (REPOSITORY / "src" / "e3sse").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "analysis_seed"
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)):
                assert node.args[1].value in frozen, (path.name, node.lineno, node.args[1].value)


def test_pilot_boundary_refuses_final_evaluation_frames_and_non_endpoint_mde_frames():
    from e3sse.production.runners.pilot import pilot_boundary

    membership = [
        {"record_kind": "mplitrj_frame", "frame_id": "t:0", "published_split": "train",
         "chemical_system_id": "Li-O", "hop_id": "h1", "raw_file_kind": "source"},
        {"record_kind": "mplitrj_frame", "frame_id": "t:1", "published_split": "train",
         "chemical_system_id": "Li-O", "hop_id": "h1", "raw_file_kind": "band_optim_step"},
        {"record_kind": "mplitrj_frame", "frame_id": "x:0", "published_split": "test",
         "chemical_system_id": "Li-S", "hop_id": "h2", "raw_file_kind": "source"},
        {"record_kind": "nebdft2k_hop", "hop_id": "h2", "published_split": "test",
         "chemical_system_id": "Li-S"}]
    record = pilot_boundary(membership, ["t:0", "t:1"], ["t:0"])
    assert record["study2_pilot_path_frames_of_training_hops"] == 1  # permitted
    assert record["study2_pilot_forbidden_overlap"] == 0
    with pytest.raises(RunnerValidationError, match="final-evaluation"):
        pilot_boundary(membership, ["t:0", "x:0"], ["t:0"])
    with pytest.raises(RunnerValidationError, match="endpoint"):
        pilot_boundary(membership, ["t:0", "t:1"], ["t:1"])


def test_bvse_pool_tolerance_preflight_depends_on_pinned_server_evidence(tmp_path):
    import copy

    from e3sse.production.preflight import check_bvse_pool_tolerance

    protocol = json.loads((REPOSITORY / "configs" /
                           "reliability_gated_internal_freeze_protocol.json").read_text())
    recipe = protocol["amendments"]["prospective"]["ml_neb"]["bvse_recipe"]
    assert recipe["pool_barrier_tolerance_eV"] is None
    check = check_bvse_pool_tolerance(protocol, config_dir=tmp_path)
    assert check.status == "passed" and not check.blocking  # recorded only, excludes nothing
    chosen = copy.deepcopy(protocol)
    chosen["amendments"]["prospective"]["ml_neb"]["bvse_recipe"]["pool_barrier_tolerance_eV"] = 0.05
    assert check_bvse_pool_tolerance(chosen, config_dir=tmp_path).status == "failed"
    recipe = chosen["amendments"]["prospective"]["ml_neb"]["bvse_recipe"]
    # The stage's own report (pool differences incl. calibration units) is never evidence.
    own = tmp_path / "bvse_reproduction_report.json"
    own.write_text("{}")
    recipe["pool_barrier_tolerance_evidence"] = {
        "source": "bvse_reproduction_server_audit", "path": str(own),
        "sha256": hashlib.sha256(own.read_bytes()).hexdigest(), "git_commit": "c" * 40}
    assert check_bvse_pool_tolerance(chosen, config_dir=tmp_path).status == "failed"
    audit = tmp_path / "bvse_server_reproduction_audit.json"
    audit.write_text("{}")
    recipe["pool_barrier_tolerance_evidence"] = {
        "source": "bvse_reproduction_server_audit", "path": str(audit),
        "sha256": hashlib.sha256(audit.read_bytes()).hexdigest(), "git_commit": "c" * 40}
    assert check_bvse_pool_tolerance(chosen, config_dir=tmp_path).status == "passed"
    recipe["pool_barrier_tolerance_evidence"]["sha256"] = "0" * 64
    assert check_bvse_pool_tolerance(chosen, config_dir=tmp_path).status == "failed"


def test_study4_threshold_slice_is_identity_only_and_training_side_only():
    from e3sse.production.runners import study4

    calibration = [f"Li-X{i}" for i in range(40)]
    test = [f"Li-T{i}" for i in range(10)]
    split = study4.repeat_split("p" * 64, 3, calibration, test, 0.2)
    assert split == study4.repeat_split("p" * 64, 3, list(reversed(calibration)), test, 0.2)
    assert split["threshold"] and split["threshold"] <= set(calibration)
    assert not split["threshold"] & (split["calibration"] | split["test"])
    assert split["test"] == set(test)
    assert split["threshold"] != study4.repeat_split("p" * 64, 4, calibration, test, 0.2)["threshold"]
    with pytest.raises(RunnerValidationError):
        study4.repeat_split("p" * 64, 0, ["Li-O"], ["Li-O"], 0.2)


def test_reference_unit_missing_covariates_is_excluded_and_never_shapes_the_envelope(monkeypatch):
    monkeypatch.setattr(population, "weight_fit_slice",
                        lambda parent, system, fraction: system in {"Li-P-S", "Br-Li"})
    units = [_unit("development", "Li-O", "mp-1"), _unit("P", "Li-P-S", "mp-2"),
             _unit("P", "Br-Li", "mp-3"), _unit("Q", "Br-Li-O", "mp-4")]
    descriptors = {
        "development:Li-O": {"vector": _vector(hop=2.0), "chemical_system": "Li-O"},
        "P:Li-P-S": {"vector": _vector("sulfide", 3.0), "chemical_system": "Li-P-S"},
        "P:Br-Li": {"vector": None, "chemical_system": "Br-Li"},  # slice unit, no covariates
        "Q:Br-Li-O": {"vector": _vector(hop=2.5), "chemical_system": "Br-Li-O"}}
    built, _ = population.build_population(units, descriptors, parent_protocol_id="p" * 64,
                                           fraction=0.2)
    roles = {u["unit_key"]: u["role"] for u in built["units"]}
    assert roles["P:Br-Li"] == "excluded_missing_covariate"
    assert roles["Q:Br-Li-O"] == "excluded_gate0"  # Br only came from the incomplete unit
    assert built["gate0"]["element_support"] == ["Li", "O", "P", "S"]
    assert built["gate0"]["reference_units"] == 2


def test_density_features_add_critical_barrier_and_tm_indicator_never_the_gap():
    from e3sse.production.runners.density import density_features

    row = {**_unit("P", "Fe-Li-O", "mp-1"), "vector": _vector(), "tm_hubbard_u": True,
           "bottleneck_gap_eV": -0.1}
    features = density_features(row)
    assert features == [*_vector(), 0.4, 1.0]  # gap is not a feature
    assert density_features({**row, "critical_bvse_barrier_eV": None}) is None  # fail closed
    assert density_features({**row, "tm_hubbard_u": None}) is None
    assert density_features({**row, "vector": None}) is None
