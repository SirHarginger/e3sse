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


def test_population_roles_gate0_and_development_manifest_hash():
    from e3sse.protocol.contracts import _verify_development_manifest

    units = [{"side": "development", "chemical_system": "Li-O", "unit_edge_id": "mp-1_0_1_0_0_0",
              "material_id": "mp-1", "rate_limiting_edge_ids": ["mp-1_0_1_0_0_0"]},
             {"side": "P", "chemical_system": "Li-P-S", "unit_edge_id": "mp-2_0_1_0_0_0",
              "material_id": "mp-2", "rate_limiting_edge_ids": ["mp-2_0_1_0_0_0"]},
             {"side": "Q", "chemical_system": "Li-O-S", "unit_edge_id": "mp-7_0_1_0_0_0",
              "material_id": "mp-7", "rate_limiting_edge_ids": ["mp-7_0_1_0_0_0"]},
             {"side": "Q", "chemical_system": "Ge-Li", "unit_edge_id": "mp-8_0_1_0_0_0",
              "material_id": "mp-8", "rate_limiting_edge_ids": ["mp-8_0_1_0_0_0"]}]
    descriptors = {
        "development:Li-O": {"vector": _vector(hop=2.0), "elements": ["Li", "O"], "chemical_system": "Li-O"},
        "P:Li-P-S": {"vector": _vector("sulfide", 3.0), "elements": ["Li", "P", "S"],
                     "chemical_system": "Li-P-S"},
        "Q:Li-O-S": {"vector": _vector(hop=2.5), "elements": ["Li", "O", "S"], "chemical_system": "Li-O-S"},
        "Q:Ge-Li": {"vector": _vector(hop=2.5), "elements": ["Ge", "Li"], "chemical_system": "Ge-Li"}}
    built, manifest = population.build_population(units, descriptors, parent_protocol_id="p" * 64,
                                                  fraction=0.2)
    roles = {u["unit_key"]: u["role"] for u in built["units"]}
    assert roles["development:Li-O"] == "development"
    assert roles["Q:Ge-Li"] == "excluded_gate0"  # Ge has no calibration-side support
    assert roles["Q:Li-O-S"] in ("pool", "pool_weight_slice")
    internal = {"amendments": {"prospective": {"development_manifest_sha256": manifest["manifest_sha256"]}}}
    _verify_development_manifest(manifest, internal, ["mp-1_0_1_0_0_0"])  # contract-compatible hash
    with pytest.raises(ValueError):
        _verify_development_manifest(manifest, internal, ["mp-2_0_1_0_0_0"])  # not development


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


def test_pilot_mde_simulation_is_deterministic():
    first = mde_simulation([0.8, 0.85, 0.9], 0.02, [0.3, 0.5, 0.7, 0.4] * 10, noise_eV=0.05,
                           seed=4, simulations=200)
    assert first == mde_simulation([0.8, 0.85, 0.9], 0.02, [0.3, 0.5, 0.7, 0.4] * 10,
                                   noise_eV=0.05, seed=4, simulations=200)
    assert first["rho_srt_median"] > 0


def test_holm_and_cluster_bootstrap():
    adjusted = inference_stats.holm({"a": 0.01, "b": 0.04, "c": None})
    assert adjusted == {"a": 0.02, "b": 0.04, "c": None}
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
