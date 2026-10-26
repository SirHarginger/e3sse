"""Study-1 estimator, sufficiency, RNG, support, deciles, leakage and parser invariants."""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
from ase.io import read
from protocol_fixtures import frozen_universe

from e3sse.protocol.contracts import (
    audit_softening_frame_eligibility,
    audit_softening_frame_eligibility_batch,
    validate_softening_support,
)
from e3sse.protocol.partitions import analysis_seed
from e3sse.softening import frame_statistics as fs
from e3sse.softening import study1_analysis as sa
from e3sse.softening.extxyz_frames import (
    FrameFormatError,
    parse_frame_geometry,
    parse_labelled_frame,
    scan_frame_index,
)

SUPPORT = {"full_estimate_min_frames": 10, "full_estimate_min_informative_components": 100,
           "informative_force_eV_per_A": 0.1, "pooled_min_materials": 3,
           "label_budget_k": [1, 3, 5, 10, 30], "label_budget_min_population": 60}


def _frames(n=12, atoms=6, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        numbers = rng.choice([3, 8, 15], size=atoms)
        x = rng.normal(size=(atoms, 3))
        y = 0.8 * x + rng.normal(scale=0.05, size=(atoms, 3))
        frames.append((numbers, x, y))
    return frames


def test_eq17_orientation_recovers_known_scale():
    x = np.random.default_rng(1).normal(size=(20, 3))
    rows = fs.element_statistics([3] * 20, x, 0.8 * x)
    total = fs.combine(rows, "all")
    assert fs.eq17(total["sxy"], total["sxx"]) == pytest.approx(0.8, abs=1e-14)
    assert fs.eq17(total["sxy"], total["sxx"]) != pytest.approx(1 / 0.8)
    assert fs.eq17(1.0, 0.0) is None


@pytest.mark.parametrize("component", ["all", "Li", "host", "Z8"])
def test_sufficient_statistics_equal_concatenated_raw_forces(component):
    frames = _frames()
    rng = np.random.default_rng(5)
    subset = sorted(rng.choice(len(frames), size=7, replace=False))
    counts = np.bincount(rng.integers(0, len(subset), size=len(subset)), minlength=len(subset))
    raw_x, raw_y, sums = [], [], {"sxy": 0.0, "sxx": 0.0, "syy": 0.0}
    for weight, i in zip(counts, subset, strict=True):  # bootstrap: frame i drawn c_i times
        numbers, x, y = frames[i]
        mask = np.array([fs.component_set_members(component, int(z)) for z in numbers])
        for _ in range(weight):
            raw_x.append(x[mask].ravel())
            raw_y.append(y[mask].ravel())
        combined = fs.combine(fs.element_statistics(numbers, x, y), component)
        for key in sums:
            sums[key] += weight * combined[key]
    x, y = np.concatenate(raw_x), np.concatenate(raw_y)
    assert fs.eq17(sums["sxy"], sums["sxx"]) == pytest.approx((x @ y) / (x @ x), rel=1e-13)
    best = (x @ y) / (x @ x)
    direct = float(((y - best * x) ** 2).sum() / (y @ y))
    assert fs.residual_fraction(sums["sxy"], sums["sxx"], sums["syy"]) == pytest.approx(
        direct, rel=1e-9)


def test_per_frame_sums_are_exact_and_count_informative_components():
    x = np.array([[0.05, -0.1, 0.2]] * 2)
    (row,) = fs.element_statistics([3, 3], x)
    assert row["n_informative"] == 4 and row["n_atoms"] == 2
    assert row["sxx"] == math.fsum((x * x).ravel().tolist())
    with pytest.raises(ValueError, match="finite"):
        fs.element_statistics([3], [[np.nan, 0, 0]])


def test_bootstrap_stream_is_block_size_invariant_and_unbiased():
    big = np.concatenate(list(fs.iter_bootstrap_indices(42, 7, 300)))
    small = np.concatenate(list(fs.iter_bootstrap_indices(42, 7, 300, max_values=10)))
    assert np.array_equal(big, small) and big.shape == (300, 7)
    assert set(np.unique(big)) == set(range(7))
    assert abs(np.bincount(big.ravel()) / big.size - 1 / 7).max() < 0.02
    assert not np.array_equal(big, fs.bootstrap_indices(43, 7, 300))


def test_without_replacement_draws_exactly_k_distinct_frames():
    for k in (1, 3, 5, 10, 30):
        draw = fs.draw_without_replacement(11, 60, k)
        assert len(draw) == k == len(set(draw.tolist())) and draw.max() < 60
        assert np.array_equal(draw, fs.draw_without_replacement(11, 60, k))
    with pytest.raises(ValueError):
        fs.draw_without_replacement(1, 5, 6)


def _context(models, sxy, sxx, success=None):
    n = sxy.shape[1]
    stats = np.zeros((len(models), n, 3, 3))
    stats[:, :, 0, 0], stats[:, :, 0, 1], stats[:, :, 0, 2] = sxy, sxx, sxy
    return {"models": models, "set_stats": stats,
            "set_counts": np.tile(np.array([[4, 12]]), (n, 3, 1)),
            "success": np.ones((len(models), n), bool) if success is None else success,
            "material_code": np.zeros(n, dtype=np.int64)}


def test_label_budget_uses_common_random_numbers_and_the_frozen_seed():
    rng = np.random.default_rng(3)
    sxx = rng.uniform(1, 2, size=(2, 70))
    sxx[1] = sxx[0]
    sxy = np.stack([0.8 * sxx[0] * rng.uniform(0.9, 1.1, 70), 0.9 * sxx[0]])
    payload = {"unit": "label_budget", "population": "primary", "hop_id": "mp-1_0_1",
               "material_id": "mp-1", "rows": list(range(70)), "selection_sha256": "x",
               "k_values": [1, 3], "repeats": 5, "base_seed": 2026}
    first = sa.label_budget_unit(payload, _context(["a", "b"], sxy, sxx))
    swapped = sa.label_budget_unit(payload, _context(["b", "a"], sxy[::-1], sxx[::-1]))
    assert first["draws_sha256"] == swapped["draws_sha256"]  # model-independent draws
    assert first["models"]["a"] == swapped["models"]["a"]
    assert first["models"]["b"]["s_k"]["3"] == pytest.approx([0.9] * 5)
    seed = analysis_seed(2026, "study1_label_budget", "material", "mp-1", "mp-1_0_1", 3, 0)
    index = fs.draw_without_replacement(seed, 70, 3)
    assert first["models"]["a"]["s_k"]["3"][0] == pytest.approx(sxy[0, index].sum()
                                                                / sxx[0, index].sum())
    failed = _context(["a", "b"], sxy, sxx, success=np.array([[True] * 70, [True] * 69 + [False]]))
    assert sa.label_budget_unit(payload, failed)["models"]["b"]["status"] == "incomplete_inference"


def _estimate_payload(rows, scope="material"):
    return {"unit": "estimate", "analysis_id": "study1|t", "kind": "target_hop",
            "population": "primary", "scope": scope, "scope_target": "mp-1",
            "target_hop_id": "h", "material_id": "mp-1", "estimator_kind": "primary",
            "decile": None, "component_sets": ["all"], "rows": rows, "selection_sha256": "s",
            "bootstrap_seed": 9, "bootstrap_resamples": 50, "confidence_level": 0.95,
            "support": SUPPORT}


def test_unsupported_scope_reports_reasons_and_never_falls_back():
    sxx = np.full((1, 12), 2.0)
    context = _context(["a"], 0.8 * sxx, sxx)
    (small,) = sa.estimate_unit(_estimate_payload(list(range(9))), context)
    assert small["status"] == "unsupported" and small["s_hat"] is None
    assert small["unsupported_reasons"] == ["fewer_than_10_frames"]
    (pooled,) = sa.estimate_unit(_estimate_payload(list(range(12)), scope="global"), context)
    assert pooled["unsupported_reasons"] == ["fewer_than_3_materials"]
    (ok,) = sa.estimate_unit(_estimate_payload(list(range(12))), context)
    assert ok["status"] == "estimated" and ok["s_hat"] == pytest.approx(0.8)
    assert ok["ci_low"] == pytest.approx(0.8) and ok["ci_high"] == pytest.approx(0.8)
    again = sa.estimate_unit(_estimate_payload(list(range(12))), context)
    assert again == [ok]  # deterministic, independent of any scheduling
    failed = _context(["a"], 0.8 * sxx, sxx, success=np.array([[True] * 11 + [False]]))
    (row,) = sa.estimate_unit(_estimate_payload(list(range(12))), failed)
    assert row["status"] == "incomplete_inference" and row["s_hat"] is None


def test_support_rule_matches_the_frozen_contract_on_all_component_estimates():
    records = [
        {"frame_id": f"f{i}", "hop_id": "other", "material_id": "mp-1",
         "chemical_system_id": "Li-O", "published_split": "train",
         "mapping_status": "mapped_unique", "raw_file_kind": "band_optim_step",
         "origin_evidence_status": "unequivocal", "origin_evidence": "x",
         "partition_role": "training_or_calibration"}
        for i in range(12)
    ]
    protocol, membership = frozen_universe(records)
    selection = audit_softening_frame_eligibility(
        records, target_hop_id="target", estimator_scope="material",
        scope_evidence={"status": "verified", "scope": "material", "target": "mp-1",
                        "source": "test"},
        allowed_partition_role="training_or_calibration", target_material_id="mp-1",
        protocol=protocol, partition_records=membership, partition=0)
    rng = np.random.default_rng(2)
    for n_frames, scale in ((12, 1.0), (9, 1.0), (12, 0.01)):
        frames = [{"frame_id": f"f{i}", "dft_forces": scale * rng.normal(size=(4, 3))}
                  for i in range(n_frames)]
        informative = sum(int((np.abs(f["dft_forces"]) >= 0.1).sum()) for f in frames)
        reasons = fs.support_status(n_frames=n_frames, n_informative=informative, n_materials=1,
                                    pooled=False, parameters=SUPPORT)
        try:
            validate_softening_support(selection, protocol, frames, purpose="full_estimate")
            contract = True
        except ValueError:
            contract = False
        assert contract == (not reasons)


def test_batch_audit_equals_single_audit_and_unknown_provenance_fails_closed():
    base = {"material_id": "mp-1", "chemical_system_id": "Li-O", "published_split": "train",
            "mapping_status": "mapped_unique", "origin_evidence": "x",
            "origin_evidence_status": "unequivocal", "partition_role": "training_or_calibration"}
    records = [
        {**base, "frame_id": "own-path", "hop_id": "t", "raw_file_kind": "band_optim_step"},
        {**base, "frame_id": "own-end", "hop_id": "t", "raw_file_kind": "source"},
        {**base, "frame_id": "own-unknown", "hop_id": "t", "raw_file_kind": "source",
         "origin_evidence_status": "unresolved"},
        {**base, "frame_id": "other", "hop_id": "u", "raw_file_kind": "band_optim_step"},
    ]
    protocol, membership = frozen_universe(records)
    request = {"target_hop_id": "t", "estimator_scope": "material",
               "scope_evidence": {"status": "verified", "scope": "material", "target": "mp-1",
                                  "source": "test"},
               "allowed_partition_role": "training_or_calibration", "target_material_id": "mp-1"}
    single = audit_softening_frame_eligibility(records, protocol=protocol,
                                               partition_records=membership, partition=0,
                                               **request)
    (batch,) = audit_softening_frame_eligibility_batch(
        [{"records": records, **request}], protocol=protocol, partition_records=membership,
        partition=0)
    assert batch == single
    assert {r["frame_id"] for r in single["eligible_records"]} == {"own-end", "other"}
    assert single["reason_counts"]["excluded_target_hop_unidentified_origin"] == 1


def _meta(material, population="primary"):
    return {"material_id": material, "population": population}


def test_energy_deciles_use_dft_energy_population_and_right_closed_bins():
    metadata = [_meta("a")] * 11 + [_meta("b", "tm_hubbard_u_secondary")] * 3
    n_atoms = np.array([2] * 14)
    energies = np.array([float(i) for i in range(11)] + [100.0, 50.0, 0.0])
    record, deciles = sa.energy_deciles(metadata, n_atoms, energies, "primary")
    # dE = i/2 for i = 0..10; numpy linear cut points at 0.5, 1.0, ..., 4.5.
    assert record["cut_points_eV_per_atom"] == pytest.approx([0.5 * k for k in range(1, 10)])
    assert deciles[1] == 1 and deciles[2] == 2  # 0.5 equals a cut point -> lower decile
    assert deciles.tolist()[11:] == [0, 0, 0]  # other population untouched
    ties = np.array([0.0] * 11 + [0.0] * 3)
    record, _ = sa.energy_deciles(metadata, n_atoms, ties, "primary")
    assert record["empty_deciles"] == list(range(2, 11))


def test_parser_matches_ase_and_reads_no_label_for_geometry():
    text = (
        "2\n"
        'Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0" Properties=species:S:1:pos:R:3:forces:R:3'
        ' energy=-7.25 material_id=mp-1 pbc="T T T"\n'
        "Li 0.1 0.2 0.3 0.5 -0.25 0.125\n"
        "O 1.0 1.5 2.0 -0.5 0.25 -0.125\n"
    )
    frame = parse_labelled_frame(text)
    atoms = read(io.StringIO(text), format="extxyz")
    assert np.array_equal(frame["positions"], atoms.positions)
    assert np.array_equal(frame["dft_forces"], atoms.get_forces())
    assert frame["dft_energy"] == atoms.get_potential_energy() == -7.25
    geometry = parse_frame_geometry(text)
    assert "dft_forces" not in geometry and "dft_energy" not in geometry
    with pytest.raises(FrameFormatError, match="forces"):
        parse_labelled_frame(text.replace(":forces:R:3", "").replace(" 0.5 -0.25 0.125", "")
                             .replace(" -0.5 0.25 -0.125", ""))


def test_frame_index_offsets_and_digest(tmp_path):
    text = "".join(
        f"1\nLattice=\"2 0 0 0 2 0 0 0 2\" Properties=species:S:1:pos:R:3:forces:R:3 "
        f"energy=-{i}.0 material_id=mp-{i}\nLi 0 0 0 0 0 0\n" for i in range(3)
    )
    path = tmp_path / "x.xyz"
    path.write_text(text)
    digest, entries = scan_frame_index(path)
    assert [e.material_id for e in entries] == ["mp-0", "mp-1", "mp-2"]
    with path.open("rb") as handle:
        handle.seek(entries[1].byte_offset)
        assert parse_labelled_frame(handle.read(entries[1].byte_length).decode())[
            "dft_energy"] == -1.0
    assert entries[2].cell_volume == pytest.approx(8.0)
    assert len(digest) == 64
    assert json.dumps(digest)


@pytest.mark.parametrize("n", [1, 2, 16, 64])
def test_bootstrap_handles_power_of_two_cluster_counts(n):
    draws = fs.bootstrap_indices(5, n, 40)
    assert draws.shape == (40, n) and draws.min() >= 0 and draws.max() < n


def test_non_finite_dft_energy_fails_closed():
    text = ('1\nLattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3:forces:R:3 '
            "energy=nan material_id=mp-1\nLi 0 0 0 0 0 0\n")
    with pytest.raises(FrameFormatError, match="not finite"):
        parse_labelled_frame(text)


def test_out_of_memory_is_infrastructure_not_a_frame_failure():
    from e3sse.softening.study1_inference import _resource_failure

    assert _resource_failure(MemoryError())
    assert _resource_failure(RuntimeError("DefaultCPUAllocator: can't allocate memory"))
    assert not _resource_failure(ValueError("element 118 not supported"))


def test_npz_write_is_reproducible_and_never_replaced(tmp_path):
    from e3sse.softening.study1_inference import write_npz_once

    arrays = {"a": np.arange(3), "b": np.array(["x", "y"])}
    first = write_npz_once(tmp_path / "s.npz", arrays)
    assert write_npz_once(tmp_path / "s.npz", arrays) == first  # crash-safe resume
    with pytest.raises(FileExistsError):
        write_npz_once(tmp_path / "s.npz", {"a": np.arange(4)})
    assert list(tmp_path.iterdir()) == [tmp_path / "s.npz"]
