"""Rate-limiting BVSE hop on synthetic periodic Li-site networks (no real data)."""

from __future__ import annotations

import pytest

from e3sse.screening.rate_limiting_hop import parse_edge_id, rate_limiting_hop


def _hop(src, dst, offset, em, material="mp-7"):
    return {"edge_id": f"{material}_{src}_{dst}_{offset[0]}_{offset[1]}_{offset[2]}", "em": em}


def test_edge_id_parsing_and_rejection():
    assert parse_edge_id("mp-1_0_1_1_0_-1") == {"material_id": "mp-1", "source": 0, "target": 1,
                                                 "offset": (1, 0, -1)}
    with pytest.raises(ValueError):
        parse_edge_id("mp-1_0_1")


def test_one_site_lattice_percolates_dimension_by_dimension():
    hops = [_hop(0, 0, (1, 0, 0), 0.2), _hop(0, 0, (0, 1, 0), 0.5), _hop(0, 0, (0, 0, 1), 0.9)]
    assert rate_limiting_hop(hops, 1)["rate_limiting_edge_ids"] == ["mp-7_0_0_1_0_0"]
    assert rate_limiting_hop(hops, 2)["critical_barrier_eV"] == 0.5
    assert rate_limiting_hop(hops, 3)["critical_barrier_eV"] == 0.9


def test_bottleneck_is_minimax_not_the_largest_barrier():
    # Two sites; a cheap 1-D channel 0-1-0' plus an expensive redundant hop.
    hops = [_hop(0, 1, (0, 0, 0), 0.3), _hop(1, 0, (1, 0, 0), 0.4), _hop(0, 1, (0, 1, 0), 2.0)]
    result = rate_limiting_hop(hops, 1)
    assert result["critical_barrier_eV"] == 0.4  # not 2.0
    assert result["rate_limiting_edge_ids"] == ["mp-7_1_0_1_0_0"]


def test_ties_are_counted_and_broken_by_edge_id_and_order_free():
    hops = [_hop(0, 0, (1, 0, 0), 0.5), _hop(0, 0, (0, 1, 0), 0.5)]
    forward, backward = rate_limiting_hop(hops, 1), rate_limiting_hop(hops[::-1], 1)
    assert forward == backward
    assert len(forward["tied_hops"]) == 2  # each alone completes 1-D percolation


def test_non_percolating_missing_and_mixed_material_cases_fail_closed():
    closed = [_hop(0, 1, (0, 0, 0), 0.1), _hop(1, 2, (0, 0, 0), 0.2),
              _hop(2, 0, (0, 0, 0), 0.3)]  # closed triangle inside one cell: zero-offset cycle
    assert rate_limiting_hop(closed, 1)["status"] == "no_percolating_network"
    conflict = [_hop(0, 1, (1, 0, 0), 0.2), _hop(1, 0, (-1, 0, 0), 0.3)]
    assert rate_limiting_hop(conflict, 1)["status"] == "direction_conflict"
    assert rate_limiting_hop([], 1)["status"] == "no_hops"
    assert rate_limiting_hop([_hop(0, 0, (1, 0, 0), float("nan"))], 1)["status"] == \
        "non_finite_barrier"
    with pytest.raises(ValueError, match="more than one material"):
        rate_limiting_hop([_hop(0, 0, (1, 0, 0), 0.1), _hop(0, 0, (1, 0, 0), 0.1, "mp-8")], 1)


def _audit_module():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "audit_rate_limiting_hop_linkage",
        Path(__file__).resolve().parents[1] / "scripts" / "audit_rate_limiting_hop_linkage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit_configs(tmp_path, *, tamper=False):
    import hashlib
    import json
    from pathlib import Path

    data_root = Path(__file__).resolve().parent / "fixtures" / "g0" / "data"
    raw = data_root / "raw"
    identities = {name: [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in sorted((raw / name).glob("*.csv"))]
                  for name in ("BVEL13k", "nebBVSE122k", "nebDFT2k")}
    if tamper:
        identities["nebDFT2k"][0]["sha256"] = "0" * 64
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"inputs": identities}))
    protocol = tmp_path / "protocol_freeze.json"
    protocol.write_text(json.dumps({"inputs": {"g0_input_manifest": {
        "path": str(inputs), "sha256": hashlib.sha256(inputs.read_bytes()).hexdigest()}}}))
    server = tmp_path / "server.json"
    server.write_text(json.dumps({"paths": {
        "data_root": str(data_root), "outputs_root": str(tmp_path / "outputs"),
        "logs_root": str(tmp_path / "logs"), "scratch_root": str(tmp_path / "scratch")}}))
    return server, protocol


def test_linkage_audit_uses_canonical_bindings_and_never_reads_outcomes(tmp_path):
    import json

    module = _audit_module()
    server, protocol = _audit_configs(tmp_path)
    args = ["--config", str(server), "--protocol-config", str(protocol)]
    assert module.main(args) == 0
    out = tmp_path / "outputs" / "gate1_linkage" / "rate_limiting_hop_linkage_audit.json"
    report = json.loads(out.read_text())
    assert report["outcome_columns_read"] is False
    assert "em_dft" not in report["inputs"]["nebDFT2k"]["columns_read"]
    pool = report["pool"]
    assert pool["linked"] + pool["unlinked"] == pool["total_candidate_structures"]
    assert len(report["provenance"]["code_sha256"]) == 2
    with pytest.raises(SystemExit, match="never overwritten"):
        module.main(args)


def test_linkage_audit_stops_when_inputs_differ_from_g0(tmp_path):
    module = _audit_module()
    server, protocol = _audit_configs(tmp_path, tamper=True)
    with pytest.raises(SystemExit, match="G0"):
        module.main(["--config", str(server), "--protocol-config", str(protocol)])
    with pytest.raises(module.AuditError, match="outcome columns"):
        module.read_columns(tmp_path / "x.csv", ("edge_id", "em_dft"))

def test_reverse_direction_duplicate_edges_create_no_spurious_percolation():
    # a -> b (+x) and b -> a (-x) are the same physical hop: no cycle, no percolation.
    hops = [_hop(0, 1, (1, 0, 0), 0.2), _hop(1, 0, (-1, 0, 0), 0.2)]
    assert rate_limiting_hop(hops, 1)["status"] == "no_percolating_network"


def test_multiple_percolating_paths_use_the_cheapest_bottleneck():
    cheap = [_hop(0, 1, (0, 0, 0), 0.3), _hop(1, 0, (1, 0, 0), 0.35)]
    costly = [_hop(2, 3, (0, 0, 0), 0.1), _hop(3, 2, (1, 0, 0), 0.8)]
    result = rate_limiting_hop(cheap + costly, 1)
    assert result["critical_barrier_eV"] == 0.35
    assert result["rate_limiting_edge_ids"] == ["mp-7_1_0_1_0_0"]


def test_symmetry_equivalent_hops_are_ties_with_a_stable_identifier():
    # Two symmetry-equivalent channels with equal barriers: a recorded tie, stable choice.
    hops = [_hop(0, 0, (1, 0, 0), 0.4), _hop(1, 1, (1, 0, 0), 0.4)]
    runs = {tuple(rate_limiting_hop(order, 1)["rate_limiting_edge_ids"])
            for order in (hops, hops[::-1])}
    assert runs == {("mp-7_0_0_1_0_0",)}
    assert rate_limiting_hop(hops, 1)["tied_hops"] == ["mp-7_0_0_1_0_0", "mp-7_1_1_1_0_0"]


def test_reverse_rows_merge_into_one_physical_hop_keeping_both_ids():
    hops = [_hop(0, 0, (1, 0, 0), 0.3), _hop(0, 0, (-1, 0, 0), 0.3)]
    result = rate_limiting_hop(hops, 1)
    assert result["n_physical_hops"] == 1 and result["n_rows"] == 2
    assert result["rate_limiting_edge_ids"] == ["mp-7_0_0_-1_0_0", "mp-7_0_0_1_0_0"]


def test_non_percolating_hop_at_the_critical_barrier_is_not_a_tie():
    hops = [_hop(0, 0, (1, 0, 0), 0.5), _hop(1, 2, (0, 0, 0), 0.5)]  # second: no cycle
    assert rate_limiting_hop(hops, 1)["tied_hops"] == ["mp-7_0_0_1_0_0"]


def test_long_chains_do_not_hit_the_recursion_limit():
    n = 5000
    hops = [_hop(i, i + 1, (0, 0, 0), 0.1) for i in range(n)] + [_hop(n, 0, (1, 0, 0), 0.2)]
    assert rate_limiting_hop(hops, 1)["critical_barrier_eV"] == 0.2
