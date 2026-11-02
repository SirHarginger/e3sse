"""Outcome-free Gate-1 structural populations on a synthetic periodic graph."""

from __future__ import annotations

import pytest

from e3sse.protocol.chemistry import primary_exclusion_elements
from e3sse.screening.gate1_populations import choose_structure, edge_tables, structural_populations

EXCLUDED = primary_exclusion_elements()


def _cubic(material: str, barrier: float) -> list[dict]:
    """Three hops of one site to its images along x, y, z: a 3-D network."""

    rows = []
    for offset, scale in (("1_0_0", 1.0), ("0_1_0", 1.0), ("0_0_1", 1.2)):
        rows.append({"material_id": material, "edge_id": f"{material}_0_0_{offset}",
                     "chemsys": "Li-O-P", "bvse_migration_barrier_eV": str(barrier * scale)})
    return rows


def _rows(material, barrier, chemsys):
    return [{**row, "chemsys": chemsys} for row in _cubic(material, barrier)]


def test_structure_rule_picks_lowest_critical_barrier_then_id():
    rows = _cubic("mp-2", 0.3) + _cubic("mp-1", 0.3) + _cubic("mp-3", 0.2)
    table, ambiguous = edge_tables(rows)
    chosen = choose_structure(["mp-1", "mp-2", "mp-3"], table, ambiguous)
    assert chosen["material_id"] == "mp-3"
    chosen = choose_structure(["mp-1", "mp-2"], table, ambiguous)
    assert chosen["material_id"] == "mp-1"  # tie on E*_3 -> ascending material_id


def test_ambiguous_material_is_never_analysed():
    rows = _cubic("mp-1", 0.3)
    rows.append({**rows[0], "bvse_migration_barrier_eV": "0.9"})  # conflicting duplicate
    table, ambiguous = edge_tables(rows)
    assert ambiguous == {"mp-1": "conflicting_duplicate_edge"}
    assert choose_structure(["mp-1"], table, ambiguous)["status"] == "no_percolating_structure"


def test_population_classes_and_unique_chemical_systems():
    bvse = (_rows("mp-1", 0.3, "Li-O-P") + _rows("mp-2", 0.2, "Li-O-P")
            + _rows("mp-3", 0.4, "Li-S") + _rows("mp-4", 0.5, "Cl-Li")
            + _rows("mp-5", 0.5, "Fe-Li-O") + _rows("mp-6", 0.1, "Br-Li"))
    # nebDFT2k holds the rate-limiting hop of mp-1 (but the structure rule picks mp-2),
    # and of mp-3, mp-4, mp-5; mp-6 is a pool-only system.
    dft = [{"material_id": m, "edge_id": f"{m}_0_0_0_0_1", "chemsys": c}
           for m, c in (("mp-1", "Li-O-P"), ("mp-3", "Li-S"), ("mp-4", "Cl-Li"),
                        ("mp-5", "Fe-Li-O"))]
    bvel = [{"material_id": "mp-6", "chemsys": "Br-Li"}, {"material_id": "mp-2",
                                                         "chemsys": "Li-O-P"}]
    membership = [{"chemical_system_id": "Li-O-P", "published_split": "train"},
                  {"chemical_system_id": "Li-S", "published_split": "test"},
                  {"chemical_system_id": "Cl-Li", "published_split": "test"},
                  {"chemical_system_id": "Cl-Li", "published_split": "train"},
                  {"chemical_system_id": "Fe-Li-O", "published_split": "train"}]
    report = structural_populations(
        nebbvse_rows=bvse, bvel_rows=bvel, nebdft_rows=dft, membership_records=membership,
        primary_assignment={"Fe-Li-O": "tm_hubbard_u_secondary", "Li-O-P": "primary",
                            "Li-S": "primary", "Cl-Li": "primary"},
        excluded_elements=EXCLUDED)
    counts = report["counts"]
    assert counts["nebDFT2k_materials_rate_limiting_hop_in_nebDFT2k"] == 4
    assert counts["unique_chemical_systems_among_them"] == 4
    populations = report["populations"]
    assert populations["development"]["structural_upper_bound"]["systems"] == ["Li-S"]
    # Li-O-P: the chosen structure (mp-2) has no nebDFT2k hop -> ineligible, never re-chosen.
    assert populations["P"]["rate_limiting_hop_not_in_nebDFT2k"]["systems"] == ["Li-O-P"]
    assert report["excluded_from_P"]["spanning_published_splits"]["systems"] == ["Cl-Li"]
    assert populations["P"]["tm_hubbard_u_excluded"]["systems"] == ["Fe-Li-O"]
    assert populations["Q"]["structural_upper_bound"]["systems"] == ["Br-Li"]
    assert report["outcome_columns_read"] is False and report["dimension"] == 3
    assert "nebDFT2k element criteria" in report["gate0_not_applied"]


def _populations(bvse, bvel, dft, membership):
    return structural_populations(nebbvse_rows=bvse, bvel_rows=bvel, nebdft_rows=dft,
                                  membership_records=membership, primary_assignment={},
                                  excluded_elements=EXCLUDED)


def test_material_with_two_chemical_systems_is_never_counted_twice():
    bvse = _rows("mp-1", 0.3, "Li-O-P")
    bvel = [{"material_id": "mp-1", "chemsys": "Li-O-S"}]  # disagrees with nebBVSE122k
    dft = [{"material_id": "mp-1", "edge_id": "mp-1_0_0_1_0_0", "chemsys": "Li-O-P"}]
    report = _populations(bvse, bvel, dft, [{"chemical_system_id": "Li-O-P",
                                             "published_split": "train"}])
    assert report["material_quality"]["inconsistent_chemical_system_materials"]["materials"] == [
        "mp-1"]
    eligible = [s for side in report["populations"].values()
                for s in side.get("structural_upper_bound", {}).get("systems", [])]
    assert eligible == []


def test_dft_labelled_system_can_never_reach_the_pool():
    bvse = _rows("mp-1", 0.3, "Li-O-P")
    bvel = [{"material_id": "mp-1", "chemsys": "Li-O-P"}]
    dft = [{"material_id": "mp-1", "edge_id": "mp-1_0_0_1_0_0", "chemsys": "Li-O-P"}]
    with pytest.raises(ValueError, match="outside the partition universe"):
        _populations(bvse, bvel, dft, [{"chemical_system_id": "Li-S", "published_split": "train"}])


def test_pool_system_inside_partition_universe_is_counted_not_dropped():
    report = _populations(_rows("mp-9", 0.3, "Br-Li"), [{"material_id": "mp-9",
                                                         "chemsys": "Br-Li"}], [],
                          [{"chemical_system_id": "Br-Li", "published_split": "train"}])
    assert report["not_a_candidate"]["systems"] == ["Br-Li"]
    assert report["counts"]["Q_structural_upper_bound_systems"] == 0


def test_validation_systems_never_enter_p():
    bvse = _rows("mp-1", 0.3, "Li-O-P")
    dft = [{"material_id": "mp-1", "edge_id": "mp-1_0_0_1_0_0", "chemsys": "Li-O-P"}]
    report = _populations(bvse, [], dft, [{"chemical_system_id": "Li-O-P",
                                           "published_split": "val"}])
    assert report["excluded_from_P"]["published_validation"]["systems"] == ["Li-O-P"]
    assert report["counts"]["P_structural_upper_bound_systems"] == 0
