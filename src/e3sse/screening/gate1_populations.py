"""Outcome-free Gate-1 structural populations at the frozen dimension d = 3.

Inputs are identifiers and the BVSE covariate only: nebBVSE122k ``material_id``,
``edge_id``, ``chemsys`` and the BVSE migration barrier; BVEL13k ``material_id``,
``chemsys``; nebDFT2k ``material_id``, ``edge_id``, ``chemsys``; the frozen partition
membership (published split per chemical system); and the frozen TM/Hubbard-U
assignment.  No DFT barrier, energy or force is an input.

Rules (``amendments.prospective.populations``, Study-5 revision of 2026-11-03,
count-informed and outcome-free):

* chemical systems are normalized with ``protocol.partitions.normalize_chemical_system``;
* Q (pool): BVEL13k systems outside the partition universe; the structure rule picks,
  among every material with that system in nebBVSE122k union BVEL13k, the structure
  with the lowest critical BVSE barrier E*_3 among those with a 3-D percolating
  network (ties by ascending ``material_id``); the unit hop is its h*_3;
* P (calibration side): published-train-only systems with nebDFT2k hops;
  development: systems with published-test contact (test-only or spanning), built
  only on non-validation materials; validation-only systems are the sealed reserve;
* P/development structure: the lowest-E*_3 3-D-percolating nebDFT2k material of the
  system (ties by ``material_id``); unit hop: its h*_3 if a nebDFT2k hop
  (``rate_limiting_exact``), else its nebDFT2k hop nearest the bottleneck
  (``bottleneck_proxy``, see ``unit_hop``); a material with an ambiguous edge table
  is never analysed;
* TM/Hubbard-U systems are a reported stratum (``tm_hubbard_u``) on every side, not
  an exclusion; Gate 0 is the nebDFT2k element/hop-length envelope only, applied
  later from descriptors, so these counts are structural upper bounds.

ML-NEB convergence and BVSE-path reproduction filters come later and are not
applied here.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from e3sse.protocol.partitions import normalize_chemical_system
from e3sse.screening.rate_limiting_hop import parse_edge_id, physical_hops, rate_limiting_hop

POPULATION_AUDIT_SCHEMA = "e3sse.gate1-structural-population-audit"
DIMENSION = 3
BVSE_FIELD = "bvse_migration_barrier_eV"
RATE_LIMITING_EXACT = "rate_limiting_exact"
BOTTLENECK_PROXY = "bottleneck_proxy"


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def _system(value: str) -> str | None:
    try:
        return normalize_chemical_system(value)
    except ValueError:
        return None


def edge_tables(nebbvse_rows: Iterable[Mapping[str, str]]) -> tuple[dict[str, list], dict[str, str]]:
    """Per-material BVSE edges, and materials made ineligible by an ambiguous table.

    Identical in rule to ``scripts/audit_rate_limiting_hop_linkage.py``.
    """

    rows_by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ambiguous: dict[str, str] = {}
    seen: dict[str, float] = {}
    for row in nebbvse_rows:
        material, edge = row["material_id"], row["edge_id"]
        try:
            parsed = parse_edge_id(edge)
            barrier = float(row[BVSE_FIELD])
        except ValueError:
            ambiguous.setdefault(material, "unparseable_edge_or_barrier")
            continue
        if parsed["material_id"] != material:
            ambiguous.setdefault(material, "edge_prefix_mismatch")
            continue
        if edge in seen:
            if seen[edge] != barrier:
                ambiguous.setdefault(material, "conflicting_duplicate_edge")
            continue
        seen[edge] = barrier
        rows_by_material[material].append({"edge_id": edge, "em": barrier})
    return dict(rows_by_material), ambiguous


def choose_structure(materials: Iterable[str], rows_by_material: Mapping[str, list],
                     ambiguous: Mapping[str, str]) -> dict[str, Any]:
    """Structure rule over one system's structure universe (lowest E*_3, then id)."""

    candidates = []
    statuses: Counter[str] = Counter()
    for material in sorted(set(materials)):
        if material in ambiguous:
            statuses["ambiguous_edges"] += 1
            continue
        result = rate_limiting_hop(rows_by_material.get(material, []), DIMENSION)
        statuses[result["status"]] += 1
        if result["status"] == "percolating":
            candidates.append((result["critical_barrier_eV"], material, result))
    if not candidates:
        return {"status": "no_percolating_structure", "structure_statuses": dict(statuses)}
    critical, material, result = min(candidates, key=lambda item: (item[0], item[1]))
    return {"status": "chosen", "material_id": material, "critical_barrier_eV": critical,
            "rate_limiting_edge_ids": result["rate_limiting_edge_ids"],
            "tie": len(result["tied_hops"]) > 1, "structure_statuses": dict(statuses)}


def unit_hop(chosen: Mapping[str, Any], material_rows: list[dict[str, Any]],
             dft_edges: set[str]) -> dict[str, Any] | None:
    """Calibration-side unit hop of a chosen nebDFT2k structure (frozen Study-5 rule).

    h*_3 when it is a nebDFT2k hop (``rate_limiting_exact``); otherwise the structure's
    nebDFT2k hop with the smallest |em_bvse - E*_3|, hops above and below E*_3 both
    allowed, ties by ascending canonical hop key (``bottleneck_proxy``).  Covariate
    only: BVSE barriers and identifiers.  The returned edge IDs are the nebDFT2k
    directions of the one physical hop.
    """

    exact = sorted(e for e in chosen["rate_limiting_edge_ids"] if e in dft_edges)
    if exact:
        return {"unit_edge_ids": exact, "unit_hop_role": RATE_LIMITING_EXACT,
                "unit_bvse_barrier_eV": chosen["critical_barrier_eV"]}
    hops, failure = physical_hops(material_rows)
    if failure is not None:
        return None
    critical = chosen["critical_barrier_eV"]
    labelled = [h for h in hops if any(e in dft_edges for e in h["edge_ids"])]
    if not labelled:
        return None
    best = min(labelled, key=lambda h: (abs(h["em"] - critical), h["key"]))
    return {"unit_edge_ids": sorted(e for e in best["edge_ids"] if e in dft_edges),
            "unit_hop_role": BOTTLENECK_PROXY, "unit_bvse_barrier_eV": best["em"]}


def material_systems(*tables: Iterable[Mapping[str, str]]) -> tuple[dict[str, str], dict[str, Any]]:
    """One normalized chemical system per material across every table.

    A material whose rows disagree on the normalized system, or carry an unparseable
    chemsys, is never assigned; it is reported and kept out of every universe.
    """

    seen: dict[str, set[str]] = defaultdict(set)
    unparseable: set[str] = set()
    for table in tables:
        for row in table:
            system = _system(row["chemsys"])
            if system is None:
                unparseable.add(row["material_id"])
            else:
                seen[row["material_id"]].add(system)
    inconsistent = sorted(m for m, systems in seen.items() if len(systems) > 1)
    rejected = set(inconsistent) | unparseable
    mapping = {m: next(iter(systems)) for m, systems in seen.items() if m not in rejected}
    return mapping, {"inconsistent_chemical_system_materials": inconsistent,
                     "unparseable_chemsys_materials": sorted(unparseable)}


def structural_populations(
    *,
    nebbvse_rows: Iterable[Mapping[str, str]],
    bvel_rows: Iterable[Mapping[str, str]],
    nebdft_rows: Iterable[Mapping[str, str]],
    membership_records: Iterable[Mapping[str, Any]],
    primary_assignment: Mapping[str, str],
    excluded_elements: Iterable[str],
) -> dict[str, Any]:
    """See the module docstring; ``excluded_elements`` is the frozen TM/Hubbard-U list."""

    nebbvse_rows, bvel_rows, nebdft_rows = list(nebbvse_rows), list(bvel_rows), list(nebdft_rows)
    excluded = set(excluded_elements)
    if not excluded:
        raise ValueError("the frozen TM/Hubbard-U element list is empty")
    rows_by_material, ambiguous = edge_tables(nebbvse_rows)
    material_system, rejected = material_systems(nebbvse_rows, bvel_rows, nebdft_rows)
    for material in rejected["inconsistent_chemical_system_materials"]:
        ambiguous.setdefault(material, "inconsistent_chemical_system")
    dft_edges = {row["edge_id"] for row in nebdft_rows}

    # ---- the calibration-side materials whose own h*_3 is a nebDFT2k hop
    dft_materials = sorted({row["material_id"] for row in nebdft_rows})
    hop_in_dft = []
    for material in dft_materials:
        if material in ambiguous or material not in material_system:
            continue
        result = rate_limiting_hop(rows_by_material.get(material, []), DIMENSION)
        if result["status"] == "percolating" and any(
                edge in dft_edges for edge in result["rate_limiting_edge_ids"]):
            hop_in_dft.append(material)
    hop_in_dft_systems = {material_system[m] for m in hop_in_dft}

    # ---- structure universe per normalized system (each material in exactly one)
    universe: dict[str, set[str]] = defaultdict(set)
    for table in (nebbvse_rows, bvel_rows):
        for row in table:
            if row["material_id"] in material_system:
                universe[material_system[row["material_id"]]].add(row["material_id"])
    dft_systems = {material_system[m] for m in dft_materials if m in material_system}
    bvel_systems = {material_system[r["material_id"]] for r in bvel_rows
                    if r["material_id"] in material_system}

    splits: dict[str, set[str]] = defaultdict(set)
    for record in membership_records:
        system = _system(str(record["chemical_system_id"]))
        if system is None:
            raise ValueError(f"unparseable membership chemical system {record['chemical_system_id']!r}")
        splits[system].add(record["published_split"])
    partition_universe = set(splits)
    outside = sorted(dft_systems - partition_universe)
    if outside:
        # A DFT-labelled system must never reach the label-free prospective pool.
        raise ValueError(f"{len(outside)} nebDFT2k chemical system(s) lie outside the partition "
                         f"universe, e.g. {outside[0]}")

    def tm_hubbard_u(system: str) -> bool:
        if system in primary_assignment:
            return primary_assignment[system] != "primary"
        return bool(set(system.split("-")) & excluded)

    def classify(system: str) -> str:
        if system not in partition_universe:
            return "outside_partition_universe"
        if len(splits[system]) > 1:
            return "spanning_published_splits"
        return {"test": "published_test", "train": "published_train",
                "val": "published_validation"}[next(iter(splits[system]))]

    # Material-level published split (nebDFT2k hop records of the frozen membership):
    # a development unit is never built on a published-validation material.
    material_splits: dict[str, set[str]] = defaultdict(set)
    for record in membership_records:
        if (record.get("material_id") is not None
                and record.get("record_kind", "nebdft2k_hop") == "nebdft2k_hop"):
            material_splits[str(record["material_id"])].add(record["published_split"])
    dft_by_system: dict[str, list[str]] = defaultdict(list)
    for material in dft_materials:
        if material in material_system:
            dft_by_system[material_system[material]].append(material)

    sides: dict[str, dict[str, list[str]]] = {
        name: defaultdict(list) for name in ("development", "P", "Q")}
    not_a_candidate: list[str] = []
    ties = 0
    excluded_from_p: dict[str, list[str]] = defaultdict(list)
    units: list[dict[str, Any]] = []
    for system in sorted(dft_systems | bvel_systems):
        in_dft = system in dft_systems
        membership = classify(system)
        if in_dft and membership == "published_train":
            side = "P"  # calibration_set: published-train-only systems
        elif in_dft and membership in ("published_test", "spanning_published_splits"):
            # Development: every system with published-test contact (its outcomes already
            # enter Study 2a / G1, so it can never be a calibration unit).
            side = "development"
        elif in_dft:
            # Validation-only systems: sealed reserve, excluded from Study 5 (parent rule).
            excluded_from_p[membership].append(system)
            continue
        elif membership == "outside_partition_universe":
            side = "Q"
        else:
            # BVEL13k-only system inside the partition universe: excluded from the pool
            # (Study-5 exclusion) and, without nebDFT2k hops, not a calibration unit.
            not_a_candidate.append(system)
            continue
        record = sides[side]
        record["candidate_systems"].append(system)
        # TM/Hubbard-U: a reported stratum and density-ratio covariate, never an
        # exclusion in Study 5 (counted below among structurally eligible units).
        tm = tm_hubbard_u(system)
        if side == "Q":
            chosen = choose_structure(universe.get(system, ()), rows_by_material, ambiguous)
            if chosen["status"] != "chosen":
                record["no_percolating_structure"].append(system)
                continue
            ties += chosen["tie"]
            hop = {"unit_edge_ids": sorted(chosen["rate_limiting_edge_ids"]),
                   "unit_hop_role": RATE_LIMITING_EXACT,
                   "unit_bvse_barrier_eV": chosen["critical_barrier_eV"]}
        else:
            materials = [m for m in dft_by_system[system] if side == "P"
                         or "val" not in material_splits.get(m, set())]
            chosen = choose_structure(materials, rows_by_material, ambiguous)
            if chosen["status"] != "chosen":
                record["no_percolating_nebDFT2k_structure"].append(system)
                continue
            ties += chosen["tie"]
            hop = unit_hop(chosen, rows_by_material[chosen["material_id"]], dft_edges)
            if hop is None:  # cannot happen for a nebDFT2k material; fail visibly if it does
                record["no_nebDFT2k_hop_in_structure"].append(system)
                continue
        record["structural_upper_bound"].append(system)
        record[hop["unit_hop_role"]].append(system)
        if tm:
            record["tm_hubbard_u_stratum"].append(system)
        units.append({"chemical_system": system, "side": side,
                      "material_id": chosen["material_id"],
                      # the nebDFT2k direction of the unit hop on the calibration side,
                      # else the first stored direction (both are one physical hop)
                      "unit_edge_id": hop["unit_edge_ids"][0],
                      "unit_edge_ids": hop["unit_edge_ids"],
                      "unit_hop_role": hop["unit_hop_role"],
                      "unit_bvse_barrier_eV": hop["unit_bvse_barrier_eV"],
                      "rate_limiting_edge_ids": sorted(chosen["rate_limiting_edge_ids"]),
                      "critical_bvse_barrier_eV": chosen["critical_barrier_eV"],
                      "bottleneck_gap_eV": hop["unit_bvse_barrier_eV"]
                      - chosen["critical_barrier_eV"],
                      "tm_hubbard_u": tm,
                      "structure_rule_tie": chosen["tie"]})

    def summary(record: Mapping[str, list[str]]) -> dict[str, Any]:
        return {key: {"count": len(values), "sha256": _digest(values), "systems": sorted(values)}
                for key, values in sorted(record.items())}

    return {
        "schema": POPULATION_AUDIT_SCHEMA,
        "dimension": DIMENSION,
        "outcome_columns_read": False,
        "counts": {
            "nebDFT2k_materials_rate_limiting_hop_in_nebDFT2k": len(hop_in_dft),
            "unique_chemical_systems_among_them": len(hop_in_dft_systems),
            **{f"{side}_structural_upper_bound_systems": len(sides[side]["structural_upper_bound"])
               for side in ("development", "P", "Q")},
            **{f"{side}_{role}_systems": len(sides[side][role])
               for side in ("development", "P", "Q") for role in
               (RATE_LIMITING_EXACT, BOTTLENECK_PROXY, "tm_hubbard_u_stratum")},
            "structure_rule_ties": ties,
        },
        "calibration_side_materials": {"count": len(hop_in_dft), "sha256": _digest(hop_in_dft),
                                       "materials": sorted(hop_in_dft)},
        "populations": {side: summary(record) for side, record in sides.items()},
        "units": sorted(units, key=lambda u: (u["side"], u["chemical_system"])),
        "excluded_from_P": {key: {"count": len(values), "sha256": _digest(values),
                                  "systems": sorted(values)}
                            for key, values in sorted(excluded_from_p.items())},
        "not_a_candidate": {"count": len(not_a_candidate), "sha256": _digest(not_a_candidate),
                            "systems": sorted(not_a_candidate),
                            "reason": "BVEL13k-only system inside the partition universe"},
        "material_quality": {key: {"count": len(values), "materials": values}
                             for key, values in rejected.items()},
        "tm_hubbard_u": ("reported stratum (frozen assignment for partition-universe systems; "
                         "frozen excluded_elements for the others), not an exclusion"),
        "gate0_not_applied": ("nebDFT2k element and hop-length envelope (applied by the "
                              "development_population stage from descriptors)"),
        "filters_not_applied": ["ML-NEB convergence", "BVSE path reproduction (P only)",
                                "weight-fit hold-out slices"],
        "interpretation_flags": [
            ("P counts published-train-only systems (calibration_set); development counts "
             "test-only and spanning systems; published-validation systems are the sealed "
             "reserve (parent published_validation_split_role rule)"),
            "not a population manifest: counts are upper bounds (see filters_not_applied)"],
        "interpretation": "structural upper bounds before the remaining frozen filters; never "
                          "chosen from or compared with any outcome",
    }
