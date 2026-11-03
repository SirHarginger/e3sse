"""descriptor_availability: label-free covariates for every Gate-1 structural unit.

For each development, P and Q candidate unit of the Gate-1 structural audit this
computes ``screening.applicability_support.descriptor_vector`` from the configured
pristine host structure, the hop length of the unit's rate-limiting edge
(``|r_t + o.L - r_s|``, own cell offset) and the nebBVSE122k BVSE barrier of that
edge.  No DFT or ML quantity is read.  A unit whose structure, edge or BVSE value
is unavailable keeps ``None`` in that feature (fail closed, never imputed).

Outputs: ``descriptors.json`` (vectors per unit, consumed by the population,
support and density-ratio stages) and ``descriptor_availability.json`` (table
R_descriptor_availability: per feature and population, units, missing, finite
fraction; counts only).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from e3sse.production.runners.context import (
    MissingEvidence,
    RunnerContext,
    RunnerValidationError,
    canonical_sha256,
)
from e3sse.production.runners.gate1_tables import linkage_tables, pinned_g0_inputs
from e3sse.production.runners.structures import StructureSource, hop_length
from e3sse.screening.applicability_support import DESCRIPTOR_NAMES, descriptor_vector
from e3sse.screening.rate_limiting_hop import parse_edge_id

DESCRIPTOR_SCHEMA = "e3sse.production.unit-descriptors"


def unit_key(unit: dict[str, Any]) -> str:
    return f"{unit['side']}:{unit['chemical_system']}"


def gate1_units(ctx: RunnerContext) -> list[dict[str, Any]]:
    audit = ctx.upstream("gate1_population_audit", "gate1_structural_population_audit")
    if audit.get("outcome_columns_read") is not False or audit.get("dimension") != 3:
        raise ValueError("the Gate-1 structural audit is not an outcome-free d = 3 audit")
    return list(audit["units"])


def unit_descriptors(units, *, structures: StructureSource, bvse: dict[str, float]) -> dict:
    rows = {}
    for unit in units:
        key = unit_key(unit)
        try:
            atoms = structures.load(unit["material_id"])
        except (MissingEvidence, OSError, ValueError, KeyError) as exc:
            rows[key] = {**unit, "vector": None, "missing_reason": f"structure: {exc}"}
            continue
        try:
            length, reason = hop_length(atoms, parse_edge_id(unit["unit_edge_id"])), None
        except (RunnerValidationError, ValueError, IndexError) as exc:
            length, reason = None, f"hop_length: {exc}"
        vector = descriptor_vector(atoms.get_chemical_symbols(),
                                   cell_volume_A3=float(atoms.get_volume()),
                                   hop_length_A=length, em_bvse_eV=bvse.get(unit["unit_edge_id"]))
        rows[key] = {**unit, "vector": vector, "missing_reason": reason,
                     "elements": sorted(set(atoms.get_chemical_symbols()))}
    return rows


def availability_table(rows: dict[str, Any]) -> list[dict[str, Any]]:
    table = []
    by_side: dict[str, list] = defaultdict(list)
    for row in rows.values():
        by_side[row["side"]].append(row)
    for side, members in sorted(by_side.items()):
        for index, name in enumerate(DESCRIPTOR_NAMES):
            values = [None if m["vector"] is None else m["vector"][index] for m in members]
            finite = sum(v is not None and math.isfinite(v) for v in values)
            table.append({"feature": name, "population": side, "n_units": len(values),
                          "n_missing": len(values) - finite,
                          "finite_fraction": finite / len(values) if values else None})
    return table


def run(ctx: RunnerContext) -> dict[str, Any]:
    units = gate1_units(ctx)
    g0 = pinned_g0_inputs(ctx.freeze_config)
    structures = StructureSource(ctx.runner_settings("structures").get("source"),
                                 ctx.data_root, g0)
    tables = linkage_tables(ctx.data_root / "raw", g0)
    bvse = {}
    for row in tables["nebBVSE122k"]:
        try:
            bvse[row["edge_id"]] = float(row["bvse_migration_barrier_eV"])
        except ValueError:
            continue
    rows = unit_descriptors(units, structures=structures, bvse=bvse)
    descriptors = {"schema": DESCRIPTOR_SCHEMA, "feature_names": list(DESCRIPTOR_NAMES),
                   "label_free": True, "units": rows,
                   "structure_source": structures.identity(),
                   "units_sha256": canonical_sha256(sorted(rows))}
    table = availability_table(rows)
    outputs = {"descriptors": ctx.publish("descriptors.json", descriptors),
               "descriptor_availability": ctx.publish("descriptor_availability.json",
                                                      {"table": "R_descriptor_availability",
                                                       "rows": table})}
    complete = sum(r["vector"] is not None and all(v is not None for v in r["vector"])
                   for r in rows.values())
    return {"outputs": outputs, "summary": {"units": len(rows), "complete_units": complete}}
