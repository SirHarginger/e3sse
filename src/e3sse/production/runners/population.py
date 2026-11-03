"""development_population: frozen unit manifests before any outcome or ML-NEB.

From the Gate-1 structural audit (one structure per chemical system, d = 3
rate-limiting hop) and the label-free descriptors:

* development units: published-test systems (h*_3 a nebDFT2k hop);
* P units: published-train-only systems (owner decision 2026-11-02);
* Q units: BVEL13k systems outside the partition universe that pass Gate 0
  (``amendments.prospective.populations.gate0_nebdft2k_criteria``);
* weight-fit slices (``density_ratio.settings.holdout_rule``): a P or Q system is in
  the slice iff SHA256('e3sse.weight-holdout' NUL parent protocol ID NUL system),
  read as an integer over 2^256, is below ``holdout_fraction``.  Membership depends
  on the system identifier only.  P-slice systems are never calibration units;
  Q-slice systems are never selectable.

The development manifest (hop IDs of the development units) is hashed exactly as
``contracts._verify_development_manifest`` hashes it, so its SHA-256 can be pinned
in ``amendments.prospective.development_manifest_sha256`` before any development
outcome is read.
"""

from __future__ import annotations

import hashlib
from typing import Any

from e3sse.production.runners.context import (
    RunnerContext,
    RunnerValidationError,
    canonical_sha256,
)
from e3sse.production.runners.descriptors import gate1_units, unit_key

POPULATION_SCHEMA = "e3sse.production.population-manifest"
DEVELOPMENT_SCHEMA = "e3sse.development-manifest"
HOLDOUT_DOMAIN = b"e3sse.weight-holdout"
HOP_LENGTH_INDEX = 5  # DESCRIPTOR_NAMES.index("hop_length_A")


def weight_fit_slice(parent_protocol_id: str, chemical_system: str, fraction: float) -> bool:
    digest = hashlib.sha256(HOLDOUT_DOMAIN + b"\0" + parent_protocol_id.encode() + b"\0"
                            + chemical_system.encode()).hexdigest()
    return int(digest, 16) / float(2 ** 256) < fraction


def gate0_pool_decisions(units: list[dict[str, Any]], descriptors: dict[str, Any]
                         ) -> dict[str, dict[str, Any]]:
    """Q units: calibration-side element support and h*_3 length range (covariates)."""

    calibration = [descriptors[unit_key(u)] for u in units if u["side"] in ("development", "P")]
    elements = {e for row in calibration for e in row.get("elements", [])}
    lengths = [row["vector"][HOP_LENGTH_INDEX] for row in calibration
               if row.get("vector") and row["vector"][HOP_LENGTH_INDEX] is not None]
    if not elements or not lengths:
        raise RunnerValidationError("no calibration-side elements or hop lengths for Gate 0")
    low, high = min(lengths), max(lengths)
    decisions = {}
    for unit in units:
        if unit["side"] != "Q":
            continue
        row = descriptors[unit_key(unit)]
        reasons = []
        unit_elements = set(row.get("elements", []))
        if not unit_elements:
            reasons.append("elements_unknown")
        elif not unit_elements <= elements:
            reasons.append("element_outside_calibration_support")
        length = row["vector"][HOP_LENGTH_INDEX] if row.get("vector") else None
        if length is None:
            reasons.append("hop_length_undefined")
        elif not low <= length <= high:
            reasons.append("hop_length_outside_calibration_range")
        decisions[unit_key(unit)] = {"passes": not reasons, "reasons": reasons}
    return {"decisions": decisions, "element_support": sorted(elements),
            "hop_length_range_A": [low, high]}


def build_population(units, descriptors, *, parent_protocol_id: str, fraction: float
                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    gate0 = gate0_pool_decisions(units, descriptors)
    records = []
    for unit in units:
        key = unit_key(unit)
        side = unit["side"]
        record = {"unit_key": key, "unit_id": unit["unit_edge_id"], "side": side,
                  "chemical_system": unit["chemical_system"], "material_id": unit["material_id"],
                  "rate_limiting_edge_ids": unit["rate_limiting_edge_ids"]}
        if side == "development":
            record["role"] = "development"
        else:
            if side == "Q" and not gate0["decisions"][key]["passes"]:
                record.update(role="excluded_gate0", gate0_reasons=gate0["decisions"][key]["reasons"])
                records.append(record)
                continue
            in_slice = weight_fit_slice(parent_protocol_id, unit["chemical_system"], fraction)
            base = "calibration" if side == "P" else "pool"
            record["role"] = f"{base}_weight_slice" if in_slice else base
        records.append(record)
    development = sorted(r["unit_id"] for r in records if r["role"] == "development")
    manifest = {"schema": DEVELOPMENT_SCHEMA, "parent_protocol_id": parent_protocol_id,
                "rule": "published-test systems with nebDFT2k hops, spanning systems excluded, one "
                        "structure per system by structure_rule, its h*_3 (a nebDFT2k hop)",
                "hop_ids": development,
                "chemical_systems": sorted(r["chemical_system"] for r in records
                                           if r["role"] == "development")}
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["role"]] = counts.get(record["role"], 0) + 1
    population = {"schema": POPULATION_SCHEMA, "units": records, "counts": dict(sorted(counts.items())),
                  "weight_slice_rule": {"domain": HOLDOUT_DOMAIN.decode(), "fraction": fraction,
                                        "salt": parent_protocol_id},
                  "gate0": {k: v for k, v in gate0.items() if k != "decisions"},
                  "label_free": True}
    return population, manifest


def run(ctx: RunnerContext) -> dict[str, Any]:
    units = gate1_units(ctx)
    descriptors = ctx.upstream("descriptor_availability", "descriptors")["units"]
    prospective = ctx.active_protocol["amendments"]["prospective"]
    fraction = float(prospective["density_ratio"]["settings"]["holdout_fraction"])
    parent_id = ctx.active_protocol["parent_protocol"]["protocol_id"]
    population, manifest = build_population(units, descriptors, parent_protocol_id=parent_id,
                                            fraction=fraction)
    pinned = prospective.get("development_manifest_sha256")
    if pinned not in (None, "server_evidence_pending") and pinned != manifest["manifest_sha256"]:
        raise RunnerValidationError("the development manifest differs from its pinned SHA-256")
    outputs = {"population_manifest": ctx.publish("population_manifest.json", population),
               "development_manifest": ctx.publish("development_manifest.json", manifest)}
    return {"outputs": outputs,
            "summary": {"counts": population["counts"],
                        "development_manifest_sha256": manifest["manifest_sha256"],
                        "pin_status": "matches" if pinned == manifest["manifest_sha256"]
                        else "unpinned"}}
