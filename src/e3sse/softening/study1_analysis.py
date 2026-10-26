"""Study-1 analysis: leakage-free Eq. 17 estimates, bootstrap and label budget.

Inputs are the merged per-frame statistics of the DFT reference and each frozen
model.  No model is evaluated here.  The steps are:

1. consolidate the statistics in ascending frame-ID order (the frozen cluster order)
   and check that each model's DFT-derived sums equal the reference exactly;
2. derive frame eligibility with the frozen leakage audit
   (``contracts.audit_softening_frame_eligibility_batch`` + the primary guard):
   one material-scope selection per published-train target hop (primary and,
   labelled separately, oracle own-path), and pooled global/family/element atlas
   selections per population;
3. freeze configurational-energy decile cut points from DFT energies only;
4. freeze the label-budget hop set (model-independent) before any draw;
5. evaluate every unit (all models at once) in deterministic, resumable shards.

Scopes are independent: an unsupported estimate is reported as unsupported with its
reasons and is never replaced by another scope's estimate.

Pooled atlas selections describe a model's softening; they correct no hop, so they
are audited against a sentinel target hop that matches no hop.  Leave-target-hop-out
pooled scales for barrier correction belong to Study 2 and follow from the same
sufficient statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from e3sse.protocol.contracts import (
    audit_softening_frame_eligibility_batch,
    require_primary_softening_selection,
)
from e3sse.protocol.partitions import analysis_seed
from e3sse.softening.frame_statistics import (
    COMPONENT_SETS,
    LITHIUM_Z,
    draw_without_replacement,
    eq17,
    iter_bootstrap_indices,
    percentile_interval,
    residual_fraction,
    support_status,
)

ANALYSIS_SCHEMA = "e3sse.study1.analysis.v1"
POOLED_SENTINEL_HOP = "study1-pooled-atlas-without-target-hop"
POPULATIONS = ("primary", "tm_hubbard_u_secondary")
F_BLOCK_SENSITIVITY = "primary_f_block_excluded"
F_BLOCK_RULE = "pre-specified secondary: additionally exclude f-block elements Ce-Lu"
F_BLOCK_Z = tuple(range(58, 72))  # Ce .. Lu
DECILES = 10
SET_INDEX = {name: index for index, name in enumerate(COMPONENT_SETS)}
COMMON_HOP_SCHEMA = "e3sse.study1.common-hop-population.v1"
ANALYSIS_CODE_FILES = (
    "src/e3sse/softening/frame_statistics.py",
    "src/e3sse/softening/study1_analysis.py",
    "src/e3sse/softening/study1_inputs.py",
    "src/e3sse/protocol/contracts.py",
    "src/e3sse/protocol/partitions.py",
    "src/e3sse/softening/study1_tables.py",
    "scripts/analyze_study1_softening_atlas.py",
)
ATLAS_DESCRIPTIVE_LABEL = "atlas_descriptive_not_for_correction"
PATH_FILE_KINDS = ("band_optim_step", "traj_init")


class Study1AnalysisError(RuntimeError):
    """A hard provenance or consistency failure; distinct from an unsupported estimate."""


# --------------------------------------------------------------------------- consolidation


def consolidate(
    frames: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Arrays in ascending frame-ID order; models are checked against the reference.

    ``frames`` are the frozen published-train membership rows; ``reference`` and each
    ``models[name]`` are merged ``frame_statistics.npz`` arrays (source-index order).
    """

    import numpy as np

    by_index = {int(row["source_frame_index"]): row for row in frames}
    reference_index = reference["frame_index"].tolist()
    if sorted(by_index) != reference_index:
        raise Study1AnalysisError("reference frames differ from the published-train membership")
    order = sorted(range(len(reference_index)), key=lambda i: by_index[reference_index[i]]["frame_id"])
    position = np.empty(len(order), dtype=np.int64)
    position[np.array(order, dtype=np.int64)] = np.arange(len(order))
    names = sorted(models)
    n, m = len(order), len(names)

    def element_table(arrays: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
        rows = position[arrays["element_frame_row"]]
        keys = np.lexsort((rows, arrays["element_Z"]))
        table = {"row": rows[keys], "Z": arrays["element_Z"][keys]}
        for name in columns:
            table[name] = arrays[f"element_{name}"][keys]
        return table

    ref = element_table(reference, ("n_atoms", "sxx", "n_informative"))
    set_counts = np.zeros((n, len(COMPONENT_SETS), 2), dtype=np.int64)  # n_atoms, n_informative
    for set_name, s in SET_INDEX.items():
        member = _members(set_name, ref["Z"])
        np.add.at(set_counts[:, s, 0], ref["row"][member], ref["n_atoms"][member])
        np.add.at(set_counts[:, s, 1], ref["row"][member], ref["n_informative"][member])
    set_stats = np.zeros((m, n, len(COMPONENT_SETS), 3), dtype=np.float64)  # sxy, sxx, syy
    success = np.zeros((m, n), dtype=bool)
    element_stats = np.zeros((m, len(ref["Z"]), 3), dtype=np.float64)
    reference_hashes = reference["atomic_numbers_sha256"][order]
    for k, name in enumerate(names):
        arrays = models[name]
        if arrays["frame_index"].tolist() != reference_index:
            raise Study1AnalysisError(f"{name}: frames differ from the reference")
        if not (arrays["atomic_numbers_sha256"][order] == reference_hashes).all():
            raise Study1AnalysisError(f"{name}: atomic numbers differ from the reference")
        ok = arrays["success"][order]
        success[k] = ok
        table = element_table(arrays, ("n_atoms", "sxy", "sxx", "syy", "n_informative"))
        # Successful frames carry every element row; the DFT parts must equal the reference.
        mask = ok[ref["row"]]
        expected = {key: ref[key][mask] for key in ("row", "Z", "n_atoms", "sxx", "n_informative")}
        for key, value in expected.items():
            if not np.array_equal(table[key], value):
                raise Study1AnalysisError(f"{name}: DFT-derived {key} differs from the reference")
        element_stats[k][mask] = np.stack([table["sxy"], table["sxx"], table["syy"]], axis=1)
        for set_name, s in SET_INDEX.items():
            member = _members(set_name, table["Z"])
            for column, key in enumerate(("sxy", "sxx", "syy")):
                # Unbuffered, in fixed (Z, row) order: deterministic across runs.
                np.add.at(set_stats[k, :, s, column], table["row"][member], table[key][member])
    return {
        "models": names,
        "frame_id": np.array([by_index[reference_index[i]]["frame_id"] for i in order], dtype=str),
        "source_frame_index": reference["frame_index"][order],
        "n_atoms": reference["n_atoms"][order],
        "dft_energy_eV": reference["dft_energy_eV"][order],
        "set_counts": set_counts,
        "set_stats": set_stats,
        "success": success,
        "element_row": ref["row"],
        "element_Z": ref["Z"],
        "element_counts": np.stack([ref["n_atoms"], ref["n_informative"]], axis=1),
        "element_stats": element_stats,
    }


def _members(set_name: str, z: Any) -> Any:
    import numpy as np

    if set_name == "all":
        return np.ones(len(z), dtype=bool)
    if set_name == "Li":
        return z == LITHIUM_Z
    return z != LITHIUM_Z


# --------------------------------------------------------------------------- frame metadata


def frame_metadata(
    frames: Sequence[Mapping[str, Any]],
    frame_ids: Sequence[str],
    partition_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Membership rows in consolidated order, with frozen population and family."""

    population = partition_manifest["primary_population"]["assignment_by_chemical_system"]
    family = partition_manifest["anion_families"]["assignment_by_material"]
    by_id = {row["frame_id"]: row for row in frames}
    rows = []
    for frame_id in frame_ids:
        row = dict(by_id[frame_id])
        if row["chemical_system_id"] not in population or row["material_id"] not in family:
            raise Study1AnalysisError(f"{frame_id}: no frozen population or anion family")
        row["population"] = population[row["chemical_system_id"]]
        row["anion_family"] = family[row["material_id"]]
        row["anion_family_evidence_status"] = "verified"
        row["elements"] = row["chemical_system_id"].split("-")
        row["element_evidence_status"] = "verified"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- deciles


def energy_deciles(
    metadata: Sequence[Mapping[str, Any]], n_atoms: Any, energies: Any, population: str
) -> tuple[dict[str, Any], Any]:
    """Frozen ``configurational_energy_deciles`` for one population (DFT energies only).

    dE_f = [E(f) - min E over R(m, N_f)] / N_f, R(m, N) = published-train frames of
    material m with N atoms in this population; cut points numpy.quantile 0.1..0.9
    (linear) over all such frames; right-closed bins (a value equal to a cut point
    goes to the lower decile); empty deciles stay empty.  Returns (record, deciles)
    where deciles holds 1..10 for population frames and 0 elsewhere.
    """

    import numpy as np

    rows = [i for i, row in enumerate(metadata) if row["population"] == population]
    minimum: dict[tuple[str, int], float] = {}
    for i in rows:
        key = (metadata[i]["material_id"], int(n_atoms[i]))
        minimum[key] = min(minimum.get(key, math.inf), float(energies[i]))
    delta = np.full(len(metadata), np.nan)
    for i in rows:
        key = (metadata[i]["material_id"], int(n_atoms[i]))
        delta[i] = (float(energies[i]) - minimum[key]) / int(n_atoms[i])
    values = delta[rows]
    if not np.isfinite(values).all():
        raise Study1AnalysisError(f"non-finite DFT energy in the {population} population")
    if len(values) == 0:
        return {"population": population, "frames": 0, "cut_points_eV_per_atom": None}, \
            np.zeros(len(metadata), dtype=np.int64)
    cuts = np.quantile(values, np.arange(1, DECILES) / DECILES, method="linear")
    deciles = np.zeros(len(metadata), dtype=np.int64)
    deciles[rows] = np.searchsorted(cuts, values, side="left") + 1
    counts = Counter(deciles[rows].tolist())
    return {
        "population": population,
        "frames": len(rows),
        "reference_groups": len(minimum),
        "cut_points_eV_per_atom": cuts.tolist(),
        "frames_per_decile": {str(d): counts.get(d, 0) for d in range(1, DECILES + 1)},
        "empty_deciles": [d for d in range(1, DECILES + 1) if counts.get(d, 0) == 0],
    }, deciles


# --------------------------------------------------------------------------- eligibility


def _scope_evidence(scope: str, target: str | None) -> dict[str, Any]:
    sources = {
        "global": "frozen published split",
        "material": "frozen partition_membership material_id",
        "family": "frozen partition_manifest.anion_families (dominant-anion-count-v1)",
        "element": "frozen chemical_system_id (sorted-unique-elements-v1)",
    }
    return {"status": "verified", "scope": scope, "target": target, "source": sources[scope]}


def eligibility_units(
    *,
    protocol: Mapping[str, Any],
    partition_records: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    target_hops: Sequence[Mapping[str, Any]],
    population_of_system: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audited units: (unit list, accounting).  Units carry consolidated row indices."""

    if any(hop["hop_id"] == POOLED_SENTINEL_HOP for hop in target_hops):
        raise Study1AnalysisError("the pooled sentinel collides with a real hop ID")
    row_of = {row["frame_id"]: i for i, row in enumerate(metadata)}
    by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_population: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        by_material[row["material_id"]].append(row)
        by_population[row["population"]].append(row)
    requests, describe = [], []
    for hop in target_hops:
        population = population_of_system[hop["chemical_system_id"]]
        for kind in ("primary", "oracle"):
            requests.append({
                "records": by_material.get(hop["material_id"], []),
                "target_hop_id": hop["hop_id"], "estimator_scope": "material",
                "scope_evidence": _scope_evidence("material", hop["material_id"]),
                "allowed_partition_role": "training_or_calibration",
                "target_material_id": hop["material_id"], "estimator_kind": kind,
            })
            describe.append({"kind": "target_hop", "population": population, "scope": "material",
                             "scope_target": hop["material_id"], "target_hop_id": hop["hop_id"],
                             "material_id": hop["material_id"], "estimator_kind": kind})
    for population in POPULATIONS:
        records = by_population.get(population, [])
        if not records:
            continue
        families = sorted({row["anion_family"] for row in records})
        elements = sorted({e for row in records for e in row["elements"]})
        targets = [("global", None)] + [("family", f) for f in families] + [
            ("element", e) for e in elements
        ]
        for scope, target in targets:
            requests.append({
                "records": records, "target_hop_id": POOLED_SENTINEL_HOP, "estimator_scope": scope,
                "scope_evidence": _scope_evidence(scope, target),
                "allowed_partition_role": "training_or_calibration",
                **({"target_family": target} if scope == "family" else {}),
                **({"target_element": target} if scope == "element" else {}),
            })
            describe.append({"kind": "pooled", "population": population, "scope": scope,
                             "scope_target": target, "target_hop_id": None,
                             "material_id": None, "estimator_kind": "primary"})
    units, reasons = [], defaultdict(Counter)
    selections = audit_softening_frame_eligibility_batch(
        requests, protocol=protocol, partition_records=partition_records, partition="published"
    )
    for info, selection in zip(describe, selections, strict=True):
        if info["estimator_kind"] == "primary":
            require_primary_softening_selection(selection, protocol)
        elif selection.get("analysis_label") != "oracle_upper_bound":
            raise Study1AnalysisError("oracle selection is not labelled as an oracle")
        rows = sorted(row_of[record["frame_id"]] for record in selection["eligible_records"])
        # Diagnostic (F11): admissible frames from other hops' NEB paths of the structure.
        other_path = sum(
            1 for record in selection["eligible_records"]
            if record["raw_file_kind"] in PATH_FILE_KINDS
            and record["hop_id"] != info["target_hop_id"]
        ) if info["kind"] == "target_hop" else None
        reasons[f"{info['kind']}:{info['estimator_kind']}"].update(selection["reason_counts"])
        units.append({**info, "rows": rows, "selection_sha256": selection["selection_sha256"],
                      "other_hop_path_frames": other_path,
                      "reason_counts": selection["reason_counts"]})
    accounting = {key: dict(sorted(value.items())) for key, value in sorted(reasons.items())}
    return units, accounting


def f_block_free(chemical_system: str) -> bool:
    from ase.data import chemical_symbols

    symbols = {chemical_symbols[z] for z in F_BLOCK_Z}
    return not set(chemical_system.split("-")) & symbols


# --------------------------------------------------------------------------- unit payloads


def analysis_id(unit: Mapping[str, Any]) -> str:
    """Bootstrap analysis ID: excludes model and component set (common random numbers)."""

    parts = [unit["population"], unit["scope"], unit.get("scope_target") or "-",
             unit.get("target_hop_id") or "-", unit["estimator_kind"]]
    if unit.get("decile"):
        parts.append(f"decile={unit['decile']}")
    return "study1|" + "|".join(str(part) for part in parts)


def estimate_payloads(
    units: Sequence[Mapping[str, Any]],
    deciles: Any,
    metadata: Sequence[Mapping[str, Any]],
    *,
    constants: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Estimate units: hops ([all, Li, host]), pooled scopes, and decile scopes."""

    import numpy as np

    payloads = []
    for unit in units:
        if unit["kind"] == "target_hop":
            sets = ["all"] if unit["estimator_kind"] == "oracle" else list(COMPONENT_SETS)
        elif unit["scope"] == "global":
            sets = list(COMPONENT_SETS)
        elif unit["scope"] == "element":
            sets = [f"Z{_atomic_number(unit['scope_target'])}"]
        else:
            sets = ["all"]
        payloads.append(_payload(unit, unit["rows"], sets, constants))
        if unit["kind"] == "pooled" and unit["scope"] == "global":
            rows = np.asarray(unit["rows"], dtype=np.int64)
            for d in range(1, DECILES + 1):
                chosen = rows[deciles[rows] == d].tolist()
                payloads.append(_payload({**unit, "decile": d}, chosen, ["all"], constants))
            if unit["population"] == "primary":
                chosen = [i for i in unit["rows"] if f_block_free(metadata[i]["chemical_system_id"])]
                payloads.append(_payload({**unit, "population": F_BLOCK_SENSITIVITY}, chosen,
                                         list(COMPONENT_SETS), constants))
    return payloads


def _atomic_number(symbol: str) -> int:
    from ase.data import atomic_numbers

    return int(atomic_numbers[symbol])


def _payload(unit: Mapping[str, Any], rows: Sequence[int], sets: Sequence[str],
             constants: Mapping[str, Any]) -> dict[str, Any]:
    identifier = analysis_id(unit)
    return {
        "unit": "estimate",
        "analysis_id": identifier,
        "kind": unit["kind"],
        "population": unit["population"],
        "scope": unit["scope"],
        "scope_target": unit.get("scope_target"),
        "target_hop_id": unit.get("target_hop_id"),
        "material_id": unit.get("material_id"),
        "estimator_kind": unit["estimator_kind"],
        "decile": unit.get("decile"),
        "component_sets": list(sets),
        "rows": [int(i) for i in rows],
        "selection_sha256": unit["selection_sha256"],
        "other_hop_path_frames": unit.get("other_hop_path_frames"),
        "bootstrap_seed": analysis_seed(
            constants["base_seed"], "cluster_bootstrap", "study1", identifier, "published"
        ),
        "bootstrap_resamples": constants["bootstrap_resamples"],
        "confidence_level": constants["confidence_level"],
        "support": constants["support"],
    }


def label_budget_hop_set(
    units: Sequence[Mapping[str, Any]], informative_frames: Any, population: str,
    minimum: int,
) -> dict[str, Any]:
    """Frozen hop set (``hop_set_rule``): fixed before any draw, identical for every model.

    Population per hop (``population_rule``): its leakage-eligible material-scope frames
    with at least one |F_DFT| >= 0.1 eV/A component (DFT reference only).
    """

    hops = []
    for unit in units:
        if (unit["kind"], unit["estimator_kind"], unit["population"]) != (
            "target_hop", "primary", population
        ):
            continue
        rows = [i for i in unit["rows"] if informative_frames[i]]
        if len(rows) >= minimum:
            hops.append({"hop_id": unit["target_hop_id"], "material_id": unit["material_id"],
                         "population_rows": rows, "selection_sha256": unit["selection_sha256"]})
    hops.sort(key=lambda hop: hop["hop_id"])
    ids = [hop["hop_id"] for hop in hops]
    return {
        "population": population,
        "minimum_population_frames": minimum,
        "hop_ids": ids,
        "hop_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "hops": hops,
    }


def label_budget_payloads(hop_set: Mapping[str, Any], constants: Mapping[str, Any]) -> list[dict]:
    return [
        {
            "unit": "label_budget",
            "population": hop_set["population"],
            "hop_id": hop["hop_id"],
            "material_id": hop["material_id"],
            "rows": hop["population_rows"],
            "selection_sha256": hop["selection_sha256"],
            "k_values": constants["label_budget_k"],
            "repeats": constants["label_budget_repeats"],
            "base_seed": constants["base_seed"],
        }
        for hop in hop_set["hops"]
    ]


# --------------------------------------------------------------------------- worker


def analysis_worker_setup(inputs_dir: str) -> dict[str, Any]:
    """Memory-mapped consolidated arrays shared through the page cache."""

    import numpy as np

    names = ("set_stats", "set_counts", "success", "material_code", "element_row",
             "element_Z", "element_counts", "element_stats")
    arrays = {name: np.load(Path(inputs_dir) / f"{name}.npy", mmap_mode="r") for name in names}
    arrays["models"] = json.loads((Path(inputs_dir) / "models.json").read_text())
    return arrays


def run_unit(payload: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    if payload["unit"] == "estimate":
        return {"rows": estimate_unit(payload, context)}
    return label_budget_unit(payload, context)


def _set_columns(payload: Mapping[str, Any], context: Mapping[str, Any], set_name: str):
    """(stats[m, frame, (sxy, sxx, syy)], counts[frame, (n_atoms, n_informative)])."""

    import numpy as np

    rows = np.asarray(payload["rows"], dtype=np.int64)
    if set_name in SET_INDEX:
        s = SET_INDEX[set_name]
        return np.asarray(context["set_stats"][:, rows, s, :]), np.asarray(
            context["set_counts"][rows, s, :]
        )
    z = int(set_name[1:])
    element_z = context["element_Z"]
    start, stop = np.searchsorted(element_z, z, "left"), np.searchsorted(element_z, z, "right")
    block_rows = np.asarray(context["element_row"][start:stop])
    where = np.searchsorted(block_rows, rows)
    found = (where < len(block_rows)) & (block_rows[np.minimum(where, len(block_rows) - 1)] == rows)
    stats = np.zeros((len(context["models"]), len(rows), 3))
    counts = np.zeros((len(rows), 2), dtype=np.int64)
    positions = start + where[found]
    stats[:, found, :] = np.asarray(context["element_stats"][:, positions, :])
    counts[found] = np.asarray(context["element_counts"][positions])
    return stats, counts


def _support(counts, all_counts, materials, *, pooled: bool, parameters: Mapping[str, Any]):
    """(present, n_frames, n_materials, n_informative, n_support, unmet reasons)."""

    present = counts[:, 0] > 0
    n_frames = int(present.sum())
    n_materials = len(set(materials[present].tolist()))
    n_informative = int(counts[:, 1].sum())
    # Frozen rule (contracts.validate_softening_support): informative components
    # are counted over all atoms of the frames the estimate uses.
    n_support = int(all_counts[present, 1].sum())
    reasons = support_status(n_frames=n_frames, n_informative=n_support,
                             n_materials=n_materials, pooled=pooled, parameters=parameters)
    return present, n_frames, n_materials, n_informative, n_support, reasons


def common_hop_population(
    payloads: Sequence[Mapping[str, Any]], set_counts: Any, material_code: Any,
    f_block_free_hops: set[str],
) -> dict[str, list[str]]:
    """Common paired hop set per population, fixed before any estimate.

    A primary target-hop "all" estimate is ``estimated`` for every model exactly when
    it meets the frozen support rule: support uses DFT counts only, and analysis
    refuses to run while any model frame failed.  So the hops estimated for every
    model are the supported hops; ``tables.model_summary`` checks this afterwards.
    """

    import numpy as np

    common: dict[str, list[str]] = {population: [] for population in POPULATIONS}
    for payload in payloads:
        if (payload.get("unit"), payload["kind"], payload["estimator_kind"], payload["decile"]) != (
            "estimate", "target_hop", "primary", None
        ) or payload["population"] not in common:
            continue
        rows = np.asarray(payload["rows"], dtype=np.int64)
        counts = np.asarray(set_counts[rows, SET_INDEX["all"], :])
        *_, reasons = _support(counts, counts, np.asarray(material_code[rows]),
                               pooled=payload["scope"] != "material",
                               parameters=payload["support"])
        if not reasons:
            common[payload["population"]].append(payload["target_hop_id"])
    common[F_BLOCK_SENSITIVITY] = [h for h in common["primary"] if h in f_block_free_hops]
    return {population: sorted(set(hops)) for population, hops in sorted(common.items())}


def validate_common_hop_population(artifact: Mapping[str, Any]) -> dict[str, list[str]]:
    """Hop lists of a persisted common-hop population, after internal consistency checks."""

    from e3sse.execution.sharded_execution import canonical_sha256

    if artifact.get("schema_version") != COMMON_HOP_SCHEMA:
        raise Study1AnalysisError("common-hop population has an unexpected schema")
    expected = (*POPULATIONS, F_BLOCK_SENSITIVITY)
    populations = artifact.get("populations", {})
    if sorted(populations) != sorted(expected):
        raise Study1AnalysisError("common-hop population lists the wrong populations")
    hops = {}
    for population in expected:
        record = populations[population]
        ids = record["hop_ids"]
        if (ids != sorted(set(ids)) or record["n_hops"] != len(ids)
                or record["hop_ids_sha256"] != canonical_sha256(ids)):
            raise Study1AnalysisError(f"corrupt common-hop population for {population}")
        hops[population] = list(ids)
    if not set(hops[F_BLOCK_SENSITIVITY]) <= set(hops["primary"]):
        raise Study1AnalysisError("f-block-free common hops are not a subset of primary")
    return hops


def estimate_unit(payload: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    rows = np.asarray(payload["rows"], dtype=np.int64)
    models = context["models"]
    success = np.asarray(context["success"][:, rows])
    materials = np.asarray(context["material_code"][rows])
    support = payload["support"]
    pooled = payload["scope"] != "material"
    results, bootstrap_targets = [], []
    _, all_counts = _set_columns(payload, context, "all")
    for set_name in payload["component_sets"]:
        stats, counts = _set_columns(payload, context, set_name)
        present, n_frames, n_materials, n_informative, n_support, reasons = _support(
            counts, all_counts, materials, pooled=pooled, parameters=support
        )
        for m, model in enumerate(models):
            failed = int((~success[m] & present).sum())
            sxy, sxx, syy = (math.fsum(stats[m, :, c].tolist()) for c in range(3))
            row = {
                "model": model, "component_set": set_name, "n_frames": n_frames,
                "n_materials": n_materials, "n_components": int(3 * counts[:, 0].sum()),
                "n_informative_components": n_informative,
                "n_support_informative_components": n_support, "n_failed_frames": failed,
                "numerator_sxy": None, "denominator_sxx": sxx, "syy": None, "s_hat": None,
                "residual_fraction": None, "ci_low": None, "ci_high": None,
                "unsupported_reasons": reasons,
            }
            if failed:
                row["status"] = "incomplete_inference"
            elif reasons:
                row["status"] = "unsupported"
            else:
                row.update({
                    "status": "estimated", "numerator_sxy": sxy, "syy": syy,
                    "s_hat": eq17(sxy, sxx), "residual_fraction": residual_fraction(sxy, sxx, syy),
                })
                bootstrap_targets.append((row, stats[m, :, 0], stats[m, :, 1]))
            results.append(row)
    if bootstrap_targets:
        _bootstrap(payload, bootstrap_targets)
    common = {key: payload[key] for key in (
        "analysis_id", "kind", "population", "scope", "scope_target", "target_hop_id",
        "material_id", "estimator_kind", "decile", "selection_sha256", "bootstrap_seed",
        "bootstrap_resamples", "confidence_level",
    )}
    common["interval_type"] = (
        "whole_frame_cluster_percentile_bootstrap; conditional on the observed paths and "
        "materials (frozen study_1_contract.bootstrap_unit)"
    )
    if payload["estimator_kind"] == "oracle":
        common["analysis_label"] = "oracle_upper_bound"
    elif payload["kind"] == "pooled":
        # Pooled atlas scales describe a model; they were audited against a sentinel hop
        # and must never be used to correct a hop (Study 2 derives leave-hop-out scales).
        common["analysis_label"] = ATLAS_DESCRIPTIVE_LABEL
    else:
        common["analysis_label"] = "primary"
    common["other_hop_path_frames"] = payload.get("other_hop_path_frames")
    common["eligible_frames"] = len(payload["rows"])
    return [{**common, **row} for row in results]


def _bootstrap(payload: Mapping[str, Any], targets: list) -> None:
    """Whole-frame cluster bootstrap; one index stream shared by every model and set."""

    import numpy as np

    n = len(payload["rows"])
    numerators = np.stack([t[1] for t in targets], axis=1)  # frames x targets
    denominators = np.stack([t[2] for t in targets], axis=1)
    values = []
    for block in iter_bootstrap_indices(payload["bootstrap_seed"], n, payload["bootstrap_resamples"]):
        for draw in block:
            # numpy pairwise reductions, not BLAS: identical on every host.
            counts = np.bincount(draw, minlength=n).astype(np.float64)[:, None]
            values.append(((counts * numerators).sum(axis=0), (counts * denominators).sum(axis=0)))
    num = np.array([v[0] for v in values])
    den = np.array([v[1] for v in values])
    for j, (row, _, _) in enumerate(targets):
        if (den[:, j] <= 0).any():
            row["bootstrap_status"] = "zero_denominator_resample"
            continue
        row["ci_low"], row["ci_high"] = percentile_interval(num[:, j] / den[:, j],
                                                            payload["confidence_level"])
        row["bootstrap_status"] = "complete"


def label_budget_unit(payload: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Repeated K-frame subsamples of one hop; draws exclude model (common random numbers)."""

    import numpy as np

    rows = np.asarray(payload["rows"], dtype=np.int64)
    n = len(rows)
    stats = np.asarray(context["set_stats"][:, rows, SET_INDEX["all"], :])
    success = np.asarray(context["success"][:, rows])
    models = context["models"]
    result = {"hop_id": payload["hop_id"], "material_id": payload["material_id"],
              "population": payload["population"], "population_frames": n,
              "selection_sha256": payload["selection_sha256"], "draws_sha256": {},
              "models": {}}
    draws = {}
    for k in payload["k_values"]:
        digest = hashlib.sha256()
        chosen = []
        for repeat in range(payload["repeats"]):
            seed = analysis_seed(payload["base_seed"], "study1_label_budget", "material",
                                 payload["material_id"], payload["hop_id"], k, repeat)
            index = draw_without_replacement(seed, n, k)
            digest.update(index.astype("<i8").tobytes())
            chosen.append(index)
        draws[k] = np.stack(chosen)
        result["draws_sha256"][str(k)] = digest.hexdigest()
    for m, model in enumerate(models):
        if not success[m].all():
            result["models"][model] = {"status": "incomplete_inference",
                                       "failed_population_frames": int((~success[m]).sum())}
            continue
        sxy, sxx = stats[m, :, 0], stats[m, :, 1]
        entry = {"status": "complete", "s_all": eq17(math.fsum(sxy.tolist()), math.fsum(sxx.tolist())),
                 "s_k": {}}
        for k, index in draws.items():
            denominator = sxx[index].sum(axis=1)
            if (denominator <= 0).any():
                raise Study1AnalysisError(f"{payload['hop_id']}: zero Eq. 17 denominator at K={k}")
            entry["s_k"][str(k)] = (sxy[index].sum(axis=1) / denominator).tolist()
        result["models"][model] = entry
    return result


# --------------------------------------------------------------------------- summaries


def label_budget_summary(
    results: Sequence[Mapping[str, Any]], models: Sequence[str], k_values: Sequence[int],
    plateau_sd: float, supported_max_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-hop SDs and the frozen aggregate: median over materials of material-mean SD.

    SD uses ddof=1; the finite-population correction divides by sqrt((N-K)/(N-1)).
    A model with any incomplete hop has no plateau (the hop set is fixed).
    """

    import numpy as np

    per_hop, aggregate = [], []
    for model in models:
        complete = all(r["models"][model]["status"] == "complete" for r in results)
        for k in k_values:
            by_material: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for r in results:
                entry = r["models"][model]
                if entry["status"] != "complete":
                    continue
                values = np.asarray(entry["s_k"][str(k)])
                n = r["population_frames"]
                sd = float(values.std(ddof=1))
                corrected = sd / math.sqrt((n - k) / (n - 1))
                bias = abs(float(np.median(values)) - entry["s_all"])
                by_material[r["material_id"]].append((corrected, bias))
                per_hop.append({"population": r["population"], "model": model, "hop_id": r["hop_id"],
                                "material_id": r["material_id"], "K": k, "population_frames": n,
                                "s_all": entry["s_all"], "median_s_k": float(np.median(values)),
                                "sd_s_k": sd, "sd_s_k_fpc": corrected,
                                "abs_median_s_k_minus_s_all": bias})
            material_sd = [float(np.mean([v[0] for v in items])) for items in by_material.values()]
            material_bias = [float(np.mean([v[1] for v in items])) for items in by_material.values()]
            aggregate.append({
                "population": results[0]["population"] if results else None,
                "model": model, "K": k, "complete": complete,
                "hops": sum(len(v) for v in by_material.values()),
                "materials": len(by_material),
                "median_material_mean_sd_fpc": float(np.median(material_sd)) if material_sd else None,
                "median_material_mean_abs_bias": float(np.median(material_bias))
                if material_bias else None,
            })
    for model in models:
        rows = [row for row in aggregate if row["model"] == model]
        plateau = None
        if rows and all(row["complete"] for row in rows):
            plateau = min((row["K"] for row in rows if row["median_material_mean_sd_fpc"]
                           is not None and row["median_material_mean_sd_fpc"] <= plateau_sd),
                          default=None)
        complete = bool(rows) and all(row["complete"] for row in rows)
        for row in rows:
            if not complete:
                # Medians over a partial hop set are not results on the frozen hop set.
                row["median_material_mean_sd_fpc"] = None
                row["median_material_mean_abs_bias"] = None
            row["plateau_K"] = plateau
            # H4 is registered for the primary population only; others are descriptive.
            row["h4_supported"] = (
                None if not complete or row["population"] != "primary"
                else plateau is not None and plateau <= supported_max_k
            )
    return per_hop, aggregate


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman correlation with average ranks for ties (descriptive only)."""

    import numpy as np

    def ranks(values):
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values))
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            result[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return result

    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])
