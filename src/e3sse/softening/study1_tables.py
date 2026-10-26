"""Study-1 derived tables (full precision) and paper tables (rounded), from unit results.

Every figure and paper table is computed from these saved tables, never from raw data.
H3 (pretraining coverage) is descriptive: architecture and pretraining data are
confounded across groups, and the per-model registry limitations are carried along.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from e3sse.softening.study1_analysis import F_BLOCK_SENSITIVITY, spearman_rho
from e3sse.softening.study1_inputs import GROUP_ORDER

ESTIMATE_COLUMNS = (
    "model", "model_group", "population", "kind", "scope", "scope_target", "target_hop_id",
    "material_id", "estimator_kind", "analysis_label", "decile", "component_set", "status",
    "unsupported_reasons", "eligible_frames", "n_frames", "n_materials", "n_components",
    "n_informative_components", "n_support_informative_components", "n_failed_frames", "numerator_sxy", "denominator_sxx", "syy",
    "s_hat", "residual_fraction", "ci_low", "ci_high", "confidence_level",
    "bootstrap_resamples", "bootstrap_status", "bootstrap_seed", "interval_type",
    "other_hop_path_frames", "analysis_id",
    "selection_sha256",
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, float):
        return repr(value)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def estimate_rows(results: Mapping[str, Any], groups: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = [
        {**row, "model_group": groups[row["model"]]}
        for result in results.values() if "rows" in result
        for row in result["rows"]
    ]
    return sorted(rows, key=lambda r: (
        GROUP_ORDER[r["model_group"]], r["model"], r["population"], r["kind"], r["scope"],
        str(r["scope_target"]), str(r["target_hop_id"]), r["estimator_kind"], r["decile"] or 0,
        r["component_set"],
    ))


def _median(values: Sequence[float]) -> float | None:
    import numpy as np

    return float(np.median(values)) if values else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    import numpy as np

    return float(np.quantile(values, q, method="linear")) if values else None


def model_summary(
    rows: Sequence[Mapping[str, Any]], models: Sequence[str], groups: Mapping[str, str],
    f_block_free_hops: set[str], common_hops: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """s-bar (frozen ``s_bar_definition``) and pooled global scales per model/population.

    ``common_hops`` is the persisted common-hop population
    (``frozen/study1_common_hop_population.json``); it must equal the hops every
    model actually estimated, or the summary refuses to proceed.
    """

    summary = []
    hop_rows = defaultdict(list)
    pooled = {}
    for row in rows:
        if row["kind"] == "target_hop" and row["estimator_kind"] == "primary":
            hop_rows[(row["model"], row["population"], row["component_set"])].append(row)
        if row["kind"] == "pooled" and row["scope"] == "global" and row["decile"] is None:
            pooled[(row["model"], row["population"], row["component_set"])] = row
    # Paired cross-model comparisons use one common set of hops estimated for every
    # model (estimator_minimum_support); per-model s-bar keeps each model's own hops.
    observed: dict[str, set[str]] = {}
    for population in ("primary", "tm_hubbard_u_secondary"):
        per_model = [
            {r["target_hop_id"] for r in hop_rows.get((m, population, "all"), [])
             if r["status"] == "estimated"}
            for m in models
        ]
        observed[population] = set.intersection(*per_model) if per_model else set()
    observed[F_BLOCK_SENSITIVITY] = observed["primary"] & f_block_free_hops
    common = {population: set(common_hops[population]) for population in observed}
    if common != observed:
        raise ValueError("persisted common-hop population differs from the hops estimated "
                         "for every model; refusing to summarise")
    for model in models:
        for population in ("primary", F_BLOCK_SENSITIVITY, "tm_hubbard_u_secondary"):
            source = "primary" if population == F_BLOCK_SENSITIVITY else population
            hops = hop_rows.get((model, source, "all"), [])
            if population == F_BLOCK_SENSITIVITY:
                hops = [r for r in hops if r["target_hop_id"] in f_block_free_hops]
            estimated = [r["s_hat"] for r in hops if r["status"] == "estimated"]
            shared = [r["s_hat"] for r in hops if r["target_hop_id"] in common[population]]
            li = {r["target_hop_id"]: r for r in hop_rows.get((model, source, "Li"), [])}
            host = {r["target_hop_id"]: r for r in hop_rows.get((model, source, "host"), [])}
            gaps = [
                li[h]["s_hat"] - host[h]["s_hat"] for h in li
                if h in host and li[h]["status"] == host[h]["status"] == "estimated"
                and (population != F_BLOCK_SENSITIVITY or h in f_block_free_hops)
            ]
            entry = {
                "model": model, "model_group": groups[model], "population": population,
                "target_hops": len(hops), "hops_estimated": len(estimated),
                "hops_unsupported": sum(r["status"] == "unsupported" for r in hops),
                "hops_incomplete_inference": sum(r["status"] == "incomplete_inference" for r in hops),
                "s_bar": _median(estimated), "common_hops": len(shared),
                "s_bar_common_hops": _median(shared),
                "s_hop_q25": _quantile(estimated, 0.25),
                "s_hop_q75": _quantile(estimated, 0.75),
                "hops_with_s_li_and_s_host": len(gaps), "median_hop_s_li_minus_s_host": _median(gaps),
            }
            for component in ("all", "Li", "host"):
                g = pooled.get((model, population, component), {})
                prefix = "global_s" if component == "all" else f"global_s_{component.lower()}"
                entry[prefix] = g.get("s_hat")
                entry[f"{prefix}_ci_low"] = g.get("ci_low")
                entry[f"{prefix}_ci_high"] = g.get("ci_high")
                entry[f"{prefix}_status"] = g.get("status")
            if population == "primary":
                g = pooled.get((model, population, "all"), {})
                entry["global_frames"] = g.get("n_frames")
                entry["global_materials"] = g.get("n_materials")
            summary.append(entry)
    return summary


def pretraining_coverage(
    summary: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Descriptive H3 evidence: s-bar ordering by pretraining group (no inference)."""

    primary = [row for row in summary if row["population"] == "primary"]
    key = "s_bar_common_hops"
    by_group = defaultdict(list)
    for row in primary:
        if row[key] is not None:
            by_group[row["model_group"]].append(row[key])
    medians = {group: _median(values) for group, values in sorted(by_group.items())}
    defined = [row for row in primary if row[key] is not None]
    pairs = (("M3GNet-MP-2021.2.8-PES", "M3GNet-MatPES-PBE-v2025.1-PES"),
             ("SevenNet-0-11July2024", "SevenNet-omat"))
    s_bar = {row["model"]: row[key] for row in primary}
    ordered = None
    if all(medians.get(group) is not None for group in GROUP_ORDER):
        ordered = medians["A"] < medians["B1"] < medians["B2"]
    return {
        "classification": "descriptive; RQ3/H3 ordering of s-bar by pretraining group",
        "hypothesis": "s-bar ordered MPtrj (A) < MatPES-type (B1) < OMat24-type (B2)",
        "comparison_hop_set": "primary target hops estimated for every model (paired)",
        "common_hops": primary[0]["common_hops"] if primary else 0,
        "excluded_hops_by_model": {row["model"]: row["target_hops"] - row["common_hops"]
                                   for row in primary},
        "per_model_s_bar_on_own_hops": {row["model"]: row["s_bar"] for row in primary},
        "group_ordinal": dict(GROUP_ORDER),
        "s_bar_by_model": s_bar,
        "group_median_s_bar": medians,
        "group_medians_strictly_ordered_A_lt_B1_lt_B2": ordered,
        "spearman_rho_group_ordinal_vs_s_bar": spearman_rho(
            [GROUP_ORDER[row["model_group"]] for row in defined], [row[key] for row in defined]
        ),
        "spearman_note": "average ranks for ties; six models with B1 and B2 one model each: "
                         "descriptive and low power, no p-value is computed",
        "architecture_matched_contrasts": [
            {"group_a_model": a, "other_model": b,
             "s_bar_difference_other_minus_a": None if s_bar.get(a) is None or s_bar.get(b) is None
             else s_bar[b] - s_bar[a],
             "registered_limitations": list(registry[b].get("limitations", []))}
            for a, b in pairs if a in registry and b in registry
        ],
        "decision_note": "the H3 decision also requires Study 2 SRT gains in reverse order; "
                         "Study 1 supplies only the s-bar ordering",
        "limitation": "architecture and pretraining data are confounded across groups; "
                      "B1/B2 differ from their Group-A partners in capacity",
    }


def inclusion_accounting(
    metadata: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]],
    eligibility_reasons: Mapping[str, Mapping[str, int]], rows: Sequence[Mapping[str, Any]],
    hop_sets: Mapping[str, Mapping[str, Any]], failures: Mapping[str, int],
) -> list[dict[str, Any]]:
    out = []

    def add(category: str, population: str | None, count: int, model: str | None = None) -> None:
        out.append({"category": category, "population": population, "model": model, "count": count})

    for population, count in sorted(Counter(r["population"] for r in metadata).items()):
        add("published_train_frames", population, count)
        add("published_train_materials", population,
            len({r["material_id"] for r in metadata if r["population"] == population}))
    hops = [u for u in units if u["kind"] == "target_hop" and u["estimator_kind"] == "primary"]
    for population, count in sorted(Counter(u["population"] for u in hops).items()):
        add("target_hops", population, count)
        add("target_hops_without_eligible_frames", population,
            sum(1 for u in hops if u["population"] == population and not u["rows"]))
    for label, reasons in eligibility_reasons.items():
        for reason, count in reasons.items():
            add(f"eligibility:{label}:{reason}", None, count)
    for population, hop_set in sorted(hop_sets.items()):
        add("label_budget_hops", population, len(hop_set["hop_ids"]))
    for model, count in sorted(failures.items()):
        add("inference_failed_frames", None, count, model)
    status = Counter((r["model"], r["population"], r["kind"], r["scope"], r["component_set"],
                      r["status"]) for r in rows if r["estimator_kind"] == "primary")
    for (model, population, kind, scope, component, state), count in sorted(status.items()):
        add(f"estimates:{kind}:{scope}:{component}:{state}", population, count, model)
    return out


def rounded(value: Any, digits: int) -> Any:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def interval(low: Any, high: Any, digits: int = 3) -> str:
    if low in (None, "") or high in (None, ""):
        return ""
    return f"[{float(low):.{digits}f}, {float(high):.{digits}f}]"
