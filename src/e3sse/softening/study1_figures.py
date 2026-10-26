"""Predetermined Study-1 figures, drawn only from the saved derived tables.

Seaborn draws the statistical marks on Axes; Matplotlib's object-oriented API owns
layout, reference lines, panel letters, legends and export (no pyplot state).
Precomputed bootstrap intervals are drawn as error bars and never recomputed.
Models keep one order (group A, B1, B2; then canonical name), one colour and one
marker each, so panels stay distinguishable in greyscale.

``study1_softening_atlas`` (IEEE double column):
  A  hop-level leakage-free material s by model (primary population)
  B  pooled family s with 95% whole-frame bootstrap intervals
  C  pooled s by configurational-energy decile
  D  pooled s_Li and s_host per model
``study1_label_budget_stability`` (IEEE single column): estimator stability only,
the median over materials of the finite-population-corrected SD of s_K versus K,
with the frozen 0.03 plateau line.  It is not a barrier-error result (Study 2).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from e3sse.softening.study1_inputs import GROUP_ORDER

FIGURE_SCHEMA = "e3sse.study1.figure.v1"
IEEE_SINGLE_COLUMN_IN = 3.5
IEEE_DOUBLE_COLUMN_IN = 7.16
RASTER_DPI = 600
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
FAMILIES = ("oxide", "sulfide", "halide", "nitride", "other")
SHORT_NAMES = {
    "CHGNet-0.3.0": "CHGNet",
    "M3GNet-MP-2021.2.8-PES": "M3GNet-MP",
    "MACE-MP-0-small-2023-12-10": "MACE-MP-0 s",
    "SevenNet-0-11July2024": "SevenNet-0",
    "M3GNet-MatPES-PBE-v2025.1-PES": "M3GNet-MatPES",
    "SevenNet-omat": "SevenNet-omat",
}


def short_name(model: str) -> str:
    return SHORT_NAMES.get(model, model)


def _empty_note(ax: Any, rows: Sequence[Any]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "no supported estimate", transform=ax.transAxes, ha="center",
                va="center", color="0.4")


def _float(value: str) -> float | None:
    return None if value in ("", None) else float(value)


def model_order(summary: Sequence[Mapping[str, str]]) -> list[str]:
    groups = {row["model"]: row["model_group"] for row in summary}
    return sorted(groups, key=lambda name: (GROUP_ORDER[groups[name]], name))


def _style(models: Sequence[str]) -> dict[str, dict[str, Any]]:
    import seaborn as sns

    colours = sns.color_palette("colorblind", n_colors=len(models))
    return {m: {"color": colours[i], "marker": MARKERS[i % len(MARKERS)]}
            for i, m in enumerate(models)}


def _theme() -> None:
    import seaborn as sns

    sns.set_theme(context="paper", style="ticks", font_scale=0.8,
                  rc={"pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6})


def _letter(ax: Any, letter: str) -> None:
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontweight="bold", va="bottom")


def _dodged(ax, x, rows, models, style, value="s_hat", low="ci_low", high="ci_high"):
    width = 0.7
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width / max(1, len(models))
        points = [(x[r["key"]] + offset, _float(r[value]), _float(r[low]), _float(r[high]))
                  for r in rows if r["model"] == model and _float(r[value]) is not None]
        if not points:
            continue
        xs, ys, lo, hi = zip(*points, strict=True)
        yerr = [[y - (l if l is not None else y) for y, l in zip(ys, lo, strict=True)],
                [(h if h is not None else y) - y for y, h in zip(ys, hi, strict=True)]]
        ax.errorbar(xs, ys, yerr=yerr, fmt=style[model]["marker"], color=style[model]["color"],
                    ms=3, lw=0.8, capsize=1.5, label=model)


def softening_atlas(estimates: Sequence[Mapping[str, str]],
                    summary: Sequence[Mapping[str, str]]) -> Any:
    import seaborn as sns
    from matplotlib.figure import Figure

    _theme()
    models = model_order(summary)
    style = _style(models)
    groups = {row["model"]: row["model_group"] for row in summary}
    figure = Figure(figsize=(IEEE_DOUBLE_COLUMN_IN, 5.6), layout="constrained")
    (ax_a, ax_b), (ax_c, ax_d) = figure.subplots(2, 2)

    hop = [r for r in estimates if r["kind"] == "target_hop" and r["estimator_kind"] == "primary"
           and r["population"] == "primary" and r["component_set"] == "all"
           and r["status"] == "estimated"]
    data = {"model": [r["model"] for r in hop], "s": [float(r["s_hat"]) for r in hop]}
    if hop:
        sns.boxplot(data=data, x="model", y="s", order=models, hue="model", hue_order=models,
                    palette=[style[m]["color"] for m in models], legend=False, ax=ax_a,
                    fliersize=1.0, linewidth=0.6, width=0.6)
    ax_a.axhline(1.0, color="0.4", lw=0.6, ls="--")
    ax_a.set_xticks(range(len(models)), [short_name(m) for m in models], rotation=30, ha="right")
    ax_a.set(xlabel="", ylabel="hop-level material $s$ (leakage-free)")
    _letter(ax_a, "A")

    family = [dict(r, key=r["scope_target"]) for r in estimates
              if r["kind"] == "pooled" and r["scope"] == "family" and r["population"] == "primary"
              and r["component_set"] == "all" and r["status"] == "estimated"]
    present = [f for f in FAMILIES if any(r["scope_target"] == f for r in family)]
    _dodged(ax_b, {f: i for i, f in enumerate(present)}, family, models, style)
    ax_b.set_xticks(range(len(present)), present)
    _empty_note(ax_b, family)
    ax_b.axhline(1.0, color="0.4", lw=0.6, ls="--")
    ax_b.set(xlabel="anion family", ylabel="pooled family $s$")
    _letter(ax_b, "B")

    decile = [dict(r, key=int(r["decile"])) for r in estimates
              if r["kind"] == "pooled" and r["decile"] and r["population"] == "primary"
              and r["status"] == "estimated"]
    _dodged(ax_c, {d: d for d in range(1, 11)}, decile, models, style)
    ax_c.set_xticks(range(1, 11))
    _empty_note(ax_c, decile)
    ax_c.axhline(1.0, color="0.4", lw=0.6, ls="--")
    ax_c.set(xlabel="configurational-energy decile (DFT)", ylabel="pooled $s$")
    _letter(ax_c, "C")

    primary = [r for r in summary if r["population"] == "primary"]
    for j, (prefix, label, face) in enumerate((("global_s_li", "$s_{Li}$", None),
                                               ("global_s_host", "$s_{host}$", "none"))):
        rows = [dict(r, key=r["model"], model=r["model"]) for r in primary
                if _float(r[prefix]) is not None]
        for r in rows:
            i = models.index(r["model"])
            y, lo, hi = _float(r[prefix]), _float(r[f"{prefix}_ci_low"]), _float(
                r[f"{prefix}_ci_high"])
            ax_d.errorbar([i + (j - 0.5) * 0.3], [y],
                          yerr=[[y - (lo if lo is not None else y)], [(hi if hi is not None else y) - y]],
                          fmt=style[r["model"]]["marker"], color=style[r["model"]]["color"],
                          mfc=face, ms=4, lw=0.8, capsize=1.5)
        ax_d.plot([], [], "o", mfc=face, color="0.3", ls="", label=label)
    ax_d.axhline(1.0, color="0.4", lw=0.6, ls="--")
    ax_d.set_xticks(range(len(models)), [short_name(m) for m in models], rotation=30, ha="right")
    ax_d.set(xlabel="", ylabel="pooled element-resolved $s$")
    ax_d.legend(frameon=False, loc="best")
    _letter(ax_d, "D")

    from matplotlib.lines import Line2D

    handles = [Line2D([], [], ls="", marker=style[m]["marker"], color=style[m]["color"],
                      label=f"{short_name(m)} ({groups[m]})") for m in models]
    figure.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False)
    return figure


def label_budget_stability(stability: Sequence[Mapping[str, str]],
                           summary: Sequence[Mapping[str, str]]) -> Any:
    import seaborn as sns
    from matplotlib.figure import Figure

    _theme()
    models = model_order(summary)
    style = _style(models)
    figure = Figure(figsize=(IEEE_SINGLE_COLUMN_IN, 2.8), layout="constrained")
    ax = figure.subplots()
    rows = [r for r in stability if r["population"] == "primary"
            and _float(r["median_material_mean_sd_fpc"]) is not None]
    data = {"model": [r["model"] for r in rows], "K": [int(r["K"]) for r in rows],
            "sd": [float(r["median_material_mean_sd_fpc"]) for r in rows]}
    if rows:
        sns.lineplot(data=data, x="K", y="sd", hue="model", hue_order=models, style="model",
                     style_order=models, markers=[style[m]["marker"] for m in models],
                     dashes=False, palette=[style[m]["color"] for m in models],
                     estimator=None, errorbar=None, ax=ax, lw=0.8, ms=4)
    ax.axhline(0.03, color="0.3", lw=0.7, ls="--")
    ax.text(1.02, 0.03, "0.03", va="center", fontsize=6, transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ks = sorted({int(r["K"]) for r in stability}) or [1, 3, 5, 10, 30]
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set(xlabel="label budget $K$ (frames)",
           ylabel="median SD($s_K$), finite-population corrected")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [short_name(label) for label in labels], frameon=False, fontsize=5,
              loc="upper right")
    return figure


def export(figure: Any, directory: Path, stem: str, inputs: Mapping[str, Path],
           provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Vector PDF (canonical) plus a 600-dpi review PNG and a JSON figure manifest."""

    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, options in ((".pdf", {}), (".png", {"dpi": RASTER_DPI})):
        path = directory / f"{stem}{suffix}"
        if path.exists():
            raise FileExistsError(f"{path} exists; figures are never overwritten")
        figure.savefig(path, **options)
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": FIGURE_SCHEMA,
        "figure": stem,
        "inputs_sha256": {name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for name, p in sorted(inputs.items())},
        "outputs_sha256": outputs,
        "raster_dpi": RASTER_DPI,
        **provenance,
    }
    path = directory / f"{stem}.figure.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
