"""Central publication style for every E3-SSE figure (IEEE two-column).

Seaborn draws statistical marks.  Matplotlib's object-oriented Figure/Axes API owns
layout, annotation, legends and export; no pyplot state is used.  Figures read only
saved ``derived/`` tables and never recompute science.

Semantic identity is constant across all figures.  A *method* (reference DFT, raw
ML, SRT, gated SRT, supervised, oracle, random) always has the same colour, marker
and line style.  *Models* have a separate palette with markers.  A panel encodes
either methods or models by colour, never both.  Colours are Okabe-Ito
(colourblind-safe), and markers or line styles keep every series distinguishable in
greyscale.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FIGURE_MANIFEST_SCHEMA = "e3sse.figure-manifest"
IEEE_SINGLE_COLUMN_IN = 3.5
IEEE_DOUBLE_COLUMN_IN = 7.16
RASTER_DPI = 600
FONT_SIZE_PT = 8
MIN_FONT_SIZE_PT = 6.5
PANEL_LETTERS = "ABCDEFGH"

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999",
}

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "reference_dft": {"label": "DFT reference", "color": OKABE_ITO["black"], "marker": None,
                      "linestyle": "-"},
    "raw": {"label": "Raw MLIP", "color": OKABE_ITO["vermillion"], "marker": "o",
            "linestyle": "--"},
    "srt": {"label": "SRT", "color": OKABE_ITO["blue"], "marker": "s", "linestyle": "-"},
    "srt_gated": {"label": "Reliability-gated SRT", "color": OKABE_ITO["green"], "marker": "D",
                  "linestyle": "-"},
    "supervised": {"label": "Supervised recalibration", "color": OKABE_ITO["grey"],
                   "marker": "^", "linestyle": ":"},
    "oracle": {"label": "Oracle (own path)", "color": OKABE_ITO["purple"], "marker": "x",
               "linestyle": "-."},
    "naive": {"label": "Naive threshold", "color": OKABE_ITO["orange"], "marker": "v",
              "linestyle": "--"},
    "random": {"label": "random abstention baseline", "color": OKABE_ITO["grey"], "marker": None,
               "linestyle": "--"},
}

MODEL_ORDER = (
    "CHGNet-0.3.0",
    "M3GNet-MP-2021.2.8-PES",
    "MACE-MP-0-small-2023-12-10",
    "SevenNet-0-11July2024",
    "M3GNet-MatPES-PBE-v2025.1-PES",
    "SevenNet-omat",
)
MODEL_SHORT_NAMES = {
    "CHGNet-0.3.0": "CHGNet",
    "M3GNet-MP-2021.2.8-PES": "M3GNet-MP",
    "MACE-MP-0-small-2023-12-10": "MACE-MP-0 s",
    "SevenNet-0-11July2024": "SevenNet-0",
    "M3GNet-MatPES-PBE-v2025.1-PES": "M3GNet-MatPES",
    "SevenNet-omat": "SevenNet-omat",
}
_MODEL_COLOURS = ("vermillion", "orange", "sky", "blue", "green", "purple")
_MODEL_MARKERS = ("o", "s", "^", "D", "v", "P")


def method_style(method: str) -> dict[str, Any]:
    try:
        return dict(METHOD_STYLES[method])
    except KeyError as exc:
        raise KeyError(f"no registered style for method {method!r}") from exc


def model_style(model: str) -> dict[str, Any]:
    """Fixed colour and marker by canonical model order (never by plotting order)."""

    index = MODEL_ORDER.index(model)
    return {"label": MODEL_SHORT_NAMES[model], "color": OKABE_ITO[_MODEL_COLOURS[index]],
            "marker": _MODEL_MARKERS[index]}


def apply_theme() -> None:
    import seaborn as sns

    sns.set_theme(context="paper", style="ticks", rc={
        "font.size": FONT_SIZE_PT, "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT, "xtick.labelsize": FONT_SIZE_PT - 1,
        "ytick.labelsize": FONT_SIZE_PT - 1, "legend.fontsize": FONT_SIZE_PT - 1,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def new_figure(width: str, height_in: float, *, nrows: int = 1, ncols: int = 1, **kwargs) -> Any:
    """Figure at final physical size: ``width`` is 'single' (3.5 in) or 'double' (7.16 in)."""

    from matplotlib.figure import Figure

    widths = {"single": IEEE_SINGLE_COLUMN_IN, "double": IEEE_DOUBLE_COLUMN_IN}
    if width not in widths:
        raise ValueError("width must be 'single' or 'double'")
    apply_theme()
    figure = Figure(figsize=(widths[width], height_in), layout="constrained")
    axes = figure.subplots(nrows, ncols, squeeze=False, **kwargs)
    return figure, axes


def panel_letter(ax: Any, letter: str) -> None:
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=FONT_SIZE_PT + 1, ha="right", va="bottom")


def parity_line(ax: Any, low: float, high: float) -> None:
    style = method_style("reference_dft")
    ax.plot([low, high], [low, high], color=style["color"], lw=0.6, ls="-", zorder=0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")


def smallest_font_pt(figure: Any) -> float:
    """Smallest text size (pt) on the figure at its final physical size."""

    sizes = [text.get_fontsize() for text in figure.findobj(
        match=lambda artist: hasattr(artist, "get_fontsize") and hasattr(artist, "get_text")
        and artist.get_text())]
    return min(sizes) if sizes else float("inf")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_figure(
    figure: Any,
    directory: Path,
    figure_id: str,
    *,
    inputs: Mapping[str, Path],
    plot_code: Sequence[Path],
    provenance: Mapping[str, Any],
    svg: bool = False,
) -> dict[str, Any]:
    """Vector PDF, 600-dpi PNG (optional SVG) and the figure provenance manifest.

    ``provenance`` must carry ``protocol_id``, ``revision_id``, ``git_sha`` and
    ``analysis_fingerprint``.  Existing outputs are never overwritten.
    """

    missing = [k for k in ("protocol_id", "revision_id", "git_sha", "analysis_fingerprint")
               if k not in provenance]
    if missing:
        raise ValueError(f"figure provenance lacks {missing}")
    if smallest_font_pt(figure) < MIN_FONT_SIZE_PT:
        raise ValueError(f"{figure_id}: text below {MIN_FONT_SIZE_PT} pt at final size")
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    formats = [(".pdf", {}), (".png", {"dpi": RASTER_DPI})] + ([(".svg", {})] if svg else [])
    for suffix, options in formats:
        path = directory / f"{figure_id}{suffix}"
        if path.exists():
            raise FileExistsError(f"{path} exists; figures are never overwritten")
        figure.savefig(path, **options)
        outputs[path.name] = _sha256(path)
    width, height = figure.get_size_inches()
    manifest = {
        "schema_version": FIGURE_MANIFEST_SCHEMA,
        "figure_id": figure_id,
        **{k: provenance[k] for k in sorted(provenance)},
        "input_files": {name: str(path) for name, path in sorted(inputs.items())},
        "input_sha256": {name: _sha256(path) for name, path in sorted(inputs.items())},
        "plot_code_sha256": {str(p): _sha256(p) for p in sorted(plot_code)},
        "output_sha256": outputs,
        "dimensions_in": [round(float(width), 4), round(float(height), 4)],
        "raster_dpi": RASTER_DPI,
    }
    path = directory / f"{figure_id}.figure.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
