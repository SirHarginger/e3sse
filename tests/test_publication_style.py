"""Central figure style: semantic identity, final-size readability, provenance manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from e3sse.figures.publication_style import (
    IEEE_DOUBLE_COLUMN_IN,
    METHOD_STYLES,
    MODEL_ORDER,
    export_figure,
    method_style,
    model_style,
    new_figure,
    panel_letter,
    parity_line,
)

REPOSITORY = Path(__file__).resolve().parents[1]
PROVENANCE = {"protocol_id": "p" * 64, "revision_id": "r" * 64, "git_sha": "a" * 40,
              "analysis_fingerprint": "f" * 16}


def test_methods_and_models_have_distinct_fixed_greyscale_safe_identities():
    styles = [(s["color"], s["marker"], s["linestyle"]) for s in METHOD_STYLES.values()]
    assert len(set(styles)) == len(styles)
    assert len({(model_style(m)["color"], model_style(m)["marker"]) for m in MODEL_ORDER}) == 6
    assert len({model_style(m)["marker"] for m in MODEL_ORDER}) == 6
    with pytest.raises(KeyError):
        method_style("unknown")


def test_export_writes_pdf_png_manifest_once_and_checks_text_size(tmp_path):
    table = tmp_path / "derived.csv"
    table.write_text("x,y\n0,0\n1,1\n")
    figure, axes = new_figure("double", 2.4, ncols=2)
    assert figure.get_size_inches()[0] == pytest.approx(IEEE_DOUBLE_COLUMN_IN)
    for letter, ax in zip("AB", axes[0], strict=True):
        parity_line(ax, 0.0, 1.0)
        ax.set_xlabel("DFT barrier (eV)")
        panel_letter(ax, letter)
    code = [REPOSITORY / "src/e3sse/figures/publication_style.py"]
    manifest = export_figure(figure, tmp_path / "out", "fig_test", inputs={"table": table},
                             plot_code=code, provenance=PROVENANCE)
    assert set(manifest["output_sha256"]) == {"fig_test.pdf", "fig_test.png"}
    assert manifest["raster_dpi"] == 600 and manifest["revision_id"] == "r" * 64
    assert json.loads((tmp_path / "out" / "fig_test.figure.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        export_figure(figure, tmp_path / "out", "fig_test", inputs={"table": table},
                      plot_code=code, provenance=PROVENANCE)
    with pytest.raises(ValueError, match="provenance"):
        export_figure(figure, tmp_path / "o2", "f", inputs={}, plot_code=code, provenance={})
    axes[0][0].text(0.5, 0.5, "tiny", fontsize=4)
    with pytest.raises(ValueError, match="pt at final size"):
        export_figure(figure, tmp_path / "o3", "f", inputs={}, plot_code=code,
                      provenance=PROVENANCE)


def test_registries_are_json_with_required_semantic_fields():
    figures = json.loads((REPOSITORY / "configs/figure_registry.json").read_text())
    required = {"figure_id", "scientific_question", "input_artifact", "panel", "plot_type",
                "x_quantity", "y_quantity", "grouping", "uncertainty", "reference_lines",
                "intended_width", "main_or_supplement"}
    assert all(required <= set(entry) for entry in figures["figures"])
    main = {e["figure_id"] for e in figures["figures"] if e["main_or_supplement"] == "main"}
    assert len(main) == 4
    inputs = [e["input_artifact"] for e in figures["figures"]]
    assert all(i.startswith(("derived/", "none")) for i in inputs)
    tables = json.loads((REPOSITORY / "configs/table_registry.json").read_text())
    assert {t["table_id"] for t in tables["tables"]} >= {"B_transfer", "C_applicability",
                                                         "E_selection", "G_prior_work"}
