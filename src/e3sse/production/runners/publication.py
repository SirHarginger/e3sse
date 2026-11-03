"""publication: registered derived tables and figures, built only from production artifacts.

Rules (configs/figure_registry.json, configs/table_registry.json): figures read only
``derived/`` tables written here from the stage artifacts; plotting recomputes no
science; every figure is a vector PDF + 600-dpi PNG + provenance manifest
(``figures.publication_style.export_figure``: IEEE sizes, Seaborn on Matplotlib
Figure/Axes objects); projections are visualization only.  Registered entries whose
inputs the frozen pipeline does not produce are listed with their reason, never
drawn from substitute data.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from e3sse.production.runners.context import RunnerContext, RunnerValidationError, canonical_sha256

NOT_BUILT = {
    "fig1_e3sse_method": "method schematic: TikZ source docs/figures/e3sse_method.tex, not a data figure",
    "S1_force_parity": "needs per-component force bins; the frozen Study-1 inference stores only "
                       "Eq. 17 sufficient statistics",
    "S2_force_residuals": "needs per-component residual bins (same reason as S1)",
    "S15_representation_support_umap": "optional: umap-learn is not a pinned dependency",
    "P3D_representation": "optional presentation view (interactive HTML), not a paper figure",
    "S6_paired_error_distributions": "needs a derived per-hop paired-error table; not produced "
                                     "yet (declared gap, never substituted by S5/S7)",
    "S10_latent_maps": "registered latent maps of development, calibration and prospective "
                       "units; only development path vectors reach the derived tables",
    "S11_latent_support": "registered support-distance view; support distances of calibration "
                          "and pool units are not in the derived tables (declared gap)",
    "S13_candidate_space": "registered candidate-space view; not produced (declared gap, never "
                           "substituted by fig4 D or S17)",
    "S16_support_vs_errors": "registered as support distance vs three error panels with tau_d; "
                             "per-unit support distances with errors are not in the derived "
                             "tables (declared gap; the rho_F plot is not a substitute)",
}
PARTIAL = {
    "S14_representation_support_pca": "development panel only; calibration and prospective "
                                      "panels need their path vectors in the derived tables",
    "fig3_transfer_reliability": "panel C shows deployable methods only (B1-B4); the B5 "
                                 "recalibration comparator and the B6 oracle are reported in "
                                 "tables, never beside deployable methods",
}


def _cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return repr(value) if isinstance(value, float) else value


def render_table(rows: list[dict[str, Any]]) -> str:
    import io

    columns = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _cell(row.get(k)) for k in columns})
    return buffer.getvalue()


def write_table(path: Path, rows: list[dict[str, Any]], *, root: Path) -> Path:
    """Create-once CSV (temp file, fsync, hard link).  On resume an existing table must
    be byte-identical to a fresh rendering of the same rows, never silently reused."""

    from e3sse.execution.sharded_execution import publish_text_once

    text = render_table(rows)
    if path.exists():
        if path.read_bytes() != text.encode("utf-8"):
            raise RunnerValidationError(f"{path} differs from its re-rendered rows")
        return path
    publish_text_once(path, text, root=root)
    return path


def derived_tables(ctx: RunnerContext) -> dict[str, list[dict[str, Any]]]:
    """Flatten the stage artifacts into registered derived tables (full precision)."""

    transfer = ctx.upstream("transfer", "transfer_results")
    reliability = ctx.upstream("reliability", "reliability_results")
    tables: dict[str, list[dict[str, Any]]] = {
        "transfer_predictions": list(ctx.upstream("transfer", "transfer_predictions")),
        "transfer_paired_effects": [], "transfer_classification": [],
        "risk_coverage_summary": [], "risk_coverage_curves": [], "label_budget_downstream": [],
        "applicability_records": [], "prospective_funnel": [], "prospective_candidates": [],
        "density_ratio_diagnostics": [], "relaxed_vs_fixed_geometry": [],
        "conformal_draw_records": [], "representation_projection": [], "endpoints": []}
    for model, result in transfer["results"].items():
        for method, entry in result["methods"].items():
            if "method" not in entry:
                continue
            bootstrap = entry.get("bootstrap", {})
            tables["transfer_paired_effects"].append({
                "model": model, "method": method, "role": entry["role"], "n": entry["method"]["n"],
                "mae_raw_eV": entry["raw"]["mae_eV"], "mae_method_eV": entry["method"]["mae_eV"],
                "delta_mae_eV": entry["raw"]["mae_eV"] - entry["method"]["mae_eV"],
                "rho_srt": entry["rho_srt"],
                "ci_low": bootstrap.get("mae_reduction_ci95", {}).get("low"),
                "ci_high": bootstrap.get("mae_reduction_ci95", {}).get("high"),
                "population_sha256": entry["population"]["population_sha256"]})
            for c in entry["method"]["classification"]:
                tables["transfer_classification"].append({"model": model, "method": method, **c})
        for row in result["label_budget_downstream"]["rows"]:
            tables["label_budget_downstream"].append({"model": model, **{k: v for k, v in row.items()
                                                                        if k != "per_hop"}})
    for model, endpoint in transfer["P1"].items():
        tables["endpoints"].append({"endpoint": "P1", "model": model, **endpoint})
    for model, result in reliability["models"].items():
        for score, entry in result["scores"].items():
            summary = entry.get("selective_primary", {})
            tables["risk_coverage_summary"].append({
                "model": model, "score": score, **{k: summary.get(k) for k in
                                                   ("aurc", "e_aurc", "normalized_gain",
                                                    "risk_at_target_coverage", "n_undefined_scores")},
                "spearman_with_benefit": entry.get("spearman_with_benefit"),
                "spearman_ci_low": entry.get("spearman_ci95", {}).get("low"),
                "spearman_ci_high": entry.get("spearman_ci95", {}).get("high")})
            curve = entry.get("selective_primary_curve")
            if curve:
                for coverage, risk in zip(curve["coverage"], curve["risk"]):
                    tables["risk_coverage_curves"].append({"model": model, "score": score,
                                                           "coverage": coverage, "risk_eV": risk})
        if "P4" in result:
            tables["endpoints"].append({"endpoint": "P4", "model": model,
                                        "estimate": result["P4"]["delta_aurc"],
                                        "p_one_sided": result["P4"]["p_one_sided"],
                                        "p_holm": reliability["P4_H6"]["holm_adjusted_p"].get(model)})
    for record in ctx.upstream("transfer", "transfer_hop_records"):
        tables["applicability_records"].append({
            "model": record["model"], "hop_id": record["hop_id"],
            "raw_error_eV": abs(record["raw_barrier_eV"] - record["dft_barrier_eV"]),
            "srt_error_eV": None if record["srt_material_eV"] is None else
            abs(record["srt_material_eV"] - record["dft_barrier_eV"]), **record["scores"]})
    selection = ctx.upstream("discovery_selection", "discovery_selection")
    funnel = selection["funnel"]
    for gate in ("gate1", "ml_neb_attempted", "ml_neb_converged", "applicability_supported",
                 "conformal_predictor_eligible", "weight_fit_holdout", "no_density_weight",
                 "wcs_run", "selected"):
        tables["prospective_funnel"].append({"gate": gate, "count": funnel["counts"][gate],
                                             "denominator": "gate1"})
    for category, count in funnel["counts"]["failures_by_category"].items():
        tables["prospective_funnel"].append({"gate": f"failed:{category}", "count": count,
                                             "denominator": "ml_neb_attempted"})
    tables["prospective_candidates"] = list(selection.get("candidates", []))
    density = ctx.upstream("density_ratio", "density_ratio")
    for key, value in density.get("diagnostics", {}).items():
        tables["density_ratio_diagnostics"].append({"diagnostic": key, "value": value
                                                    if not isinstance(value, (dict, list))
                                                    else canonical_sha256(value)})
    deployment = ctx.upstream("deployment_calibration", "deployed_units")
    tables["relaxed_vs_fixed_geometry"] = list(deployment.get("relaxed_path_comparison", {})
                                               .get("rows", []))
    draws = ctx.upstream("controlled_selection", "conformal_draw_records")
    tables["conformal_draw_records"] = [{"model": m, **r} for m, rows in draws.items() for r in rows]
    tables["representation_projection"] = representation_projection(ctx)
    return tables


def representation_projection(ctx: RunnerContext) -> list[dict[str, Any]]:
    """2-D PCA of the development path vectors (fitted on them; visualization only)."""

    import numpy as np

    rows = []
    for model, branch in ctx.upstream("path_representation", "path_representation")["models"].items():
        reference = branch.get("reference")
        if not reference:
            continue
        z = np.asarray(reference["reference_z"], dtype=float)
        centred = z - z.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        scores = centred @ vt[:2].T
        for key, (pc1, pc2) in zip(reference["unit_keys"], scores):
            rows.append({"model": model, "unit_key": key, "space": reference["source"],
                         "pc1": float(pc1), "pc2": float(pc2), "split": "development"})
    return rows


# --------------------------------------------------------------------- figures

def _frame(path: Path):
    """A derived table as a DataFrame; an empty table (no rows, no header) is empty."""

    import pandas as pd

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _primary(df):
    return df[df["population"] == "primary"] if "population" in df.columns else df


def verify_figure(manifest_path: Path, inputs: dict[str, Path], code: list[Path]) -> None:
    """An existing figure is reused only if its inputs, plotting code and outputs are
    exactly the current ones (never silently)."""

    import json

    from e3sse.production.runners.context import file_sha256

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_inputs = {name: file_sha256(path) for name, path in sorted(inputs.items())}
    expected_code = {str(p): file_sha256(p) for p in sorted(code)}
    outputs_ok = all(
        (manifest_path.parent / name).is_file() and file_sha256(manifest_path.parent / name) == digest
        for name, digest in manifest.get("output_sha256", {}).items())
    if (manifest.get("input_sha256") != expected_inputs
            or manifest.get("plot_code_sha256") != expected_code or not outputs_ok):
        raise RunnerValidationError(f"{manifest_path.name}: inputs, code or outputs differ from "
                                    "the current derived tables")


def figure_builders():
    """figure_id -> (inputs, draw(frames) -> Figure); drawing only, no science."""

    import numpy as np
    import seaborn as sns

    from e3sse.figures.publication_style import new_figure, panel_letter, parity_line

    def parity(ax, df, method, label):
        sub = _primary(df)  # metrics are primary-population only; so are the parity plots
        sub = sub[sub["method"] == method] if not sub.empty else sub
        if sub.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            return
        low, high = 0.0, float(np.nanmax([sub["dft_eV"].max(), sub["predicted_eV"].max()])) * 1.05
        sns.scatterplot(data=sub, x="dft_eV", y="predicted_eV", hue="model", s=6, ax=ax,
                        legend=False, linewidth=0)
        parity_line(ax, low, high)
        ax.set_xlabel("DFT barrier (eV)")
        ax.set_ylabel(label)

    def fig3(frames):
        figure, axes = new_figure("double", 5.0, nrows=2, ncols=2)
        df = frames["transfer_predictions"]
        parity(axes[0, 0], df, "B0_raw", "raw ML barrier (eV)")
        parity(axes[0, 1], df, "B3_srt_material", "SRT barrier (eV)")
        effects = frames["transfer_paired_effects"]
        deployable = effects[effects["role"] == "deployable"] if not effects.empty else effects
        sns.pointplot(data=deployable, x="model", y="delta_mae_eV", hue="method", ax=axes[1, 0],
                      linestyle="none", dodge=0.4, markersize=3)
        axes[1, 0].axhline(0, lw=0.5, color="0.5")
        axes[1, 0].set_ylabel("MAE reduction raw to method (eV)")
        axes[1, 0].tick_params(axis="x", rotation=45)
        curves = frames["risk_coverage_curves"]
        sns.lineplot(data=curves, x="coverage", y="risk_eV", hue="score", style="model",
                     ax=axes[1, 1], legend=False)  # every registered score
        axes[1, 1].set_ylabel("selective MAE (eV)")
        for ax, letter in zip(axes.flat, "ABCD"):
            panel_letter(ax, letter)
        return figure

    def fig4(frames):
        figure, axes = new_figure("double", 5.0, nrows=2, ncols=2)
        draws = frames["conformal_draw_records"]
        bh = draws[draws["method"] == "BH"]
        grouped = bh.groupby(["model", "predictor"]).agg(fdr=("fdp", "mean"),
                                                         power=("true_discoveries", "mean")).reset_index()
        sns.scatterplot(data=grouped, x="fdr", y="power", hue="predictor", ax=axes[0, 0], s=12)
        axes[0, 0].set_xlabel("realized FDR")
        axes[0, 0].set_ylabel("mean true discoveries")
        sns.boxplot(data=bh, x="predictor", y="fdp", ax=axes[0, 1], fliersize=1, linewidth=0.5)
        axes[0, 1].tick_params(axis="x", rotation=45)
        funnel = frames["prospective_funnel"]
        sns.barplot(data=funnel, y="gate", x="count", ax=axes[1, 0], color="0.6")
        candidates = frames["prospective_candidates"]
        if not candidates.empty:
            sns.scatterplot(data=candidates, x="deployed_prediction_eV", y="conformal_p",
                            hue="selected", ax=axes[1, 1], s=8)
        axes[1, 1].set_xlabel("deployed barrier (eV)")
        axes[1, 1].set_ylabel("weighted conformal p")
        for ax, letter in zip(axes.flat, "ABCD"):
            panel_letter(ax, letter)
        return figure

    def simple(kind):
        def draw(frames):
            figure, axes = new_figure("single", 2.8)
            ax = axes[0, 0]
            if kind == "S5":
                parity(ax, frames["transfer_predictions"], "B3_srt_material", "SRT barrier (eV)")
            elif kind == "S7":
                df = frames["transfer_predictions"]
                df = df[df["method"] == "B3_srt_material"]
                sns.scatterplot(x=(df["predicted_eV"] + df["dft_eV"]) / 2,
                                y=df["predicted_eV"] - df["dft_eV"], hue=df["model"], s=5, ax=ax,
                                legend=False)
                ax.axhline(0, lw=0.5, color="0.5")
                ax.set_xlabel("mean of SRT and DFT (eV)")
                ax.set_ylabel("SRT - DFT (eV)")
            elif kind == "S8":
                df = frames["transfer_classification"]
                sns.barplot(data=df, x="threshold_eV", y="f1", hue="method", ax=ax)
            elif kind == "S9":
                sns.lineplot(data=frames["risk_coverage_curves"], x="coverage", y="risk_eV",
                             hue="score", ax=ax)
            elif kind == "S12":
                sns.boxplot(data=frames["conformal_draw_records"], x="alpha", y="fdp",
                            hue="method", ax=ax, fliersize=1, linewidth=0.5)
            elif kind == "S14":
                sns.scatterplot(data=frames["representation_projection"], x="pc1", y="pc2",
                                hue="model", s=5, ax=ax)
                ax.set_xlabel("PC1 (visualization only)")
                ax.set_ylabel("PC2")
            elif kind == "fig2D":
                df = frames["label_budget_downstream"]
                sns.lineplot(data=df, x="K", y="mean_abs_srt_error_eV", hue="model", marker="o",
                             ax=ax)
                ax.set_xscale("log")
                ax.set_xlabel("frames per material K")
                ax.set_ylabel("mean |SRT - DFT| (eV)")
            elif kind == "S17":
                sns.barplot(data=frames["prospective_funnel"], y="gate", x="count", ax=ax,
                            color="0.6")
            elif kind == "S18":
                df = frames["density_ratio_diagnostics"]
                numeric = df[df["value"].apply(lambda v: isinstance(v, (int, float)))]
                sns.barplot(data=numeric, y="diagnostic", x="value", ax=ax, color="0.6")
            elif kind == "S19":
                sns.pointplot(data=frames["risk_coverage_summary"], x="score", y="aurc",
                              hue="model", ax=ax, linestyle="none", dodge=0.4, markersize=3)
                ax.tick_params(axis="x", rotation=45)
            elif kind == "S20":
                df = frames["relaxed_vs_fixed_geometry"]
                if not df.empty:
                    sns.histplot(df["raw_difference_eV"], ax=ax)
                ax.set_xlabel("ML-NEB minus fixed-geometry barrier (eV)")
            return figure
        return draw

    return {
        "fig3_transfer_reliability": (("transfer_predictions", "transfer_paired_effects",
                                       "risk_coverage_curves"), fig3),
        "fig4_selection_discovery": (("conformal_draw_records", "prospective_funnel",
                                      "prospective_candidates"), fig4),
        "S5_barrier_parity_per_model": (("transfer_predictions",), simple("S5")),
        "S7_signed_residuals": (("transfer_predictions",), simple("S7")),
        "S8_classification": (("transfer_classification",), simple("S8")),
        "S9_risk_coverage_all": (("risk_coverage_curves",), simple("S9")),
        "S12_fdr_diagnostics": (("conformal_draw_records",), simple("S12")),
        "S14_representation_support_pca": (("representation_projection",), simple("S14")),
        "fig2_panel_D_label_budget_downstream": (("label_budget_downstream",), simple("fig2D")),
        "S17_pipeline_attrition": (("prospective_funnel",), simple("S17")),
        "S18_density_ratio_diagnostics": (("density_ratio_diagnostics",), simple("S18")),
        "S19_aurc_comparison": (("risk_coverage_summary",), simple("S19")),
        "S20_relaxed_vs_fixed_geometry": (("relaxed_vs_fixed_geometry",), simple("S20")),
    }


ALSO_FROM_STUDY1 = {
    "fig2_calibration": "panels A-C built by the estimate stage (study1_softening_atlas + "
                        "study1_label_budget_stability); panel D (downstream barrier error) "
                        "is built here as fig2_panel_D_label_budget_downstream",
    "S3_scale_bootstrap": "Study-1 bootstrap intervals (derived/study1_softening_estimates.csv)",
    "S4_li_host_scales": "Study-1 s_Li vs s_host (derived/study1_softening_estimates.csv)",
}


def run(ctx: RunnerContext) -> dict[str, Any]:
    import json

    from e3sse.figures.publication_style import export_figure

    tables = derived_tables(ctx)
    derived_dir = ctx.work_dir / "derived"
    outputs: dict[str, Path] = {}
    for name, rows in tables.items():
        path = derived_dir / f"{name}.csv"
        write_table(path, rows, root=ctx.work_dir)
        outputs[f"derived:{name}"] = path
    frames = {name: _frame(path) for name, path in
              ((n, derived_dir / f"{n}.csv") for n in tables)}
    import matplotlib.pyplot as plt
    internal = ctx.internal_freeze()
    provenance = {"protocol_id": internal["protocol_id"], "revision_id": internal["protocol_id"],
                  "git_sha": internal["git_sha"],
                  "analysis_fingerprint": canonical_sha256({n: outputs[f"derived:{n}"].name
                                                            for n in tables})}
    figures_dir = ctx.work_dir / "figures"
    built, failed = [], {}
    code = [Path(__file__), Path(__file__).parents[2] / "figures" / "publication_style.py"]
    for figure_id, (inputs, draw) in sorted(figure_builders().items()):
        manifest_path = figures_dir / f"{figure_id}.figure.json"
        input_paths = {n: derived_dir / f"{n}.csv" for n in inputs}
        if manifest_path.exists():
            verify_figure(manifest_path, input_paths, code)
            built.append(figure_id)
            outputs[f"figure:{figure_id}"] = manifest_path
            continue
        orphans = sorted(p.name for p in figures_dir.glob(f"{figure_id}.*")) \
            if figures_dir.is_dir() else []
        if orphans:  # an interrupted export: never overwritten, never silently reused
            failed[figure_id] = f"orphan outputs without a manifest: {orphans} (inspect and move)"
            continue
        figure = None
        try:
            figure = draw(frames)
            export_figure(figure, figures_dir, figure_id, inputs=input_paths, plot_code=code,
                          provenance=provenance)
        except (MemoryError, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001 -- recorded per figure; the stage then stops
            failed[figure_id] = f"{type(exc).__name__}: {exc}"[:500]
            continue
        finally:
            if figure is not None:
                plt.close(figure)
        built.append(figure_id)
        outputs[f"figure:{figure_id}"] = manifest_path
    if failed:
        # Every registered figure is built or declared; a failure stops the stage (exit 3)
        # and is retried on resume, so no report claims an incomplete figure set.
        raise RunnerValidationError(f"figures failed: {dict(sorted(failed.items()))}")
    report = {"built": built, "partial_with_reason": PARTIAL, "not_built_with_reason": NOT_BUILT,
              "provided_elsewhere": ALSO_FROM_STUDY1}
    outputs["publication_report"] = ctx.publish("publication_report.json",
                                                json.loads(json.dumps(report)))
    return {"outputs": outputs, "summary": {"figures_built": len(built), "failed": failed}}
