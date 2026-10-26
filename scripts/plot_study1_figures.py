#!/usr/bin/env python3
"""Render the Study-1 figures from a completed analysis's saved derived tables.

Writes ``study1_softening_atlas`` and ``study1_label_budget_stability`` (vector PDF,
600-dpi PNG and a JSON manifest with input-table SHA-256) under
``<analysis>/figures/``.  Each input table is checked against the analysis manifest
first.  No figure number comes from anywhere but those tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from e3sse.softening import study1_figures as figures
from e3sse.softening import study1_inputs as inputs
from e3sse.softening.study1_tables import read_csv

REPOSITORY = Path(__file__).resolve().parents[1]
TABLES = {
    "estimates": "derived/study1_softening_estimates.csv",
    "summary": "derived/study1_model_summary.csv",
    "stability": "derived/study1_label_budget_stability.csv",
}


def locate_analysis(config_path: Path, output_root: str | None) -> Path:
    config, _ = inputs.load_study1_config(config_path)
    root = Path(output_root or config["output_root"]).expanduser()
    found = sorted((root / config["frozen_protocol"]["protocol_id"][:16] / "analysis").glob(
        "*/study1_analysis_manifest.json"))
    if len(found) != 1:
        raise SystemExit(f"expected one completed analysis, found {len(found)}; "
                         "pass --analysis-dir")
    return found[0].parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPOSITORY / "configs/study1_softening_atlas.json"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    analysis = args.analysis_dir or locate_analysis(Path(args.config), args.output_root)
    manifest = json.loads((analysis / "study1_analysis_manifest.json").read_text())
    paths = {name: analysis / relative for name, relative in TABLES.items()}
    for name, relative in TABLES.items():
        if hashlib.sha256(paths[name].read_bytes()).hexdigest() != manifest["outputs_sha256"][
            relative
        ]:
            raise SystemExit(f"{relative} differs from the analysis manifest")
    data = {name: read_csv(path) for name, path in paths.items()}
    provenance = {
        "analysis_fingerprint": manifest["analysis_fingerprint"],
        "protocol_id": manifest["identity"]["protocol_id"],
        "implementation_git_sha": manifest["implementation_git_sha"],
    }
    out = analysis / "figures"
    atlas = figures.softening_atlas(data["estimates"], data["summary"])
    figures.export(atlas, out, "study1_softening_atlas",
                   {k: paths[k] for k in ("estimates", "summary")}, provenance)
    budget = figures.label_budget_stability(data["stability"], data["summary"])
    figures.export(budget, out, "study1_label_budget_stability",
                   {k: paths[k] for k in ("stability", "summary")}, provenance)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
