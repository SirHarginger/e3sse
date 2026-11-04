"""Production ``recipe()`` + ``bvse_path_task`` on public nebDFT2k (harness-format output).

Usage: <bvse-runtime>/bin/python run_prod.py UNITS OUT DATA_ROOT [--workers N]

Unlike ``harness.py`` (which re-implements the candidate variants), this runs the
production task unchanged with the protocol's frozen recipe; only the server G0
manifest check of the structure source is replaced (see ``harness.LocalStructureSource``).
Reads geometry, ions arrays and BVSE energies only; no DFT quantity.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import LocalStructureSource, archived_bvse_barrier

from e3sse.production.runners import bvse_paths
from e3sse.production.runners.hop_files import read_geometry
from e3sse.screening.rate_limiting_hop import parse_edge_id

_STATE: dict = {}


def _init(data_root: str) -> None:
    warnings.filterwarnings("ignore")
    protocol = json.loads((REPO / "configs/reliability_gated_internal_freeze_protocol.json").read_text())
    _STATE["recipe"] = bvse_paths.recipe(protocol["amendments"])
    _STATE["init"] = Path(data_root) / "raw/nebDFT2k"
    _STATE["source"] = LocalStructureSource(
        {"kind": "ase_extxyz_shards", "id_field": "material_id",
         "paths": [f"raw/BVEL13k/BVEL13k_{s}.xyz" for s in ("train", "val", "test")]},
        Path(data_root), {})


def one(unit_id: str) -> dict:
    init = _STATE["init"] / f"{unit_id}_init.xyz"
    atoms = _STATE["source"].load(unit_id.split("_")[0])
    payload = {"unit_key": unit_id, "unit_id": unit_id, "edge": dict(parse_edge_id(unit_id)),
               "structure": bvse_paths._geometry(atoms), "reference": read_geometry(init),
               "recipe": _STATE["recipe"]}
    result = bvse_paths.bvse_path_task(payload, None)
    result.pop("images", None)
    result["variant"] = "PROD"
    if result["status"] == "generated":
        result["archived_init_bvse_barrier_eV"] = archived_bvse_barrier(init)
        result["barrier_minus_archived_init_eV"] = (result["bvse_barrier_eV"]
                                                    - result["archived_init_bvse_barrier_eV"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("units")
    parser.add_argument("out")
    parser.add_argument("data_root")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    units = Path(args.units).read_text().split()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.data_root,)) as pool:
        results = list(pool.map(one, units, chunksize=1))
    index = Path(args.data_root) / "raw/nebBVSE122k/nebBVSE122k_index.csv"
    with index.open(newline="") as handle:
        pool_barriers = {row["edge_id"]: float(row["em"]) for row in csv.DictReader(handle)}
    for result in results:
        if result["status"] == "generated" and result["unit_id"] in pool_barriers:
            result["barrier_minus_nebbvse122k_eV"] = (result["bvse_barrier_eV"]
                                                      - pool_barriers[result["unit_id"]])
    _init(args.data_root)
    Path(args.out).write_text(json.dumps({"python": platform.python_version(),
                                          "recipe": _STATE["recipe"], "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
