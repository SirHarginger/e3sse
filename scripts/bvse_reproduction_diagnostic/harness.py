"""Outcome-free BVSE-path reproduction diagnostic against public nebDFT2k _init.xyz.

Reads only geometry, ions arrays and BVSE energies (_init.xyz), pristine BVEL13k
structures (geometry) and the nebBVSE122k BVSE barrier (label-free covariate).  Never
reads em_dft, *_relaxed.xyz or any DFT quantity.  Scoring is production's own
compare_paths with the frozen tolerance read from the protocol.  Pristine structures
come from production's StructureSource; only its server G0-manifest check is replaced
by recording each local file's SHA-256.  Run with a BVSE runtime interpreter, e.g.
``<runtime>/bin/python harness.py --data-root D --units U --variants C2,C5 --out R``
where D/raw holds BVEL13k/, nebDFT2k/ and nebBVSE122k/ (index CSV only).

Variants (all ions >= 0.3.5 SaddleFinder.bvse_neb, k=2.0, default=True, gm=False,
FIRE fmax=0.1 steps=100, superedge 8.0 A, n_images=5):
  C0  distort=True, perturbation kept               (current pin; production task)
  C1  distort=False                                  (production task, recipe flag only)
  C2  distort=True, FIRE, subtract perturbation, FIRE (= SaddleFinder.run, relax_endpoints=False)
  C3  distort=True, FIRE, subtract perturbation       (no second relaxation)
  C4  SaddleFinder.run(relax_endpoints=True)          (ions' own default driver)
  C5  C2 with pymatgen BVAnalyzer oxidation states    (amended recipe, 2026-11-04)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from e3sse.production.runners import bvse_paths
from e3sse.production.runners.hop_files import read_geometry
from e3sse.production.runners.structures import StructureSource, check_edge_sites
from e3sse.screening.rate_limiting_hop import parse_edge_id

VARIANTS = ("C0", "C1", "C2", "C3", "C4", "C5")  # C5 = C2 with pymatgen BVAnalyzer oxidation states
_STATE: dict = {}


class LocalStructureSource(StructureSource):
    """Production loader; only the server G0-manifest check is replaced by recording
    the local file SHA-256 (compared with the server manifest separately)."""

    def _verified(self, path):
        if str(path) not in self.files_read:
            self.files_read[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return path


def base_recipe(distort: bool) -> dict:
    """The pre-amendment recipe every variant starts from (n_images=5, ions defaults);
    independent of the production recipe so the recorded candidates stay reproducible.
    Only the frozen reproduction tolerance is read from the protocol."""

    protocol = json.loads((REPO / "configs/reliability_gated_internal_freeze_protocol.json").read_text())
    tolerance = protocol["amendments"]["prospective"]["ml_neb"]["frozen_numeric_settings"][
        "bvse_reproduction_tolerance"]
    return {"mobile_symbol": "Li", "mobile_charge": 1, "superedge_cutoff_A": 8.0, "n_images": 5,
            "bvse_neb_options": {"k": 2.0, "default": True, "distort": distort, "gm": False},
            "fire_fmax": 0.1, "fire_steps": 100,
            "cell_tolerance_A": float(tolerance["cell_A"]),
            "position_tolerance_A": float(tolerance["max_atom_displacement_A"])}


def _variant_task(payload: dict, variant: str) -> dict:
    """One candidate: C0/C1 relax once (C0 = the pre-amendment production task), C2-C5
    add ions' perturbation-removal step."""

    from ase import Atoms
    from ase.optimize import FIRE

    settings = payload["recipe"]
    result = {"unit_key": payload["unit_key"], "unit_id": payload["unit_id"]}
    s = payload["structure"]
    atoms = Atoms(numbers=s["numbers"], positions=s["positions"], cell=s["cell"], pbc=s["pbc"])
    try:
        if variant == "C5":
            import numpy as np
            from pymatgen.analysis.bond_valence import BVAnalyzer
            from pymatgen.io.ase import AseAtomsAdaptor

            valences = BVAnalyzer().get_valences(AseAtomsAdaptor.get_structure(atoms))
            atoms.set_array("oxi_states", np.array(valences))
        else:
            from ions import Decorator
            atoms = Decorator().decorate(atoms)
    except Exception as exc:  # noqa: BLE001
        return {**result, "status": "decorator_failure", "detail": f"{type(exc).__name__}: {exc}"}
    try:
        from ions.geom.edge import Edge
        from ions.tools import SaddleFinder
        from ions.utils import collect_bvse_params

        atoms = collect_bvse_params(atoms, settings["mobile_symbol"], settings["mobile_charge"])
        edge = Edge(atoms, payload["edge"]["source"], payload["edge"]["target"], payload["edge"]["offset"])
        images = edge.superedge(settings["superedge_cutoff_A"]).interpolate(n_images=settings["n_images"])
        finder = SaddleFinder()
        neb = finder.bvse_neb(images, **settings["bvse_neb_options"])
        perturbation = max((float(abs(im.info["perturbation"]).max()) for im in images
                            if "perturbation" in im.info), default=0.0)
        fmax, steps = settings["fire_fmax"], settings["fire_steps"]
        if variant in ("C0", "C1"):
            FIRE(neb, logfile=None).run(fmax=fmax, steps=steps)
        elif variant == "C4":
            finder.run(neb, steps=steps, fmax=fmax, relax_endpoints=True, logfile=None)
        else:
            FIRE(neb, logfile=None).run(fmax=fmax, steps=steps)
            for image in images:
                image.positions -= image.info["perturbation"]
                image.info["perturbation"] = 0.0 * image.info["perturbation"]
            if variant in ("C2", "C5"):
                FIRE(neb, logfile=None).run(fmax=fmax, steps=steps)
        if variant not in ("C0", "C1"):  # as recorded; ASE would also recompute on change
            for image in images:
                image.calc.reset()
        bvse_fmax = float(abs(neb.get_forces()).max())
        bvse_barrier = float(finder.get_barrier(images))
    except Exception as exc:  # noqa: BLE001
        return {**result, "status": "ions_failure", "detail": f"{type(exc).__name__}: {exc}"}
    geometry = [bvse_paths._geometry(image) for image in images]
    result.update(status="generated", bvse_fmax_eV_per_A=bvse_fmax,
                  bvse_converged=bvse_fmax <= settings["fire_fmax"],
                  max_framework_perturbation_A=perturbation, bvse_barrier_eV=bvse_barrier)
    result["reproduction"] = bvse_paths.compare_paths(geometry, payload["reference"], settings)
    return result


def _init(args) -> None:
    _STATE["source"] = LocalStructureSource(
        {"kind": "ase_extxyz_shards", "paths": [f"raw/BVEL13k/BVEL13k_{s}.xyz" for s in ("train", "val", "test")],
         "id_field": "material_id"}, Path(args.data_root), {})
    _STATE["recipes"] = {False: base_recipe(False), True: base_recipe(True)}


def archived_bvse_barrier(path: Path) -> float | None:
    from ase.io import read

    energies = [im.get_potential_energy() for im in read(path, ":")]
    return float(max(energies) - energies[0])


def run_unit(job):
    edge_id, variant, init_path, count_oracle = job
    started = time.perf_counter()
    unit = {"unit_key": edge_id, "unit_id": edge_id}
    material = edge_id.split("_")[0]
    try:
        atoms = _STATE["source"].load(material)
    except Exception as exc:  # noqa: BLE001
        return {**unit, "variant": variant, "status": "structure_missing", "detail": str(exc)}
    edge = parse_edge_id(edge_id)
    try:
        check_edge_sites(atoms, edge)
    except Exception as exc:  # noqa: BLE001
        return {**unit, "variant": variant, "status": "edge_sites_not_li", "detail": str(exc)}
    reference = read_geometry(Path(init_path))
    payload = {**unit, "edge": dict(edge), "structure": bvse_paths._geometry(atoms),
               "reference": reference, "recipe": dict(_STATE["recipes"][variant != "C1"])}
    if count_oracle:
        payload["recipe"]["n_images"] = len(reference)
    result = _variant_task(payload, variant)
    result["variant"] = variant
    result["n_atoms"] = len(reference[0]["numbers"])
    result["wall_s"] = round(time.perf_counter() - started, 2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--units", required=True, help="text file, one edge_id per line")
    parser.add_argument("--variants", default="C0")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count-from-reference", action="store_true",
                        help="DIAGNOSTIC ONLY: n_images = archived image count (not a production rule)")
    args = parser.parse_args()
    root = Path(args.data_root)
    init_dir = root / "raw/nebDFT2k"
    edges = [line.strip() for line in Path(args.units).read_text().split() if line.strip()]
    jobs = [(e, v, str(init_dir / f"{e}_init.xyz"), args.count_from_reference)
            for v in args.variants.split(",") for e in edges]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args,)) as pool:
        results = list(pool.map(run_unit, jobs, chunksize=1))
    pool_barriers = {}
    with open(root / "raw/nebBVSE122k/nebBVSE122k_index.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            pool_barriers[row["edge_id"]] = float(row["em"])
    for r in results:
        if r.get("status") == "generated":
            r["archived_init_bvse_barrier_eV"] = archived_bvse_barrier(init_dir / f"{r['unit_id']}_init.xyz")
            r["barrier_minus_archived_init_eV"] = r["bvse_barrier_eV"] - r["archived_init_bvse_barrier_eV"]
            nb = pool_barriers.get(r["unit_id"])
            r["barrier_minus_nebbvse122k_eV"] = None if nb is None else r["bvse_barrier_eV"] - nb
    Path(args.out).write_text(json.dumps({"python": platform.python_version(),
                                       "count_from_reference": args.count_from_reference, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
