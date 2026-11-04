"""Per-candidate JSON record from harness outputs (frozen acceptance rule unchanged)."""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

MIN_RATE = 0.5  # preflight.BVSE_PIN_MIN_REPRODUCTION_RATE (frozen 2026-11-03)


def quantiles(values):
    v = sorted(values)
    if not v:
        return {"n": 0}
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]  # production _quantiles
    return {"n": len(v), "min": v[0], "median": pick(0.5), "p90": pick(0.9), "max": v[-1]}


def summarize(results_path: Path, candidate: dict, lock: Path, variant: str) -> dict:
    data = json.loads(results_path.read_text())
    rows = [r for r in data["results"] if r["variant"] == variant]
    counts = collections.Counter()
    reasons = collections.Counter()
    for r in rows:
        if r["status"] != "generated":
            counts[r["status"]] += 1
            continue
        rep = r["reproduction"]
        counts["reproduced" if rep["matches"] else "reproduction_mismatch"] += 1
        if not rep["matches"]:
            reasons[rep["reason"]] += 1
        if r.get("bvse_converged") is False:
            counts["bvse_not_converged"] += 1
    checked = counts["reproduced"] + counts["reproduction_mismatch"]
    rate = counts["reproduced"] / checked if checked else None
    gen = [r for r in rows if r["status"] == "generated"]
    return {
        **candidate,
        "variant": variant,
        "count_from_reference_DIAGNOSTIC_ONLY": data.get("count_from_reference", False),
        "python": data["python"],
        "environment_lock_path": str(lock),
        "environment_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "environment_lock_pins": [l for l in lock.read_text().splitlines()
                                  if l.split("==")[0].lower() in
                                  ("ions", "ase", "numpy", "scipy", "pymatgen", "spglib",
                                   "networkx", "pyyaml")],
        "units": len(rows),
        "counts": dict(sorted(counts.items())),
        "mismatch_reasons": dict(sorted(reasons.items())),
        "reproduction_rate": rate,
        "frozen_rule": f"per unit: identical image/atom count+order, cell <= 1e-4 A, every "
                       f"atom <= 0.01 A; pin confirmed iff rate > {MIN_RATE}",
        "frozen_rule_passes": rate is not None and rate > MIN_RATE,
        "max_position_difference_A": quantiles(
            [r["reproduction"]["max_position_difference_A"] for r in gen
             if r["reproduction"].get("max_position_difference_A") is not None]),
        "abs_barrier_minus_nebbvse122k_eV": quantiles(
            [abs(r["barrier_minus_nebbvse122k_eV"]) for r in gen
             if r.get("barrier_minus_nebbvse122k_eV") is not None]),
        "abs_barrier_minus_archived_init_energy_profile_eV": quantiles(
            [abs(r["barrier_minus_archived_init_eV"]) for r in gen
             if r.get("barrier_minus_archived_init_eV") is not None]),
    }


if __name__ == "__main__":
    results, candidate_json, lock, variant, out = sys.argv[1:6]
    record = summarize(Path(results), json.loads(candidate_json), Path(lock), variant)
    Path(out).write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("candidate", "variant", "counts", "reproduction_rate",
                                             "frozen_rule_passes", "max_position_difference_A",
                                             "abs_barrier_minus_nebbvse122k_eV")}, indent=None))
