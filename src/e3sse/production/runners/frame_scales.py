"""Eq. 17 scales and diagnostics from per-frame sufficient statistics.

Inputs are merged ``frame_statistics.npz`` files in the Study-1 format (one per model,
plus the DFT reference): per frame and element, Sxy, Sxx, Syy and informative-component
counts.  Every estimate uses ``softening.frame_statistics`` (Eq. 17, support rule,
bootstrap indices) and ``softening.applicability`` (rho_F, scale uncertainty, Li/host
and energy disagreement).  Frame eligibility always comes from the frozen leakage
audit (``contracts.audit_softening_frame_eligibility_batch``); the s_scope rule then
keeps only endpoint-relaxation (non-path) frames, own path forbidden
(``deployment_gate.s_scope``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from e3sse.softening.frame_statistics import (
    COMPONENT_SETS,
    LITHIUM_Z,
    eq17,
    iter_bootstrap_indices,
    support_status,
)

ENDPOINT_KINDS = ("source", "target")  # contracts.ENDPOINT_RELAXATION_FILE_KINDS


def thresholds_or_closed(population, supported, quantile_level: float) -> dict[str, Any]:
    """``applicability.gate_thresholds``; a threshold population without any supported,
    defined value leaves the thresholds undefined (recorded), so the gate is closed."""

    from e3sse.softening.applicability import GATED_DIAGNOSTICS, gate_thresholds

    empty = [name for name in GATED_DIAGNOSTICS
             if not any(ok and v is not None for v, ok in zip(population[name], supported))]
    if empty:  # checked directly, never inferred from an error message
        return {"thresholds": {name: None for name in GATED_DIAGNOSTICS},
                "quantile_level": quantile_level, "status": "undefined",
                "reason": f"no supported, defined value of {empty}",
                "n_population": len(supported), "n_supported": sum(map(bool, supported))}
    return {**gate_thresholds(population, supported, quantile_level), "status": "defined"}


def gate_or_closed(*, supported: bool, diagnostics, thresholds) -> tuple[bool, list[str]]:
    """``applicability.gate_decision``; undefined thresholds close the gate (no SRT)."""

    from e3sse.softening.applicability import gate_decision

    if any(value is None for value in thresholds.values()):
        return False, ["thresholds_undefined"]
    return gate_decision(supported=supported, diagnostics=diagnostics, thresholds=thresholds)


def aligned_frame_stats(frame_ids: Sequence[str], model: Mapping[str, Any],
                        reference: Mapping[str, Any]) -> FrameStats:
    """FrameStats whose frame IDs (``<source>:<index>``) must name the statistics rows'
    own ``frame_index`` in order; a misaligned row would attach labels to another frame."""

    indices = [int(i) for i in model["frame_index"]]
    try:
        named = [int(str(frame_id).rsplit(":", 1)[1]) for frame_id in frame_ids]
    except (IndexError, ValueError) as exc:
        raise ValueError("frame IDs must have the form <source>:<index>") from exc
    if named != indices:
        raise ValueError("frame IDs do not match the statistics rows' frame_index")
    return FrameStats(frame_ids, model, reference)


class FrameStats:
    """One model's per-frame sums aligned to frame IDs (plus DFT reference energies)."""

    def __init__(self, frame_ids: Sequence[str], model: Mapping[str, Any],
                 reference: Mapping[str, Any]) -> None:
        import numpy as np

        if not np.array_equal(model["frame_index"], reference["frame_index"]):
            raise ValueError("model and reference statistics cover different frames")
        n = len(model["frame_index"])
        if len(frame_ids) != n:
            raise ValueError("one frame ID per statistics row is required")
        self.frame_ids = list(frame_ids)
        self.row = {frame_id: i for i, frame_id in enumerate(self.frame_ids)}
        z = np.asarray(model["element_Z"], dtype=np.int64)
        owner = np.asarray(model["element_frame_row"], dtype=np.int64)
        self.sums: dict[str, Any] = {}
        for name in COMPONENT_SETS:
            mask = np.ones_like(z, dtype=bool) if name == "all" else (
                z == LITHIUM_Z if name == "Li" else z != LITHIUM_Z)
            block = np.zeros((n, 3))
            for column, key in enumerate(("element_sxy", "element_sxx", "element_syy")):
                np.add.at(block[:, column], owner[mask], np.asarray(model[key])[mask])
            self.sums[name] = block
        informative = np.zeros(n, dtype=np.int64)
        np.add.at(informative, owner, np.asarray(model["element_n_informative"], dtype=np.int64))
        self.informative = informative  # all-set count: the frozen support basis
        self.success = np.asarray(model["success"], dtype=bool)
        self.dft_energy = np.asarray(reference["dft_energy_eV"], dtype=float)
        self.n_atoms = np.asarray(reference["n_atoms"], dtype=np.int64)

    def rows(self, frame_ids: Sequence[str]) -> Any:
        import numpy as np

        return np.asarray(sorted(self.row[f] for f in frame_ids), dtype=np.int64)

    def estimate(self, rows: Any, component_set: str, *, materials: Sequence[str],
                 pooled: bool, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Eq. 17 with the frozen support rule; undefined values carry their reasons."""

        import numpy as np

        from e3sse.softening.applicability import rho_f

        rows = np.asarray(rows, dtype=np.int64)
        failed = int((~self.success[rows]).sum())
        reasons = support_status(n_frames=int(rows.size),
                                 n_informative=int(self.informative[rows].sum()),
                                 n_materials=len(set(materials)), pooled=pooled,
                                 parameters=parameters)
        if failed:
            reasons.append("model_frame_failure")
        sxy, sxx, syy = (math.fsum(self.sums[component_set][rows, c].tolist()) for c in range(3))
        s = eq17(sxy, sxx) if sxx > 0 else None
        return {"component_set": component_set, "n_frames": int(rows.size),
                "n_informative": int(self.informative[rows].sum()), "failed_frames": failed,
                "sxy": sxy, "sxx": sxx, "syy": syy, "supported": not reasons,
                "unsupported_reasons": reasons,
                "s_hat": s if not reasons else None,
                "rho_F": rho_f(sxy, sxx, syy) if not reasons else None}

    def bootstrap(self, rows: Any, component_set: str, *, seed: int, resamples: int) -> list[float | None]:
        """Whole-frame percentile bootstrap of Eq. 17 (frozen generator)."""

        import numpy as np

        rows = np.asarray(rows, dtype=np.int64)
        block_sums = self.sums[component_set][rows][:, :2]
        values: list[float | None] = []
        for block in iter_bootstrap_indices(seed, int(rows.size), resamples):
            drawn = block_sums[np.asarray(block)].sum(axis=1)  # (resamples, 2)
            positive = drawn[:, 1] > 0
            ratio = np.divide(drawn[:, 0], drawn[:, 1], out=np.zeros(len(drawn)), where=positive)
            values.extend(float(r) if ok else None for r, ok in zip(ratio, positive))
        return values


def energy_reference(stats: FrameStats, rows: Any) -> list[float]:
    """dE_f = [E_f - min E over the rows of the same atom count] / N_f (eV/atom)."""

    groups: dict[int, float] = {}
    for i in rows:
        n = int(stats.n_atoms[i])
        groups[n] = min(groups.get(n, math.inf), float(stats.dft_energy[i]))
    return [(float(stats.dft_energy[i]) - groups[int(stats.n_atoms[i])]) / int(stats.n_atoms[i])
            for i in rows]


def hop_diagnostics(stats: FrameStats, rows: Any, *, material: str, seed: int,
                    resamples: int, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Material-scope s (all/Li/host) and the frozen retrospective diagnostics."""

    import numpy as np

    from e3sse.softening.applicability import (
        energy_median_split,
        log_ratio_disagreement,
        scale_uncertainty,
    )

    rows = np.asarray(rows, dtype=np.int64)
    materials = [material] * int(rows.size)
    est = {name: stats.estimate(rows, name, materials=materials, pooled=False,
                                parameters=parameters) for name in COMPONENT_SETS}
    s_all = est["all"]["s_hat"]
    uncertainty = (scale_uncertainty(s_all, stats.bootstrap(rows, "all", seed=seed,
                                                            resamples=resamples))
                   if s_all is not None else {"value": None, "reason": "s_undefined"})
    pair = lambda e: (e["sxy"], e["sxx"])
    li_host = log_ratio_disagreement(pair(est["Li"]), pair(est["host"]),
                                     numerator_supported=est["Li"]["supported"],
                                     denominator_supported=est["host"]["supported"],
                                     label="li_host")
    energy: dict[str, Any] = {"value": None, "reason": "too_few_frames"}
    if rows.size >= 2:
        low, high = energy_median_split(energy_reference(stats, rows))
        low_e = stats.estimate(rows[low], "all", materials=materials[: len(low)],
                               pooled=False, parameters=parameters)
        high_e = stats.estimate(rows[high], "all", materials=materials[: len(high)],
                                pooled=False, parameters=parameters)
        energy = log_ratio_disagreement(pair(high_e), pair(low_e),
                                        numerator_supported=high_e["supported"],
                                        denominator_supported=low_e["supported"],
                                        label="energy_halves")
    return {"estimates": est, "s_hat": s_all,
            "rho_F": None if est["all"]["rho_F"] is None else est["all"]["rho_F"]["value"],
            "scale_uncertainty": uncertainty.get("value"),
            "li_host_disagreement": li_host.get("value"),
            "energy_dependence": energy.get("value"),
            "undefined_reasons": {
                "rho_F": None if est["all"]["rho_F"] is None else est["all"]["rho_F"].get("reason"),
                "scale_uncertainty": uncertainty.get("reason"),
                "li_host_disagreement": li_host.get("reason"),
                "energy_dependence": energy.get("reason")}}


def scope_frames(*, protocol: Mapping[str, Any], membership: Sequence[Mapping[str, Any]],
                 hops: Sequence[Mapping[str, Any]], available: set[str], partition: str | int,
                 side: str, estimator_kind: str = "primary") -> dict[str, list[str]]:
    """Per target hop, the frame IDs of its material-scope selection under s_scope.

    ``side`` is the frozen partition role (``training_or_calibration`` or ``test``).
    Primary selections keep endpoint-relaxation frames only (s_scope); the oracle
    keeps the hop's own path frames, labelled by the audit.
    """

    from e3sse.protocol.contracts import (
        audit_softening_frame_eligibility_batch,
        require_primary_softening_selection,
    )
    from e3sse.softening.study1_analysis import _scope_evidence

    by_material: dict[str, list[Mapping[str, Any]]] = {}
    for row in membership:
        if row.get("record_kind") == "mplitrj_frame" and row["frame_id"] in available:
            by_material.setdefault(row["material_id"], []).append(row)
    requests = [{"records": by_material.get(hop["material_id"], []), "target_hop_id": hop["hop_id"],
                 "estimator_scope": "material",
                 "scope_evidence": _scope_evidence("material", hop["material_id"]),
                 "allowed_partition_role": side, "target_material_id": hop["material_id"],
                 "estimator_kind": estimator_kind} for hop in hops]
    selections = audit_softening_frame_eligibility_batch(
        requests, protocol=protocol, partition_records=membership, partition=partition)
    frames = {}
    for hop, selection in zip(hops, selections):
        if estimator_kind == "primary":
            require_primary_softening_selection(selection, protocol)
            chosen = [r["frame_id"] for r in selection["eligible_records"]
                      if r["raw_file_kind"] in ENDPOINT_KINDS]
        else:
            if selection.get("analysis_label") != "oracle_upper_bound":
                raise ValueError("oracle selection is not labelled as an oracle")
            chosen = [r["frame_id"] for r in selection["eligible_records"]]
        frames[hop["hop_id"]] = sorted(chosen)
    return frames
