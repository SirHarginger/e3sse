"""The deployed predictor, defined once for model selection, calibration and discovery.

``amendments.prospective.deployment_predictor``: E_deploy = E_ML-NEB / s_family if the
unit passes the support gate and its anion family has a supported scale, otherwise
the raw ML-NEB barrier, labelled with its reason
(``screening.prospective_contract.prediction_source``).  If gate G1 failed, every
prediction is the raw ML-NEB barrier (``g1_contingency``).  Support: the frozen
reference (development population only); a development unit's own distance is its
leave-one-chemical-system-out distance.  A unit with no converged band has no
prediction (ML-NEB failure, never abstention).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def support_rows(branch: Mapping[str, Any], records: list[Mapping[str, Any]],
                 descriptors: Mapping[str, Any]) -> dict[str, list[float | None] | None]:
    """The unit's support covariates in the model's branch (original space)."""

    from e3sse.production.runners.representation import path_vector

    rows: dict[str, Any] = {}
    for record in records:
        if record["status"] != "converged":
            continue
        key = record["unit_key"]
        if branch["branch"] == "latent_support":
            embeddings = record.get("migrating_li_embeddings")
            rows[key] = path_vector(embeddings, record["energies_eV"]) if embeddings else None
        elif branch["branch"] == "descriptor_space":
            rows[key] = (descriptors.get(key) or {}).get("vector")
        else:
            rows[key] = None
    return rows


def support_decisions(branch: Mapping[str, Any], rows: Mapping[str, Any], *,
                      development: bool = False) -> dict[str, dict[str, Any]]:
    from e3sse.screening.applicability_support import support_decisions as decide
    from e3sse.screening.applicability_support import support_distances

    if branch.get("branch") not in ("latent_support", "descriptor_space"):
        return {key: {"supported": False, "distance": None, "reason": "support_unavailable"}
                for key in rows}
    reference = branch["reference"]
    keys = sorted(rows)
    if development:
        own = dict(zip(reference["unit_keys"], reference["loo_distances"]))
        distances = [own.get(key) for key in keys]
    else:
        distances = support_distances(reference, [rows[k] if rows[k] is not None and
                                                  all(v is not None for v in rows[k]) else
                                                  [None] for k in keys]) if keys else []
    return dict(zip(keys, decide(reference, distances)))


def predict(record: Mapping[str, Any], *, family: str | None, scales: Mapping[str, Any],
            supported: bool, g1_passes: bool) -> dict[str, Any] | None:
    from e3sse.barriers.fixed_geometry import srt_barrier
    from e3sse.screening.prospective_contract import prediction_source

    if record["status"] != "converged":
        return None
    raw = record["barrier_eV"]
    scale = (scales.get("family", {}).get(family) or {}).get("s_hat") if family else None
    source = prediction_source(supported=supported, family_scale_available=scale is not None)
    if not g1_passes:
        return {"prediction_eV": raw, "source": "raw_g1_contingency", "raw_eV": raw,
                "s_family": scale}
    value = srt_barrier(raw, scale) if source == "srt_family" else raw
    return {"prediction_eV": value, "source": source, "raw_eV": raw, "s_family": scale}


def unit_family(descriptors: Mapping[str, Any], key: str) -> str | None:
    """Frozen dominant-anion-count family from the unit's descriptor one-hot (composition)."""

    from e3sse.screening.applicability_support import DESCRIPTOR_NAMES

    vector = (descriptors.get(key) or {}).get("vector")
    if vector is None:
        return None
    hot = [name[len("family_"):] for name, value in zip(DESCRIPTOR_NAMES, vector)
           if name.startswith("family_") and value == 1.0]
    return hot[0] if len(hot) == 1 else None
