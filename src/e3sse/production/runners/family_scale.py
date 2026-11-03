"""family_scale: the prospective family-scope s (``amendments.prospective.s_scope``).

Pooled Eq. 17 per anion family over the Study-1 frame statistics of published-train,
primary-population MPLiTrj frames whose chemical system has no nebDFT2k hop (so the
scale is disjoint from every calibration, development and evaluation hop).  The
frozen pooled support rule applies (>= 10 frames, >= 100 informative components,
>= 3 materials).  An unsupported family has no scale: units of that family get the
explicitly labelled raw fallback (``raw_family_scale_unavailable``), never another
scope's scale.  A pooled global scale over the same frames is reported alongside.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from e3sse.production.runners.context import MissingEvidence, RunnerContext

FAMILY_SCALE_KIND = "mplitrj_train_systems_without_nebdft2k_forces"


def scale_frames(membership: Sequence[Mapping[str, Any]], population: Mapping[str, str],
                 *, split_of_system: Mapping[str, str] | None = None,
                 training_systems: set[str] | None = None) -> list[Mapping[str, Any]]:
    """Primary-population frames of systems without nebDFT2k hops on the training side.

    The published split (default) or, for a Study-4 repeat, an explicit set of
    training-side chemical systems defines the training side.
    """

    with_hops = {r["chemical_system_id"] for r in membership if r.get("record_kind") == "nebdft2k_hop"}
    frames = []
    for row in membership:
        if row.get("record_kind") != "mplitrj_frame":
            continue
        system = row["chemical_system_id"]
        if system in with_hops or population.get(system) != "primary":
            continue
        training = (system in training_systems if training_systems is not None
                    else row.get("published_split") == "train")
        if training:
            frames.append(row)
    return frames


def pooled_scales(stats: Any, frames: Sequence[Mapping[str, Any]], family_of: Mapping[str, str],
                  parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Pooled global and per-family Eq. 17 estimates with the frozen support rule."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frames:
        if row["frame_id"] in stats.row:
            groups[family_of[row["material_id"]]].append(row)
    result: dict[str, Any] = {"family": {}}
    present = [row for rows in groups.values() for row in rows]
    result["global"] = stats.estimate(stats.rows([r["frame_id"] for r in present]), "all",
                                      materials=[r["material_id"] for r in present], pooled=True,
                                      parameters=parameters)
    for family, rows in sorted(groups.items()):
        result["family"][family] = stats.estimate(
            stats.rows([r["frame_id"] for r in rows]), "all",
            materials=[r["material_id"] for r in rows], pooled=True, parameters=parameters)
    return result


def study1_frame_stats(ctx: RunnerContext, model: str) -> Any:
    """FrameStats over the Study-1 published-train frames of ``model``."""

    from e3sse.production.runners.frame_scales import FrameStats
    from e3sse.softening.study1_inference import verify_statistics_file

    model_dir = ctx.upstream_path("estimate", f"frame_statistics:{model}").parent
    reference_dir = ctx.upstream_path("estimate", "frame_statistics:dft_reference").parent
    _, model_arrays = verify_statistics_file(model_dir)
    _, reference = verify_statistics_file(reference_dir)
    name = ctx.study1_config["inference"]["source_file"]
    frame_ids = [f"{name}:{int(i)}" for i in model_arrays["frame_index"]]
    return FrameStats(frame_ids, model_arrays, reference)


def run(ctx: RunnerContext) -> dict[str, Any]:
    from e3sse.softening.study1_inputs import support_parameters

    population = ctx.partition_manifest["primary_population"]["assignment_by_chemical_system"]
    family_of = ctx.partition_manifest["anion_families"]["assignment_by_material"]
    frames = scale_frames(ctx.membership, population)
    if not frames:
        raise MissingEvidence("no training-side frames of systems without nebDFT2k hops")
    ctx.require_outcome(role="train", outcome="family_scale_frame_statistics",
                        outcome_kind=FAMILY_SCALE_KIND,
                        unit_ids=[r["frame_id"] for r in frames])
    parameters = support_parameters(ctx.parent_protocol)
    scales = {}
    for model in sorted(ctx.models):
        stats = study1_frame_stats(ctx, model)
        scales[model] = pooled_scales(stats, frames, family_of, parameters)
    record = {"scope": "family (pooled, MPLiTrj training-side primary systems without nebDFT2k hops)",
              "fallback": "family scale unavailable -> raw_family_scale_unavailable (labelled)",
              "frames": len(frames), "scales": scales}
    available = {m: sorted(f for f, e in s["family"].items() if e["s_hat"] is not None)
                 for m, s in scales.items()}
    return {"outputs": {"family_scales": ctx.publish("family_scales.json", record)},
            "summary": {"families_with_scale": available}}
