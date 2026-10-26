"""Frozen inputs of Study 1 (Softening Atlas): protocol, membership, models, gate.

Everything scientific comes from the frozen protocol directory written by
``scripts/build_protocol_freeze.py``: ``protocol.json`` (pinned by protocol ID and
file SHA-256 in ``configs/study1_softening_atlas.json``), ``partition_membership.json``
(bound by ``partition_universe.membership_sha256``) and ``partition_manifest.json``
(bound by ``manifest_id``).  Nothing here reads a DFT label; reading one requires
:func:`require_study1_outcome_access`, the repository's outcome seal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from e3sse.protocol.contracts import (
    OutcomeAccessDenied,
    require_outcome_access,
    validate_protocol_freeze_for_outcome_access,
)

STUDY1_CONFIG_SCHEMA = "e3sse.study1.config.v1"
PUBLISHED_PARTITION = "published"
FROZEN_MODEL_GROUPS = ("A", "B1", "B2")
GROUP_ORDER = {"A": 1, "B1": 2, "B2": 3}
STUDY1_OUTCOME = "study1_mplitrj_published_train_dft_forces_and_energies"
REGISTRATION_FILE = "registration_registered.json"


class Study1InputError(RuntimeError):
    """A frozen Study-1 input is missing, stale or inconsistent; never continue."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Study1InputError(f"missing input {path}") from exc


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_study1_config(path: Path) -> tuple[dict[str, Any], str]:
    config = load_json(path)
    if config.get("schema_version") != STUDY1_CONFIG_SCHEMA:
        raise Study1InputError("unsupported Study-1 configuration schema")
    return config, file_sha256(path)


def load_frozen_protocol(config: Mapping[str, Any], *, directory: Path | None = None) -> dict:
    """``protocol.json`` verified against its pinned ID, file SHA-256 and freeze commit."""

    pins = config["frozen_protocol"]
    root = Path(directory or pins["directory"]).expanduser()
    path = root / "protocol.json"
    if not path.is_file():
        raise Study1InputError(f"missing frozen protocol {path}")
    if file_sha256(path) != pins["protocol_json_sha256"]:
        raise Study1InputError("protocol.json SHA-256 differs from the pinned frozen protocol")
    protocol = load_json(path)
    try:
        validate_protocol_freeze_for_outcome_access(protocol)
    except (KeyError, TypeError, ValueError) as exc:
        raise Study1InputError(f"protocol.json is not a valid frozen protocol: {exc}") from exc
    if protocol["protocol_id"] != pins["protocol_id"]:
        raise Study1InputError("protocol_id differs from the pinned frozen protocol")
    if protocol["git_sha"] != pins["freeze_git_sha"]:
        raise Study1InputError("frozen protocol Git SHA differs from the pinned freeze commit")
    if protocol["study_1_contract"].get("estimator") != "zero_intercept_force_parity_projection":
        raise Study1InputError("frozen Study-1 estimator is not Eq. 17")
    return protocol


def membership_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """The frozen membership digest (``contracts._verify_membership`` formula)."""

    return hashlib.sha256(
        json.dumps(list(records), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_membership(root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = load_json(Path(root) / "partition_membership.json")
    records = payload.get("records")
    expected = protocol["partition_definition"]["partition_universe"]["membership_sha256"]
    if not isinstance(records, list) or membership_sha256(records) != expected:
        raise Study1InputError("partition_membership.json differs from the frozen membership")
    return records


def load_partition_manifest(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = load_json(Path(root) / "partition_manifest.json")
    if manifest != protocol["partition_definition"]:
        raise Study1InputError("partition_manifest.json differs from the frozen protocol")
    for key, label in (("primary_population", "TM/Hubbard-U population"),
                       ("anion_families", "anion families")):
        if manifest.get(key, {}).get("status") != "resolved":
            raise Study1InputError(f"frozen {label} are not resolved")
    return manifest


def registration_addendum(root: Path) -> dict[str, Any] | None:
    """The ``registered`` addendum written by ``scripts/record_registration.py``, if any."""

    path = Path(root) / REGISTRATION_FILE
    return load_json(path) if path.is_file() else None


def require_study1_outcome_access(
    protocol: Mapping[str, Any],
    addendum: Mapping[str, Any] | None,
    records: Sequence[Mapping[str, Any]],
    frame_ids: Sequence[str],
) -> None:
    """The repository's seal 1 for published-train MPLiTrj labels; raises when sealed."""

    try:
        require_outcome_access(
            role="train",
            outcome=STUDY1_OUTCOME,
            frozen_protocol=protocol,
            purpose="preregistered_analysis",
            registration_addendum=addendum,
            partition_records=records,
            unit_ids=frame_ids,
            outcome_kind="mplitrj_published_train_dft_forces_energies",
        )
    except OutcomeAccessDenied as exc:
        raise Study1InputError(f"Study 1 outcomes are sealed: {exc}") from exc


def frozen_models(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The six frozen Study-1 models (Groups A, B1, B2), keyed by canonical name."""

    models: dict[str, dict[str, Any]] = {}
    for group in FROZEN_MODEL_GROUPS:
        for model in protocol["model_groups"].get(group, []):
            if model.get("resolution_status") != "resolved" or model.get("fine_tuned") is not False:
                raise Study1InputError(f"unresolved or fine-tuned model in group {group}")
            if model.get("reference_compatibility", {}).get("primary_analysis_eligible") is not True:
                raise Study1InputError(f"{model['canonical_name']} is not primary-analysis eligible")
            models[model["canonical_name"]] = dict(model)
    for model in protocol["model_groups"].get("C", []):
        if model.get("canonical_name"):
            raise Study1InputError("Group C is not enabled for Study 1")
    return models


def model_runtime(protocol: Mapping[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    """(environment name, frozen environment record) of the model's pinned runtime."""

    environments = protocol["production_evidence"]["runtime_environments"]
    matches = [(env, record) for env, record in environments.items() if name in record["models"]]
    if len(matches) != 1:
        raise Study1InputError(f"model {name} needs exactly one frozen runtime environment")
    return matches[0][0], dict(matches[0][1])


def mplitrj_source(protocol: Mapping[str, Any], source_file: str) -> dict[str, Any]:
    """Frozen identity (identifier, SHA-256, frames, split) of one flattened source."""

    datasets = protocol["dataset_manifest"]["datasets"]
    provenance = next(item for item in datasets if item["dataset"] == "MPLiTrj")["provenance"]
    matches = [
        source for source in provenance["flattened_sources"]
        if Path(source["identifier"]).name == source_file
    ]
    if len(matches) != 1:
        raise Study1InputError(f"{source_file} is not a frozen flattened MPLiTrj source")
    source = dict(matches[0])
    source["provenance_index_id"] = provenance["index_id"]
    return source


def published_train_frames(
    records: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Published-train frames of one source file, ascending source frame index."""

    name = Path(source["identifier"]).name
    frames = [
        dict(row) for row in records
        if row.get("record_kind") == "mplitrj_frame"
        and row.get("published_split") == "train"
        and str(row.get("frame_id", "")).startswith(f"{name}:")
    ]
    frames.sort(key=lambda row: row["source_frame_index"])
    for row in frames:
        index = row["source_frame_index"]
        if (
            type(index) is not int or not 0 <= index < source["frames"]
            or row["frame_id"] != f"{name}:{index}"
            or row["source_file_sha256"] != source["sha256"]
            or row.get("provenance_index_id") != source["provenance_index_id"]
        ):
            raise Study1InputError(f"frozen frame identity is inconsistent: {row['frame_id']}")
    if source.get("published_split") != "train":
        raise Study1InputError(f"{name} is not the published training source")
    everywhere = sum(1 for row in records if row.get("record_kind") == "mplitrj_frame"
                     and row.get("published_split") == "train")
    if everywhere != len(frames) or len(frames) != source["frames"]:
        raise Study1InputError("published-train frames are not exactly the frames of the source")
    return frames


def pilot_amendment_ids(root: Path) -> list[str]:
    """IDs of pilot amendments recorded by registered OSF updates (identity, not access)."""

    ids = set()
    for path in sorted(Path(root).glob("registration_*_pilot-*.json")):
        amendment = load_json(path).get("pilot_amendment_id")
        if amendment:
            ids.add(amendment)
    return sorted(ids)


def study1_target_hops(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Hops to be corrected (Research Design 5.6): published-train nebDFT2k hops."""

    hops = [
        dict(row) for row in records
        if row.get("record_kind") == "nebdft2k_hop"
        and row.get("published_split") == "train"
        and row.get("eligibility_status") == "included"
    ]
    return sorted(hops, key=lambda row: row["hop_id"])


def support_parameters(protocol: Mapping[str, Any]) -> dict[str, Any]:
    support = protocol["study_1_contract"]["minimum_support"]
    resolution = support.get("resolution", {})
    if resolution.get("status") != "resolved":
        raise Study1InputError("estimator_minimum_support is not resolved")
    return dict(resolution["parameters"])


def statistical_constants(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Every Study-1 statistical constant, read from the frozen protocol."""

    contract = protocol["study_1_contract"]
    randomness = protocol["statistical_plan"]["randomness"]
    budget = contract["label_budget_stability"]
    if contract.get("bootstrap_unit") != "whole_frame":
        raise Study1InputError("Study-1 bootstrap unit is not the whole frame")
    if randomness["bootstrap"]["generator"] != "numpy.random.Generator(numpy.random.PCG64DXSM(seed))":
        raise Study1InputError("unexpected frozen bootstrap generator")
    if randomness["bootstrap"]["interval"] != "percentile_2.5_97.5":
        raise Study1InputError("unexpected frozen bootstrap interval")
    if budget.get("status") != "resolved":
        raise Study1InputError("label_budget_stability is not resolved")
    # The plateau threshold and H4 bound exist only in the registered rule text; bind
    # the constants below to that exact text so a different freeze cannot reuse them.
    rule = budget.get("selection_rule", "")
    if "median SD <= 0.03" not in rule or "H4 supported iff K* <= 10" not in rule:
        raise Study1InputError("label-budget plateau rule differs from the implemented one")
    if "divided by sqrt((N-K)/(N-1))" not in budget.get("stability_metric", ""):
        raise Study1InputError("label-budget stability metric differs from the implemented one")
    if int(contract["bootstrap_resamples"]) < int(contract["minimum_bootstrap_resamples"]):
        raise Study1InputError("frozen bootstrap resamples are below the registered minimum")
    return {
        "base_seed": int(randomness["base_seed"]),
        "bootstrap_resamples": int(contract["bootstrap_resamples"]),
        "confidence_level": float(protocol["statistical_plan"]["confidence_level"]),
        "label_budget_k": [int(k) for k in contract["label_budget_k"]],
        "label_budget_repeats": int(budget["repeat_count"]),
        "label_budget_plateau_sd": 0.03,
        "label_budget_supported_max_k": 10,
        "support": support_parameters(protocol),
    }


def refuse_tmpfs(path: Path) -> None:
    """Study-1 outputs are several GB; a RAM-backed tmpfs would consume worker memory."""

    target = Path(path).expanduser().resolve()
    best, kind = "", None
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return
    for line in lines:
        fields = line.split()
        mount = Path(fields[1]) if len(fields) >= 3 else None
        if mount and (target == mount or mount in target.parents) and len(fields[1]) > len(best):
            best, kind = fields[1], fields[2]
    if kind in {"tmpfs", "ramfs"}:
        raise Study1InputError(f"output {target} is on {kind} ({best}); use the ext4 output disk")
