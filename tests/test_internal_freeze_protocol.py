"""Reliability-gated internal-freeze protocol layered on the unchanged registration-gated parent."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from test_protocol_safeguards import _clean_repository, _production_fixture

from e3sse.protocol.contracts import (
    OutcomeAccessDenied,
    build_pilot_amendment,
    require_outcome_access,
)
from e3sse.protocol.freeze import materialize_protocol
from e3sse.protocol.internal_freeze import (
    InternalFreezeError,
    build_internal_freeze_protocol,
    build_internal_pilot_record,
    internal_freeze_blockers,
    internal_freeze_provenance,
    local_engineering_blockers,
    validate_internal_freeze_config,
    validate_internal_freeze_protocol,
)
from e3sse.protocol.registration import protocol_file_sha256

REPOSITORY = Path(__file__).resolve().parents[1]
CHECKED_IN = REPOSITORY / "configs" / "reliability_gated_internal_freeze_protocol.json"
SHA = "b" * 40


def _parent(tmp_path, monkeypatch):
    _clean_repository(monkeypatch)
    protocol, path = materialize_protocol(_production_fixture(tmp_path, complete=True))
    membership = json.loads((path.parent / "partition_membership.json").read_text())["records"]
    return protocol, path, membership


def _config(parent: dict, *, resolved: bool = True) -> dict:
    config = json.loads(CHECKED_IN.read_text())
    config["parent_protocol"]["protocol_id"] = parent["protocol_id"]
    config["parent_protocol"]["protocol_sha256"] = protocol_file_sha256(parent)
    if resolved:
        for item in config["open_items"]:
            item["blocking"] = False
            item.setdefault("resolution", {"status": "resolved", "rule": "synthetic test",
                                           "evidence": "test fixture"})
    return config


def _frozen_internal_protocol(parent: dict) -> dict:
    return build_internal_freeze_protocol(parent, _config(parent), git_sha=SHA)


def _nonpath_frames(membership):
    return [r["frame_id"] for r in membership if r.get("frame_id") and r.get("raw_file_kind") == "source"]


def test_checked_in_config_is_internal_freeze_and_blocked_until_reviews():
    config = json.loads(CHECKED_IN.read_text())
    validate_internal_freeze_config(config)
    governance = config["governance"]
    assert governance["governance_mode"] == "internal_version_controlled_freeze"
    assert governance["external_registration_required"] is False
    assert governance["preregistration_claimed"] is False
    # Engineering checkpoint rule: no blocking local item; the freeze still needs server evidence.
    assert local_engineering_blockers(config) == []
    assert {"rate_limiting_hop_algorithm", "bvse_reproduction", "descriptor_availability",
            "path_representation_mapping", "development_manifest"} <= set(
        internal_freeze_blockers(config))
    # Embedding evidence is pinned (path + SHA-256), so its probe items no longer block.
    pins = config["amendments"]["representation"]["embedding_capability_evidence"]["reports"]
    assert len(pins) == 6 and all(len(pin["sha256"]) == 64 for pin in pins.values())
    assert not {"embedding_probe_server_run", "embedding_extraction_verification",
                "applicability_fallback_hierarchy"} & set(internal_freeze_blockers(config))
    # Remaining blockers are server evidence or recorded owner decisions, never local work.
    assert all(i["blocking"] is False or i["blocker_class"] in ("server_evidence", "owner_decision")
               for i in config["open_items"])
    # Owner decisions of 2026-11-02 are recorded as resolved protocol rules.
    assert not {"p_population_split_membership", "weight_slice_definition_order",
                "ase_single_version"} & set(internal_freeze_blockers(config))
    # The parent pins are the registration-gated historical identity used by Study 1.
    study1 = json.loads((REPOSITORY / "configs" / "study1_softening_atlas.json").read_text())
    assert config["parent_protocol"]["protocol_id"] == study1["frozen_protocol"]["protocol_id"]
    assert (config["parent_protocol"]["protocol_sha256"]
            == study1["frozen_protocol"]["protocol_json_sha256"])


def test_internal_freeze_leaves_parent_bytes_and_id_unchanged_and_is_content_addressed(
    tmp_path, monkeypatch
):
    parent, path, _ = _parent(tmp_path, monkeypatch)
    before = path.read_bytes()
    internal_protocol = _frozen_internal_protocol(parent)
    assert path.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == protocol_file_sha256(parent)
    assert internal_protocol["parent_protocol_id"] == parent["protocol_id"]
    assert internal_protocol["supersedes_protocol_id"] == parent["protocol_id"]
    assert internal_protocol["parent_protocol_sha256"] == protocol_file_sha256(parent)
    assert "version" not in json.dumps(sorted(internal_protocol))
    assert internal_protocol["protocol_id"] != parent["protocol_id"]
    assert internal_protocol["freeze_status"] == "frozen"
    assert _frozen_internal_protocol(parent)["protocol_id"] == internal_protocol["protocol_id"]
    changed = _config(parent)
    changed["amendments"]["deployment_gate"]["quantile_level"] = 0.9
    other = build_internal_freeze_protocol(parent, changed, git_sha=SHA)
    assert other["protocol_id"] != internal_protocol["protocol_id"]
    validate_internal_freeze_protocol(internal_protocol, parent)


def test_internal_freeze_training_access_needs_no_osf_while_parent_path_still_does(tmp_path, monkeypatch):
    parent, _, membership = _parent(tmp_path, monkeypatch)
    frames = _nonpath_frames(membership)
    kwargs = {"role": "train", "outcome": "study1", "frozen_protocol": parent,
              "partition_records": membership, "unit_ids": frames,
              "outcome_kind": "mplitrj_nonpath_dft_force"}
    with pytest.raises(OutcomeAccessDenied, match="registered on OSF"):
        require_outcome_access(**kwargs)
    with pytest.raises(OutcomeAccessDenied, match="pre-specified, not pre-registered"):
        require_outcome_access(**kwargs, internal_freeze_protocol=_frozen_internal_protocol(parent))
    with pytest.raises(OutcomeAccessDenied, match="requires the internal-freeze protocol"):
        require_outcome_access(**{**kwargs, "purpose": "prespecified_analysis"})
    kwargs["purpose"] = "prespecified_analysis"
    require_outcome_access(**kwargs, internal_freeze_protocol=_frozen_internal_protocol(parent))
    provenance = internal_freeze_provenance(_frozen_internal_protocol(parent))
    assert provenance["governance"].startswith("internal_freeze:")
    assert provenance["preregistration_claimed"] is False
    # Unverified units stay sealed under internal freeze exactly as under the parent.
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(**{**kwargs, "partition_records": None},
                               internal_freeze_protocol=_frozen_internal_protocol(parent))


def test_unfrozen_tampered_or_foreign_internal_protocols_grant_nothing(tmp_path, monkeypatch):
    parent, _, membership = _parent(tmp_path, monkeypatch)
    kwargs = {"role": "train", "outcome": "study1", "frozen_protocol": parent,
              "partition_records": membership, "unit_ids": _nonpath_frames(membership),
              "purpose": "prespecified_analysis"}
    candidate = build_internal_freeze_protocol(parent, _config(parent, resolved=False), git_sha=SHA)
    assert candidate["freeze_status"] == "candidate_unfrozen"
    assert candidate["freeze_blockers"]
    synthetic = build_internal_freeze_protocol(parent, _config(parent), git_sha=SHA,
                                        materialization_mode="synthetic")
    tampered = copy.deepcopy(_frozen_internal_protocol(parent))
    tampered["amendments"]["deployment_gate"]["quantile_level"] = 0.99
    flags = copy.deepcopy(_frozen_internal_protocol(parent))
    flags["governance"]["preregistration_claimed"] = 0
    for internal_protocol, message in ((candidate, "frozen production"), (synthetic, "frozen production"),
                              (tampered, "content hash"), (flags, "content hash")):
        with pytest.raises(OutcomeAccessDenied, match=message):
            require_outcome_access(**kwargs, internal_freeze_protocol=internal_protocol)
    foreign = copy.deepcopy(parent)
    foreign["git_sha"] = "c" * 40
    with pytest.raises(ValueError):
        build_internal_freeze_protocol(foreign, _config(parent), git_sha=SHA)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("deployment_gate", "barrier_labels_used"), True, "barrier_labels_used"),
        (("deployment_gate", "quantile_level"), 1.5, "quantile_level"),
        (("applicability", "diagnostics", "combined", "learned_weights"), True, "learned weights"),
        (("representation", "projections", "quantitative_use"), "allowed", "projections"),
        (("transfer_benchmark", "methods", "B6_oracle_own_path", "role"), "deployable", "oracle"),
    ],
)
def test_internal_freeze_config_cannot_relax_scientific_invariants(tmp_path, monkeypatch, path, value,
                                                            message):
    parent, _, _ = _parent(tmp_path, monkeypatch)
    config = _config(parent)
    node = config["amendments"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(InternalFreezeError, match=message):
        build_internal_freeze_protocol(parent, config, git_sha=SHA)


@pytest.mark.parametrize("key", ["external_registration_required", "preregistration_claimed"])
def test_governance_flags_are_exact_booleans(key):
    config = json.loads(CHECKED_IN.read_text())
    config["governance"][key] = 0
    with pytest.raises(InternalFreezeError, match=key):
        validate_internal_freeze_config(config)


def test_internal_freeze_test_outcomes_need_pilot_amendment_bound_by_record(tmp_path, monkeypatch):
    parent, _, membership = _parent(tmp_path, monkeypatch)
    internal_protocol = _frozen_internal_protocol(parent)
    train_hops = sorted({r["hop_id"] for r in membership if r["published_split"] == "train"})
    kwargs = {"role": "test", "outcome": "stage2a", "frozen_protocol": parent,
              "internal_freeze_protocol": internal_protocol, "purpose": "prespecified_analysis"}
    with pytest.raises(OutcomeAccessDenied, match="pilot amendment"):
        require_outcome_access(**kwargs)
    amendment = build_pilot_amendment(
        parent, partition_records=membership, training_unit_ids=train_hops,
        frozen_thresholds={"study2_noise_floor_eV": 0.01},
        mde_analysis={"study2_mde_eV": "reported"},
    )
    with pytest.raises(OutcomeAccessDenied, match="internal pilot record"):
        require_outcome_access(**kwargs, pilot_amendment=amendment)
    record = build_internal_pilot_record(internal_protocol, parent, amendment)
    require_outcome_access(**kwargs, pilot_amendment=amendment, internal_pilot_record=record)
    stale = {**record, "internal_freeze_protocol_id": "d" * 64}
    with pytest.raises(OutcomeAccessDenied, match="different internal-freeze protocol"):
        require_outcome_access(**kwargs, pilot_amendment=amendment, internal_pilot_record=stale)
    # FPMD still needs the transport addendum.
    with pytest.raises(OutcomeAccessDenied, match="transport"):
        require_outcome_access(**{**kwargs, "role": "fpmd_reference"}, pilot_amendment=amendment,
                               internal_pilot_record=record)


def _cli():
    spec = importlib.util.spec_from_file_location(
        "build_internal_freeze_protocol", REPOSITORY / "scripts" / "build_internal_freeze_protocol.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _study1_and_server(tmp_path, root):
    study1 = tmp_path / "study1_config.json"
    study1.write_text(json.dumps({"output_root": str(root)}))
    server = tmp_path / "server_config.json"
    server.write_text(json.dumps({"paths": {"data_root": str(tmp_path / "data")}}))
    return study1, server


def test_cli_writes_once_verifies_on_resume_and_never_touches_parent(tmp_path, monkeypatch):
    parent, path, _ = _parent(tmp_path, monkeypatch)
    cli = _cli()
    monkeypatch.setattr(cli, "_validate_repository_state",
                        lambda _r, *, expected_sha, require_clean: (SHA, "main", False))
    config_path = tmp_path / "internal_freeze.json"
    config_path.write_text(json.dumps(_config(parent)))
    before = path.read_bytes()
    study1, server = _study1_and_server(tmp_path, tmp_path / "study1_outputs")
    args = ["build", "--config", str(config_path), "--parent-protocol", str(path),
            "--output-root", str(tmp_path / "internal_freeze"), "--study1-config", str(study1),
            "--server-config", str(server)]
    assert cli.main(args) == 1  # production needs --git-sha
    args += ["--git-sha", SHA]
    assert cli.main(args) == 0
    assert cli.main(args) == 1  # exists, no --resume
    assert cli.main([*args, "--resume"]) == 0
    assert path.read_bytes() == before
    written = list((tmp_path / "internal_freeze").glob("*/internal_freeze_protocol.json"))
    assert len(written) == 1
    validate_internal_freeze_protocol(json.loads(written[0].read_text()), parent)
    unsafe = [*args[:6], str(tmp_path / "data" / "x"), *args[7:]]
    assert cli.main(unsafe) == 1  # output under the data root
    path.write_text(path.read_text().rstrip("\n"))
    assert cli.main([*args, "--resume"]) == 1

def test_declaration_is_checked_against_every_study1_protocol_directory(tmp_path, monkeypatch):
    cli = _cli()
    root = tmp_path / "study1_outputs"
    study1, _ = _study1_and_server(tmp_path, root)
    absent = cli.outcome_access_evidence(study1)
    assert absent["checked"] and absent["root_exists"] is False
    record = root / "any-protocol" / "inference" / "CHGNet-0.3.0" / "outcome_access" / "x.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}")
    with pytest.raises(InternalFreezeError, match="falsified"):
        cli.outcome_access_evidence(study1)
    record.unlink()
    present = cli.outcome_access_evidence(study1)
    assert present["root_exists"] and len(present["listing_sha256"]) == 64

def test_cleared_blockers_need_a_trace_and_pilot_cannot_override_frozen_amendments(tmp_path, monkeypatch):
    parent, _, membership = _parent(tmp_path, monkeypatch)
    config = _config(parent)
    item = next(i for i in config["open_items"] if i["id"] == "ml_neb_settings")
    item.pop("resolution", None)
    item.pop("non_blocking_reason", None)
    item["blocking"] = False
    item_without_trace = copy.deepcopy(config)
    for other in item_without_trace["open_items"]:
        if other["id"] != "ml_neb_settings":
            other.setdefault("non_blocking_reason", "test")
    with pytest.raises(InternalFreezeError, match="ml_neb_settings"):
        validate_internal_freeze_config(item_without_trace)
    internal_protocol = _frozen_internal_protocol(parent)
    train_hops = sorted({r["hop_id"] for r in membership if r["published_split"] == "train"})
    clash = build_pilot_amendment(
        parent, partition_records=membership, training_unit_ids=train_hops,
        frozen_thresholds={"study2_applicability_gate_quantile": 0.9},
        mde_analysis={"study2_mde_eV": "reported"},
    )
    with pytest.raises(InternalFreezeError, match="override internal-freeze"):
        build_internal_pilot_record(internal_protocol, parent, clash)
    harmless = build_pilot_amendment(
        parent, partition_records=membership, training_unit_ids=train_hops,
        frozen_thresholds={"study2_aggregate_noise_floor_eV": 0.01},
        mde_analysis={"study2_mde_eV": "reported"},
    )
    build_internal_pilot_record(internal_protocol, parent, harmless)


def test_validation_rederives_blockers_and_invariants_not_just_the_hash(tmp_path, monkeypatch):
    from e3sse.protocol.internal_freeze import _content_id

    parent, _, _ = _parent(tmp_path, monkeypatch)
    forged = copy.deepcopy(_frozen_internal_protocol(parent))
    forged["open_items"][0]["blocking"] = True
    forged["protocol_id"] = _content_id(forged, "protocol_id")
    with pytest.raises(InternalFreezeError, match="blocking open items"):
        validate_internal_freeze_protocol(forged, parent)
    relaxed = copy.deepcopy(_frozen_internal_protocol(parent))
    relaxed["amendments"]["deployment_gate"]["barrier_labels_used"] = True
    relaxed["protocol_id"] = _content_id(relaxed, "protocol_id")
    with pytest.raises(InternalFreezeError, match="barrier_labels_used"):
        validate_internal_freeze_protocol(relaxed, parent)


def _development_manifest(hop_ids):
    content = {"schema": "e3sse.development-manifest", "hop_ids": sorted(hop_ids)}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()
    return {**content, "manifest_sha256": digest}, digest


def test_development_model_selection_reads_only_pinned_development_hops(tmp_path, monkeypatch):
    parent, _, membership = _parent(tmp_path, monkeypatch)
    test_hops = sorted({r["hop_id"] for r in membership if r["published_split"] == "test"})
    train_hops = sorted({r["hop_id"] for r in membership if r["published_split"] == "train"})
    manifest, digest = _development_manifest(test_hops)
    config = _config(parent)
    config["amendments"]["prospective"]["development_manifest_sha256"] = digest
    internal = build_internal_freeze_protocol(parent, config, git_sha=SHA)
    amendment = build_pilot_amendment(
        parent, partition_records=membership, training_unit_ids=train_hops,
        frozen_thresholds={"study2_noise_floor_eV": 0.01},
        mde_analysis={"study2_mde_eV": "reported"})
    record = build_internal_pilot_record(internal, parent, amendment)
    kwargs = {"role": "test", "outcome": "development_model_selection",
              "frozen_protocol": parent, "purpose": "prospective_model_selection",
              "pilot_amendment": amendment, "internal_pilot_record": record,
              "unit_ids": test_hops}
    require_outcome_access(**kwargs, internal_freeze_protocol=internal,
                           development_manifest=manifest)
    with pytest.raises(OutcomeAccessDenied, match="requires the internal-freeze"):
        require_outcome_access(**kwargs, development_manifest=manifest)
    with pytest.raises(OutcomeAccessDenied, match="restricted to the pinned"):
        require_outcome_access(**{**kwargs, "unit_ids": test_hops + ["not-a-development-hop"]},
                               internal_freeze_protocol=internal, development_manifest=manifest)
    tampered = {**manifest, "hop_ids": test_hops + ["extra"]}
    with pytest.raises(OutcomeAccessDenied, match="not the one pinned"):
        require_outcome_access(**kwargs, internal_freeze_protocol=internal,
                               development_manifest=tampered)


def test_study1_runs_under_internal_freeze_without_registration(tmp_path, monkeypatch):
    from e3sse.softening import study1_inputs

    parent, _, membership = _parent(tmp_path, monkeypatch)
    internal = _frozen_internal_protocol(parent)
    directory = tmp_path / "internal"
    directory.mkdir()
    artifact = directory / "internal_freeze_protocol.json"
    artifact.write_text(json.dumps(internal))
    pin = {"protocol_id": internal["protocol_id"], "directory": str(directory),
           "file_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    bound = study1_inputs.internal_freeze_binding({"internal_freeze_protocol": pin}, parent)
    assert bound["protocol_id"] == internal["protocol_id"]
    frames = _nonpath_frames(membership)
    study1_inputs.require_study1_outcome_access(parent, None, membership, frames,
                                                internal_freeze_protocol=bound)
    with pytest.raises(study1_inputs.Study1InputError, match="registered on OSF"):
        study1_inputs.require_study1_outcome_access(parent, None, membership, frames)
    provenance = study1_inputs.outcome_governance_provenance(bound)
    assert provenance["preregistration_claimed"] is False
    assert study1_inputs.internal_freeze_binding({}, parent) is None
    artifact.write_text(json.dumps({**internal, "git_sha": "c" * 40}))
    with pytest.raises(study1_inputs.Study1InputError, match="pinned SHA-256"):
        study1_inputs.internal_freeze_binding({"internal_freeze_protocol": pin}, parent)
