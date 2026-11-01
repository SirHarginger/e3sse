"""The scientific freeze never waits for OSF; registration is a bound addendum."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_protocol_safeguards import _clean_repository, _production_fixture

from e3sse.protocol.freeze import materialize_protocol
from e3sse.protocol.registration import (
    build_registration_addendum,
    protocol_file_sha256,
    registration_status,
    validate_registration_addendum,
)

SUBMITTED = "2026-10-20T09:00:00Z"


def _record_registration_main():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "record_registration", Path("scripts/record_registration.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _frozen(tmp_path, monkeypatch):
    _clean_repository(monkeypatch)
    config = _production_fixture(tmp_path, complete=True)
    protocol, path = materialize_protocol(config)
    return config, protocol, path


def test_scientific_protocol_freezes_without_any_osf_identifier(tmp_path, monkeypatch):
    _config, protocol, path = _frozen(tmp_path, monkeypatch)
    assert protocol["freeze_status"] == "frozen"
    assert protocol["freeze_blockers"] == []
    text = json.dumps(protocol)
    assert "osf.io/" not in text and "osf_registration_id" not in text
    assert "registration_status" not in text
    assert "preregistration" not in json.dumps(protocol["unresolved_items"])
    assert registration_status(protocol) == "protocol_frozen_registration_pending"
    # protocol.json bytes are exactly what the addendum hashes.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol_file_sha256(protocol)


def test_recording_registration_later_leaves_the_protocol_id_unchanged(tmp_path, monkeypatch):
    record = _record_registration_main()
    config, protocol, path = _frozen(tmp_path, monkeypatch)
    before = path.read_bytes()
    common = ["--protocol", str(path)]
    assert record([*common, "--status", "submitted", "--submitted-at-utc", SUBMITTED]) == 0
    assert record(
        [
            *common,
            "--status",
            "registered",
            "--osf-id",
            "fixture12",
            "--submitted-at-utc",
            SUBMITTED,
            "--visibility",
            "embargoed",
        ]
    ) == 0
    registered = json.loads((path.parent / "registration_registered.json").read_text())
    assert registered["parent_protocol_id"] == protocol["protocol_id"]
    assert registered["osf_url"] == "https://osf.io/fixture12"
    assert registration_status(protocol, registered) == "protocol_registered"
    assert path.read_bytes() == before
    resumed, resumed_path = materialize_protocol(config, resume=True)
    assert resumed["protocol_id"] == protocol["protocol_id"]
    assert resumed_path.read_bytes() == before
    # The same status is never overwritten with different content.
    with pytest.raises(SystemExit, match="use --resume"):
        record([*common, "--status", "submitted", "--submitted-at-utc", SUBMITTED])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        record(
            [*common, "--status", "submitted", "--submitted-at-utc", "2026-10-21T09:00:00Z",
             "--resume"]
        )


def test_registration_must_reference_the_exact_frozen_protocol(tmp_path, monkeypatch):
    _config, protocol, _path = _frozen(tmp_path, monkeypatch)
    addendum = build_registration_addendum(
        protocol,
        status="registered",
        osf_registration_id="fixture12",
        submitted_at_utc=SUBMITTED,
        visibility="public",
        doi="10.17605/OSF.IO/FIXTURE12",
    )
    validate_registration_addendum(addendum, protocol)
    other = copy.deepcopy(protocol)
    other["known_deviations"] = [*other["known_deviations"], "different content"]
    content = {key: value for key, value in other.items() if key != "protocol_id"}
    other["protocol_id"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="different protocol ID"):
        validate_registration_addendum(addendum, other)

    def rehashed(**changes):
        forged = {**addendum, **changes}
        content = {key: value for key, value in forged.items() if key != "addendum_id"}
        forged["addendum_id"] = hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return forged

    with pytest.raises(ValueError, match="different protocol file hash"):
        validate_registration_addendum(rehashed(protocol_sha256="0" * 64), protocol)
    with pytest.raises(ValueError, match="different Git SHA"):
        validate_registration_addendum(rehashed(git_sha="b" * 40), protocol)
    with pytest.raises(ValueError, match="content hash"):
        validate_registration_addendum({**addendum, "visibility": "embargoed"}, protocol)
    with pytest.raises(ValueError, match="OSF registration identifier"):
        build_registration_addendum(
            protocol, status="registered", submitted_at_utc=SUBMITTED, visibility="public"
        )
    with pytest.raises(ValueError, match="pending submission"):
        build_registration_addendum(
            protocol, status="pending_submission", osf_registration_id="fixture12"
        )
    unfrozen = {**protocol, "freeze_status": "candidate_unfrozen"}
    with pytest.raises(ValueError):
        registration_status(unfrozen)


def test_pilot_update_record_needs_a_submitted_record_and_a_hash_id(tmp_path, monkeypatch):
    _config, protocol, _path = _frozen(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="pilot amendment update"):
        build_registration_addendum(
            protocol, status="pending_submission", pilot_amendment_id="a" * 64
        )
    with pytest.raises(ValueError, match="pilot amendment update"):
        build_registration_addendum(
            protocol,
            status="registered",
            osf_registration_id="fixture12",
            submitted_at_utc=SUBMITTED,
            visibility="embargoed",
            pilot_amendment_id="not-a-hash",
        )
    update = build_registration_addendum(
        protocol,
        status="registered",
        osf_registration_id="fixture12",
        submitted_at_utc=SUBMITTED,
        visibility="embargoed",
        pilot_amendment_id="a" * 64,
    )
    assert registration_status(protocol, update) == "protocol_registered"


def test_record_refuses_a_protocol_file_that_is_not_the_builders_bytes(tmp_path, monkeypatch):
    record = _record_registration_main()
    _config, protocol, _path = _frozen(tmp_path, monkeypatch)
    edited = tmp_path / "edited" / "protocol.json"
    edited.parent.mkdir()
    edited.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")  # same content
    with pytest.raises(SystemExit, match="not byte-identical"):
        record(["--protocol", str(edited), "--status", "pending_submission"])
    assert not list(edited.parent.glob("registration_*.json"))
