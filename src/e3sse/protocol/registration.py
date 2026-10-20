"""Registration provenance addendum for a frozen scientific protocol.

State flow: scientific decisions + production evidence -> content-derived frozen
protocol ID -> OSF submission of that exact protocol -> OSF identifier/DOI -> this
addendum.  The addendum binds the frozen protocol's ID, the SHA-256 of its exact
``protocol.json`` bytes, and its Git SHA; it is never part of the protocol content,
so recording or updating a registration cannot change the scientific protocol ID.
OSF identifiers are validated for form only and are never generated here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from .contracts import validate_protocol_freeze_for_outcome_access

REGISTRATION_ADDENDUM_SCHEMA = "e3sse.protocol.registration-addendum.v1"
REGISTRATION_STATUSES = ("pending_submission", "submitted", "registered")
VISIBILITIES = ("embargoed", "public")
_OSF_ID = re.compile(r"[a-z0-9]{5,}")
_DOI = re.compile(r"10\.\d{4,9}/\S+")


def protocol_file_sha256(protocol: Mapping[str, Any]) -> str:
    """SHA-256 of ``protocol.json`` exactly as the builder writes it.

    Computed from the parsed protocol; ``scripts/record_registration.py`` checks that
    it equals the SHA-256 of the file bytes on disk before recording anything.
    """

    payload = json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _content_id(value: Mapping[str, Any]) -> str:
    content = {key: item for key, item in value.items() if key != "addendum_id"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def build_registration_addendum(
    protocol: Mapping[str, Any],
    *,
    status: str,
    osf_registration_id: str | None = None,
    doi: str | None = None,
    submitted_at_utc: str | None = None,
    visibility: str | None = None,
    pilot_amendment_id: str | None = None,
) -> dict[str, Any]:
    """Record the registration state of exactly this frozen protocol.

    ``pilot_amendment_id`` marks an OSF update depositing the training-only pilot
    amendment; test outcomes need a registered record naming that exact amendment.
    """

    addendum = {
        "schema_version": REGISTRATION_ADDENDUM_SCHEMA,
        "parent_protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_file_sha256(protocol),
        "git_sha": protocol["git_sha"],
        "registration_status": status,
        "osf_registration_id": osf_registration_id,
        "osf_url": f"https://osf.io/{osf_registration_id}" if osf_registration_id else None,
        "doi": doi,
        "submitted_at_utc": submitted_at_utc,
        "visibility": visibility,
        "pilot_amendment_id": pilot_amendment_id,
    }
    addendum["addendum_id"] = _content_id(addendum)
    validate_registration_addendum(addendum, protocol)
    return addendum


def validate_registration_addendum(addendum: Any, protocol: Mapping[str, Any]) -> None:
    """Reject an addendum that does not reference this exact frozen protocol."""

    validate_protocol_freeze_for_outcome_access(protocol)
    if not isinstance(addendum, Mapping):
        raise TypeError("registration addendum must be an object")
    if addendum.get("schema_version") != REGISTRATION_ADDENDUM_SCHEMA:
        raise ValueError("unsupported registration addendum schema")
    if addendum.get("parent_protocol_id") != protocol["protocol_id"]:
        raise ValueError("registration addendum references a different protocol ID")
    if addendum.get("protocol_sha256") != protocol_file_sha256(protocol):
        raise ValueError("registration addendum references a different protocol file hash")
    if addendum.get("git_sha") != protocol["git_sha"]:
        raise ValueError("registration addendum references a different Git SHA")
    status = addendum.get("registration_status")
    if status not in REGISTRATION_STATUSES:
        raise ValueError(f"invalid registration status {status!r}")
    osf_id = addendum.get("osf_registration_id")
    if status == "pending_submission":
        if any(addendum.get(f) is not None for f in ("osf_registration_id", "doi", "visibility")):
            raise ValueError("a pending submission cannot carry OSF identifiers")
    else:
        if not _utc(addendum.get("submitted_at_utc")):
            raise ValueError("submitted registrations need an RFC 3339 UTC submission time")
    if status == "registered":
        if not isinstance(osf_id, str) or _OSF_ID.fullmatch(osf_id) is None:
            raise ValueError("a registered protocol needs its OSF registration identifier")
        if addendum.get("visibility") not in VISIBILITIES:
            raise ValueError("a registered protocol needs embargoed/public visibility")
    if osf_id is not None and addendum.get("osf_url") != f"https://osf.io/{osf_id}":
        raise ValueError("OSF URL does not match the registration identifier")
    pilot = addendum.get("pilot_amendment_id")
    if pilot is not None and (
        status == "pending_submission"
        or not isinstance(pilot, str)
        or re.fullmatch(r"[0-9a-f]{64}", pilot) is None
    ):
        raise ValueError("a pilot amendment update needs a submitted record and its SHA-256 ID")
    doi = addendum.get("doi")
    if doi is not None and (not isinstance(doi, str) or _DOI.fullmatch(doi) is None):
        raise ValueError("DOI must have the form 10.<registrant>/<suffix>")
    if addendum.get("addendum_id") != _content_id(addendum):
        raise ValueError("registration addendum content hash mismatch")


def registration_status(
    protocol: Mapping[str, Any], addendum: Mapping[str, Any] | None = None
) -> str:
    """``protocol_frozen_registration_pending`` or ``protocol_registered``."""

    validate_protocol_freeze_for_outcome_access(protocol)
    if addendum is None:
        return "protocol_frozen_registration_pending"
    validate_registration_addendum(addendum, protocol)
    if addendum["registration_status"] == "registered":
        return "protocol_registered"
    return "protocol_frozen_registration_pending"
