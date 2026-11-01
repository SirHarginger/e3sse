from __future__ import annotations

import hashlib
import json

import pytest

from e3sse.protocol.linkage import (
    EvaluationLinkageError,
    combine_partition_universe,
    derive_nebdft2k_linkage,
    validate_nebdft2k_linkage,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _payload() -> dict:
    source = {
        "edge_id": "mp-1_0_1_0_0_0",
        "material_id": "mp-1",
        "chemical_system_id": "O-Li",
        "published_split": "train",
        "eligibility_status": "included",
        "exclusion_reason": None,
        "source_identifier": "nebDFT2k/mp-1_0_1_0_0_0",
        "source_sha256": "a" * 64,
    }
    normalized = {
        "record_kind": "nebdft2k_hop",
        "frame_id": None,
        "hop_id": source["edge_id"],
        "material_id": source["material_id"],
        "chemical_system_id": "Li-O",
        "published_split": "train",
        "eligibility_status": "included",
        "exclusion_reason": None,
        "source_identifier": source["source_identifier"],
        "source_sha256": source["source_sha256"],
    }
    return {
        "schema_version": "e3sse.nebdft2k.partition-linkage.v1",
        "contains_outcomes": False,
        "source_g0_run_id": "g0",
        "source_input_identity_digest": "digest",
        "chemical_system_algorithm": "sorted-unique-elements-v1",
        "record_count": 1,
        "records_sha256": _sha([normalized]),
        "records": [source],
    }


def test_outcome_free_nebdft2k_linkage_is_normalized_and_bound():
    rows = validate_nebdft2k_linkage(_payload(), g0_run_id="g0", g0_input_identity_digest="digest")
    assert rows[0]["chemical_system_id"] == "Li-O"
    assert rows[0]["hop_id"] == "mp-1_0_1_0_0_0"


@pytest.mark.parametrize("mutation", ["outcomes", "g0", "hash"])
def test_nebdft2k_linkage_rejects_untrusted_or_outcome_bearing_input(mutation):
    payload = _payload()
    if mutation == "outcomes":
        payload["contains_outcomes"] = True
    elif mutation == "g0":
        payload["source_g0_run_id"] = "wrong"
    else:
        payload["records_sha256"] = "0" * 64
    with pytest.raises(EvaluationLinkageError):
        validate_nebdft2k_linkage(payload, g0_run_id="g0", g0_input_identity_digest="digest")


def test_cross_dataset_material_split_propagation_is_fail_closed():
    neb = validate_nebdft2k_linkage(_payload(), g0_run_id="g0", g0_input_identity_digest="digest")
    mpl = [
        {
            "frame_id": "train.xyz:0",
            "hop_id": "mp-1_0_1_0_0_0",
            "material_id": "mp-1",
            "chemical_system_id": "Li-O",
            "published_split": "train",
        }
    ]
    combined, evidence = combine_partition_universe(mpl, neb)
    assert len(combined) == 2
    assert evidence["counts"]["nebdft2k_hops"] == 1
    assert evidence["counts"]["mplitrj_frames"] == 1
    mpl[0]["published_split"] = "test"
    with pytest.raises(EvaluationLinkageError, match="propagation failed"):
        combine_partition_universe(mpl, neb)


def test_nebdft2k_linkage_derivation_reads_identifiers_only(tmp_path):
    index = tmp_path / "raw" / "nebDFT2k" / "index.csv"
    index.parent.mkdir(parents=True)
    index.write_text(
        "material_id,edge_id,chemsys,_split,em_dft\nmp-1,mp-1_0_1_0_0_0,Li-O,train,0.41\n",
        encoding="utf-8",
    )
    stat = index.stat()
    identity = {
        "path": str(index),
        "identifier": "raw/nebDFT2k/index.csv",
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
    }
    payload = derive_nebdft2k_linkage(
        [identity],
        data_root=tmp_path,
        g0_run_id="g0",
        g0_input_identity_digest="digest",
    )
    assert payload["contains_outcomes"] is False
    assert set(payload["records"][0]) == {
        "edge_id",
        "material_id",
        "chemical_system_id",
        "published_split",
        "eligibility_status",
        "exclusion_reason",
        "source_identifier",
        "source_sha256",
    }
    validate_nebdft2k_linkage(payload, g0_run_id="g0", g0_input_identity_digest="digest")
