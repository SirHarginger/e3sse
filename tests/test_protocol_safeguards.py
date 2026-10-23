from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from protocol_fixtures import production_evidence, unit_evidence_artifact
from test_protocol_configuration import _model, _production_config
from test_protocol_freeze import _phase_b_fixture, _sha, _write

from e3sse.protocol import freeze
from e3sse.protocol.contracts import (
    OutcomeAccessDenied,
    build_pilot_amendment,
    build_transport_addendum,
    require_outcome_access,
    unit_conversion_factor,
    unit_conversion_id,
    validate_unit_contract,
)
from e3sse.protocol.decisions import DECISION_REGISTER, validate_decision_register
from e3sse.protocol.freeze import ProtocolFreezeError, materialize_protocol
from e3sse.protocol.partitions import (
    build_repeated_partitions,
    classify_primary_population,
    derive_study5_exclusions,
    normalize_chemical_system,
)
from e3sse.protocol.registration import build_registration_addendum

_ELEMENTS = ["O", "S", "Cl", "N", "F", "P", "Br", "I", "Se", "B", "Si", "Ge", "Al"]


def _units() -> dict:
    return copy.deepcopy(_production_config()["protocol"]["unit_contract"])


def _universe() -> list[dict]:
    """37 chemical systems, each with several materials, hops, and frames."""

    rows = []
    pairs = [(a, b) for i, a in enumerate(_ELEMENTS) for b in _ELEMENTS[i + 1 :]]
    systems = [normalize_chemical_system(["Li", a]) for a in _ELEMENTS]
    systems += [normalize_chemical_system(["Li", a, b]) for a, b in pairs[:24]]
    assert len(systems) == 37
    for s_index, system in enumerate(systems):
        for m_index in range(1 + s_index % 3):
            material = f"mp-{s_index}-{m_index}"
            for h_index in range(2):
                hop = f"{material}_{h_index}_1_0_0_0"
                for _frame in range(3):
                    rows.append(
                        {
                            "frame_id": f"MPLiTrj_train.xyz:{len(rows)}",
                            "material_id": material,
                            "hop_id": hop,
                            "chemical_system_id": system,
                            "published_split": "train",
                        }
                    )
    return rows


# --- chemical systems and partitions ---------------------------------------------------


def test_chemical_system_normalization_is_deterministic_and_chemically_distinct():
    equivalent = ["Li-O-P", "P-O-Li", "Li3PO4", ["O", "P", "Li", "O"], "Li-P-O"]
    assert {normalize_chemical_system(item) for item in equivalent} == {"Li-O-P"}
    assert normalize_chemical_system("Co-Li-O") != normalize_chemical_system("C-Li-O")
    assert normalize_chemical_system("LiCoO2") == "Co-Li-O"
    assert normalize_chemical_system("Li-S") != normalize_chemical_system("Li-Se")
    for invalid in ("li-o", "Li-Xx", "", "Li-O; rm"):
        with pytest.raises(ValueError):
            normalize_chemical_system(invalid)


def test_ten_grouped_partitions_are_exact_inherited_and_seed_frozen():
    rows = _universe()
    result = build_repeated_partitions(rows, base_seed=2026)
    assert result["repeats"] == len(result["partitions"]) == 10
    assert result["chemical_system_count"] == 37
    seeds = [repeat["resolved_seed"] for repeat in result["partitions"]]
    assert len(set(seeds)) == 10
    assert [repeat["repeat_index"] for repeat in result["partitions"]] == list(range(10))
    test_sets = set()
    for repeat in result["partitions"]:
        train = set(repeat["training_or_calibration_chemical_systems"])
        test = set(repeat["test_chemical_systems"])
        # round(0.40 * 37) = 15 chemical systems; observations never rebalance groups.
        assert (len(train), len(test)) == (22, 15)
        assert train.isdisjoint(test)
        assert train | test == {row["chemical_system_id"] for row in rows}
        assert repeat["unit_overlap_counts"] == {"frames": 0, "materials": 0, "hops": 0}
        side_of = repeat["assignment_by_chemical_system"]
        for unit in ("frame_id", "material_id", "hop_id"):
            sides: dict[str, set[str]] = {}
            for row in rows:
                sides.setdefault(row[unit], set()).add(side_of[row["chemical_system_id"]])
            assert all(len(value) == 1 for value in sides.values())
        test_sets.add(frozenset(test))
    assert len(test_sets) > 1
    assert build_repeated_partitions(list(reversed(rows)), base_seed=2026) == result
    assert build_repeated_partitions(rows, base_seed=2027) != result
    with pytest.raises(ValueError, match="exactly 10"):
        build_repeated_partitions(rows, base_seed=2026, repeats=9)
    with pytest.raises(ValueError, match="0.40"):
        build_repeated_partitions(rows, base_seed=2026, test_fraction=0.3)


def test_study5_excludes_the_whole_partition_universe():
    rows = _universe()
    universe = sorted({row["chemical_system_id"] for row in rows})
    repeated = build_repeated_partitions(rows, base_seed=1)
    published = ["Li-O", "Li-S"]
    result = derive_study5_exclusions(published, repeated, universe)
    union = set(published).union(*(set(r["test_chemical_systems"]) for r in repeated["partitions"]))
    assert union <= set(result["chemical_system_ids"]) == set(universe)
    assert result["rule"] == "entire-partition-universe-v1"
    assert result["candidate_pool_inspected"] is False
    with pytest.raises(ValueError, match="non-empty"):
        derive_study5_exclusions([], repeated, universe)
    with pytest.raises(TypeError):
        derive_study5_exclusions("Li-O", repeated, universe)
    with pytest.raises(ValueError, match="outside the partition universe"):
        derive_study5_exclusions(published, repeated, universe[:3])
    truncated = {**repeated, "partitions": repeated["partitions"][:9]}
    with pytest.raises(ValueError, match="incomplete"):
        derive_study5_exclusions(published, truncated, universe)

def test_primary_population_requires_a_registered_element_set():
    population = classify_primary_population(["Li-O", "Fe-Li-O-P", "Li-S"], ["Fe", "Mn"])
    assert population == {
        "Fe-Li-O-P": "tm_hubbard_u_secondary",
        "Li-O": "primary",
        "Li-S": "primary",
    }
    for invalid in ([], ["Xx"]):
        with pytest.raises(ValueError):
            classify_primary_population(["Li-O"], invalid)


# --- units -----------------------------------------------------------------------------


def _evidence(units, name):
    return units[name]["production_source_unit_evidence"]


def test_only_registered_unit_conversions_resolve():
    units = _units()
    _evidence(units, "dft_energy").update(
        {"source_unit": "hartree", "conversion": unit_conversion_id("hartree", "eV")}
    )
    _evidence(units, "dft_force").update(
        {"source_unit": "hartree/bohr", "conversion": "hartree/bohr->eV/angstrom"}
    )
    validate_unit_contract(units)
    assert unit_conversion_factor("dft_energy", units["dft_energy"]) == pytest.approx(
        27.211386245988
    )
    assert unit_conversion_factor("dft_force", units["dft_force"]) == pytest.approx(
        27.211386245988 / 0.529177210903
    )
    _evidence(units, "dft_force")["conversion"] = "multiply by 51.4"
    with pytest.raises(ValueError, match="dft_force"):
        validate_unit_contract(units)
    units = _units()
    _evidence(units, "temperature").update({"source_unit": "degC", "conversion": "degC->K"})
    with pytest.raises(ValueError, match="temperature"):
        validate_unit_contract(units)
    units = _units()
    _evidence(units, "barrier")["evidence"] = None
    with pytest.raises(ValueError, match="barrier"):
        validate_unit_contract(units)
    units = _units()
    _evidence(units, "stress").update({"status": "not_retained", "evidence": None})
    with pytest.raises(ValueError, match="stress"):
        validate_unit_contract(units)
    _evidence(units, "stress")["evidence"] = "stress not retained by the static studies"
    validate_unit_contract(units)
    _evidence(units, "position").update({"status": "not_retained"})
    with pytest.raises(ValueError, match="position"):
        validate_unit_contract(units)
    with pytest.raises(ValueError, match="unrecognized"):
        unit_conversion_id("kcal/mol", "eV")


def test_canonical_units_are_frozen_and_source_units_are_production_evidenced():
    template = json.loads(Path("configs/protocol_freeze.json").read_text(encoding="utf-8"))
    units = template["protocol"]["unit_contract"]
    assert {name: record["canonical_analysis_unit"] for name, record in units.items()} == {
        "position": "angstrom",
        "cell": "angstrom",
        "dft_energy": "eV",
        "ml_energy": "eV",
        "dft_force": "eV/angstrom",
        "ml_force": "eV/angstrom",
        "stress": "eV/angstrom^3",
        "barrier": "eV",
        "temperature": "K",
    }
    validate_unit_contract(units)
    for name in ("position", "cell", "dft_energy", "dft_force", "stress", "barrier"):
        evidence = units[name]["production_source_unit_evidence"]
        assert evidence["status"] == "verified" and evidence["conversion"] == "identity"
        assert "4b7c4b9c558e5e749cc0065062d9acc63aeccde41a4fca0d451856aa28fa330d" in (
            evidence["evidence"]
        )
    unproven = copy.deepcopy(units)
    unproven["dft_force"]["production_source_unit_evidence"].update(
        {"source_unit": None, "conversion": None, "status": "unresolved", "evidence": None}
    )
    with pytest.raises(ValueError, match="production source-unit evidence unresolved"):
        validate_unit_contract(unproven)
    shifted = copy.deepcopy(units)
    shifted["barrier"]["canonical_analysis_unit"] = "meV"
    with pytest.raises(ValueError, match="canonical analysis units differ"):
        validate_unit_contract(shifted)


# --- decision register -----------------------------------------------------------------


def test_decision_register_uses_the_five_way_vocabulary_with_citations():
    register = validate_decision_register(DECISION_REGISTER)
    classes = {item["classification"] for item in register}
    assert {"paper_specified", "repository_existing", "engineering_choice"} <= classes
    for item in register:
        if item["classification"] == "paper_specified":
            assert "Research Design" in item["source"]
    by_id = {item["id"]: item for item in register}
    assert by_id["partition_rounding_rule"]["classification"] == "engineering_choice"
    assert by_id["robustness_partitions_10_grouped_60_40"]["classification"] == "paper_specified"
    with pytest.raises(ValueError, match="cite the design"):
        validate_decision_register(
            [{"id": "x", "classification": "paper_specified", "source": "x", "statement": "x"}]
        )
    with pytest.raises(ValueError, match="classification"):
        validate_decision_register(
            [{"id": "x", "classification": "assumed", "source": "x", "statement": "x"}]
        )


# --- materialization ---------------------------------------------------------------------


def _clean_repository(monkeypatch):
    monkeypatch.setattr(
        freeze,
        "_validate_repository_state",
        lambda _root, *, expected_sha, require_clean: ("a" * 40, "main", False),
    )


def _unit_evidence_for(config: dict) -> dict:
    """Producer-format unit evidence collected from exactly the fixture's pinned files."""

    root = Path(config["inputs"]["mplitrj_provenance"]["root"])
    sources = json.loads((root / "manifest.json").read_text())["flattened_sources"]
    inventory_path = Path(config["inputs"]["g0_input_manifest"]["path"])
    g0_inputs = json.loads(inventory_path.read_text())
    return unit_evidence_artifact(
        {
            "mplitrj": {s["published_split"]: {"sha256": s["sha256"]} for s in sources},
            "nebdft2k": {
                "index_sha256": g0_inputs["inputs"]["nebDFT2k"][0]["sha256"],
                "index_columns": ["edge_id", "material_id", "chemsys", "_split"],
            },
        },
        inventory_sha256=_sha(inventory_path),
        files=[f"/srv/host/data/{s['published_split']}.xyz" for s in sources],
    )


def _production_fixture(tmp_path: Path, *, complete: bool) -> Path:
    config_path, _audit = _phase_b_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["materialization_mode"] = "production"
    if complete:
        # A complete linkage carries each material's composition (anion families).
        linkage_path = Path(config["inputs"]["material_chemical_system_linkage"]["path"])
        linkage = json.loads(linkage_path.read_text())
        for record in linkage["materials"]:
            elements = record["chemical_system_id"].split("-")
            record["composition"] = {element: 1 for element in elements}
        _write(linkage_path, linkage)
        config["inputs"]["material_chemical_system_linkage"]["sha256"] = _sha(linkage_path)
        reference = _production_config()["protocol"]
        config["inputs"]["repository_git_sha"] = "a" * 40
        config["protocol"]["scientific_resolutions"] = reference["scientific_resolutions"]
        config["protocol"]["unresolved_items"] = reference["unresolved_items"]
        config["protocol"]["model_groups"] = {
            "A": [_model("A", index) for index in range(4)],
            "B1": [_model("B1")],
            "B2": [_model("B2")],
            "C": [],
        }
        config["inputs"]["production_evidence"] = production_evidence(
            [m for group in ("A", "B1", "B2") for m in config["protocol"]["model_groups"][group]],
            root=tmp_path,
            unit_evidence=_unit_evidence_for(config),
        )
    else:
        config["inputs"]["g0_audit"]["sha256"] = None
        config["protocol"]["unit_contract"]["dft_force"]["production_source_unit_evidence"].update(
            {"source_unit": None, "status": "unresolved", "evidence": None}
        )
        config["protocol"]["unresolved_items"] = [
            {
                "id": "s_host_definition",
                "blocking": True,
                "resolution_category": "requires_preanalysis_scientific_decision",
            }
        ]
    _write(config_path, config)
    return config_path


def test_production_candidate_is_materialized_sealed_and_lists_every_blocker(
    tmp_path, monkeypatch
):
    _clean_repository(monkeypatch)
    config = _production_fixture(tmp_path, complete=False)
    inputs = {path: _sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    protocol, path = materialize_protocol(config)
    assert protocol["freeze_status"] == "candidate_unfrozen"
    assert protocol["production_eligible"] is False
    blockers = "\n".join(protocol["freeze_blockers"])
    for expected in (
        "repository_git_sha",
        "g0_audit.sha256",
        "dft_force",
        "s_host_definition",
        "model group A",
    ):
        assert expected in blockers
    assert path.parent.name == protocol["protocol_id"]
    assert all(_sha(source) == digest for source, digest in inputs.items())
    for role in ("test", "validation", "repeated_test", "fpmd_reference", "calibration"):
        with pytest.raises(OutcomeAccessDenied):
            require_outcome_access(role=role, outcome="summary", frozen_protocol=protocol)
    resumed, _ = materialize_protocol(config, resume=True)
    assert resumed == protocol


def test_complete_production_freeze_opens_only_preregistered_static_outcomes(
    tmp_path, monkeypatch
):
    _clean_repository(monkeypatch)
    config = _production_fixture(tmp_path, complete=True)
    protocol, _path = materialize_protocol(config)
    assert protocol["freeze_status"] == "frozen"
    assert protocol["freeze_blockers"] == []
    assert protocol["g0_status"] == "partial"
    assert protocol["static_branch_status"] == "ready"
    assert protocol["transport_branch_status"] == "partial"
    population = protocol["partition_definition"]["primary_population"]
    assert population["status"] == "resolved"
    membership = json.loads((_path.parent / "partition_membership.json").read_text())["records"]
    train_hops = sorted(
        {row["hop_id"] for row in membership if row["published_split"] == "train"}
    )
    test_hops = sorted({row["hop_id"] for row in membership if row["published_split"] == "test"})
    nonpath_frames = [
        row["frame_id"]
        for row in membership
        if row.get("frame_id") and row.get("raw_file_kind") == "source"
    ]
    assert nonpath_frames
    # Registration gates every outcome, training included (Research Design 4, 5.1, 8).
    with pytest.raises(OutcomeAccessDenied, match="registered on OSF"):
        require_outcome_access(
            role="train",
            outcome="pilot",
            frozen_protocol=protocol,
            purpose="threshold_selection",
            partition_records=membership,
            unit_ids=nonpath_frames,
            outcome_kind="mplitrj_nonpath_dft_force",
        )
    registered = build_registration_addendum(
        protocol,
        status="registered",
        osf_registration_id="fixture12",
        submitted_at_utc="2026-10-20T09:00:00Z",
        visibility="embargoed",
    )
    # Seal 1: training outcomes need the freeze and verified membership, not a declared role.
    require_outcome_access(
        role="train",
        outcome="pilot",
        frozen_protocol=protocol,
        registration_addendum=registered,
        purpose="threshold_selection",
        partition_records=membership,
        unit_ids=nonpath_frames,
        outcome_kind="mplitrj_nonpath_dft_force",
    )
    # A permitted label cannot smuggle hops or path frames into tuning.
    path_frames = [r["frame_id"] for r in membership if r.get("raw_file_kind") == "band_optim_step"]
    train_path = [f for f in path_frames if f.startswith("MPLiTrj_train")]
    for units in (train_hops, train_path):
        if not units:
            continue
        with pytest.raises(OutcomeAccessDenied, match="non-path frames"):
            require_outcome_access(
                role="train",
                outcome="pilot",
                frozen_protocol=protocol,
                registration_addendum=registered,
                purpose="method_tuning",
                partition_records=membership,
                unit_ids=units,
                outcome_kind="mplitrj_nonpath_dft_energy",
            )
    # Hop units imply barriers: no amendment, no access, whatever the label says.
    with pytest.raises(OutcomeAccessDenied, match="pilot amendment"):
        require_outcome_access(
            role="train",
            outcome="anything",
            frozen_protocol=protocol,
            registration_addendum=registered,
            partition_records=membership,
            unit_ids=train_hops,
        )
    # nebDFT2k barriers are never tuning inputs, even for published-train hops.
    for kind in ("nebdft2k_barrier_em_dft", None, "mplitrj_path_dft_energy"):
        with pytest.raises(OutcomeAccessDenied, match="may use only"):
            require_outcome_access(
                role="train",
                outcome="pilot",
                frozen_protocol=protocol,
                registration_addendum=registered,
                purpose="threshold_selection",
                partition_records=membership,
                unit_ids=nonpath_frames,
                outcome_kind=kind,
            )
    with pytest.raises(ValueError, match="only Study 1/2"):
        build_pilot_amendment(
            protocol,
            partition_records=membership,
            training_unit_ids=nonpath_frames,
            frozen_thresholds={"study4_alpha": 0.1},
            mde_analysis={"study2_mde_eV": "reported"},
        )
    for units, records in ((test_hops, membership), (train_hops, membership[:-1]), (None, None)):
        with pytest.raises(OutcomeAccessDenied):
            require_outcome_access(
                role="train",
                outcome="pilot",
                frozen_protocol=protocol,
                registration_addendum=registered,
                partition_records=records,
                unit_ids=units,
            )
    # Published-train barriers (recalibration comparator) need the pilot amendment.
    with pytest.raises(OutcomeAccessDenied, match="pilot amendment"):
        require_outcome_access(
            role="train",
            outcome="recalibration_fit",
            frozen_protocol=protocol,
            registration_addendum=registered,
            partition_records=membership,
            unit_ids=train_hops,
            outcome_kind="nebdft2k_barrier_em_dft",
        )
    # Seal 2: test outcomes need the training-only pilot amendment as well.
    with pytest.raises(OutcomeAccessDenied, match="pilot amendment"):
        require_outcome_access(
            role="test",
            outcome="stage2a_rho_srt",
            frozen_protocol=protocol,
            registration_addendum=registered,
        )
    amendment = build_pilot_amendment(
        protocol,
        partition_records=membership,
        training_unit_ids=train_hops,
        frozen_thresholds={"study1_decile_count": 10},
        mde_analysis={"study2_mde_eV": "reported"},
    )
    # The initial registration predates the pilot: test outcomes need the OSF update
    # that deposits this exact pilot amendment.
    with pytest.raises(OutcomeAccessDenied, match="OSF update"):
        require_outcome_access(
            role="test",
            outcome="stage2a_rho_srt",
            frozen_protocol=protocol,
            pilot_amendment=amendment,
            registration_addendum=registered,
        )
    registered = build_registration_addendum(
        protocol,
        status="registered",
        osf_registration_id="fixture12",
        submitted_at_utc="2026-11-02T09:00:00Z",
        visibility="embargoed",
        pilot_amendment_id=amendment["amendment_id"],
    )
    require_outcome_access(
        role="test",
        outcome="stage2a_rho_srt",
        frozen_protocol=protocol,
        pilot_amendment=amendment,
        registration_addendum=registered,
    )
    require_outcome_access(
        role="train",
        outcome="recalibration_fit",
        frozen_protocol=protocol,
        registration_addendum=registered,
        partition_records=membership,
        unit_ids=train_hops,
        outcome_kind="nebdft2k_barrier_em_dft",
        pilot_amendment=amendment,
    )
    for purpose in ("threshold_selection", "method_tuning", "hyperparameter_selection"):
        with pytest.raises(OutcomeAccessDenied, match="training-split"):
            require_outcome_access(
                role="test",
                outcome="barrier_mae",
                frozen_protocol=protocol,
                pilot_amendment=amendment,
                registration_addendum=registered,
                purpose=purpose,
                outcome_kind="mplitrj_nonpath_dft_force",
            )
    # Study 3: transport readiness is a separate addendum bound to the static protocol.
    with pytest.raises(OutcomeAccessDenied, match="transport addendum"):
        require_outcome_access(
            role="fpmd_reference",
            outcome="log10_D",
            frozen_protocol=protocol,
            pilot_amendment=amendment,
            registration_addendum=registered,
        )
    audit_path = tmp_path / "g0" / "audit.json"
    partial = build_transport_addendum(protocol, g0_audit_path=audit_path)
    assert partial["fpmd_g0_status"] == "partial"
    assert partial["g0_audit_sha256"] == _sha(audit_path)
    with pytest.raises(OutcomeAccessDenied, match="not ready"):
        require_outcome_access(
            role="fpmd_reference",
            outcome="log10_D",
            frozen_protocol=protocol,
            pilot_amendment=amendment,
            registration_addendum=registered,
            transport_addendum=partial,
        )
    later_audit = json.loads(audit_path.read_text())
    for dataset in later_audit["datasets"]:
        if dataset["dataset"] == "FPMD":
            dataset.update({"status": "pass", "blockers": []})
    later_path = tmp_path / "g0-later" / "audit.json"
    _write(later_path, later_audit)
    ready = build_transport_addendum(protocol, g0_audit_path=later_path)
    forged = {**partial, "transport_branch_status": "ready", "blockers": []}
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(
            role="fpmd_reference",
            outcome="log10_D",
            frozen_protocol=protocol,
            pilot_amendment=amendment,
            registration_addendum=registered,
            transport_addendum=forged,
        )
    require_outcome_access(
        role="fpmd_reference",
        outcome="log10_D",
        frozen_protocol=protocol,
        pilot_amendment=amendment,
        registration_addendum=registered,
        transport_addendum=ready,
    )
    with pytest.raises(OutcomeAccessDenied, match="purpose"):
        require_outcome_access(role="train", outcome="pilot", purpose="peek")
    tampered = {**protocol, "freeze_blockers": ["late edit"]}
    with pytest.raises(OutcomeAccessDenied):
        require_outcome_access(
            role="test",
            outcome="barrier_mae",
            frozen_protocol=tampered,
            pilot_amendment=amendment,
        )


def test_protocol_snapshot_records_contracts_and_content_identity(tmp_path):
    config, _ = _phase_b_fixture(tmp_path)
    protocol, path = materialize_protocol(config)
    for field in (
        "decision_register",
        "unit_conversion_registry",
        "leakage_rules",
        "study_1_contract",
        "study_2a_contract",
        "statistical_plan",
        "decision_gates",
        "study5_exclusions",
        "future_mlip_inference_contract",
    ):
        assert protocol[field], field
    rules = protocol["leakage_rules"]
    assert rules["eligibility_rule"] == "research_design_3.6_v1"
    assert rules["target_hop"]["endpoint_relaxation_frames"] == "admissible"
    assert rules["target_hop"]["neb_path_frames"] == "prohibited_for_primary_oracle_only"
    assert protocol["canonical_analysis_units"]["barrier"]["classification"] == "paper_specified"
    assert "repeat_fit_guard" in protocol["inference_roles"]
    assert protocol["study_1_contract"]["published_split"] == "train"
    assert protocol["study_1_contract"]["label_budget_k"] == [1, 3, 5, 10, 30]
    subsampling = protocol["study_1_contract"]["label_budget_subsampling"]
    assert subsampling["estimator_scope"] == "material"
    assert "model_id" not in subsampling["seed_key"]  # common random numbers across models
    assert protocol["study_1_contract"]["production_inference_executed"] is False
    assert "created_at_utc" not in json.dumps(protocol)
    manifest = json.loads((path.parent / "run_manifest.json").read_text())
    assert manifest["created_at_utc"]
    assert manifest["protocol_id"] == protocol["protocol_id"]


def test_resume_rejects_changed_config_and_new_inputs_get_a_separate_protocol_id(tmp_path):
    config_path, _ = _phase_b_fixture(tmp_path)
    first, first_path = materialize_protocol(config_path)
    config = json.loads(config_path.read_text())
    config["operator_note"] = "edited after materialization"
    _write(config_path, config)
    with pytest.raises(ProtocolFreezeError, match="stale"):
        materialize_protocol(config_path, resume=True)
    config.pop("operator_note")
    config["protocol"]["known_deviations"].append("second synthetic deviation")
    _write(config_path, config)
    second, second_path = materialize_protocol(config_path)
    assert second["protocol_id"] != first["protocol_id"]
    assert second_path.parent != first_path.parent
    assert json.loads(first_path.read_text()) == first


def test_transport_branch_is_blocked_when_fpmd_fails_g0(tmp_path):
    config_path, audit_path = _phase_b_fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    for dataset in audit["datasets"]:
        if dataset["dataset"] == "FPMD":
            dataset["status"] = "fail"
    _write(audit_path, audit)
    config = json.loads(config_path.read_text())
    config["inputs"]["g0_audit"]["sha256"] = _sha(audit_path)
    _write(config_path, config)
    protocol, _ = materialize_protocol(config_path)
    assert protocol["transport_branch_status"] == "blocked"
    assert protocol["static_branch_status"] == "ready"


def test_checked_in_template_leaves_only_server_only_input_pins_as_blockers():
    from e3sse.protocol.configuration import assess_production_configuration

    template = json.loads(Path("configs/protocol_freeze.json").read_text(encoding="utf-8"))
    state = assess_production_configuration(template)
    assert state.production_eligible is False
    joined = "\n".join(state.freeze_blockers)
    # Only the reviewed commit remains: a commit cannot contain its own SHA, so it is
    # supplied by --git-sha at materialization (the tracked value stays CURRENT).
    assert list(state.freeze_blockers) == ["production repository_git_sha must be pinned"]
    assert template["inputs"]["repository_git_sha"] == "CURRENT"
    inputs = template["inputs"]
    assert inputs["g0_audit"]["sha256"] == (
        "9caca0f02d242af7d70c5149a70ded230f7316d27b410f7cad4755207e40d468"
    )
    assert inputs["g0_input_manifest"]["sha256"] == (
        "5dc454f6c70aa91e26d151118eb8236010c3f28ec1c53767aa574e16049d4c8d"
    )
    assert inputs["mplitrj_provenance"]["manifest_sha256"] == (
        "9fa71143d1bb75281577d0258e9e8c23fe17e1d7b34278c9193fea2d7f5a2491"
    )
    with_commit = copy.deepcopy(template)
    with_commit["inputs"]["repository_git_sha"] = "a" * 40
    assert assess_production_configuration(with_commit).production_eligible is True
    assert "/srv/" not in json.dumps(template["protocol"])  # host paths stay out of identity
    assert template["inputs"]["server_config"]["sha256"] == _sha(Path("configs/server.json"))
    assert "preregistration" not in joined
    for resolved in (
        "transition_metal_hubbard_u_machine_predicate",
        "mixed_anion_family",
        "s_bar_definition",
        "study4_predictor_calibration_independence",
        "g4_monte_carlo_standard_error_definition",
        "model_checkpoint",
        "fpmd_transport_reference_recoverability",
        "group_c_fine_tuned_comparator",
        "ml_energy",
        "temperature",
        "production_source_unit_evidence",
        "nebdft2k_convergence_flag_availability",
        "production_evidence",
    ):
        assert resolved not in joined
    units = template["protocol"]["unit_contract"]
    assert units["temperature"]["production_source_unit_evidence"]["status"] == "deferred_study3"

def test_inference_contract_rejects_per_atom_corrected_and_unpinned_predictions():
    from test_protocol_contracts import _record, _run_manifest

    from e3sse.protocol.contracts import validate_inference_record, validate_inference_run_manifest

    record = _record()
    validate_inference_record(record)
    per_atom = {**record, "raw_energy_kind": "per_atom", "n_atoms": 1, "raw_energy": -0.5}
    with pytest.raises(ValueError, match="total configuration energy"):
        validate_inference_record(per_atom)
    corrected = {**record, "energy_corrections_applied": ["MP2020"]}
    with pytest.raises(ValueError, match="uncorrected"):
        validate_inference_record(corrected)
    with pytest.raises(ValueError, match="record_key"):
        validate_inference_record({**record, "record_key": "0" * 64})
    with pytest.raises(ValueError, match="fallback_reason"):
        validate_inference_record({**record, "effective_dtype": "float32"})
    validate_inference_record(
        {**record, "effective_dtype": "float32", "fallback_reason": "fixture CPU fallback"}
    )
    wrong_target = copy.deepcopy(record)
    for holder in (wrong_target, wrong_target["model_provenance"]):
        holder["reference_compatibility"]["target_hubbard_u_policy"] = "PBE+U"
    with pytest.raises(ValueError, match="differ from LiTraj"):
        validate_inference_record(wrong_target)
    manifest = _run_manifest(record)
    validate_inference_run_manifest(
        manifest, records=[record], expected_protocol_id=record["protocol_id"]
    )
    with pytest.raises(ValueError, match="different protocol_id"):
        validate_inference_run_manifest(manifest, expected_protocol_id="other")
    with pytest.raises(ValueError, match="clean Git"):
        validate_inference_run_manifest({**manifest, "git_dirty": True})
    duplicate = {**record, "attempt_id": "attempt-2", "attempt_number": 2}
    counts = {"success": 2, "failed": 0}
    with pytest.raises(ValueError, match="more than one successful"):
        validate_inference_run_manifest(
            {**manifest, "record_counts_by_status": counts}, records=[record, duplicate]
        )


def _edit_shard(tmp_path: Path, mutate) -> Path:
    config_path, audit_path = _phase_b_fixture(tmp_path)
    root = tmp_path / "provenance" / "index-1"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    descriptor = manifest["shards"][0]
    shard_path = root / descriptor["path"]
    shard = json.loads(shard_path.read_text())
    mutate(shard)
    _write(shard_path, shard)
    descriptor["sha256"] = _sha(shard_path)
    _write(manifest_path, manifest)
    audit = json.loads(audit_path.read_text())
    audit["mplitrj_provenance_index"]["manifest_sha256"] = _sha(manifest_path)
    _write(audit_path, audit)
    config = json.loads(config_path.read_text())
    config["inputs"]["g0_audit"]["sha256"] = _sha(audit_path)
    config["inputs"]["mplitrj_provenance"]["manifest_sha256"] = _sha(manifest_path)
    linkage = json.loads(Path(config["inputs"]["material_chemical_system_linkage"]["path"]).read_text())
    linkage["provenance_manifest_sha256"] = _sha(manifest_path)
    linkage_path = Path(config["inputs"]["material_chemical_system_linkage"]["path"])
    _write(linkage_path, linkage)
    config["inputs"]["material_chemical_system_linkage"]["sha256"] = _sha(linkage_path)
    _write(config_path, config)
    return config_path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda shard: shard.update({"material_bucket": "zz"}), "wrong material bucket"),
        (lambda shard: shard.update({"schema_version": "old"}), "shard schema"),
        (lambda shard: shard.update({"split": None}), "valid published split"),
    ],
)
def test_provenance_shard_body_is_validated(tmp_path, mutation, message):
    config = _edit_shard(tmp_path, mutation)
    with pytest.raises(ProtocolFreezeError, match=message):
        materialize_protocol(config)


def test_g0_failure_limited_to_fpmd_keeps_static_branch_and_static_failure_blocks(tmp_path):
    config_path, audit_path = _phase_b_fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    audit["status"] = "fail"
    for dataset in audit["datasets"]:
        if dataset["dataset"] == "FPMD":
            dataset["status"] = "fail"
    _write(audit_path, audit)
    config = json.loads(config_path.read_text())
    config["inputs"]["g0_audit"]["sha256"] = _sha(audit_path)
    _write(config_path, config)
    protocol, _ = materialize_protocol(config_path)
    assert (protocol["static_branch_status"], protocol["transport_branch_status"]) == (
        "ready",
        "blocked",
    )
    for dataset in audit["datasets"]:
        if dataset["dataset"] == "BVEL13k":
            dataset["status"] = "fail"
    _write(audit_path, audit)
    config["inputs"]["g0_audit"]["sha256"] = _sha(audit_path)
    _write(config_path, config)
    with pytest.raises(ProtocolFreezeError, match="static branch"):
        materialize_protocol(config_path)


def test_partitions_apply_60_40_within_nebdft2k_strata():
    rows = _universe()
    hop_systems = sorted({row["chemical_system_id"] for row in rows})[:20]
    rows += [
        {
            "record_kind": "nebdft2k_hop",
            "frame_id": None,
            "material_id": f"neb-{index}",
            "hop_id": f"neb-{index}_0_1_0_0_0",
            "chemical_system_id": system,
            "published_split": "train",
        }
        for index, system in enumerate(hop_systems)
    ]
    result = build_repeated_partitions(rows, base_seed=2026)
    assert result["strata"] == {"with_nebdft2k_hops": 20, "without_nebdft2k_hops": 17}
    for repeat in result["partitions"]:
        test = set(repeat["test_chemical_systems"])
        # round(0.4 * 20) = 8 hop systems and round(0.4 * 17) = 7 others.
        assert len(test & set(hop_systems)) == 8
        assert len(test - set(hop_systems)) == 7
        assert repeat["achieved_test_fraction_by_stratum"]["with_nebdft2k_hops"] == 0.4


def test_cli_git_pin_reaches_frozen_and_empty_run_root_is_not_a_blocker(tmp_path, monkeypatch):
    _clean_repository(monkeypatch)
    config_path = _production_fixture(tmp_path, complete=True)
    config = json.loads(config_path.read_text())
    config["inputs"]["repository_git_sha"] = "CURRENT"
    _write(config_path, config)
    candidate, _ = materialize_protocol(config_path)
    assert candidate["freeze_status"] == "candidate_unfrozen"
    assert any("repository_git_sha" in item for item in candidate["freeze_blockers"])
    with pytest.raises(ProtocolFreezeError, match="40-character"):
        materialize_protocol(config_path, git_sha="HEAD")
    frozen, path = materialize_protocol(config_path, git_sha="a" * 40)
    assert frozen["freeze_status"] == "frozen"
    resolved = json.loads((path.parent / "resolved_config.json").read_text())
    assert resolved["inputs"]["repository_git_sha"] == "a" * 40
    config["protocol"]["known_deviations"].append("forces a new protocol id")
    _write(config_path, config)
    empty = tmp_path / "outputs" / "freeze"
    target_id = None
    from e3sse.protocol import freeze as module

    original = module._atomic_json

    def crash(path, value):
        nonlocal target_id
        target_id = path.parent.name
        path.parent.mkdir(parents=True, exist_ok=True)
        raise OSError("crash before checkpoint")

    monkeypatch.setattr(module, "_atomic_json", crash)
    with pytest.raises(OSError, match="crash before checkpoint"):
        materialize_protocol(config_path, git_sha="a" * 40)
    assert (empty / target_id).is_dir() and not any((empty / target_id).iterdir())
    monkeypatch.setattr(module, "_atomic_json", original)
    recovered, _ = materialize_protocol(config_path, git_sha="a" * 40)
    assert recovered["protocol_id"] == target_id


def _stratified(n_with: int, n_without: int) -> list[dict]:
    rows = []
    pairs = [(a, b) for i, a in enumerate(_ELEMENTS) for b in _ELEMENTS[i + 1 :]]
    for index in range(n_with + n_without):
        system = normalize_chemical_system(["Li", *pairs[index]])
        rows.append(
            {
                "record_kind": "nebdft2k_hop" if index < n_with else "mplitrj_frame",
                "frame_id": None if index < n_with else f"f{index}",
                "material_id": f"m{index}",
                "hop_id": f"m{index}_0_1_0_0_0",
                "chemical_system_id": system,
                "published_split": "train",
            }
        )
    assert len({row["chemical_system_id"] for row in rows}) == n_with + n_without
    return rows


@pytest.mark.parametrize(("n_with", "n_without"), [(1, 1), (3, 3), (1, 6), (20, 17), (7, 0)])
def test_sixty_forty_is_defined_over_chemical_system_groups(n_with, n_without):
    rows = _stratified(n_with, n_without)
    total = n_with + n_without
    expected_test = min(total // 2, max(1, int(0.4 * total + 0.5)))
    result = build_repeated_partitions(rows, base_seed=2026)
    assert result["partition_fraction_unit"] == "chemical_system"
    assert result["stratification"] == "has_nebDFT2k_hop"
    assert result["stratification_classification"] == "engineering_choice"
    for repeat in result["partitions"]:
        counts = repeat["realized_chemical_system_counts"]
        # Stratification never changes the group-level total of the unstratified rule.
        assert counts == {"training_or_calibration": total - expected_test, "test": expected_test}
        assert repeat["achieved_fractions"]["test"] == expected_test / total
        assert repeat["rounding_deviation_from_target_test_fraction"] == pytest.approx(
            expected_test / total - 0.4
        )
        assert set(repeat["training_or_calibration_chemical_systems"]).isdisjoint(
            repeat["test_chemical_systems"]
        )
        assert repeat["unit_overlap_counts"] == {"frames": 0, "materials": 0, "hops": 0}


def test_repeat_specific_fits_may_use_only_that_repeats_training_side(tmp_path):
    from e3sse.protocol.contracts import require_repeat_training_side

    config, _ = _phase_b_fixture(tmp_path)
    protocol, _ = materialize_protocol(config)
    repeat = protocol["partition_definition"]["repeated_partitions"]["partitions"][3]
    train = repeat["training_or_calibration_chemical_systems"]
    test = repeat["test_chemical_systems"]
    for fitted in ("softening_estimate", "recalibration", "conformal_object"):
        require_repeat_training_side(
            protocol, repeat_index=3, fitted_object=fitted, chemical_system_ids=train
        )
        with pytest.raises(ValueError, match="test-side or unassigned"):
            require_repeat_training_side(
                protocol, repeat_index=3, fitted_object=fitted, chemical_system_ids=[*train, test[0]]
            )
    with pytest.raises(ValueError, match="not in the frozen"):
        require_repeat_training_side(
            protocol, repeat_index=10, fitted_object="recalibration", chemical_system_ids=train
        )
    with pytest.raises(ValueError, match="unknown repeat-fitted"):
        require_repeat_training_side(
            protocol, repeat_index=3, fitted_object="anything", chemical_system_ids=train
        )
    for not_refit in ("threshold", "predictor"):
        with pytest.raises(ValueError, match="unknown repeat-fitted"):
            require_repeat_training_side(
                protocol, repeat_index=3, fitted_object=not_refit, chemical_system_ids=train
            )
