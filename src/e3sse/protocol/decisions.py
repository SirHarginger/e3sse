"""Classified register of every consequential protocol decision.

Each rule names where its authority comes from, so an engineering choice can
never be reported as a paper-defined scientific rule.  ``paper_specified``
entries cite a section of ``docs/research/E3-SSE_Research_Design.docx``;
``repository_existing`` entries cite the module that already implements them.
Items still awaiting registration live in the configuration's
``unresolved_items`` and are classified ``unresolved`` there.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

DECISION_REGISTER_SCHEMA = "e3sse.protocol.decision-register.v1"
CLASSIFICATIONS = (
    "paper_specified",
    "repository_existing",
    "engineering_choice",
    "production_derived",
    "unresolved",
)

_D = "Research Design"
_PAPER_CITATION = re.compile(r"Research Design \d")

RESOLUTION_CATEGORIES = (
    "resolve_from_production_artifact",
    "resolve_from_paper",
    "requires_preanalysis_scientific_decision",
    "study3_only",
    "optional_later",
)
_BLOCKING_CATEGORIES = {
    "resolve_from_production_artifact",
    "requires_preanalysis_scientific_decision",
}
# Every register item and its category, fixed in code: deleting or reclassifying a
# configuration entry can never clear a blocker.
PROTOCOL_ITEM_CATEGORIES: dict[str, str] = {
    "production_source_unit_evidence": "resolve_from_production_artifact",
    "nebdft2k_convergence_flag_availability": "resolve_from_production_artifact",
    "endpoint_origin_policy": "resolve_from_paper",
    "study1_repeat_population_scope": "resolve_from_paper",
    "study5_evaluation_system_scope": "requires_preanalysis_scientific_decision",
    "test_side_material_scope_frame_population": "resolve_from_paper",
    "transition_metal_hubbard_u_machine_predicate": "requires_preanalysis_scientific_decision",
    "mixed_anion_family_precedence": "requires_preanalysis_scientific_decision",
    "configurational_energy_decile_definition": "requires_preanalysis_scientific_decision",
    "label_budget_repetition_and_stability_rule": "requires_preanalysis_scientific_decision",
    "estimator_minimum_support": "requires_preanalysis_scientific_decision",
    "s_bar_definition": "requires_preanalysis_scientific_decision",
    "s_host_definition": "requires_preanalysis_scientific_decision",
    "holm_family_interpretation": "requires_preanalysis_scientific_decision",
    "published_validation_split_role": "requires_preanalysis_scientific_decision",
    "published_split_chemical_system_overlap_rule": "requires_preanalysis_scientific_decision",
    "study4_predictor_calibration_independence": "requires_preanalysis_scientific_decision",
    "g4_monte_carlo_standard_error_definition": "requires_preanalysis_scientific_decision",
    "repeated_test_pilot_overlap_rule": "requires_preanalysis_scientific_decision",
    "model_checkpoint_and_reference_compatibility_resolution": (
        "requires_preanalysis_scientific_decision"
    ),
    "preregistration_record": "resolve_from_production_artifact",
    "fpmd_transport_reference_recoverability": "study3_only",
    "group_c_fine_tuned_comparator": "optional_later",
}
REQUIRED_RESOLUTION_ITEMS = tuple(
    identifier
    for identifier, category in PROTOCOL_ITEM_CATEGORIES.items()
    if category in _BLOCKING_CATEGORIES
)

DECISION_REGISTER: tuple[dict[str, str], ...] = (
    # Partitions and leakage.
    {"id": "primary_partition_published_litraj_split", "classification": "paper_specified",
     "source": f"{_D} 5.4", "statement": "the published LiTraj split is the primary partition "
     "and is preserved literally; it is never regrouped or replaced"},
    {"id": "robustness_partitions_10_grouped_60_40", "classification": "paper_specified",
     "source": f"{_D} 5.4", "statement": "ten repeated partitions grouped by chemical system, "
     "60% calibration or training and 40% test"},
    {"id": "chemical_system_is_atomic_partition_unit", "classification": "paper_specified",
     "source": f"{_D} 5.4; AGENTS.md", "statement": "all frames and hops of a chemical system "
     "share a partition"},
    {"id": "study5_pool_excludes_evaluation_chemical_systems",
     "classification": "paper_specified", "source": f"{_D} 5.4, 5.10 gate 0",
     "statement": "the Study 5 pool excludes every chemical system present in any evaluation "
     "partition"},
    {"id": "leakage_exclude_target_hop_neb_frames", "classification": "paper_specified",
     "source": f"{_D} 3.6", "statement": "the s correcting a hop is not estimated from frames "
     "generated during that hop's NEB optimisation"},
    {"id": "leakage_admissible_sources", "classification": "paper_specified",
     "source": f"{_D} 3.6", "statement": "endpoint-relaxation frames and frames of other hops "
     "of the same structure are admissible"},
    {"id": "target_hop_endpoint_relaxation_admissible", "classification": "paper_specified",
     "source": f"{_D} 3.6", "statement": "a target hop's endpoint-relaxation frames are "
     "admissible for its primary s; its NEB-path frames are prohibited; unidentified "
     "origin fails closed (eligibility rule research_design_3.6_v1)"},
    {"id": "litraj_raw_file_kinds", "classification": "repository_existing",
     "source": "src/e3sse/data/litraj_provenance.py", "statement": "raw hop files: "
     "band_optim_step and traj_init are NEB-path frames; source/target hold the relaxed "
     "endpoints; mapped_hop_unique rows carry a hop but no file kind"},
    {"id": "unidentified_origin_fails_closed", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/contracts.py", "statement": "conservative addition to "
     "3.6: a target-hop or test-side frame without a proven endpoint file (ambiguous, "
     "unmapped, mapped_hop_unique, unknown kind) is excluded; endpoint evidence requires a "
     "unique structural match"},
    {"id": "reverse_hops_are_other_hops", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/contracts.py", "statement": "accepted reading of 3.6: on "
     "the training side, frames of any other hop of the structure, including a reverse or "
     "symmetry-equivalent hop, are admissible"},
    {"id": "own_path_estimates_are_oracle_only", "classification": "paper_specified",
     "source": f"{_D} 3.6", "statement": "estimates from a hop's own path frames are reported "
     "only as an oracle upper bound"},
    {"id": "fitted_quantities_training_or_nonpath", "classification": "paper_specified",
     "source": f"{_D} 5.4", "statement": "s and the recalibration baseline are estimated on "
     "training partitions (excluding the target hop's own NEB frames) or on non-path frames; "
     "test-side material-level s uses only the target material's non-path frames"},
    {"id": "partition_fraction_unit_chemical_system", "classification": "paper_specified",
     "source": f"{_D} 5.4", "statement": "the 60/40 fractions count chemical-system groups, "
     "never hops, frames, materials, or records"},
    {"id": "repeat_fits_use_training_side_only", "classification": "paper_specified",
     "source": f"{_D} 5.4", "statement": "every fitted quantity of a repeat uses only that "
     "repeat's training/calibration side; its test-side outcomes fit nothing"},
    {"id": "global_thresholds_from_training_pilot", "classification": "paper_specified",
     "source": f"{_D} 4", "statement": "methodological choices and thresholds are fixed on "
     "permitted training data before held-out outcomes are inspected"},
    {"id": "study4_predictor_fixed_without_outcomes", "classification": "paper_specified",
     "source": f"{_D} 3.8", "statement": "the Study 4 predictor is constructed without "
     "calibration or test outcomes"},
    {"id": "outcomes_sealed_until_freeze", "classification": "paper_specified",
     "source": f"{_D} 4, 5.1", "statement": "test-partition and FPMD reference outcomes are not "
     "inspected until methods and thresholds are frozen; thresholds use training data only"},
    # Inclusion and exclusion.
    {"id": "primary_excludes_tm_and_hubbard_u_elements", "classification": "paper_specified",
     "source": f"{_D} 5.3, 3.9", "statement": "primary analysis excludes structures with "
     "transition metals or other elements for which MPtrj uses Hubbard U (executable element "
     "set unresolved)"},
    {"id": "tm_subset_is_preregistered_secondary", "classification": "paper_specified",
     "source": f"{_D} 5.3, 5.11", "statement": "transition-metal hops form a pre-registered "
     "secondary analysis"},
    {"id": "neb_convergence_exclusion_if_flagged", "classification": "paper_specified",
     "source": f"{_D} 5.3", "statement": "hops with failed or non-monotone DFT-NEB convergence "
     "flags, if provided, are excluded and counted"},
    {"id": "barrier_max_minus_min_image_energy", "classification": "paper_specified",
     "source": f"{_D} 5.3", "statement": "barrier = maximum minus minimum image energy, "
     "verified against the LiTraj loader before analysis"},
    {"id": "study3_host_melting_exclusion", "classification": "paper_specified",
     "source": f"{_D} 5.3", "statement": "structures whose FPMD reference shows host-framework "
     "melting are excluded from Study 3"},
    # Estimator and Study 1.
    {"id": "softening_zero_intercept_force_projection", "classification": "paper_specified",
     "source": f"{_D} 3.6 Eq. 17", "statement": "s = sum(F_ML F_DFT) / sum(F_DFT^2) over "
     "Cartesian force components"},
    {"id": "softening_frame_cluster_bootstrap", "classification": "paper_specified",
     "source": f"{_D} 3.6", "statement": "standard errors by a cluster bootstrap over whole "
     "frames"},
    {"id": "softening_resolution_levels", "classification": "paper_specified",
     "source": f"{_D} 3.6", "statement": "global per model, chemical family, material, and "
     "element-resolved estimates"},
    {"id": "study1_outputs", "classification": "paper_specified", "source": f"{_D} 5.6",
     "statement": "s distributions by model, anion family (oxide, sulfide, halide, nitride, "
     "other), element (s_Li, s_host) and configurational-energy decile on training-partition "
     "structures"},
    {"id": "label_budget_k_values", "classification": "paper_specified",
     "source": f"{_D} 3.6", "statement": "K in {1, 3, 5, 10, 30} by repeated random "
     "subsampling"},
    {"id": "preregistered_hypotheses", "classification": "paper_specified",
     "source": f"{_D} 4 Table 4", "statement": "RQ1-RQ5 hypotheses, estimands and decision "
     "rules are frozen verbatim in the protocol"},
    {"id": "two_stage_outcome_seal", "classification": "paper_specified",
     "source": f"{_D} 4, 5.1, 8", "statement": "registration precedes Study 1 and the pilot; "
     "thresholds fixed on the training pilot before test outcomes are unsealed"},
    {"id": "study2a_fixed_geometry", "classification": "paper_specified", "source": f"{_D} 5.7",
     "statement": "ML single points on DFT-NEB images; SRT barrier = fixed barrier / s; not "
     "ML-NEB"},
    {"id": "classification_thresholds", "classification": "paper_specified",
     "source": f"{_D} 5.7, 5.9", "statement": "c in {0.3, 0.5} eV; alpha in {0.1, 0.2}"},
    # Statistics.
    {"id": "cluster_bootstrap_intervals", "classification": "paper_specified",
     "source": f"{_D} 5.11", "statement": "95% chemical-system (Studies 2, 4) or material "
     "(Study 3) cluster-bootstrap percentile intervals with at least 2,000 resamples"},
    {"id": "paired_identical_units", "classification": "paper_specified",
     "source": f"{_D} 5.11", "statement": "comparisons are paired on identical units"},
    {"id": "study2_all_eligible_hops", "classification": "paper_specified",
     "source": f"{_D} 5.11", "statement": "Study 2 uses every eligible hop"},
    {"id": "mde_on_training_pilot", "classification": "paper_specified",
     "source": f"{_D} 5.11", "statement": "simulation-based minimum-detectable-effect analysis "
     "on the training-split pilot"},
    {"id": "primary_endpoints_and_holm", "classification": "paper_specified",
     "source": f"{_D} 5.11", "statement": "P1 rho_SRT Stage 2a Group A, P2 Study 3 paired "
     "reduction, P3 realised FDR; Holm across P1-P2 (family membership unresolved); P3 is a "
     "validity check"},
    {"id": "study4_design", "classification": "paper_specified", "source": f"{_D} 5.9, 3.8",
     "statement": "R = 200 draws of one hop per chemical system within each of ten "
     "partitions; BH with conformal p-values; Benjamini-Yekutieli sensitivity"},
    {"id": "study4_fdr_validity_gate", "classification": "paper_specified",
     "source": f"{_D} 6 Table 9", "statement": "G4: realised FDR <= alpha + 2 MC s.e."},
    {"id": "model_groups", "classification": "paper_specified", "source": f"{_D} 5.5 Table 7",
     "statement": "Group A CHGNet, M3GNet, MACE-MP-0 small, SevenNet-0; B1 one MatPES-type; "
     "B2 one OMat24-type, preferably architecture-matched; C optional fine-tuned"},
    {"id": "canonical_energy_force_temperature_units", "classification": "paper_specified",
     "source": f"{_D} Nomenclature", "statement": "energies and barriers in eV, forces in "
     "eV/angstrom, temperature in K"},
    # Existing repository conventions.
    {"id": "chemical_system_syntax_sorted_unique_elements",
     "classification": "repository_existing", "source": "src/e3sse/data/dataset_audit.py",
     "statement": "LiTraj chemsys convention: sorted unique element symbols joined by '-' "
     "(sorted-unique-elements-v1)"},
    {"id": "mplitrj_frame_to_hop_structural_match", "classification": "repository_existing",
     "source": "src/e3sse/data/mplitrj_provenance.py",
     "statement": "frame-to-hop provenance by structural match within material; only "
     "mapped_unique/mapped_hop_unique rows assert a hop"},
    {"id": "g0_audit_schema", "classification": "repository_existing",
     "source": "src/e3sse/data/dataset_audit.py",
     "statement": "dataset status, blockers, capabilities and input identities come from the "
     "G0 audit (e3sse.g0.audit.v2) and are consumed, never recomputed"},
    # Engineering choices.
    {"id": "partition_universe_union", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/linkage.py", "statement": "repeated partitions allocate the "
     "union of MPLiTrj and nebDFT2k chemical systems"},
    {"id": "partition_rounding_rule", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/partitions.py", "statement": "n_test = min(floor(n/2), "
     "max(1, floor(0.4 n + 0.5))) chemical systems; never rebalanced with observations"},
    {"id": "partition_rank_algorithm_and_seeds", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/partitions.py", "statement": "sha256-rank-v1 group ranking "
     "with domain-separated SHA-256 seeds from one configured base seed"},
    {"id": "bootstrap_resamples_10000", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/freeze.py", "statement": "10,000 resamples (paper minimum "
     "2,000) with numpy PCG64DXSM"},
    {"id": "conformal_deterministic_tie_break", "classification": "engineering_choice",
     "source": f"{_D} 3.8 Eq. 20", "statement": "U_j = 1, the deterministic conservative "
     "version the paper permits"},
    {"id": "study4_hop_draw_hashing", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/freeze.py", "statement": "one hop per chemical system "
     "selected by SHA-256 rank under a derived per-draw seed"},
    {"id": "label_budget_draw_without_replacement", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/freeze.py", "statement": "label-budget subsamples draw K "
     "whole frames without replacement from the leakage-admissible population"},
    {"id": "position_cell_stress_units", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/contracts.py", "statement": "positions and cells in "
     "angstrom, stress in eV/angstrom^3; only registered CODATA 2018 conversions"},
    {"id": "mlip_output_units", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/contracts.py", "statement": "adapters emit eV and "
     "eV/angstrom; every inference record is validated against this"},
    {"id": "protocol_identity", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/freeze.py", "statement": "protocol_id = SHA-256 of canonical "
     "JSON content; timestamps live only in the run manifest"},
    {"id": "g0_static_transport_branches", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/freeze.py", "statement": "G0 readiness split into static "
     "(nebDFT2k, MPLiTrj, BVEL13k, nebBVSE122k) and transport (FPMD) branches; Study 3 stays "
     "gated while transport is partial (known deviation from the single paper gate)"},
    {"id": "pilot_amendment_and_transport_addendum", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/contracts.py", "statement": "the pilot seal and Study 3 "
     "transport readiness are separate content-hashed artifacts bound to the protocol_id, so "
     "neither changes the static protocol identity"},
    {"id": "partition_strata", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/partitions.py", "statement": "the overall test-group count is "
     "the unstratified chemical-system rule, apportioned by largest remainder over "
     "has_nebDFT2k_hop strata so the hop universe is split approximately proportionally"},
    {"id": "base_seed_fixed_in_git", "classification": "engineering_choice",
     "source": "configs/protocol_freeze.json", "statement": "partition_base_seed 2026 is "
     "committed before any production materialization; any change is a recorded deviation"},
    {"id": "candidate_before_freeze", "classification": "engineering_choice",
     "source": "src/e3sse/protocol/freeze.py", "statement": "production runs with blocking "
     "items materialize a sealed candidate_unfrozen protocol listing freeze_blockers"},
    # Values recovered from production artifacts in Phase B.
    {"id": "dataset_counts_hashes_status", "classification": "production_derived",
     "source": "G0 audit and input manifest", "statement": "file/row/frame counts, source "
     "hashes, G0 status and blockers"},
    {"id": "chemical_system_membership", "classification": "production_derived",
     "source": "MPLiTrj frame species; nebDFT2k chemsys column", "statement": "material to "
     "chemical-system assignment and every partition membership"},
    {"id": "source_unit_evidence", "classification": "production_derived",
     "source": "production dataset schemas", "statement": "source units and conversions for "
     "every required quantity"},
)


def validate_decision_register(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return a sorted copy after checking classification vocabulary and citations."""

    result = []
    seen: set[str] = set()
    for entry in entries:
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"decision ids must be unique and non-empty: {identifier!r}")
        seen.add(identifier)
        if entry.get("classification") not in CLASSIFICATIONS:
            raise ValueError(f"invalid decision classification for {identifier}")
        for field in ("source", "statement"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"decision {identifier} lacks {field}")
        if entry["classification"] == "paper_specified" and _D not in entry["source"]:
            raise ValueError(f"paper_specified decision {identifier} must cite the design")
        result.append({key: entry[key] for key in ("id", "classification", "source", "statement")})
    return sorted(result, key=lambda item: item["id"])


def item_resolved(item: Mapping[str, Any] | None) -> bool:
    """An item is resolved only by an explicit non-blocking record with evidence."""

    if item is None or item.get("blocking") is not False:
        return False
    resolution = item.get("resolution")
    return (
        isinstance(resolution, Mapping)
        and resolution.get("status") == "resolved"
        and isinstance(resolution.get("evidence"), str)
        and bool(resolution["evidence"].strip())
    )


def required_item_blockers(items: list[Mapping[str, Any]]) -> list[str]:
    """Static-freeze blockers: every production/preanalysis item absent or unresolved.

    ``resolve_from_paper`` items must carry a resolution citing the Research Design;
    ``study3_only`` and ``optional_later`` items never block the static freeze.
    """

    by_id = {item["id"]: item for item in items}
    blockers = []
    for identifier, category in PROTOCOL_ITEM_CATEGORIES.items():
        item = by_id.get(identifier)
        if item is None:
            blockers.append(f"required protocol item missing from configuration: {identifier}")
        elif category in _BLOCKING_CATEGORIES and not item_resolved(item):
            blockers.append(f"blocking protocol item unresolved: {identifier}")
        elif category == "resolve_from_paper" and not (
            item_resolved(item) and _PAPER_CITATION.search(item["resolution"]["evidence"])
        ):
            blockers.append(f"paper-resolved item lacks a Research Design citation: {identifier}")
    return blockers


def validate_unresolved_items(items: Any) -> list[dict[str, Any]]:
    """Register items must be known, carry their fixed category, and declare blocking."""

    if not isinstance(items, list):
        raise TypeError("unresolved_items must be a list")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("unresolved items must be objects")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"unresolved item ids must be unique: {identifier!r}")
        seen.add(identifier)
        if identifier not in PROTOCOL_ITEM_CATEGORIES:
            raise ValueError(f"unknown protocol register item: {identifier}")
        if item.get("resolution_category") != PROTOCOL_ITEM_CATEGORIES[identifier]:
            raise ValueError(
                f"{identifier} must have resolution_category "
                f"{PROTOCOL_ITEM_CATEGORIES[identifier]!r}"
            )
        if not isinstance(item.get("blocking"), bool):
            raise TypeError(f"unresolved item {identifier} must declare blocking explicitly")
        if (
            PROTOCOL_ITEM_CATEGORIES[identifier] not in _BLOCKING_CATEGORIES
            and item["blocking"]
        ):
            raise ValueError(f"{identifier} cannot block the static freeze")
    return [dict(item) for item in items]
