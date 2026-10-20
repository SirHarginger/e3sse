"""Deterministic, chemical-system-grouped partitions.

The scientific design fixes ten 60/40 grouped repeats.  The paper does not
specify integer rounding or an RNG implementation; ``sha256-rank-v1`` is the
recorded engineering choice used here so results do not depend on a library's
random-number implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from ase.data import atomic_numbers

CHEMICAL_SYSTEM_ALGORITHM = "sorted-unique-elements-v1"
PARTITION_ALGORITHM = "sha256-rank-v1"
_ELEMENT = re.compile(r"[A-Z][a-z]?")
_IGNORABLE = re.compile(r"[\d\s.()\[\]{},_+\-/]+")


def normalize_chemical_system(value: str | Iterable[str]) -> str:
    """Return the versioned LiTraj element-set identifier (for example ``Li-O``)."""

    if isinstance(value, str):
        symbols = _ELEMENT.findall(value)
        remainder = _ELEMENT.sub("", value)
        if remainder and _IGNORABLE.fullmatch(remainder) is None:
            raise ValueError(f"invalid chemical-system identifier: {value!r}")
    else:
        symbols = [str(item) for item in value]
    if not symbols or any(atomic_numbers.get(symbol, 0) <= 0 for symbol in symbols):
        raise ValueError(f"invalid chemical-system elements: {symbols!r}")
    return "-".join(sorted(set(symbols)))


def _resolved_seed(base_seed: int, repeat_index: int) -> int:
    payload = f"e3sse.partition.seed.v1\0{base_seed}\0{repeat_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def analysis_seed(base_seed: int, domain: str, *identifiers: str | int) -> int:
    """Versioned seed derivation for draws and bootstrap analyses, without outcomes."""

    if domain not in {"study4_hop_draw", "cluster_bootstrap", "study1_label_budget"}:
        raise ValueError("unknown scientific random-operation domain")
    payload = json.dumps(
        ["e3sse.analysis.seed.v1", base_seed, domain, *identifiers],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rank(seed: int, chemical_system: str) -> str:
    return hashlib.sha256(
        f"e3sse.partition.rank.v1\0{seed}\0{chemical_system}".encode()
    ).hexdigest()


def _test_count(n_systems: int, test_fraction: float) -> int:
    """Round half up, keeping the training/calibration side at least as large as test."""

    if n_systems < 2:
        return 0
    return min(n_systems // 2, max(1, math.floor(test_fraction * n_systems + 0.5)))


def _apportion(total: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Largest-remainder split of ``total`` test groups over strata (ties by name)."""

    population = sum(sizes.values())
    exact = {name: total * size / population for name, size in sizes.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    order = sorted(sizes, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[: total - sum(counts.values())]:
        counts[name] += 1
    return counts


def build_repeated_partitions(
    records: Iterable[Mapping[str, Any]],
    *,
    base_seed: int,
    repeats: int = 10,
    test_fraction: float = 0.40,
) -> dict[str, Any]:
    """Partition distinct chemical systems and validate inherited unit assignments."""

    rows = [dict(row) for row in records]
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise TypeError("partition base_seed must be an integer")
    systems = sorted({normalize_chemical_system(row["chemical_system_id"]) for row in rows})
    if repeats != 10:
        raise ValueError("the frozen design requires exactly 10 repeated partitions")
    if test_fraction != 0.40:
        raise ValueError("the frozen design requires a 0.40 test fraction")
    if len(systems) < 2:
        raise ValueError("at least two chemical systems are required")
    # The 60/40 fraction is defined over chemical-system groups.  The overall test
    # count is the unstratified rule over all groups; it is then apportioned over the
    # has_nebDFT2k_hop strata so the Study 2/4 hop universe stays near 60/40 too.
    # Observations (hops, frames, materials) never rebalance whole systems.
    with_hops = {
        normalize_chemical_system(row["chemical_system_id"])
        for row in rows
        if row.get("record_kind") == "nebdft2k_hop"
    }
    strata = {
        "with_nebdft2k_hops": sorted(set(systems) & with_hops),
        "without_nebdft2k_hops": sorted(set(systems) - with_hops),
    }
    strata = {name: members for name, members in strata.items() if members}
    n_test = _test_count(len(systems), test_fraction)
    n_test_by_stratum = _apportion(
        n_test, {name: len(members) for name, members in strata.items()}
    )
    if n_test <= 0 or n_test >= len(systems):
        raise ValueError("60/40 rounding leaves an empty partition side")
    systems_by_identifier: dict[str, dict[str, set[str]]] = {
        key: defaultdict(set) for key in ("frame_id", "material_id", "hop_id")
    }
    for row in rows:
        system = normalize_chemical_system(row["chemical_system_id"])
        for key, mapping in systems_by_identifier.items():
            if row.get(key) is not None:
                mapping[str(row[key])].add(system)
    conflicts = {
        key: identifier
        for key, mapping in systems_by_identifier.items()
        for identifier, assigned in mapping.items()
        if len(assigned) != 1
    }
    if conflicts:
        raise ValueError(f"unit identifiers span chemical systems: {conflicts}")

    partitions = []
    for repeat_index in range(repeats):
        seed = _resolved_seed(base_seed, repeat_index)
        test_set: set[str] = set()
        for name, members in strata.items():
            ranked = sorted(members, key=lambda system: (_rank(seed, system), system))
            test_set.update(ranked[: n_test_by_stratum[name]])
        test = sorted(test_set)
        train = sorted(set(systems) - test_set)
        if set(train) & set(test):
            raise AssertionError("chemical-system overlap")
        assignment = {system: "test" for system in test} | {
            system: "training_or_calibration" for system in train
        }
        counts: dict[str, dict[str, int]] = {
            side: {"frames": 0, "materials": 0, "hops": 0}
            for side in ("training_or_calibration", "test")
        }
        identifiers: dict[str, dict[str, set[str]]] = {
            side: {"frames": set(), "materials": set(), "hops": set()} for side in counts
        }
        for row in rows:
            system = normalize_chemical_system(row["chemical_system_id"])
            side = assignment[system]
            for plural, key in (
                ("frames", "frame_id"),
                ("materials", "material_id"),
                ("hops", "hop_id"),
            ):
                value = row.get(key)
                if value is not None:
                    identifiers[side][plural].add(str(value))
        for side in counts:
            counts[side] = {name: len(values) for name, values in identifiers[side].items()}
        unit_overlap = {
            unit: len(identifiers["training_or_calibration"][unit] & identifiers["test"][unit])
            for unit in ("frames", "materials", "hops")
        }
        if any(unit_overlap.values()):
            raise ValueError("partition unit overlap detected")
        partitions.append(
            {
                "repeat_index": repeat_index,
                "resolved_seed": seed,
                "assignment_by_chemical_system": dict(sorted(assignment.items())),
                "training_or_calibration_chemical_systems": train,
                "test_chemical_systems": test,
                "counts": counts,
                "achieved_fractions": {
                    "training_or_calibration": len(train) / len(systems),
                    "test": len(test) / len(systems),
                },
                "realized_chemical_system_counts": {
                    "training_or_calibration": len(train),
                    "test": len(test),
                },
                "rounding_deviation_from_target_test_fraction": len(test) / len(systems)
                - test_fraction,
                "achieved_test_fraction_by_stratum": {
                    name: len(test_set & set(members)) / len(members)
                    for name, members in sorted(strata.items())
                },
                "unit_overlap_counts": unit_overlap,
                "overlap_count": 0,
            }
        )
    return {
        "schema_version": "e3sse.partitions.v1",
        "grouping_unit": "chemical_system",
        "chemical_system_algorithm": CHEMICAL_SYSTEM_ALGORITHM,
        "algorithm": PARTITION_ALGORITHM,
        "seed_strategy": "domain-separated SHA-256 seed and group ranking",
        "base_seed": base_seed,
        "seed_and_rounding_classification": "engineering_implementation_choice",
        "repeats": repeats,
        "target_training_or_calibration_fraction": 0.60,
        "target_test_fraction": test_fraction,
        "partition_fraction_unit": "chemical_system",
        "stratification": "has_nebDFT2k_hop",
        "stratification_classification": "engineering_choice",
        "stratification_rationale": (
            "keeps the nebDFT2k hop universe approximately proportionally split in every "
            "repeat (a stratum of one system necessarily lies on one side); the overall "
            "test group count is unchanged because it is fixed before apportionment"
        ),
        "rounding_rule": (
            "test_groups=min(floor(N/2),max(1,floor(0.40*N+0.5))) over all N chemical "
            "systems, apportioned over strata by largest remainder (ties by stratum name)"
        ),
        "strata": {name: len(members) for name, members in sorted(strata.items())},
        "chemical_system_count": len(systems),
        "partitions": partitions,
    }


def classify_primary_population(
    chemical_systems: Iterable[str], excluded_elements: Iterable[str]
) -> dict[str, str]:
    """Assign each chemical system to the primary or the TM/Hubbard-U secondary population.

    Research Design 5.3 excludes structures containing transition metals or other
    elements for which MPtrj uses Hubbard U from the primary analysis and analyses
    them as a pre-registered secondary subset.  The element set is not defined in
    the paper; it must be supplied from the registered resolution, never defaulted.
    """

    excluded = {str(symbol) for symbol in excluded_elements}
    if not excluded or any(atomic_numbers.get(symbol, 0) <= 0 for symbol in excluded):
        raise ValueError("excluded_elements must be a non-empty set of element symbols")
    result = {}
    for system in chemical_systems:
        normalized = normalize_chemical_system(system)
        elements = set(normalized.split("-"))
        result[normalized] = "tm_hubbard_u_secondary" if elements & excluded else "primary"
    return dict(sorted(result.items()))


def published_split_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Preserve literal LiTraj labels and report, but never repair, group overlap."""

    allowed = {"train", "val", "test"}
    counts = defaultdict(int)
    systems_by_split: dict[str, set[str]] = {name: set() for name in sorted(allowed)}
    splits_by_system: dict[str, set[str]] = defaultdict(set)
    memberships: dict[str, dict[str, set[str]]] = {
        split: {unit: set() for unit in ("frames", "materials", "hops")} for split in allowed
    }
    split_by_identifier: dict[str, dict[str, set[str]]] = {
        unit: defaultdict(set) for unit in ("frame_id", "material_id", "hop_id")
    }
    for row in records:
        split = row.get("published_split")
        if split not in allowed:
            raise ValueError(f"invalid published split: {split!r}")
        system = normalize_chemical_system(row["chemical_system_id"])
        counts[split] += 1
        systems_by_split[split].add(system)
        splits_by_system[system].add(split)
        for unit, field in (
            ("frames", "frame_id"),
            ("materials", "material_id"),
            ("hops", "hop_id"),
        ):
            if row.get(field) is not None:
                identifier = str(row[field])
                memberships[split][unit].add(identifier)
                split_by_identifier[field][identifier].add(split)
    conflicts = {
        field: sorted(identifier for identifier, splits in mapping.items() if len(splits) > 1)
        for field, mapping in split_by_identifier.items()
    }
    conflicts = {field: values for field, values in conflicts.items() if values}
    if conflicts:
        raise ValueError(f"identifiers have conflicting published splits: {conflicts}")
    spanning = {
        system: sorted(splits)
        for system, splits in sorted(splits_by_system.items())
        if len(splits) > 1
    }
    return {
        "source_labels_preserved": ["train", "val", "test"],
        "record_counts": {key: counts[key] for key in ("train", "val", "test")},
        "chemical_system_counts": {
            key: len(systems_by_split[key]) for key in ("train", "val", "test")
        },
        "chemical_systems_spanning_published_splits": spanning,
        "unit_membership": {
            split: {
                unit: {
                    "count": len(values),
                    "sha256": hashlib.sha256(
                        json.dumps(sorted(values), separators=(",", ":")).encode()
                    ).hexdigest(),
                }
                for unit, values in sorted(by_unit.items())
            }
            for split, by_unit in sorted(memberships.items())
        },
        "policy": "diagnose_only_do_not_regroup_published_split",
    }


STUDY5_EXCLUSION_RULE = "entire-partition-universe-v1"


def derive_study5_exclusions(
    published_test_chemical_systems: Iterable[str],
    repeated: Mapping[str, Any],
    universe_chemical_systems: Iterable[str],
) -> dict[str, Any]:
    """Exclude every chemical system of the LiTraj partition universe from Study 5.

    Research Design 5.4/5.10 exclude every chemical system present in any evaluation
    partition.  The registered conservative reading excludes the whole partition
    universe (every published split and every repeat side), a superset of
    union(published test, ten repeated tests).  Inputs are identifiers only; no
    candidate pool is read.
    """

    if isinstance(published_test_chemical_systems, (str, bytes)) or isinstance(
        universe_chemical_systems, (str, bytes)
    ):
        raise TypeError("chemical systems must be iterables of identifiers")
    published = {normalize_chemical_system(item) for item in published_test_chemical_systems}
    universe = {normalize_chemical_system(item) for item in universe_chemical_systems}
    if not published or not universe:
        raise ValueError("published test and universe chemical systems must be non-empty")
    partitions = repeated.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != repeated.get("repeats"):
        raise ValueError("repeated partition manifest is incomplete")
    evaluation = set(published)
    for repeat in partitions:
        evaluation.update(repeat["test_chemical_systems"])
    if not evaluation <= universe:
        raise ValueError("evaluation chemical systems lie outside the partition universe")
    return {
        "schema_version": "e3sse.study5.exclusions.v2",
        "rule": STUDY5_EXCLUSION_RULE,
        "derivation": "partition_universe (superset of published_test and repeated_test[0:10])",
        "chemical_system_definition": CHEMICAL_SYSTEM_ALGORITHM,
        "published_test_count": len(published),
        "evaluation_union_count": len(evaluation),
        "evaluation_union_contained": True,
        "chemical_system_ids": sorted(universe),
        "count": len(universe),
        "candidate_pool_inspected": False,
        "matching_requirement": (
            "candidate chemical systems are normalized with the same algorithm before "
            "exclusion; an unparseable candidate system is excluded (fail closed)"
        ),
    }
