"""Composition-only, outcome-independent chemistry rules frozen by the protocol.

Both rules use only element identities and counts; neither reads any energy, force,
barrier, or model prediction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ase.data import atomic_numbers, chemical_symbols

TM_RULE_ID = "pymatgen-transition-metal-union-mp2020-u-v1"
ANION_FAMILY_RULE_ID = "dominant-anion-count-v1"

# pymatgen 2024.2.8 ``Element.is_transition_metal``
# (pymatgen/core/periodic_table.py): Z in 21-30, 39-48, 57, 72-80, 89, 104-112.
PYMATGEN_TRANSITION_METAL_Z = (
    *range(21, 31),
    *range(39, 49),
    57,
    *range(72, 81),
    89,
    *range(104, 113),
)
# Elements receiving Hubbard U in Materials Project (MP2020 compatibility); all are
# already transition metals, and they are listed so the union is explicit.
MP2020_HUBBARD_U_ELEMENTS = ("Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W")

ANION_FAMILY_ELEMENTS: dict[str, tuple[str, ...]] = {
    "oxide": ("O",),
    "sulfide": ("S",),
    "halide": ("F", "Cl", "Br", "I"),
    "nitride": ("N",),
}
ANION_FAMILIES = ("oxide", "sulfide", "halide", "nitride", "other")


def primary_exclusion_elements() -> list[str]:
    """Elements whose presence moves a chemical system to the TM/Hubbard-U secondary set.

    Research Design 5.3 excludes structures with transition metals or other elements for
    which MPtrj uses Hubbard U, because LiTraj's non-spin-polarised PBE (no U) reference
    mismatches MP's spin-polarised PBE+U for d-electron transition metals.
    """

    symbols = {chemical_symbols[z] for z in PYMATGEN_TRANSITION_METAL_Z}
    return sorted(symbols | set(MP2020_HUBBARD_U_ELEMENTS))


def anion_family(composition: Mapping[str, int | float]) -> str:
    """Assign oxide/sulfide/halide/nitride/other from element counts alone.

    The family whose elements have the strictly largest atom count wins (e.g. Li6PS5Cl
    -> sulfide, Li2SO4 -> oxide); no family element -> ``other``; a tie between the two
    largest families -> ``other`` (mixed anion, no dominant family, e.g. Li3OCl).
    """

    for symbol, count in composition.items():
        if atomic_numbers.get(symbol, 0) <= 0 or count < 0:
            raise ValueError(f"invalid composition entry {symbol!r}: {count!r}")
    totals = {
        family: sum(composition.get(symbol, 0) for symbol in members)
        for family, members in ANION_FAMILY_ELEMENTS.items()
    }
    ranked = sorted(totals.items(), key=lambda item: -item[1])
    if ranked[0][1] <= 0 or ranked[0][1] == ranked[1][1]:
        return "other"
    return ranked[0][0]


def composition_counts(symbols: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[str(symbol)] = counts.get(str(symbol), 0) + 1
    return dict(sorted(counts.items()))
