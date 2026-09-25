"""Read-only hop provenance for the LiTraj ``MPLiTrj_raw.zip`` NEB archive.

The flattened ``MPLiTrj_{train,val,test}.xyz`` release carries ``material_id`` but
no hop identifier, so on its own it cannot support the E3-SSE rule that a hop's
softening correction must not use frames from that hop's own NEB optimisation.
The raw archive keeps one ``<edge_id>.neb/`` directory per hop and is the only
candidate source of that provenance.

Nothing here assumes that flattened frames come from particular archive hops.
The correspondence is tested structurally on a bounded, deterministic sample of
materials, and the archive is only opened for reading: members are decompressed
into memory one at a time and never extracted to disk.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

# Periodic image offsets in edge IDs may be negative, e.g. mp-1_0_1_-1_0_0.
EDGE_ID_PATTERN = re.compile(r"^(?P<material_id>[A-Za-z]+-\d+)_(?P<indices>-?\d+(?:_-?\d+)*)$")
STEP_FILE_PATTERN = re.compile(r"^band_optim_step_(?P<step>\d+)\.traj$")
ENDPOINT_FILES = {"source.xyz": "source", "target.xyz": "target", "traj_init.xyz": "traj_init"}
NEB_SUFFIX = ".neb"

# Extended XYZ stores positions with ~8 decimals, so 1e-4 Angstrom is far above
# formatting noise yet far below any physical displacement between NEB images.
DEFAULT_POSITION_ATOL = 1e-4
DEFAULT_CELL_ATOL = 1e-4
DEFAULT_ENERGY_ATOL = 1e-4


class BoundedExamples:
    """Exact count plus the first ``limit`` examples in deterministic input order."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0
        self.examples: list[Any] = []

    def add(self, example: Any) -> None:
        self.count += 1
        if len(self.examples) < self.limit:
            self.examples.append(example)

    def to_json(self, provenance: str) -> dict[str, Any]:
        return {
            "count": self.count,
            "example_limit": self.limit,
            "examples": list(self.examples),
            "provenance": provenance,
        }

    # State round-trips let independent work units be merged in canonical order.
    # Absorbing units in that order reproduces a serial pass exactly: counts add,
    # and the first ``limit`` examples of the concatenation are kept.
    def to_state(self) -> dict[str, Any]:
        return {"count": self.count, "limit": self.limit, "examples": list(self.examples)}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BoundedExamples:
        examples = cls(state["limit"])
        examples.count = state["count"]
        examples.examples = list(state["examples"])
        return examples

    def absorb(self, other: BoundedExamples) -> None:
        self.count += other.count
        room = self.limit - len(self.examples)
        if room > 0:
            self.examples.extend(other.examples[:room])


def parse_edge_id(edge_id: str) -> str | None:
    """Return the material ID encoded in a LiTraj edge ID, or None if it does not parse."""

    match = EDGE_ID_PATTERN.match(edge_id)
    return match.group("material_id") if match else None


@dataclass(frozen=True)
class HopMember:
    edge_id: str
    kind: str  # band_optim_step | source | target | traj_init | other | directory
    step: int | None
    name: str


def parse_member(name: str) -> HopMember | None:
    """Classify a ZIP member below an ``<edge_id>.neb/`` directory; None otherwise."""

    parts = name.rstrip("/").split("/")
    for position, part in enumerate(parts):
        if part.endswith(NEB_SUFFIX) and len(part) > len(NEB_SUFFIX):
            edge_id = part[: -len(NEB_SUFFIX)]
            rest = parts[position + 1 :]
            break
    else:
        return None
    if not rest:
        return HopMember(edge_id, "directory", None, name)
    if len(rest) == 1:
        step = STEP_FILE_PATTERN.match(rest[0])
        if step:
            return HopMember(edge_id, "band_optim_step", int(step.group("step")), name)
        if rest[0] in ENDPOINT_FILES:
            return HopMember(edge_id, ENDPOINT_FILES[rest[0]], None, name)
    return HopMember(edge_id, "other", None, name)


@dataclass
class HopEntry:
    steps: dict[int, str] = field(default_factory=dict)
    endpoints: dict[str, str] = field(default_factory=dict)
    other: list[str] = field(default_factory=list)


@dataclass
class ArchiveIndex:
    hops: dict[str, HopEntry]
    non_hop_members: list[str]
    duplicate_members: list[str]

    def materials(self) -> dict[str, list[str]]:
        """Map each parseable material ID to its sorted edge IDs."""

        grouped: dict[str, list[str]] = defaultdict(list)
        for edge_id in self.hops:
            material = parse_edge_id(edge_id)
            if material is not None:
                grouped[material].append(edge_id)
        return {material: sorted(edges) for material, edges in sorted(grouped.items())}


def index_archive(archive: zipfile.ZipFile) -> ArchiveIndex:
    """Index hop directories from the ZIP central directory without decompressing."""

    hops: dict[str, HopEntry] = defaultdict(HopEntry)
    non_hop: list[str] = []
    duplicates: list[str] = []
    for info in archive.infolist():
        member = parse_member(info.filename)
        if member is None:
            non_hop.append(info.filename)
            continue
        entry = hops[member.edge_id]
        if member.kind == "band_optim_step":
            if member.step in entry.steps:
                duplicates.append(member.name)
            else:
                entry.steps[member.step] = member.name
        elif member.kind in ENDPOINT_FILES.values():
            if member.kind in entry.endpoints:
                duplicates.append(member.name)
            else:
                entry.endpoints[member.kind] = member.name
        elif member.kind == "other":
            entry.other.append(member.name)
    return ArchiveIndex(dict(sorted(hops.items())), sorted(non_hop), sorted(duplicates))


def summarize_archive(index: ArchiveIndex, *, example_limit: int) -> dict[str, Any]:
    """Summarise hop structure with exact counts and bounded examples."""

    unparsed = BoundedExamples(example_limit)
    non_contiguous = BoundedExamples(example_limit)
    no_steps = BoundedExamples(example_limit)
    other_files = BoundedExamples(example_limit)
    steps_per_hop: Counter[int] = Counter()
    endpoint_presence: Counter[str] = Counter()
    for edge_id, entry in index.hops.items():
        if parse_edge_id(edge_id) is None:
            unparsed.add(edge_id)
        steps_per_hop[len(entry.steps)] += 1
        if not entry.steps:
            no_steps.add(edge_id)
        elif sorted(entry.steps) != list(range(len(entry.steps))):
            non_contiguous.add(edge_id)
        endpoint_presence.update(list(entry.endpoints))
        for name in entry.other:
            other_files.add(name)
    non_hop = BoundedExamples(example_limit)
    for name in index.non_hop_members:
        non_hop.add(name)
    duplicates = BoundedExamples(example_limit)
    for name in index.duplicate_members:
        duplicates.add(name)
    materials = index.materials()
    return {
        "hop_directories": len(index.hops),
        "unique_edge_ids": len(index.hops),
        "edge_id_examples": list(index.hops)[:example_limit],
        "unparseable_edge_ids": unparsed.to_json("archive directory name"),
        "material_ids": {
            "count": len(materials),
            "example_limit": example_limit,
            "examples": list(materials)[:example_limit],
            "provenance": "derived: material prefix of <edge_id>.neb directory name",
        },
        "optimization_steps_per_hop": {
            str(count): hops for count, hops in sorted(steps_per_hop.items())
        },
        "hops_without_optimization_steps": no_steps.to_json("archive listing"),
        "hops_with_non_contiguous_steps": non_contiguous.to_json("archive listing"),
        "endpoint_file_presence": {
            kind: endpoint_presence.get(kind, 0) for kind in sorted(ENDPOINT_FILES.values())
        },
        "other_hop_files": other_files.to_json("archive listing"),
        "non_hop_members": non_hop.to_json("archive listing"),
        "duplicate_members": duplicates.to_json("archive listing"),
    }


def read_member_frames(archive: zipfile.ZipFile, name: str) -> list[Atoms]:
    """Parse one archive member in memory; nothing is written to disk."""

    data = archive.read(name)
    if name.endswith(".traj"):
        frames = read(io.BytesIO(data), index=":", format="traj")
    else:
        frames = read(io.StringIO(data.decode("utf-8")), index=":", format="extxyz")
    return list(frames) if isinstance(frames, list) else [frames]


def _hop_members(entry: HopEntry) -> list[tuple[str, int | None, str]]:
    members = [("band_optim_step", step, entry.steps[step]) for step in sorted(entry.steps)]
    members.extend((kind, None, entry.endpoints[kind]) for kind in sorted(entry.endpoints))
    return members


def inspect_hops(
    archive: zipfile.ZipFile,
    index: ArchiveIndex,
    edge_ids: Iterable[str],
    *,
    example_limit: int,
) -> dict[str, Any]:
    """Describe the frames stored per file kind for a bounded set of hops."""

    frames_per_file: dict[str, Counter[int]] = defaultdict(Counter)
    atoms_per_frame: dict[str, Counter[int]] = defaultdict(Counter)
    result_keys: dict[str, set[str]] = defaultdict(set)
    info_keys: dict[str, set[str]] = defaultdict(set)
    failures = BoundedExamples(example_limit)
    sampled = []
    for edge_id in edge_ids:
        sampled.append(edge_id)
        for kind, _step, name in _hop_members(index.hops[edge_id]):
            try:
                frames = read_member_frames(archive, name)
            except Exception as exc:  # report per member; one bad file must not hide others
                failures.add(f"{name}: {type(exc).__name__}: {exc}")
                continue
            frames_per_file[kind][len(frames)] += 1
            for atoms in frames:
                atoms_per_frame[kind][len(atoms)] += 1
                result_keys[kind].update(getattr(atoms.calc, "results", None) or {})
                info_keys[kind].update(map(str, atoms.info))
    return {
        "sampled_edge_ids": sampled,
        "file_kinds": {
            kind: {
                "frames_per_file": {str(k): v for k, v in sorted(frames_per_file[kind].items())},
                "atoms_per_frame": {str(k): v for k, v in sorted(atoms_per_frame[kind].items())},
                "calculator_result_keys": sorted(result_keys[kind]),
                "info_keys": sorted(info_keys[kind]),
            }
            for kind in sorted(frames_per_file)
        },
        "read_failures": failures.to_json("in-memory ASE read of archive member"),
    }


@dataclass(frozen=True)
class FrameGeometry:
    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    energy: float | None

    def to_state(self) -> list[Any]:
        """JSON form; Python float repr round-trips float64 exactly."""

        return [
            self.numbers.tolist(),
            self.positions.tolist(),
            self.cell.tolist(),
            self.energy,
        ]

    @classmethod
    def from_state(cls, state: list[Any]) -> FrameGeometry:
        numbers, positions, cell, energy = state
        return cls(
            numbers=np.asarray(numbers, dtype=np.int64).reshape(-1),
            positions=np.asarray(positions, dtype=float).reshape(-1, 3),
            cell=np.asarray(cell, dtype=float).reshape(3, 3),
            energy=energy,
        )

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> FrameGeometry:
        results = getattr(atoms.calc, "results", None) or {}
        energy = results.get("energy")
        return cls(
            numbers=np.asarray(atoms.numbers, dtype=np.int64).copy(),
            positions=np.asarray(atoms.positions, dtype=float).copy(),
            cell=np.asarray(atoms.cell.array, dtype=float).copy(),
            energy=None if energy is None else float(energy),
        )


def composition_key(numbers: np.ndarray) -> str:
    """Order-sensitive digest of atomic numbers; stable across processes, unlike hash()."""

    return hashlib.sha256(np.asarray(numbers, dtype="<i8").tobytes()).hexdigest()


def geometries_match(
    left: FrameGeometry,
    right: FrameGeometry,
    *,
    position_atol: float = DEFAULT_POSITION_ATOL,
    cell_atol: float = DEFAULT_CELL_ATOL,
) -> bool:
    """Compare structures by species order, cell and positions modulo lattice vectors."""

    if left.numbers.shape != right.numbers.shape or not np.array_equal(left.numbers, right.numbers):
        return False
    if not np.allclose(left.cell, right.cell, rtol=0.0, atol=cell_atol):
        return False
    delta = left.positions - right.positions
    if abs(np.linalg.det(left.cell)) > 1e-8:
        # Wrapped and unwrapped copies of one configuration differ by lattice
        # vectors; remove those before measuring the displacement.
        fractional = np.linalg.solve(left.cell.T, delta.T).T
        fractional -= np.round(fractional)
        delta = fractional @ left.cell
    return bool(np.max(np.linalg.norm(delta, axis=1), initial=0.0) <= position_atol)


@dataclass(frozen=True)
class RawFrameRef:
    edge_id: str
    kind: str
    step: int | None
    image: int

    @property
    def label(self) -> str:
        step = "-" if self.step is None else str(self.step)
        return f"{self.edge_id}:{self.kind}:{step}:{self.image}"


def select_sample_materials(
    index: ArchiveIndex, count: int, *, max_members_per_material: int
) -> tuple[list[str], list[str]]:
    """First ``count`` sorted material IDs whose archive member count is within the cap.

    Returns ``(selected, skipped_as_too_large)``; the cap bounds memory and runtime
    and every skipped material is reported, so the sample is deterministic and
    its selection rule is auditable.
    """

    selected: list[str] = []
    skipped: list[str] = []
    for material, edges in index.materials().items():
        if len(selected) >= count:
            break
        members = sum(len(index.hops[e].steps) + len(index.hops[e].endpoints) for e in edges)
        if members > max_members_per_material:
            skipped.append(material)
        else:
            selected.append(material)
    return selected, skipped


def load_raw_frames(
    archive: zipfile.ZipFile,
    index: ArchiveIndex,
    materials: Iterable[str],
    *,
    example_limit: int,
) -> tuple[dict[str, list[tuple[RawFrameRef, FrameGeometry]]], BoundedExamples]:
    """Load every raw frame of every hop of the requested materials."""

    by_material = index.materials()
    loaded: dict[str, list[tuple[RawFrameRef, FrameGeometry]]] = {}
    failures = BoundedExamples(example_limit)
    for material in materials:
        frames: list[tuple[RawFrameRef, FrameGeometry]] = []
        for edge_id in by_material.get(material, []):
            for kind, step, name in _hop_members(index.hops[edge_id]):
                try:
                    atoms_list = read_member_frames(archive, name)
                except Exception as exc:
                    failures.add(f"{name}: {type(exc).__name__}: {exc}")
                    continue
                for image, atoms in enumerate(atoms_list):
                    frames.append(
                        (RawFrameRef(edge_id, kind, step, image), FrameGeometry.from_atoms(atoms))
                    )
        loaded[material] = frames
    return loaded, failures


@dataclass
class _Bucket:
    refs: list[RawFrameRef]
    positions: np.ndarray  # (frames, atoms, 3), Angstrom
    cells: np.ndarray  # (frames, 3, 3), Angstrom
    energies: np.ndarray  # (frames,), eV; NaN where absent


def _buckets(
    raw: dict[str, list[tuple[RawFrameRef, FrameGeometry]]],
) -> dict[tuple[str, str], _Bucket]:
    grouped: dict[tuple[str, str], list[tuple[RawFrameRef, FrameGeometry]]] = defaultdict(list)
    for material, frames in raw.items():
        for ref, geometry in frames:
            grouped[(material, composition_key(geometry.numbers))].append((ref, geometry))
    return {
        key: _Bucket(
            refs=[ref for ref, _ in items],
            positions=np.stack([g.positions for _, g in items]),
            cells=np.stack([g.cell for _, g in items]),
            energies=np.array([np.nan if g.energy is None else g.energy for _, g in items]),
        )
        for key, items in grouped.items()
    }


def _matching_indices(
    bucket: _Bucket,
    geometry: FrameGeometry,
    *,
    position_atol: float,
    cell_atol: float,
    chunk: int = 1024,
) -> np.ndarray:
    """Vectorised geometries_match over one bucket, chunked to bound memory."""

    cell_ok = np.all(np.abs(bucket.cells - geometry.cell) <= cell_atol, axis=(1, 2))
    candidates = np.flatnonzero(cell_ok)
    periodic = abs(np.linalg.det(geometry.cell)) > 1e-8
    inverse = np.linalg.inv(geometry.cell) if periodic else None
    hits: list[np.ndarray] = []
    for start in range(0, len(candidates), chunk):
        selected = candidates[start : start + chunk]
        delta = bucket.positions[selected] - geometry.positions
        if periodic:
            fractional = delta @ inverse
            fractional -= np.round(fractional)
            delta = fractional @ geometry.cell
        distance = np.linalg.norm(delta, axis=2).max(axis=1, initial=0.0)
        hits.append(selected[distance <= position_atol])
    return np.concatenate(hits) if hits else np.empty(0, dtype=int)


def validate_mapping(
    flattened: list[tuple[str, str, FrameGeometry]],
    raw: dict[str, list[tuple[RawFrameRef, FrameGeometry]]],
    *,
    example_limit: int,
    raw_read_failures: int = 0,
    unparseable_edge_ids: int = 0,
    position_atol: float = DEFAULT_POSITION_ATOL,
    cell_atol: float = DEFAULT_CELL_ATOL,
    energy_atol: float = DEFAULT_ENERGY_ATOL,
) -> dict[str, Any]:
    """Test whether sampled flattened frames are recoverable as raw archive frames.

    ``flattened`` holds ``(audit_frame_id, material_id, geometry)`` in file order.
    A frame may legitimately match several raw frames: NEB endpoints repeat in
    every optimisation step, and endpoints can be shared between hops of one
    material.  Hop exclusion only needs the set of hops, so frames matching more
    than one hop are reported separately rather than treated as failures.  An
    incomplete candidate set (unreadable members, or hop directories whose
    material cannot be parsed) can make a match look cleaner than it is, so it
    downgrades an otherwise clean result to ``inconclusive``.
    """

    buckets = _buckets(raw)
    single_hop = 0
    multi_hop = BoundedExamples(example_limit)
    unmatched = BoundedExamples(example_limit)
    kinds: Counter[str] = Counter()
    energy_compared = 0
    energy_agree = 0
    energy_disagree = BoundedExamples(example_limit)
    matched_raw: set[str] = set()
    hop_sequence: dict[str, list[str]] = defaultdict(list)
    for frame_id, material, geometry in flattened:
        bucket = buckets.get((material, composition_key(geometry.numbers)))
        indices = (
            np.empty(0, dtype=int)
            if bucket is None
            else _matching_indices(
                bucket, geometry, position_atol=position_atol, cell_atol=cell_atol
            )
        )
        if not len(indices):
            unmatched.add(frame_id)
            continue
        matches = [bucket.refs[i] for i in indices]
        matched_raw.update(ref.label for ref in matches)
        kinds.update({ref.kind for ref in matches})
        hops = sorted({ref.edge_id for ref in matches})
        if len(hops) == 1:
            single_hop += 1
            source_file = frame_id.rsplit(":", 1)[0]
            hop_sequence[source_file].append(hops[0])
        else:
            multi_hop.add({"frame": frame_id, "edge_ids": hops[:example_limit]})
        energies = bucket.energies[indices]
        energies = energies[~np.isnan(energies)]
        if geometry.energy is not None and len(energies):
            energy_compared += 1
            if np.any(np.abs(energies - geometry.energy) <= energy_atol):
                energy_agree += 1
            else:
                energy_disagree.add(frame_id)

    runs = sum(
        1
        for sequence in hop_sequence.values()
        for position, hop in enumerate(sequence)
        if position == 0 or sequence[position - 1] != hop
    )
    distinct = sum(len(set(sequence)) for sequence in hop_sequence.values())
    raw_total = sum(len(frames) for frames in raw.values())
    if not flattened:
        status = "not_evaluated"
    elif unmatched.count:
        status = "not_demonstrated"
    elif raw_read_failures or unparseable_edge_ids:
        status = "inconclusive"
    else:
        status = "demonstrated_on_sample"
    return {
        "status": status,
        "criteria": {
            "rule": (
                "each sampled flattened frame equals at least one raw archive frame of the same "
                "material: identical atomic numbers in order, cell within cell_atol, positions "
                "within position_atol modulo lattice translations; no unreadable members and "
                "no unparseable hop directories"
            ),
            "position_atol_angstrom": position_atol,
            "cell_atol_angstrom": cell_atol,
            "energy_atol_ev": energy_atol,
            "energy_used_for_matching": False,
        },
        "sampled_materials": sorted(raw),
        "flattened_frames_sampled": len(flattened),
        "raw_frames_loaded": raw_total,
        "raw_frames_matched": len(matched_raw),
        "raw_read_failures": raw_read_failures,
        "unparseable_edge_ids": unparseable_edge_ids,
        "matched_single_hop": single_hop,
        "matched_multi_hop": multi_hop.to_json("structural match to raw frames of >1 hop"),
        "unmatched": unmatched.to_json("no structural match among raw frames of same material"),
        "matched_frames_by_raw_file_kind": dict(sorted(kinds.items())),
        "energy_comparison": {
            "compared": energy_compared,
            "agree": energy_agree,
            "disagree": energy_disagree.to_json("calc.results energy differs beyond energy_atol"),
        },
        "single_hop_frames_contiguous_by_hop_in_file_order": (
            None if not hop_sequence else runs == distinct
        ),
    }
