"""Full frame-to-hop provenance index for the flattened MPLiTrj release.

The E3-SSE leakage rule forbids correcting a hop's barrier with a softening
estimate that used frames from that hop's own NEB optimisation.  Flattened
``MPLiTrj_{train,val,test}.xyz`` frames carry ``material_id`` but no hop, so this
module recovers, for every flattened frame, the raw ``<edge_id>.neb`` hop (and,
when unique, the optimisation step and NEB image) in ``MPLiTrj_raw.zip``.

Method (``ALGORITHM_VERSION``): a flattened frame matches a raw frame when both
belong to the same material, their atomic numbers are identical in order, every
cell component agrees within ``cell_atol`` and every atom agrees within
``position_atol`` after minimum-image wrapping in the flattened frame's cell.
Candidates are searched only among raw frames of the same material, so the cost
is a sum of per-material products rather than a global all-pairs search, and
correctness does not depend on file ordering.  Ordering is analysed and
reported as evidence but never relied on.  Energy is compared only as a
secondary consistency signal.

Statuses:
``mapped_unique``      exactly one raw candidate (hop, step and image known);
``mapped_hop_unique``  several candidates, all in one hop (e.g. NEB endpoints
                       repeated every step): hop known, step/image null;
``ambiguous``          candidates in more than one hop: no hop is asserted;
``unmapped``           no candidate; ``error``: frame or raw data unreadable.

The raw archive is read without ``zipfile`` directory parsing in workers: the
parent records each member's offset, sizes and CRC once, and workers read single
members, verifying size and CRC.  Nothing is ever written next to source data.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import re
import struct
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from e3sse.data import litraj_provenance as lp
from e3sse.data.litraj_provenance import BoundedExamples, FrameGeometry, RawFrameRef

SCHEMA_VERSION = "e3sse.mplitrj.provenance.v1"
ALGORITHM_VERSION = "structural-match-v1"
MAPPING_METHOD = (
    "same material_id; identical atomic numbers in order; cell within cell_atol; every "
    "atom within position_atol after minimum-image wrapping in the flattened cell"
)
BUCKET_HEX_DIGITS = 2
MISSING_MATERIAL = "<missing-material-id>"
MAPPED_STATUSES = ("mapped_unique", "mapped_hop_unique")
STATUSES = (*MAPPED_STATUSES, "ambiguous", "unmapped", "error")
_MATERIAL_PATTERN = re.compile(rb'(?:^|\s)material_id=(?:"([^"]*)"|(\S+))')
_LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")  # ZIP local file header, 30 bytes


def material_bucket(material_id: str | None) -> str:
    """Deterministic work/shard bucket: leading hex digits of sha256(material_id)."""

    key = MISSING_MATERIAL if material_id is None else material_id
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:BUCKET_HEX_DIGITS]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- identity


def source_identity(path: Path, data_root: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        identifier = path.relative_to(data_root).as_posix()
    except ValueError:
        identifier = path.as_posix()
    return {"identifier": identifier, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def index_inputs(sources: list[Path], archive: Path | None, data_root: Path) -> dict[str, Any]:
    return {
        "flattened": [source_identity(path, data_root) for path in sources],
        "archive": (
            source_identity(archive, data_root)
            if archive is not None and archive.is_file()
            else None
        ),
    }


def algorithm_identity() -> dict[str, str]:
    """Everything that can change record content: mapping code, split/file
    selection code and the parser/numerics versions."""

    import ase

    from e3sse.data import dataset_audit as g0

    selection = "\n".join(
        inspect.getsource(function)
        for function in (g0.split_from_filename, g0.mplitrj_source_files)
    ) + g0.SPLIT_FILENAME_PATTERN.pattern
    return {
        "version": ALGORITHM_VERSION,
        "litraj_provenance_sha256": _sha256_file(Path(lp.__file__)),
        "mplitrj_provenance_sha256": _sha256_file(Path(__file__)),
        "file_and_split_selection_sha256": hashlib.sha256(selection.encode()).hexdigest(),
        "ase_version": ase.__version__,
        "numpy_version": np.__version__,
    }


def tolerances(settings: dict[str, Any]) -> dict[str, float]:
    return {
        "position_atol_angstrom": float(settings["provenance_position_atol"]),
        "cell_atol_angstrom": float(settings["provenance_cell_atol"]),
        "energy_atol_ev": float(settings["provenance_energy_atol"]),
        # Materials whose raw hop members exceed this many uncompressed bytes are
        # reported as ``error`` instead of risking a worker out-of-memory kill.
        "max_material_raw_bytes": int(settings["provenance_max_material_raw_bytes"]),
    }


def compute_index_id(inputs: dict[str, Any], settings: dict[str, Any]) -> str:
    """Identity of an index: inputs, algorithm and tolerances (not the Git commit).

    The Git SHA is recorded for provenance and keys resume checkpoints, but an
    index stays valid across commits that do not change the mapping code.
    """

    return _canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "algorithm": algorithm_identity(),
            "tolerances": tolerances(settings),
            "inputs": inputs,
            "bucket_hex_digits": BUCKET_HEX_DIGITS,
        }
    )[:16]


# --------------------------------------------------------------------------- ZIP access


def member_entry(info: zipfile.ZipInfo) -> list[Any]:
    """Compact, picklable location of one archive member."""

    return [
        info.filename,
        info.header_offset,
        info.compress_size,
        info.file_size,
        info.compress_type,
        info.CRC,
        info.flag_bits,
    ]


def read_member(handle: Any, entry: list[Any]) -> bytes:
    """Read one member from an archive opened ``rb``, verifying size and CRC."""

    name, offset, compress_size, file_size, method, crc, flags = entry
    if flags & 0x1:
        raise ValueError(f"{name}: encrypted member")
    handle.seek(offset)
    header = _LOCAL_HEADER.unpack(handle.read(_LOCAL_HEADER.size))
    if header[0] != b"PK\x03\x04":
        raise ValueError(f"{name}: bad local file header")
    handle.seek(offset + _LOCAL_HEADER.size + header[10] + header[11])
    data = handle.read(compress_size)
    if method == zipfile.ZIP_STORED:
        payload = data
    elif method == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-15)
        payload = inflater.decompress(data) + inflater.flush()
    else:
        raise ValueError(f"{name}: unsupported compression method {method}")
    if len(payload) != file_size or zlib.crc32(payload) != crc:
        raise ValueError(f"{name}: size or CRC mismatch")
    return payload


def archive_hops(archive_path: Path) -> dict[str, Any]:
    """Hop member locations grouped by material, from one central-directory read."""

    with zipfile.ZipFile(archive_path, mode="r") as archive:
        index = lp.index_archive(archive)
        infos = {info.filename: info for info in archive.infolist()}
        by_material: dict[str, dict[str, list[list[Any]]]] = defaultdict(dict)
        unparseable: list[str] = []
        # Frames in files we do not parse could never be candidates, which could
        # make a multi-hop frame look unique; such hops block completeness.
        unrecognised = [edge for edge, entry in index.hops.items() if entry.other]
        for edge_id, entry in index.hops.items():
            material = lp.parse_edge_id(edge_id)
            if material is None:
                unparseable.append(edge_id)
                continue
            by_material[material][edge_id] = [
                [kind, step, member_entry(infos[name])]
                for kind, step, name in lp._hop_members(entry)
            ]
    return {
        "materials": dict(sorted(by_material.items())),
        "hop_count": len(index.hops),
        "unparseable_edge_ids": unparseable,
        "hops_with_unrecognised_files": unrecognised,
        "duplicate_members": index.duplicate_members,
    }


# --------------------------------------------------------------------------- flattened scan


def scan_flattened(path: str) -> dict[str, Any]:
    """Stream one extxyz file: exact SHA-256 plus offset/length/material per frame.

    Only frame boundaries and the header ``material_id`` are read here; atoms are
    parsed later, one frame at a time, from the recorded byte range.
    """

    source = Path(path)
    digest = hashlib.sha256()
    frames: dict[str, list[list[int]]] = defaultdict(list)
    index = 0
    offset = 0
    error = None
    with source.open("rb") as handle:
        try:
            while True:
                start = offset
                line = handle.readline()
                if not line:
                    break
                digest.update(line)
                offset += len(line)
                if not line.strip():
                    continue
                atoms = int(line)
                header = handle.readline()
                if not header:
                    raise ValueError(f"frame {index}: missing header line")
                digest.update(header)
                offset += len(header)
                for atom in range(atoms):
                    atom_line = handle.readline()
                    if not atom_line:
                        raise ValueError(f"frame {index}: truncated at atom {atom}")
                    digest.update(atom_line)
                    offset += len(atom_line)
                match = _MATERIAL_PATTERN.search(header)
                value = None
                if match is not None:
                    value = match.group(1) if match.group(1) is not None else match.group(2)
                material = value.decode("utf-8") if value else MISSING_MATERIAL
                frames[material].append([index, start, offset - start])
                index += 1
        except (ValueError, UnicodeDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "file": source.name,
        "frames": index,
        "sha256": digest.hexdigest(),
        "error": error,
        "material_frames": dict(sorted(frames.items())),
    }


def hash_file(path: str) -> dict[str, Any]:
    """Exact SHA-256 of a source file (read-only, streamed)."""

    return {"file": Path(path).name, "sha256": _sha256_file(Path(path))}


def read_frame_at(handle: Any, offset: int, length: int) -> Any:
    handle.seek(offset)
    text = handle.read(length).decode("utf-8")
    return read(io.StringIO(text), index=0, format="extxyz")


# --------------------------------------------------------------------------- mapping


def _parse_member(entry: list[Any], payload: bytes) -> list[Any]:
    if entry[0].endswith(".traj"):
        frames = read(io.BytesIO(payload), index=":", format="traj")
    else:
        frames = read(io.StringIO(payload.decode("utf-8")), index=":", format="extxyz")
    return list(frames) if isinstance(frames, list) else [frames]


def _load_material_raw(
    archive: Any, hops: dict[str, list[list[Any]]]
) -> tuple[dict[str, tuple[lp.RawGroup, list[str]]], list[str]]:
    """Raw frames of one material, grouped by atomic-number sequence."""

    grouped: dict[str, list[tuple[RawFrameRef, FrameGeometry, str]]] = defaultdict(list)
    failures: list[str] = []
    for edge_id in sorted(hops):
        for kind, step, entry in hops[edge_id]:
            try:
                atoms_list = _parse_member(entry, read_member(archive, entry))
            except Exception as exc:
                failures.append(f"{entry[0]}: {type(exc).__name__}: {exc}")
                continue
            for image, atoms in enumerate(atoms_list):
                geometry = FrameGeometry.from_atoms(atoms)
                grouped[lp.composition_key(geometry.numbers)].append(
                    (RawFrameRef(edge_id, kind, step, image), geometry, entry[0])
                )
    groups: dict[str, tuple[lp.RawGroup, list[str]]] = {}
    for key in sorted(grouped):
        items = grouped.pop(key)  # release per-frame copies as each group is stacked
        groups[key] = (
            lp.RawGroup(
                refs=[ref for ref, _, _ in items],
                positions=np.stack([g.positions for _, g, _ in items]),
                cells=np.stack([g.cell for _, g, _ in items]),
                energies=np.array([np.nan if g.energy is None else g.energy for _, g, _ in items]),
            ),
            [name for _, _, name in items],
        )
    return groups, failures


def classify(refs: list[RawFrameRef]) -> str:
    if not refs:
        return "unmapped"
    if len({ref.edge_id for ref in refs}) > 1:
        return "ambiguous"
    return "mapped_unique" if len(refs) == 1 else "mapped_hop_unique"


def _record(
    file: str,
    index: int,
    split: str | None,
    material: str | None,
    status: str,
    *,
    refs: list[RawFrameRef] = (),
    members: list[str] = (),
    reason: str | None = None,
    energy_consistent: bool | None = None,
) -> dict[str, Any]:
    edges = sorted({ref.edge_id for ref in refs})
    single = refs[0] if len(refs) == 1 else None
    hop = edges[0] if len(edges) == 1 else None
    return {
        "flattened_file": file,
        "flattened_frame_index": index,
        "derived_frame_id": f"{file}:{index}",
        "published_split": split,
        "material_id": material,
        "edge_id": hop,
        "raw_hop_directory": None if hop is None else members[0].rsplit("/", 1)[0] + "/",
        "raw_archive_path": None if single is None else members[0],
        "raw_file_kind": None if single is None else single.kind,
        "optimization_step": None if single is None else single.step,
        "neb_image_index": None if single is None else single.image,
        "candidate_count": len(refs),
        "candidate_edge_ids": edges if len(edges) > 1 else None,
        "mapping_status": status,
        "mapping_method": ALGORITHM_VERSION,
        "energy_consistent": energy_consistent,
        "reason": reason,
    }


def _map_material(
    item: dict[str, Any],
    files: dict[str, dict[str, Any]],
    handles: dict[str, Any],
    stack: ExitStack,
    archive: Any,
    tol: dict[str, float],
) -> list[dict[str, Any]]:
    material = None if item["material_id"] == MISSING_MATERIAL else item["material_id"]
    frames = item["frames"]

    def blanket(status: str, reason: str) -> list[dict[str, Any]]:
        return [
            _record(file, index, files[file]["split"], material, status, reason=reason)
            for file, index, _, _ in frames
        ]

    if material is None:
        return blanket("unmapped", "flattened frame has no material_id")
    if not item["hops"]:
        return blanket("unmapped", "material has no hop directory in the raw archive")
    raw_bytes = sum(entry[3] for members in item["hops"].values() for _, _, entry in members)
    if raw_bytes > tol["max_material_raw_bytes"]:
        return blanket(
            "error",
            f"raw hop members total {raw_bytes} bytes, above max_material_raw_bytes="
            f"{tol['max_material_raw_bytes']}; not mapped",
        )
    groups, failures = _load_material_raw(archive, item["hops"])
    if failures:
        # An incomplete candidate set could make a multi-hop frame look unique.
        return blanket("error", f"raw archive member unreadable: {failures[0]}")
    records = []
    for file, index, offset, length in frames:
        split = files[file]["split"]
        if file not in handles:
            # Closed by the caller's ExitStack; kept open across this bucket's materials.
            handles[file] = stack.enter_context(Path(files[file]["path"]).open("rb"))  # noqa: SIM115
        try:
            atoms = read_frame_at(handles[file], offset, length)
        except Exception as exc:
            records.append(
                _record(file, index, split, material, "error", reason=f"{type(exc).__name__}: {exc}")
            )
            continue
        if str(atoms.info.get("material_id")) != material:
            records.append(
                _record(file, index, split, material, "error", reason="material_id mismatch on re-read")
            )
            continue
        geometry = FrameGeometry.from_atoms(atoms)
        group = groups.get(lp.composition_key(geometry.numbers))
        if group is None:
            records.append(
                _record(
                    file, index, split, material, "unmapped",
                    reason="no raw frame of this material has the same atomic-number sequence",
                )
            )
            continue
        bucket, names = group
        hits = lp.matching_indices(
            bucket,
            geometry,
            position_atol=tol["position_atol_angstrom"],
            cell_atol=tol["cell_atol_angstrom"],
        )
        refs = [bucket.refs[i] for i in hits]
        members = [names[i] for i in hits]
        energies = bucket.energies[hits]
        energies = energies[~np.isnan(energies)]
        consistent = None
        if geometry.energy is not None and len(energies):
            consistent = bool(np.any(np.abs(energies - geometry.energy) <= tol["energy_atol_ev"]))
        status = classify(refs)
        records.append(
            _record(
                file, index, split, material, status,
                refs=refs, members=members, energy_consistent=consistent,
                reason="no structural match within tolerance" if status == "unmapped" else None,
            )
        )
    return records


def map_bucket(
    bucket: str,
    materials: list[dict[str, Any]],
    files: dict[str, dict[str, Any]],
    archive_path: str,
    tol: dict[str, float],
) -> dict[str, Any]:
    """Map every flattened frame of one material bucket, one material at a time."""

    records: list[dict[str, Any]] = []
    handles: dict[str, Any] = {}
    with ExitStack() as stack:
        archive = stack.enter_context(open(archive_path, "rb"))
        for item in materials:
            records.extend(_map_material(item, files, handles, stack, archive, tol))
    order = {name: position for position, name in enumerate(files)}
    records.sort(key=lambda r: (order[r["flattened_file"]], r["flattened_frame_index"]))
    return {"bucket": bucket, "records": records}


# --------------------------------------------------------------------------- analysis


def ordering_summary(sequence: list[tuple[str | None, int | None, int | None]]) -> dict[str, Any]:
    """Is a file's frame sequence ordered by hop, then (step, image)?

    ``sequence`` holds (edge_id, optimisation_step, image) per frame in file order,
    with None where unknown.  Only frames with a known hop count towards hop
    contiguity; only consecutive same-hop frames with known step and image count
    towards step/image order.
    """

    edges = [edge for edge, _, _ in sequence if edge is not None]
    runs = [edge for position, edge in enumerate(edges) if position == 0 or edges[position - 1] != edge]
    contiguous = bool(edges) and len(runs) == len(set(runs))
    compared = violations = 0
    previous: tuple[str, int, int] | None = None
    for edge, step, image in sequence:
        if edge is None or step is None or image is None:
            previous = None
            continue
        if previous is not None and previous[0] == edge:
            compared += 1
            violations += (step, image) < previous[1:]
        previous = (edge, step, image)
    if not edges:
        conclusion = "not determined: no frames with a known hop"
    elif contiguous and compared and not violations:
        conclusion = "hop-contiguous with non-decreasing (step, image) within each hop"
    elif contiguous:
        conclusion = "hop-contiguous; (step, image) order not established"
    else:
        conclusion = "not hop-contiguous: no exploitable hop ordering"
    return {
        "frames": len(sequence),
        "frames_with_hop": len(edges),
        "hop_runs": len(runs),
        "distinct_hops": len(set(runs)),
        "hop_contiguous": contiguous,
        "hops_in_sorted_edge_id_order": contiguous and runs == sorted(runs),
        "step_image_pairs_compared": compared,
        "step_image_order_violations": violations,
        "conclusion": conclusion,
        "relied_on_for_mapping": False,
    }


def nebdft2k_edge_ids(raw_root: Path) -> list[str]:
    """edge_id column of the nebDFT2k index only; no barrier values are read."""

    for path in sorted((raw_root / "nebDFT2k").rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "edge_id" in reader.fieldnames:
                return sorted({row["edge_id"] for row in reader if row.get("edge_id")})
    return []


def _set_comparison(left: set[str], right: set[str], limit: int) -> dict[str, Any]:
    def bounded(values: set[str]) -> dict[str, Any]:
        return {"count": len(values), "example_limit": limit, "examples": sorted(values)[:limit]}

    return {
        "edge_ids_in_mplitrj": len(left),
        "edge_ids_in_nebdft2k": len(right),
        "intersection": bounded(left & right),
        "mplitrj_only": bounded(left - right),
        "nebdft2k_only": bounded(right - left),
    }


# --------------------------------------------------------------------------- build


def _summarize_bucket(result: dict[str, Any], limit: int) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in result["records"]:
        by_split[record["published_split"] or "unsplit"].append(record)
    counts = {split: Counter(r["mapping_status"] for r in rows) for split, rows in by_split.items()}
    examples = {status: BoundedExamples(limit) for status in ("ambiguous", "unmapped", "error")}
    energy = Counter()
    edges: Counter[str] = Counter()
    for record in result["records"]:
        status = record["mapping_status"]
        if status in examples:
            examples[status].add(
                {
                    "frame": record["derived_frame_id"],
                    "reason": record["reason"],
                    "candidate_edge_ids": record["candidate_edge_ids"],
                }
            )
        if status in MAPPED_STATUSES:
            edges[record["edge_id"]] += 1
        if record["energy_consistent"] is not None:
            energy["compared"] += 1
            energy["agree"] += record["energy_consistent"]
    return {
        "by_split": by_split,
        "counts": {split: dict(sorted(c.items())) for split, c in sorted(counts.items())},
        "examples": {status: ex.to_state() for status, ex in examples.items()},
        "edge_frame_counts": dict(sorted(edges.items())),
        "energy": dict(energy),
    }


def build_provenance_index(
    config_path: str | Path,
    *,
    workers: int | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Build (or resume) the full MPLiTrj frame-to-hop index; returns manifest and path."""

    from e3sse.data import dataset_audit as g0  # deferred: dataset_audit imports this module

    progress = progress or (lambda message: None)
    wall_start = time.perf_counter()
    config = g0.load_config(config_path)
    settings = g0._settings(config)
    workers = g0.resolve_workers(workers, settings)
    limit = settings["example_limit"]
    data_root = Path(config["paths"]["data_root"])
    raw_root = data_root / "raw"
    output_root = g0._safe_output_root(config)
    sources = g0.mplitrj_source_files(g0.discover_datasets(raw_root)["MPLiTrj"])
    archive = data_root / settings["mplitrj_raw_archive"]
    if not sources:
        raise FileNotFoundError(f"no flattened MPLiTrj extxyz files under {raw_root}")
    if not archive.is_file():
        raise FileNotFoundError(f"raw archive not found: {archive}")
    inputs = index_inputs(sources, archive, data_root)
    index_id = compute_index_id(inputs, settings)
    root = output_root / settings["mplitrj_provenance_dir"] / index_id
    tol = tolerances(settings)
    git_sha = g0._git("rev-parse", "HEAD")
    base = {
        "index_id": index_id,
        "git_sha": git_sha,
        "algorithm": algorithm_identity(),
        "tolerances": tol,
    }
    manifest_path = root / "manifest.json"
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("git_sha") == git_sha
            and validate_manifest(root, existing)[0]
        ):
            progress(f"provenance: reused complete index {manifest_path}")
            return existing, manifest_path
    if manifest_path.exists():
        # This build supersedes the previous manifest; never leave it looking current.
        manifest_path.replace(root / "manifest.previous.json")

    split_of = {path.name: g0.split_from_filename(path.name) for path in sources}
    files = {path.name: {"path": str(path), "split": split_of[path.name]} for path in sources}
    timings: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    def fingerprint(key: str, **extra: Any) -> str:
        return _canonical_sha256({**base, "unit": key, **extra})

    with g0._worker_pool(workers) as pool:
        scan_units = [
            g0.WorkUnit(
                key=f"provenance/scan/{path.stem}",
                kind="provenance_scan",
                payload={"path": str(path)},
                checkpoint=root / "scan" / f"{path.stem}.json",
                fingerprint=fingerprint(f"scan/{path.stem}", source=inputs["flattened"][position]),
            )
            for position, path in enumerate(sources)
        ]
        archive_unit = g0.WorkUnit(
            key="provenance/archive_sha256",
            kind="provenance_hash",
            payload={"path": str(archive)},
            checkpoint=root / "scan" / "archive_sha256.json",
            fingerprint=fingerprint("archive_sha256", source=inputs["archive"]),
        )
        scans = g0._run_units(
            [archive_unit, *scan_units], pool, run_id=index_id, resume=resume,
            progress=progress, timings=timings, failures=failures,
        )
        progress("provenance: indexing raw archive central directory")
        hops = archive_hops(archive)
        if failures:
            _fail(root, index_id, failures)

        # Group frames by material bucket, in canonical file then frame order.
        buckets: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for unit, path in zip(scan_units, sources, strict=True):
            for material, frames in scans[unit.key]["material_frames"].items():
                bucket = buckets[material_bucket(None if material == MISSING_MATERIAL else material)]
                item = bucket.setdefault(
                    material,
                    {
                        "material_id": material,
                        "frames": [],
                        "hops": hops["materials"].get(material, {}),
                    },
                )
                item["frames"].extend([path.name, *frame] for frame in frames)
        archive_sha256 = scans[archive_unit.key]["sha256"]
        scan_meta = [
            {k: scans[unit.key][k] for k in ("file", "frames", "sha256", "error")}
            for unit in scan_units
        ]
        del scans

        def writer(bucket: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
            def write_shards(result: dict[str, Any]) -> dict[str, Any]:
                summary = _summarize_bucket(result, limit)
                shards = {}
                for split, rows in sorted(summary.pop("by_split").items()):
                    relative = f"{split}/material-bucket-{bucket}.json"
                    shard = {
                        "schema_version": SCHEMA_VERSION,
                        "index_id": index_id,
                        "git_sha": git_sha,
                        "source_archive": inputs["archive"],
                        "flattened_sources": inputs["flattened"],
                        "mapping_algorithm": ALGORITHM_VERSION,
                        "mapping_method": MAPPING_METHOD,
                        "tolerances": tol,
                        "split": split,
                        "material_bucket": bucket,
                        **_status_counts(Counter(r["mapping_status"] for r in rows)),
                        "created_at": g0.utc_now(),
                        "records": rows,
                    }
                    g0._atomic_json(root / relative, shard)
                    shards[split] = {
                        "path": relative,
                        "sha256": _sha256_file(root / relative),
                        "record_count": len(rows),
                    }
                summary["shards"] = shards
                return summary

            return write_shards

        def verifier(result: dict[str, Any]) -> bool:
            return all(
                (root / shard["path"]).is_file()
                and _sha256_file(root / shard["path"]) == shard["sha256"]
                for shard in result.get("shards", {}).values()
            )

        map_units = []
        for bucket in sorted(buckets):
            materials = [buckets[bucket][m] for m in sorted(buckets[bucket])]
            for item in materials:
                item["frames"].sort(key=lambda f: (list(files).index(f[0]), f[1]))
            key = f"provenance/material-bucket-{bucket}"
            map_units.append(
                g0.WorkUnit(
                    key=key,
                    kind="provenance_map",
                    payload={
                        "bucket": bucket,
                        "materials": materials,
                        "files": files,
                        "archive_path": str(archive),
                        "tol": tol,
                    },
                    checkpoint=root / "units" / f"material-bucket-{bucket}.json",
                    fingerprint=fingerprint(
                        key,
                        archive=inputs["archive"],
                        flattened=inputs["flattened"],
                        payload=_canonical_sha256(materials),
                    ),
                    on_result=writer(bucket),
                    verify=verifier,
                )
            )
        del buckets
        summaries = g0._run_units(
            map_units, pool, run_id=index_id, resume=resume,
            progress=progress, timings=timings, failures=failures,
        )
    if failures:
        _fail(root, index_id, failures)

    manifest = _assemble_manifest(
        root=root,
        index_id=index_id,
        git_sha=git_sha,
        git_dirty=bool(g0._git("status", "--porcelain")),
        inputs=inputs,
        scans=scan_meta,
        archive_sha256=archive_sha256,
        hops=hops,
        units=[(unit.key, summaries[unit.key]) for unit in map_units],
        files=files,
        tol=tol,
        limit=limit,
        nebdft2k=nebdft2k_edge_ids(raw_root),
        created_at=g0.utc_now(),
    )
    manifest["execution"] = {
        "workers": workers,
        "wall_seconds": round(time.perf_counter() - wall_start, 3),
        "units_reused": sorted(k for k, v in timings.items() if v["reused"]),
        "unit_elapsed_seconds": {k: v["elapsed_seconds"] for k, v in sorted(timings.items())},
    }
    stale = root / "failures.json"
    if stale.exists():
        stale.unlink()
    g0._atomic_json(manifest_path, manifest)
    progress(f"provenance: manifest written {manifest_path}")
    return manifest, manifest_path


def _status_counts(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    mapped = sum(counts.get(status, 0) for status in MAPPED_STATUSES)
    return {
        "record_count": total,
        "mapped_count": mapped,
        "mapped_unique_count": counts.get("mapped_unique", 0),
        "mapped_hop_unique_count": counts.get("mapped_hop_unique", 0),
        "ambiguous_count": counts.get("ambiguous", 0),
        "unmapped_count": counts.get("unmapped", 0),
        "error_count": counts.get("error", 0),
        "coverage_fraction": (mapped / total) if total else 0.0,
    }


def _fail(root: Path, index_id: str, failures: dict[str, str]) -> None:
    from e3sse.data import dataset_audit as g0

    path = root / "failures.json"
    g0._atomic_json(path, {"index_id": index_id, "failures": dict(sorted(failures.items()))})
    raise g0.G0ExecutionError(dict(sorted(failures.items())), path)


def _assemble_manifest(
    *,
    root: Path,
    index_id: str,
    git_sha: str,
    git_dirty: bool,
    inputs: dict[str, Any],
    scans: list[dict[str, Any]],
    archive_sha256: str,
    hops: dict[str, Any],
    units: list[tuple[str, dict[str, Any]]],
    files: dict[str, dict[str, Any]],
    tol: dict[str, float],
    limit: int,
    nebdft2k: list[str],
    created_at: str,
) -> dict[str, Any]:
    from e3sse.data import dataset_audit as g0

    totals: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    examples = {status: BoundedExamples(limit) for status in ("ambiguous", "unmapped", "error")}
    edges: Counter[str] = Counter()
    energy: Counter[str] = Counter()
    shards = []
    for _key, summary in units:
        for split, counts in summary["counts"].items():
            by_split[split].update(counts)
            totals.update(counts)
        for status, state in summary["examples"].items():
            examples[status].absorb(BoundedExamples.from_state(state))
        edges.update(summary["edge_frame_counts"])
        energy.update(summary["energy"])
        shards.extend(summary["shards"][split] for split in sorted(summary["shards"]))
    shards.sort(key=lambda shard: shard["path"])

    # Ordering evidence: read shards back one at a time, index by file position.
    frame_counts = {scan["file"]: scan["frames"] for scan in scans}
    sequences = {name: [(None, None, None)] * frame_counts.get(name, 0) for name in files}
    for shard in shards:
        records = json.loads((root / shard["path"]).read_text(encoding="utf-8"))["records"]
        for record in records:
            sequence = sequences[record["flattened_file"]]
            if record["flattened_frame_index"] < len(sequence):
                sequence[record["flattened_frame_index"]] = (
                    record["edge_id"],
                    record["optimization_step"] if record["raw_file_kind"] == "band_optim_step" else None,
                    record["neb_image_index"] if record["raw_file_kind"] == "band_optim_step" else None,
                )
    ordering = {name: ordering_summary(sequence) for name, sequence in sequences.items()}
    del sequences

    edges_path = root / "edges.json"
    g0._atomic_json(edges_path, {"index_id": index_id, "edge_frame_counts": dict(sorted(edges.items()))})
    scan_errors = {scan["file"]: scan["error"] for scan in scans if scan["error"]}
    counts = _status_counts(totals)
    blockers = []
    if counts["record_count"] != sum(frame_counts.values()):
        blockers.append("indexed records differ from scanned frame count")
    for status in ("ambiguous", "unmapped", "error"):
        if counts[f"{status}_count"]:
            blockers.append(f"{counts[f'{status}_count']} {status} frames")
    if scan_errors:
        blockers.append(f"flattened scan errors: {sorted(scan_errors)}")
    if hops["unparseable_edge_ids"]:
        blockers.append(
            f"{len(hops['unparseable_edge_ids'])} raw hop directories with unparseable edge_id"
        )
    if hops["hops_with_unrecognised_files"]:
        blockers.append(
            f"{len(hops['hops_with_unrecognised_files'])} raw hop directories contain "
            "unrecognised files whose frames were not candidates"
        )
    if hops["duplicate_members"]:
        blockers.append(f"{len(hops['duplicate_members'])} duplicate raw archive members")
    return {
        "schema_version": SCHEMA_VERSION,
        "index_id": index_id,
        "status": "complete",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "created_at": created_at,
        "mapping_algorithm": ALGORITHM_VERSION,
        "mapping_method": MAPPING_METHOD,
        "algorithm": algorithm_identity(),
        "tolerances": tol,
        "inputs": inputs,
        "source_archive": {
            **inputs["archive"],
            "sha256": archive_sha256,
            "hop_directories": hops["hop_count"],
            "unparseable_edge_ids": {
                "count": len(hops["unparseable_edge_ids"]),
                "example_limit": limit,
                "examples": hops["unparseable_edge_ids"][:limit],
            },
            "duplicate_members": len(hops["duplicate_members"]),
            "hops_with_unrecognised_files": {
                "count": len(hops["hops_with_unrecognised_files"]),
                "example_limit": limit,
                "examples": hops["hops_with_unrecognised_files"][:limit],
            },
        },
        "flattened_sources": [
            {
                **identity,
                "sha256": scan["sha256"],
                "frames": scan["frames"],
                "published_split": files[scan["file"]]["split"],
                "published_split_value_source": "source_filename",
                "scan_error": scan["error"],
            }
            for identity, scan in zip(inputs["flattened"], scans, strict=True)
        ],
        "counts": counts,
        "counts_by_split": {split: _status_counts(c) for split, c in sorted(by_split.items())},
        "examples": {status: ex.to_json(f"{status} frames") for status, ex in examples.items()},
        "energy_consistency": {
            "compared": energy.get("compared", 0),
            "agree": energy.get("agree", 0),
            "used_for_mapping": False,
        },
        "complete_coverage": not blockers,
        "coverage_blockers": blockers,
        "ordering": ordering,
        "nebdft2k_crosscheck": _set_comparison(set(edges), set(nebdft2k), limit),
        "edges": {"path": "edges.json", "sha256": _sha256_file(edges_path), "count": len(edges)},
        "shards": shards,
    }


# --------------------------------------------------------------------------- validation


def validate_manifest(root: Path, manifest: dict[str, Any]) -> tuple[bool, str]:
    """Every shard and the edge table must exist with the recorded SHA-256."""

    if manifest.get("status") != "complete":
        return False, "manifest status is not complete"
    for item in [*manifest.get("shards", []), manifest.get("edges", {})]:
        path = root / item.get("path", "")
        if not path.is_file() or _sha256_file(path) != item.get("sha256"):
            return False, f"missing or modified index file {item.get('path')}"
    return True, "all shards verified"


def evaluate_index(
    output_root: Path, settings: dict[str, Any], inputs: dict[str, Any], data_root: Path
) -> dict[str, Any]:
    """State of the provenance index for the current inputs, for Gate G0.

    Returns ``status`` in {valid, missing, stale, invalid, not_evaluated}; only
    ``valid`` carries counts.  Stale or modified indexes are never reused.
    """

    if inputs.get("archive") is None or not inputs.get("flattened"):
        return {"status": "not_evaluated", "reason": "flattened files or raw archive absent"}
    index_id = compute_index_id(inputs, settings)
    base = output_root / settings["mplitrj_provenance_dir"]
    root = base / index_id
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        others = sorted(
            p.name for p in base.iterdir() if p.is_dir() and p.name != index_id
        ) if base.is_dir() else []
        return {
            "status": "stale" if others else "missing",
            "expected_index_id": index_id,
            "other_index_ids": others[: settings["example_limit"]],
            "reason": (
                "only indexes built for other inputs, code or tolerances exist"
                if others
                else "no provenance index has been built"
            ),
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "expected_index_id": index_id, "reason": str(exc)}
    expected = {
        "index_id": index_id,
        "inputs": inputs,
        "algorithm": algorithm_identity(),
        "tolerances": tolerances(settings),
        "schema_version": SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return {"status": "invalid", "expected_index_id": index_id, "reason": f"{key} mismatch"}
    try:
        return _evaluate_valid_manifest(root, manifest, manifest_path, settings, data_root, index_id)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {
            "status": "invalid",
            "expected_index_id": index_id,
            "reason": f"malformed manifest or index files: {type(exc).__name__}: {exc}",
        }


def _evaluate_valid_manifest(
    root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    settings: dict[str, Any],
    data_root: Path,
    index_id: str,
) -> dict[str, Any]:
    ok, reason = validate_manifest(root, manifest)
    if not ok:
        return {"status": "invalid", "expected_index_id": index_id, "reason": reason}
    if settings.get("provenance_verify_input_sha256", True):
        # Size and mtime can survive a content change (cp -p, rsync -t), so the
        # exact hashes recorded at build time are re-checked.
        for item in [manifest["source_archive"], *manifest["flattened_sources"]]:
            path = data_root / item["identifier"]
            if _sha256_file(path) != item["sha256"]:
                return {
                    "status": "invalid",
                    "expected_index_id": index_id,
                    "reason": f"content of {item['identifier']} differs from the indexed SHA-256",
                }
    edges = json.loads((root / manifest["edges"]["path"]).read_text(encoding="utf-8"))
    unresolved = BoundedExamples(settings["example_limit"])
    for status in ("ambiguous", "unmapped", "error"):
        unresolved.absorb(BoundedExamples.from_state({
            "count": manifest["examples"][status]["count"],
            "limit": settings["example_limit"],
            "examples": [item["frame"] for item in manifest["examples"][status]["examples"]],
        }))
    return {
        "status": "valid",
        "index_id": index_id,
        "path": str(root),
        "manifest_sha256": _sha256_file(manifest_path),
        "counts": manifest["counts"],
        "complete_coverage": manifest["complete_coverage"],
        "coverage_blockers": manifest["coverage_blockers"],
        "unresolved_frames": unresolved.to_state(),
        "edge_frame_counts": edges["edge_frame_counts"],
    }
