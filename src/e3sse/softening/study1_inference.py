"""Study-1 MLIP inference over published-train MPLiTrj frames, as sufficient statistics.

Runs inside one frozen model's isolated runtime (or the project environment for the
model-independent DFT reference).  Data flow per model:

    frozen membership + frame index -> fixed contiguous shards (frames_per_shard)
      -> spawned workers (one calculator per worker, frames read by byte offset)
      -> per-frame, per-element Eq. 17 sums (frame_statistics) -> shard checkpoints
      -> canonical merge -> frame_statistics.npz + JSON manifest

Scientific identity (reuse is governed by it): frozen protocol ID and file SHA-256,
the code that computes results (content hashes), the source file SHA-256, frame
index SHA-256, model checkpoint/package/adapter identity, environment-lock SHA-256,
and the fixed shard size.  The Git SHA is recorded in every manifest; worker count,
in-flight bound, memory reserve and host are operational and never enter identity.
Per-process thread settings enter shard fingerprints (e3sse.execution).

Batching: each worker reuses one ASE calculator and evaluates frames one at a time.
The frozen adapters expose single-structure ASE calculators; package-specific batch
paths (graph batching, padding) change reduction order and memory, so they are not
used.  Results do not depend on worker count or completion order.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from e3sse.execution.sharded_execution import WorkShard, canonical_sha256, make_shard
from e3sse.softening.extxyz_frames import (
    FRAME_INDEX_SCHEMA,
    FrameFormatError,
    build_atoms,
    parse_labelled_frame,
    read_frame_text,
    scan_frame_index,
)
from e3sse.softening.frame_statistics import (
    STATISTICS_SCHEMA,
    atomic_numbers_sha256,
    element_statistics,
)

INFERENCE_MANIFEST_SCHEMA = "e3sse.study1.inference.v1"
PILOT_GATE_SCHEMA = "e3sse.study1.pilot-gate.v2"
PILOT_GATE_BINDING_SCHEMA = "e3sse.study1.pilot-gate-binding.v1"
PILOT_MIN_FORCE = 0.1  # eV/A, the frozen informative-force threshold
PILOT_MIN_INFORMATIVE_PAIRS = 3
FLOAT32_EPS = 2.0**-23
REFERENCE_RUN = "dft_reference"
SHARD_NAMESPACE = "study1.frames"
# Files whose content determines a frame-statistics result.
INFERENCE_CODE_FILES = (
    "src/e3sse/softening/extxyz_frames.py",
    "src/e3sse/softening/frame_statistics.py",
    "src/e3sse/softening/study1_inference.py",
    "src/e3sse/protocol/adapters.py",
)


def code_sha256(repository: Path, files: Sequence[str]) -> dict[str, str]:
    result = {}
    for name in files:
        digest = hashlib.sha256((Path(repository) / name).read_bytes()).hexdigest()
        result[name] = digest
    return result


# --------------------------------------------------------------------------- frame index


def build_frame_index(
    source_path: Path,
    source: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    resume: bool,
) -> tuple[dict[str, Any], Any]:
    """Byte-offset index of one frozen source, bound to its SHA-256 and the membership.

    Returns (manifest, arrays).  An existing index is reused only when its manifest,
    array digest and the source file's size and mtime are unchanged.
    """

    import numpy as np

    from e3sse.execution.sharded_execution import publish_json_once

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    arrays_path = output_dir / "frame_offsets.npz"
    stat = Path(source_path).stat()
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; use --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        arrays = load_frame_index(output_dir, source)
        if (manifest["source_bytes"], manifest["source_mtime_ns"]) != (
            stat.st_size, stat.st_mtime_ns
        ):
            raise RuntimeError("source file size/mtime changed since the frame index was built")
        return manifest, arrays
    digest, entries = scan_frame_index(Path(source_path))
    if digest != source["sha256"]:
        raise RuntimeError("source SHA-256 differs from the frozen source_file_sha256")
    if len(entries) != source["frames"]:
        raise RuntimeError("source frame count differs from the frozen provenance")
    for row in frames:
        entry = entries[row["source_frame_index"]]
        if entry.material_id != row["material_id"]:
            raise RuntimeError(f"header material_id differs from membership: {row['frame_id']}")
    arrays = {
        "frame_index": np.array([e.frame_index for e in entries], dtype=np.int64),
        "byte_offset": np.array([e.byte_offset for e in entries], dtype=np.int64),
        "byte_length": np.array([e.byte_length for e in entries], dtype=np.int64),
        "n_atoms": np.array([e.n_atoms for e in entries], dtype=np.int64),
        "cell_volume": np.array([e.cell_volume for e in entries], dtype=np.float64),
        "material_id": np.array([e.material_id for e in entries], dtype=str),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_npz_once(arrays_path, arrays)
    manifest = {
        "schema_version": FRAME_INDEX_SCHEMA,
        "classification": "structural_index_no_labels_parsed",
        "source_identifier": source["identifier"],
        "source_file_sha256": digest,
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "frames": len(entries),
        "membership_frames_checked": len(frames),
        "arrays_file": arrays_path.name,
        "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        "columns": sorted(arrays),
    }
    publish_json_once(manifest_path, manifest, root=output_dir)
    return manifest, arrays


def load_frame_index(output_dir: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    manifest = json.loads((Path(output_dir) / "manifest.json").read_text(encoding="utf-8"))
    path = Path(output_dir) / manifest["arrays_file"]
    if (
        manifest.get("schema_version") != FRAME_INDEX_SCHEMA
        or manifest.get("source_file_sha256") != source["sha256"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["arrays_sha256"]
    ):
        raise RuntimeError("stale or corrupt frame index")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# --------------------------------------------------------------------------- shards


def plan_frame_shards(
    frames: Sequence[Mapping[str, Any]], index: Mapping[str, Any], frames_per_shard: int
) -> list[WorkShard]:
    """Contiguous runs of ``frames_per_shard`` frames in source order (I/O-local).

    Membership is fixed by the configuration, so shard IDs never depend on workers.
    """

    if frames_per_shard < 1:
        raise ValueError("frames_per_shard must be positive")
    ordered = sorted(frames, key=lambda row: row["source_frame_index"])
    shards = []
    for start in range(0, len(ordered), frames_per_shard):
        block = ordered[start : start + frames_per_shard]
        entries = []
        for row in block:
            i = int(row["source_frame_index"])
            entries.append([
                row["frame_id"], i, int(index["byte_offset"][i]), int(index["byte_length"][i]),
                int(index["n_atoms"][i]), row["material_id"],
            ])
        identity = {
            "source_file_sha256": block[0]["source_file_sha256"],
            "frame_ids": [row["frame_id"] for row in block],
        }
        shards.append(make_shard(SHARD_NAMESPACE, identity, {"frames": entries}))
    return shards


# --------------------------------------------------------------------------- workers


def _torch_parameters(calc: Any) -> tuple[str | None, str | None]:
    """(device, dtype) of the calculator's torch model, read from its parameters."""

    for attribute in ("model", "potential", "models"):
        module = getattr(calc, attribute, None)
        if isinstance(module, (list, tuple)):
            module = module[0] if module else None
        parameters = getattr(module, "parameters", None)
        if callable(parameters):
            try:
                first = next(iter(parameters()))
            except StopIteration:
                continue
            return str(first.device.type), str(first.dtype).replace("torch.", "")
    return None, None


def requested_dtype(model: Mapping[str, Any]) -> str:
    return str(model.get("load_options", {}).get("default_dtype", "float32"))


def inference_worker_setup(
    model: Mapping[str, Any] | None, checkpoint: str | None, source_path: str, device: str
) -> dict[str, Any]:
    """Once per worker: verify and load the checkpoint, open the source read-only."""

    calc = None
    if model is not None:
        from e3sse.protocol.adapters import build_calculator

        calc = build_calculator(model, Path(checkpoint), device=device)
        effective = _torch_parameters(calc)
        if effective != (device, requested_dtype(model)):
            raise RuntimeError(
                f"effective device/dtype {effective} differs from requested "
                f"({device}, {requested_dtype(model)}); no fallback is registered"
            )
    return {"calc": calc, "handle": open(source_path, "rb")}


def _frame_record(entry: Sequence[Any], frame: Mapping[str, Any]) -> dict[str, Any]:
    frame_id, index, _, _, n_atoms, material = entry
    if frame["n_atoms"] != n_atoms or frame["material_id"] != material:
        raise FrameFormatError(f"{frame_id}: frame content differs from the index")
    return {
        "frame_id": frame_id,
        "frame_index": index,
        "atomic_numbers_sha256": atomic_numbers_sha256(frame["numbers"]),
    }


def _resource_failure(exc: BaseException) -> bool:
    """Out-of-memory conditions depend on the host, never on the frame or model."""

    text = str(exc).lower()
    return isinstance(exc, MemoryError) or (
        isinstance(exc, RuntimeError)
        and any(token in text for token in ("can't allocate", "out of memory", "alloc"))
    )


def infer_shard(payload: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Per-frame statistics for one shard.

    A model exception or non-finite prediction on a frame is a scientific status of
    that frame (``failed``, with metadata); a parse or provenance mismatch is an
    infrastructure failure and raises, so the shard is never stored.
    """

    import numpy as np

    calc = context["calc"]
    records = []
    for entry in payload["frames"]:
        text = read_frame_text(context["handle"], entry[2], entry[3])
        frame = parse_labelled_frame(text)
        record = _frame_record(entry, frame)
        if calc is None:
            record["n_atoms"] = frame["n_atoms"]
            record["dft_energy_eV"] = frame["dft_energy"]
            record["elements"] = [
                [row["Z"], row["n_atoms"], row["sxx"], row["n_informative"]]
                for row in element_statistics(frame["numbers"], frame["dft_forces"])
            ]
            records.append(record)
            continue
        atoms = build_atoms(frame)
        atoms.calc = calc
        try:
            ml_energy = float(atoms.get_potential_energy())
            ml_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            if not math.isfinite(ml_energy) or not np.isfinite(ml_forces).all():
                raise FloatingPointError("non-finite ML prediction")
        except Exception as exc:  # recorded as the frame's scientific status
            if _resource_failure(exc):
                raise  # infrastructure: the shard is not stored and reruns on resume
            record.update({
                "status": "failed", "ml_energy_eV": None, "elements": None,
                "error_type": type(exc).__name__, "error_message": str(exc)[:500],
            })
            records.append(record)
            continue
        record["status"] = "success"
        record["ml_energy_eV"] = ml_energy
        record["elements"] = [
            [row["Z"], row["n_atoms"], row["sxy"], row["sxx"], row["syy"], row["n_informative"]]
            for row in element_statistics(frame["numbers"], frame["dft_forces"], ml_forces)
        ]
        records.append(record)
    return {"schema_version": STATISTICS_SCHEMA, "frames": records}


# --------------------------------------------------------------------------- pilot gate


def pilot_frames(frames: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    """A fixed, label-independent pilot batch: frames ranked by SHA-256 of their ID."""

    ranked = sorted(
        frames,
        key=lambda row: hashlib.sha256(f"e3sse.study1.pilot.v1\0{row['frame_id']}".encode())
        .hexdigest(),
    )
    return ranked[:count]


def run_pilot_gate(
    model: Mapping[str, Any],
    calc: Any,
    handle: Any,
    entries: Sequence[Sequence[Any]],
    *,
    fd_step: float,
    device: str,
) -> dict[str, Any]:
    """Frozen ``pilot_gate_before_production`` on a few published-train frames.

    (1) finite-difference sign of the ML force on three components per frame;
    (2) positive Eq. 17 force-parity slope over the batch; (3) requested equals
    effective device/dtype.  It fixes no threshold and changes no estimate.
    """

    import numpy as np

    from e3sse.softening.frame_statistics import combine

    fd_pairs, energies_seen, sums = [], [], []
    for entry in entries:
        frame = parse_labelled_frame(read_frame_text(handle, entry[2], entry[3]))
        _frame_record(entry, frame)
        atoms = build_atoms(frame)
        atoms.calc = calc
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        sums.append(combine(element_statistics(frame["numbers"], frame["dft_forces"], forces),
                            "all"))
        # Largest-|F| components: the sign test is informative there.
        for flat in np.argsort(-np.abs(forces).ravel(), kind="stable")[:3]:
            atom, axis = divmod(int(flat), 3)
            energies = []
            for sign in (1.0, -1.0):
                moved = atoms.copy()
                moved.positions[atom, axis] += sign * fd_step
                moved.calc = calc
                energies.append(float(moved.get_potential_energy()))
            fd_pairs.append([-(energies[0] - energies[1]) / (2 * fd_step), float(forces[atom, axis])])
            energies_seen.append(max(abs(e) for e in energies))
    sxy = math.fsum(item["sxy"] for item in sums)
    sxx = math.fsum(item["sxx"] for item in sums)
    effective = _torch_parameters(calc)
    # A pair is informative when the energy change 2h|F| clearly exceeds float32
    # rounding of the total energy (8 ulp); the frozen gate is a sign check only.
    informative = [
        (a, b) for (a, b), energy in zip(fd_pairs, energies_seen, strict=True)
        if abs(b) >= PILOT_MIN_FORCE and 2 * fd_step * abs(b) > 8 * FLOAT32_EPS * energy
    ]
    checks = {
        "finite_difference_force_sign": len(informative) >= PILOT_MIN_INFORMATIVE_PAIRS
        and all(a * b > 0 for a, b in informative),
        "positive_force_parity_slope": sxx > 0 and sxy / sxx > 0,
        "requested_equals_effective_device_dtype": effective == (device, requested_dtype(model)),
    }
    return {
        "schema_version": PILOT_GATE_SCHEMA,
        "classification": "frozen pilot_gate_before_production; no threshold is fitted",
        "canonical_name": model["canonical_name"],
        "weights_sha256": model["weights_sha256"],
        "frame_ids": [entry[0] for entry in entries],
        "fd_step_angstrom": fd_step,
        "fd_force_pairs_eV_per_A": fd_pairs,
        "informative_pairs": len(informative),
        "fd_to_analytic_ratio_diagnostic": [a / b for a, b in informative],
        "requested_device_dtype": [device, requested_dtype(model)],
        "effective_device_dtype": list(effective),
        "checks": checks,
        "passed": all(checks.values()),
    }


class PilotGateBindingError(RuntimeError):
    """A pilot-gate report is missing, failed, or differs from the one bound to the run."""


_PILOT_REPORT_NAME = re.compile(r"pilot_gate/[0-9a-f]{16}\.json")
_BINDING_KEYS = ("schema_version", "classification", "report", "report_sha256",
                 "gate_key_sha256", "identity_sha256", "status")


def pilot_gate_binding(
    run_dir: Path, report: str, gate_key_sha256: str, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Content binding of the pilot-gate report that authorizes a model run.

    ``report`` is relative to the run directory (no host path).  The SHA-256 of the
    report bytes is the authoritative binding; ``identity_sha256`` ties it to the run's
    scientific identity.  It is provenance, not part of that identity.
    """

    if not isinstance(report, str) or _PILOT_REPORT_NAME.fullmatch(report) is None:
        raise PilotGateBindingError(f"invalid pilot-gate report name {report!r}")
    path = Path(run_dir) / report
    if not path.is_file():
        raise PilotGateBindingError(f"pilot-gate report {report} is missing")
    data = path.read_bytes()
    try:
        content = json.loads(data)
    except ValueError as exc:
        raise PilotGateBindingError(f"pilot-gate report {report} is not valid JSON") from exc
    if (content.get("schema_version"), content.get("gate_key_sha256")) != (
        PILOT_GATE_SCHEMA, gate_key_sha256
    ):
        raise PilotGateBindingError(f"pilot-gate report {report} belongs to another gate")
    return {
        "schema_version": PILOT_GATE_BINDING_SCHEMA,
        "classification": "operational provenance; not part of the scientific identity",
        "report": report,
        "report_sha256": hashlib.sha256(data).hexdigest(),
        "gate_key_sha256": gate_key_sha256,
        "identity_sha256": canonical_sha256(dict(identity)),
        "status": "passed" if content.get("passed") is True else "failed",
    }


def verify_pilot_gate_binding(
    run_dir: Path, binding: Mapping[str, Any] | None, identity: Mapping[str, Any]
) -> None:
    """Fail unless ``binding`` names a passed report, for this run's scientific
    identity, whose bytes still hash the same."""

    if (not isinstance(binding, Mapping)
            or binding.get("schema_version") != PILOT_GATE_BINDING_SCHEMA
            or sorted(binding) != sorted(_BINDING_KEYS)):
        raise PilotGateBindingError("no valid pilot-gate binding recorded for this run")
    if binding["identity_sha256"] != canonical_sha256(dict(identity)):
        raise PilotGateBindingError("pilot-gate binding belongs to another run identity")
    current = pilot_gate_binding(run_dir, binding["report"], binding["gate_key_sha256"], identity)
    if current["report_sha256"] != binding["report_sha256"]:
        raise PilotGateBindingError(
            f"pilot-gate report {binding['report']} changed after it was bound "
            f"({current['report_sha256']} != {binding['report_sha256']})"
        )
    if current != dict(binding) or current["status"] != "passed":
        raise PilotGateBindingError(f"pilot-gate report {binding['report']} did not pass")


# --------------------------------------------------------------------------- runtime


RECORDED_DEPENDENCIES = ("ase", "dgl", "e3nn", "numpy", "pymatgen", "torch", "torch_geometric")


def installed_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions = {}
    for name in sorted(names):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _lock_pins(lock: Path) -> dict[str, str]:
    """``name==version`` lines of a pip/uv freeze lock (canonical names)."""

    pins = {}
    for line in Path(lock).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line and not line.startswith("-"):
            name, version = line.split("==", 1)
            pins[name.strip().lower().replace("_", "-")] = version.split(";")[0].strip()
    return pins


def verify_runtime(
    environment: Mapping[str, Any], model: Mapping[str, Any] | None, lock: Path | None = None
) -> dict:
    """The running interpreter must be the model's frozen runtime environment.

    Checks Python, the registry pins and environment variables and, when the lock
    file is given (SHA-256 checked against the protocol by the caller), every
    ``name==version`` it records against the installed distributions.
    """

    python = sys.version.split()[0]
    installed = installed_versions(list(environment["pinned_packages"]))
    wrong = sorted(p for p, v in environment["pinned_packages"].items() if installed[p] != v)
    if python != environment["python_version"] or wrong:
        raise RuntimeError(
            f"not the frozen runtime: python {python} (frozen "
            f"{environment['python_version']}), mismatched packages {wrong}"
        )
    for name, value in environment.get("environment_variables", {}).items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"frozen runtime requires {name}={value}")
    if model is not None and installed.get(model["package_name"]) != model["package_version"]:
        raise RuntimeError("model package version differs from the frozen registry")
    drift = []
    if lock is not None:
        actual = {
            dist.metadata["Name"].lower().replace("_", "-"): dist.version
            for dist in importlib.metadata.distributions()
        }
        drift = sorted(name for name, version in _lock_pins(lock).items()
                       if actual.get(name) != version)
        if drift:
            raise RuntimeError(f"installed packages differ from the frozen lock: {drift[:10]}")
    return {"python_version": python, "pinned_packages": installed,
            "recorded_dependencies": installed_versions(RECORDED_DEPENDENCIES),
            "lock_verified": lock is not None}


# --------------------------------------------------------------------------- merge


def merge_frame_statistics(results: Mapping[str, Any], *, reference: bool) -> dict[str, Any]:
    """Columnar arrays from shard results, ordered by source frame index."""

    import numpy as np

    frames = sorted(
        (record for result in results.values() for record in result["frames"]),
        key=lambda record: record["frame_index"],
    )
    indices = [record["frame_index"] for record in frames]
    if len(set(indices)) != len(indices):
        raise RuntimeError("a frame appears in more than one shard")
    element_rows, element_frame = [], []
    for row_number, record in enumerate(frames):
        for row in record.get("elements") or []:
            element_rows.append(row)
            element_frame.append(row_number)
    arrays = {
        "frame_index": np.array(indices, dtype=np.int64),
        "atomic_numbers_sha256": np.array([r["atomic_numbers_sha256"] for r in frames], dtype=str),
        "element_frame_row": np.array(element_frame, dtype=np.int64),
    }
    columns = ("Z", "n_atoms", "sxx", "n_informative") if reference else (
        "Z", "n_atoms", "sxy", "sxx", "syy", "n_informative"
    )
    table = np.array(element_rows, dtype=np.float64).reshape(-1, len(columns))
    for position, name in enumerate(columns):
        dtype = np.float64 if name.startswith("s") else np.int64
        arrays[f"element_{name}"] = table[:, position].astype(dtype)
    if reference:
        arrays["n_atoms"] = np.array([r["n_atoms"] for r in frames], dtype=np.int64)
        arrays["dft_energy_eV"] = np.array([r["dft_energy_eV"] for r in frames], dtype=np.float64)
    else:
        arrays["success"] = np.array([r["status"] == "success" for r in frames], dtype=bool)
        arrays["ml_energy_eV"] = np.array(
            [math.nan if r["ml_energy_eV"] is None else r["ml_energy_eV"] for r in frames],
            dtype=np.float64,
        )
    return arrays


def failures_of(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {key: record[key] for key in ("frame_id", "error_type", "error_message")}
            for result in results.values()
            for record in result["frames"]
            if record.get("status") == "failed"
        ),
        key=lambda item: item["frame_id"],
    )


def write_npz_once(path: Path, arrays: Mapping[str, Any]) -> str:
    """Create a byte-reproducible ``.npz`` at ``path``; return its SHA-256.

    ``numpy.savez`` stamps the current time into each zip member, so identical
    arrays would hash differently; members here carry a fixed timestamp and order.
    An existing file is accepted only if it is byte-identical (a crash between this
    write and its manifest then resumes cleanly); it is never replaced.
    """

    import io
    import zipfile

    import numpy as np

    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, buffer.getvalue())
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"{path} exists with different content; not replaced")
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            os.unlink(temporary)
    return digest


def verify_statistics_file(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(manifest, arrays) of a merged run, after checking the file digest."""

    import numpy as np

    manifest = json.loads((Path(directory) / "frame_statistics_manifest.json").read_text())
    path = Path(directory) / manifest["statistics_file"]
    if (
        manifest.get("schema_version") != INFERENCE_MANIFEST_SCHEMA
        or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["statistics_sha256"]
    ):
        raise RuntimeError(f"stale or corrupt frame statistics in {directory}")
    with np.load(path, allow_pickle=False) as data:
        return manifest, {name: data[name] for name in data.files}


def run_identity(
    *,
    protocol: Mapping[str, Any],
    protocol_json_sha256: str,
    code: Mapping[str, str],
    source: Mapping[str, Any],
    frame_index_sha256: str,
    frames_per_shard: int,
    model: Mapping[str, Any] | None,
    environment_name: str,
    environment_lock_sha256: str | None,
    device: str,
    runtime_versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        # Numerics-relevant installed versions (numpy/ASE for the reference parser).
        "runtime_versions": dict(runtime_versions or installed_versions(RECORDED_DEPENDENCIES)),
        "schema_version": INFERENCE_MANIFEST_SCHEMA,
        "statistics_schema": STATISTICS_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_json_sha256": protocol_json_sha256,
        "membership_sha256": protocol["partition_definition"]["partition_universe"][
            "membership_sha256"
        ],
        "code_sha256": dict(code),
        "source_file_sha256": source["sha256"],
        "provenance_index_id": source["provenance_index_id"],
        "frame_index_sha256": frame_index_sha256,
        "frames_per_shard": frames_per_shard,
        "run": REFERENCE_RUN if model is None else model["canonical_name"],
        "model": None if model is None else {
            key: model[key] for key in (
                "canonical_name", "weights_sha256", "package_name", "package_version",
                "adapter_version", "frozen_group",
            )
        },
        "runtime_environment": environment_name,
        "environment_lock_sha256": environment_lock_sha256,
        "device": device,
        "requested_dtype": None if model is None else requested_dtype(model),
        "outcome_rule": "published_train_only_seal_1_frozen_governance",
    }


def results_digest(results: Mapping[str, Any]) -> str:
    return canonical_sha256({key: results[key] for key in sorted(results)})
