import hashlib
import importlib.util
import json
import multiprocessing
import os
import shutil
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from test_dataset_audit import _config, _data_tree

from e3sse.data import dataset_audit
from e3sse.data.dataset_audit import (
    DATASETS,
    FrameSchema,
    G0ExecutionError,
    IdentifierTracker,
    _mplitrj_file_unit,
    resolve_workers,
    run_audit,
)
from e3sse.data.litraj_provenance import BoundedExamples

RUNTIME_KEYS = {"started_at", "completed_at", "execution", "command"}
CHECK_G0 = Path(__file__).resolve().parents[1] / "scripts" / "check_g0.py"


def _scientific(value):
    """Drop operational metadata (timestamps, timings, command line) recursively."""

    if isinstance(value, dict):
        return {k: _scientific(v) for k, v in value.items() if k not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [_scientific(item) for item in value]
    return value


def _messages():
    messages: list[str] = []
    return messages, messages.append


def _mtimes(root: Path) -> dict[Path, int]:
    return {path: path.stat().st_mtime_ns for path in root.rglob("*.json")}


def _snapshot(root: Path) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def fork_workers(monkeypatch):
    """Fork start method so monkeypatched failures reach worker processes."""

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method unavailable")
    monkeypatch.setattr(dataset_audit, "_START_METHOD", "fork")


# --------------------------------------------------------------------------- equivalence


def test_serial_and_parallel_audits_are_scientifically_identical(tmp_path):
    config = _config(tmp_path, _data_tree(tmp_path))

    serial, path = run_audit(config, workers=1)
    shutil.rmtree(path.parent)
    parallel, parallel_path = run_audit(config, workers=2)

    assert parallel_path == path  # the worker count never changes the run ID
    assert serial["execution"]["workers"] == 1
    assert parallel["execution"]["workers"] == 2
    assert _scientific(serial) == _scientific(parallel)
    for name in DATASETS:
        checkpoint = path.parent / "datasets" / f"{name}.json"
        assert json.loads(checkpoint.read_text())["_unit_fingerprint"]


def test_output_order_is_canonical(tmp_path):
    report, _ = run_audit(_config(tmp_path, _data_tree(tmp_path)), workers=2)
    mpli = next(item for item in report["datasets"] if item["dataset"] == "MPLiTrj")

    assert [item["dataset"] for item in report["datasets"]] == list(DATASETS)
    assert list(report["gate_summary"]) == list(DATASETS)
    assert [item["file"] for item in mpli["dataset_summary"]["files"]] == [
        "MPLiTrj_test.xyz",
        "MPLiTrj_train.xyz",
        "MPLiTrj_val.xyz",
    ]


def test_state_merge_reproduces_serial_aggregation():
    serial = IdentifierTracker("material", "required", "", 3, keep_values=True)
    parts = [IdentifierTracker("material", "required", "", 3, keep_values=True) for _ in range(2)]
    records = [("mp-1", "a"), (None, "b"), ("mp-2", "c"), (None, "d"), (None, "e"), (None, "f")]
    for index, (value, label) in enumerate(records):
        for tracker in (serial, parts[index // 3]):
            tracker.observe(value, "material_id", "source_provided", label)
    merged = IdentifierTracker("material", "required", "", 3, keep_values=True)
    for part in parts:
        merged.absorb(IdentifierTracker.from_state(json.loads(json.dumps(part.to_state()))))

    assert merged.to_json() == serial.to_json()
    assert merged.missing.examples == ["b", "d", "e"]
    examples = BoundedExamples(2)
    examples.absorb(BoundedExamples.from_state({"count": 5, "limit": 2, "examples": ["x", "y"]}))
    assert examples.to_state() == {"count": 5, "limit": 2, "examples": ["x", "y"]}


# --------------------------------------------------------------------------- failures and resume


def test_worker_failure_propagates_and_resume_reuses_completed_units(
    tmp_path, monkeypatch, fork_workers
):
    config = _config(tmp_path, _data_tree(tmp_path))

    def crash(*args, **kwargs):
        raise RuntimeError("synthetic FPMD worker crash")

    monkeypatch.setattr(dataset_audit, "_audit_fpmd", crash)
    with pytest.raises(G0ExecutionError) as excinfo:
        run_audit(config, workers=2)

    run_root = excinfo.value.failures_path.parent
    assert set(excinfo.value.failures) == {"FPMD"}
    assert "synthetic FPMD worker crash" in excinfo.value.failures["FPMD"]
    assert not (run_root / "audit.json").exists()  # no partial canonical report
    failures = json.loads((run_root / "failures.json").read_text())
    assert failures["failures"] == excinfo.value.failures
    completed = _mtimes(run_root / "datasets")
    assert run_root / "datasets" / "nebDFT2k.json" in completed
    assert run_root / "datasets" / "MPLiTrj.json" in completed
    assert run_root / "datasets" / "FPMD.json" not in completed

    monkeypatch.undo()
    messages, progress = _messages()
    report, path = run_audit(config, workers=2, resume=True, progress=progress)

    assert "G0 FPMD: starting" in messages
    assert "G0 nebDFT2k: reused checkpoint" in messages
    assert "G0 MPLiTrj: reused checkpoint" in messages
    assert {p: m for p, m in _mtimes(run_root / "datasets").items() if p in completed} == completed
    assert not (run_root / "failures.json").exists()
    assert report["status"] == "partial"
    assert path == run_root / "audit.json"


def test_resume_after_partial_parallel_run_recomputes_only_missing_units(tmp_path):
    config = _config(tmp_path, _data_tree(tmp_path))
    first, path = run_audit(config, workers=2)
    run_root = path.parent
    path.unlink()
    (run_root / "checkpoints" / "MPLiTrj" / "MPLiTrj_val.json").unlink()
    (run_root / "datasets" / "MPLiTrj.json").unlink()
    kept = _mtimes(run_root / "checkpoints")

    messages, progress = _messages()
    resumed, _ = run_audit(config, workers=2, resume=True, progress=progress)

    assert "G0 MPLiTrj/MPLiTrj_val: starting" in messages
    assert "G0 MPLiTrj/MPLiTrj_train: reused checkpoint" in messages
    assert "G0 MPLiTrj/MPLiTrj_test: reused checkpoint" in messages
    assert "G0 MPLiTrj/provenance_sample: reused checkpoint" in messages
    for name in ("nebDFT2k", "FPMD", "BVEL13k", "nebBVSE122k"):
        assert f"G0 {name}: reused checkpoint" in messages
    for checkpoint, mtime in kept.items():
        assert checkpoint.stat().st_mtime_ns == mtime
    assert _scientific(resumed) == _scientific(first)


def test_worker_count_change_reuses_checkpoints(tmp_path):
    config = _config(tmp_path, _data_tree(tmp_path))
    _, path = run_audit(config, workers=1)
    path.unlink()

    messages, progress = _messages()
    run_audit(config, workers=2, resume=True, progress=progress)

    assert not [m for m in messages if m.endswith(": starting")]


def test_stale_or_corrupt_checkpoints_are_recomputed(tmp_path):
    config = _config(tmp_path, _data_tree(tmp_path))
    first, path = run_audit(config, workers=1)
    run_root = path.parent
    path.unlink()
    stale = run_root / "datasets" / "BVEL13k.json"
    stored = json.loads(stale.read_text())
    stored["_unit_fingerprint"] = "0" * 64
    stale.write_text(json.dumps(stored))
    # A valid dataset-level checkpoint supersedes its file units, so drop it to
    # exercise the corrupt file-unit checkpoint.
    (run_root / "datasets" / "MPLiTrj.json").unlink()
    (run_root / "checkpoints" / "MPLiTrj" / "MPLiTrj_test.json").write_text("{truncated")
    sidecar = run_root / "linkage" / "nebDFT2k.json"
    sidecar.write_text(json.dumps({"values": {}}))

    messages, progress = _messages()
    resumed, _ = run_audit(config, workers=1, resume=True, progress=progress)

    assert "G0 BVEL13k: starting" in messages
    assert "G0 MPLiTrj/MPLiTrj_test: starting" in messages
    assert "G0 nebDFT2k: starting" in messages  # linkage sidecar no longer matched
    assert "G0 FPMD: reused checkpoint" in messages
    assert "G0 MPLiTrj/MPLiTrj_train: reused checkpoint" in messages
    assert _scientific(resumed) == _scientific(first)


# --------------------------------------------------------------------------- safety


def test_parallel_run_is_read_only_and_leaves_no_temporary_files(tmp_path):
    data = _data_tree(tmp_path)
    before = _snapshot(data)

    _, path = run_audit(_config(tmp_path, data), workers=2)

    assert _snapshot(data) == before
    assert not list(path.parent.rglob("*.tmp"))
    assert not [p for p in data.rglob("*") if p.name.startswith(".") and p.suffix == ".tmp"]


def test_file_unit_state_does_not_grow_with_frame_count(tmp_path):
    def source(frames: int) -> Path:
        path = tmp_path / f"n{frames}" / "MPLiTrj_train.xyz"
        path.parent.mkdir()
        atoms_list = []
        for index in range(frames):
            atoms = Atoms("LiO", positions=[[0.01 * index, 0, 0], [1, 0, 0]], cell=[4, 4, 4], pbc=True)
            atoms.info["material_id"] = f"mp-{index % 3}"
            atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=[[0, 0, 0]] * 2)
            atoms_list.append(atoms)
        write(path, atoms_list, format="extxyz")
        return path

    settings = {**dataset_audit.DEFAULT_SETTINGS, "example_limit": 5}
    small = _mplitrj_file_unit(str(source(20)), settings, [])
    large = _mplitrj_file_unit(str(source(400)), settings, [])

    assert large["frames"] == 400
    assert FrameSchema.from_state(large["frame_schema"]).missing["stress"].count == 400
    assert abs(len(json.dumps(large)) - len(json.dumps(small))) < 200


def test_provenance_sample_is_capped_per_material(tmp_path):
    settings = {**dataset_audit.DEFAULT_SETTINGS, "provenance_max_frames_per_material": 1}
    source = tmp_path / "raw" / "MPLiTrj"
    shutil.copytree(Path(__file__).parent / "fixtures" / "g0" / "data" / "raw" / "MPLiTrj", source)

    state = _mplitrj_file_unit(str(source / "MPLiTrj_train.xyz"), settings, ["mp-1"])

    assert [frame[0] for frame in state["sample_frames"]] == ["MPLiTrj_train.xyz:0"]
    assert state["sample_seen_per_material"] == {"mp-1": 2}


# --------------------------------------------------------------------------- configuration


def test_workers_default_from_config_and_cli_override(tmp_path, capsys):
    data = _data_tree(tmp_path)
    config = _config(tmp_path, data, workers=2)
    report, path = run_audit(config)
    assert report["execution"]["workers"] == 2
    path.unlink()

    spec = importlib.util.spec_from_file_location("check_g0", CHECK_G0)
    check_g0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_g0)
    assert check_g0.main(["--config", str(config), "--workers", "1", "--resume"]) == 0

    output = capsys.readouterr().out
    assert "Gate G0: partial" in output
    assert json.loads(path.read_text())["execution"]["workers"] == 1
    with pytest.raises(SystemExit):
        check_g0.main(["--config", str(config), "--workers", "0"])


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_worker_counts_are_rejected(value):
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        resolve_workers(value, {})


def test_invalid_config_worker_count_is_rejected(tmp_path):
    config = _config(tmp_path, _data_tree(tmp_path), workers=0)

    with pytest.raises(ValueError, match="workers"):
        run_audit(config)
    assert resolve_workers(None, {"workers": 3}) == 3
    assert resolve_workers(None, {}) == 1


def test_killed_worker_is_reported_once_and_resume_recovers(tmp_path, monkeypatch, fork_workers):
    config = _config(tmp_path, _data_tree(tmp_path))
    _, previous = run_audit(config, workers=1)

    def die(*args, **kwargs):
        os._exit(137)  # simulate an out-of-memory kill

    monkeypatch.setattr(dataset_audit, "_audit_fpmd", die)
    with pytest.raises(G0ExecutionError) as excinfo:
        run_audit(config, workers=2)

    failures = excinfo.value.failures
    assert "out of memory" in failures["worker_pool"]
    assert all(
        message.startswith(("not completed", "not run", "a worker process"))
        for message in failures.values()
    )
    assert not previous.exists()  # superseded, never left beside failures.json
    assert (previous.parent / "audit.previous.json").exists()

    monkeypatch.undo()
    report, path = run_audit(config, workers=2, resume=True)
    assert path == previous and report["status"] == "partial"


def test_config_only_worker_change_restamps_provenance(tmp_path):
    data = _data_tree(tmp_path)
    config = _config(tmp_path, data, workers=1)
    first, path = run_audit(config)
    raw = json.loads(config.read_text())
    raw["audit"]["workers"] = 2
    config.write_text(json.dumps(raw))

    messages, progress = _messages()
    second, second_path = run_audit(config, resume=True, progress=progress)

    assert second_path == path
    assert not [m for m in messages if m.endswith(": starting")]  # every unit reused
    assert second["configuration"]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert second["execution"]["workers"] == 2
    strip = {"configuration"}
    assert {k: v for k, v in _scientific(second).items() if k not in strip} == {
        k: v for k, v in _scientific(first).items() if k not in strip
    }


def test_merge_across_files_matches_single_pass(tmp_path):
    settings = {
        **dataset_audit.DEFAULT_SETTINGS,
        "example_limit": 3,
        "provenance_max_frames_per_material": 3,
    }

    def frames(start: int, count: int) -> list[Atoms]:
        out = []
        for index in range(start, start + count):
            atoms = Atoms("LiO", positions=[[0.01 * index, 0, 0], [1, 0, 0]], cell=[4, 4, 4], pbc=True)
            atoms.info["material_id"] = "mp-1"
            out.append(atoms)  # no calculator: every frame misses energy/forces
        return out

    split_root = tmp_path / "split"
    split_root.mkdir()
    write(split_root / "MPLiTrj_a_train.xyz", frames(0, 2), format="extxyz")
    write(split_root / "MPLiTrj_b_train.xyz", frames(2, 4), format="extxyz")
    states = [
        _mplitrj_file_unit(str(path), settings, ["mp-1"])
        for path in dataset_audit.mplitrj_source_files(sorted(split_root.iterdir()))
    ]
    merged = dataset_audit._merge_mplitrj_units(states, settings)

    # The per-material cap and first-N examples fall across the file boundary.
    assert [f[0] for f in merged["sample_frames"]] == [
        "MPLiTrj_a_train.xyz:0",
        "MPLiTrj_a_train.xyz:1",
        "MPLiTrj_b_train.xyz:0",
    ]
    assert merged["frames_beyond_cap"] == 3
    energy = merged["frame_schema"].missing["energy"]
    assert energy.count == 6
    assert energy.examples == ["MPLiTrj_a_train.xyz:0", "MPLiTrj_a_train.xyz:1", "MPLiTrj_b_train.xyz:0"]
    assert merged["trackers"]["material"].values == {"mp-1": 6}
    assert [item["file"] for item in merged["per_file"]] == ["MPLiTrj_a_train.xyz", "MPLiTrj_b_train.xyz"]
