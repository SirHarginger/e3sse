"""Focused tests for the read-only FPMD AiiDA inventory boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from e3sse.data import fpmd_aiida_inventory as inventory

CAPABILITIES = tuple(inventory.DEFAULT_REQUIRED_CAPABILITIES)


def _result(
    name: str,
    family: str,
    required: bool,
    *,
    status: str = "success",
    capabilities: set[str] | tuple[str, ...] = CAPABILITIES,
) -> dict:
    query = {
        "query_status": status,
        "archive_schema_version": "main_0001",
        "group_labels": ["starting_structures", "concatenated_trajectories"],
        "capabilities": {name: name in capabilities for name in CAPABILITIES},
    }
    return {
        "name": name,
        "family": family,
        "required": required,
        "query_status": status,
        "query": query,
    }


def _settings() -> dict:
    return {
        "required_archives": [
            {"name": "FPMD_screening.aiida", "family": "legacy", "required": True},
            {"name": "FPMD2026_a.aiida", "family": "2026", "required": False},
            {"name": "FPMD2026_b.aiida", "family": "2026", "required": False},
        ],
        "required_capabilities": list(CAPABILITIES),
    }


def _config(tmp_path: Path) -> tuple[Path, list[Path]]:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    archives = [raw / item["name"] for item in _settings()["required_archives"]]
    for index, path in enumerate(archives):
        path.write_bytes(f"immutable-{index}".encode())
    lock = tmp_path / "requirements" / "aiida-archive.lock"
    lock.parent.mkdir()
    lock.write_text("aiida-core==2.9.2\n", encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "audit.json"
    config.write_text(
        json.dumps(
            {
                "paths": {
                    "data_root": str(tmp_path / "data"),
                    "outputs_root": str(tmp_path / "outputs"),
                    "logs_root": str(tmp_path / "logs"),
                    "scratch_root": str(tmp_path / "scratch"),
                },
                "fpmd_aiida": {
                    "inventory_path": "fpmd_aiida/current.json",
                    "profile_root": "fpmd_aiida/profiles",
                    "migration_root": "fpmd_aiida/migrated",
                    "requirements_lock": "requirements/aiida-archive.lock",
                    "python_version": "3.12",
                    "aiida_version": "2.9.2",
                    "example_limit": 2,
                    "allow_derived_migration": True,
                    **_settings(),
                },
            }
        ),
        encoding="utf-8",
    )
    return config, archives


def _environment(config: dict) -> dict:
    lock = Path(inventory.settings_from_config(config)["requirements_lock"])
    value = {
        "python": "3.12.14",
        "python_executable": "/isolated/e3sse-aiida/bin/python",
        "aiida_core": "2.9.2",
        "requirements_lock": str(lock),
        "requirements_lock_sha256": inventory._sha256(lock),
    }
    value["environment_id"] = inventory._digest(value)
    return value


def _query(path: Path, *, environment: dict, settings: dict) -> dict:
    del path, settings
    return {
        "query_status": "success",
        "queried_path_kind": "immutable_source",
        "environment_id": environment["environment_id"],
        "archive_schema_version": "main_0001",
        "group_labels": ["starting_structures", "concatenated_trajectories"],
        "node_type_counts": {"data.core.structure.StructureData.": 1},
        "structure_count": 1,
        "trajectory_count": 1,
        "temperature_metadata": {
            "trajectories_with_temperature": 1,
            "distinct_temperature_values": {"count": 1, "examples": [{"value": 500}]},
        },
        "diffusion_references": {"group_count": 1},
        "msd_references": {"group_count": 1},
        "provenance_linkage": {"direct_structure_to_trajectory_links": 1},
        "material_identifiers": {"count": 1, "examples": ["material-1"]},
        "capabilities": {name: True for name in CAPABILITIES},
    }


def test_optional_2026_failure_is_nonfatal_and_family_separated():
    rows = [
        _result("FPMD_screening.aiida", "legacy", True),
        _result("FPMD2026_a.aiida", "2026", False, status="error", capabilities=set()),
        _result("FPMD2026_b.aiida", "2026", False),
    ]

    assert all(inventory.aggregate_capabilities(rows, _settings()).values())
    families = inventory.aggregate_family_capabilities(rows, _settings())
    assert all(families["legacy"].values())
    assert not families["2026"]["aiida_query_available"]


def test_required_archive_query_failure_is_fatal():
    rows = [
        _result("FPMD_screening.aiida", "legacy", True, status="error", capabilities=set()),
        _result("FPMD2026_a.aiida", "2026", False),
        _result("FPMD2026_b.aiida", "2026", False),
    ]

    assert not inventory.aggregate_capabilities(rows, _settings())["aiida_query_available"]


@pytest.mark.parametrize("missing", CAPABILITIES)
def test_each_scientific_capability_is_independent(missing):
    rows = [
        _result(
            "FPMD2026_a.aiida",
            "2026",
            True,
            capabilities=set(CAPABILITIES) - {missing},
        )
    ]
    settings = {
        "required_archives": [
            {"name": "FPMD2026_a.aiida", "family": "2026", "required": True}
        ],
        "required_capabilities": list(CAPABILITIES),
    }

    result = inventory.aggregate_capabilities(rows, settings)
    assert not result[missing]
    assert all(result[name] for name in CAPABILITIES if name != missing)


def test_observed_groups_counts_and_links_drive_capabilities():
    absent = inventory._reference_capabilities(
        structure_count=0,
        trajectory_count=0,
        distinct_temperature_count=0,
        diffusion_groups=[],
        msd_groups=[],
        direct_links=0,
        process_links=0,
    )
    present = inventory._reference_capabilities(
        structure_count=2,
        trajectory_count=3,
        distinct_temperature_count=2,
        diffusion_groups=[{"label": "diffusion_coefficients"}],
        msd_groups=[{"label": "msd"}],
        direct_links=1,
        process_links=0,
    )

    assert not any(absent.values())
    assert all(present.values())


def test_temperature_metadata_is_exact_bounded_and_deterministic():
    observations = inventory._temperature_values(
        {
            "temperature": 700,
            "temperature_unit": "K",
            "nested": {"temp_k": 500},
            "bad": {"temperature": "unknown"},
            "temperature_enabled": True,
        }
    )
    values = [
        {"value": item["value"], "unit": item.get("unit")}
        for item in observations
    ]
    assert {item["unit"] for item in values} == {"K"}

    first = inventory._bounded(values, 1, "test")
    second = inventory._bounded(reversed(values), 1, "test")
    assert first == second
    assert first["count"] == 2
    assert len(first["examples"]) == 1
    assert {item["value"] for item in values} == {500, 700}


def test_build_resume_staleness_tamper_and_raw_immutability(tmp_path, monkeypatch):
    config_path, archives = _config(tmp_path)
    monkeypatch.setattr(inventory, "environment_identity", _environment)
    before = {path: path.read_bytes() for path in archives}
    calls = 0

    def counted_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _query(*args, **kwargs)

    built, final_path = inventory.build_inventory(config_path, query_adapter=counted_query)
    resumed, resumed_path = inventory.build_inventory(
        config_path, resume=True, query_adapter=counted_query
    )

    assert built["status"] == "pass"
    assert resumed == built
    assert resumed_path == final_path
    assert calls == len(archives)
    assert before == {path: path.read_bytes() for path in archives}

    config = inventory.load_config(config_path)
    assert inventory.evaluate_inventory(config, archives)["status"] == "valid"
    archives[-1].write_bytes(b"changed")
    assert inventory.evaluate_inventory(config, archives)["status"] == "invalid"

    archives[-1].write_bytes(before[archives[-1]])
    current = Path(inventory.settings_from_config(config)["inventory_path"])
    payload = json.loads(current.read_text(encoding="utf-8"))
    payload["environment"]["python_executable"] = "/tampered/python"
    payload["self_digest"] = inventory._self_digest(payload)
    current.write_text(json.dumps(payload), encoding="utf-8")
    state = inventory.evaluate_inventory(config, archives)
    assert state["status"] == "invalid"
    assert "environment identity" in state["reason"]


def test_unavailable_environment_and_failed_query_are_reported(tmp_path, monkeypatch):
    config_path, archives = _config(tmp_path)

    def unavailable(config):
        raise RuntimeError("aiida-core unavailable")

    monkeypatch.setattr(inventory, "environment_identity", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        inventory.build_inventory(config_path)

    monkeypatch.setattr(inventory, "environment_identity", _environment)
    attempts = 0

    def failing(path, **kwargs):
        nonlocal attempts
        attempts += 1
        if path.name == "FPMD_screening.aiida":
            raise RuntimeError("query failed")
        return _query(path, **kwargs)

    first, _ = inventory.build_inventory(config_path, query_adapter=failing)
    second, _ = inventory.build_inventory(config_path, resume=True, query_adapter=failing)

    assert first["status"] == second["status"] == "partial"
    assert first["errors"]
    assert attempts == len(archives) + 1


def test_optional_2026_query_failure_is_warning_and_resumable(tmp_path, monkeypatch):
    config_path, archives = _config(tmp_path)
    monkeypatch.setattr(inventory, "environment_identity", _environment)
    calls = 0

    def extension_fails(path, **kwargs):
        nonlocal calls
        calls += 1
        if path.name == "FPMD2026_a.aiida":
            raise RuntimeError("optional extension query failed")
        return _query(path, **kwargs)

    first, _ = inventory.build_inventory(config_path, query_adapter=extension_fails)
    second, _ = inventory.build_inventory(config_path, resume=True, query_adapter=extension_fails)

    assert first["status"] == "pass"
    assert first["warnings"] and not first["errors"]
    assert second == first
    assert calls == len(archives)


def test_absent_optional_archive_is_warning_and_valid_inventory(tmp_path, monkeypatch):
    config_path, archives = _config(tmp_path)
    monkeypatch.setattr(inventory, "environment_identity", _environment)
    optional = next(path for path in archives if path.name == "FPMD2026_a.aiida")
    optional.unlink()

    built, _ = inventory.build_inventory(config_path, query_adapter=_query)
    config = inventory.load_config(config_path)
    state = inventory.evaluate_inventory(config, [path for path in archives if path.exists()])

    assert built["status"] == "pass"
    assert built["warnings"] and not built["errors"]
    assert next(
        item for item in built["archives"] if item["name"] == optional.name
    )["query_status"] == "missing"
    assert state["status"] == "valid"


def test_inventory_paths_cannot_escape_output_or_scratch(tmp_path):
    config_path, _ = _config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["fpmd_aiida"]["profile_root"] = str(tmp_path / "outside")
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="profile_root"):
        inventory.settings_from_config(inventory.load_config(config_path))


def test_real_aiida_read_only_archive_smoke(tmp_path, monkeypatch):
    pytest.importorskip("aiida")
    np = pytest.importorskip("numpy")
    from aiida.manage import get_manager
    from aiida.manage.configuration import Profile
    from aiida.orm import Dict, Group, StructureData, TrajectoryData, User
    from aiida.storage.sqlite_dos.backend import SqliteDosStorage
    from aiida.tools.archive import create_archive

    profile = Profile(
        "e3sse-smoke",
        {
            "storage": {
                "backend": "core.sqlite_dos",
                "config": {"filepath": str(tmp_path / "profile")},
            },
            "process_control": {"backend": None, "config": {}},
        },
    )
    profile.default_user_email = "smoke@e3sse.invalid"
    SqliteDosStorage.initialise(profile)
    manager = get_manager()
    manager.load_profile(profile, allow_switch=True)
    User(email=profile.default_user_email).store()
    structure = StructureData(cell=np.eye(3) * 4.0)
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Li")
    structure.store()
    trajectory = TrajectoryData()
    trajectory.set_trajectory(
        symbols=["Li"],
        positions=np.array([[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]]]),
        cells=np.array([np.eye(3) * 4.0, np.eye(3) * 4.0]),
        stepids=np.array([0, 1]),
        times=np.array([0.0, 1.0]),
        pbc=[True, True, True],
    )
    trajectory.base.extras.set_many(
        {"temperature": 500.0, "temperature_unit": "K", "material_id": "smoke"}
    )
    trajectory.store()
    diffusion = Dict(
        {"diffusion_coefficient": 1.0e-7, "diffusion_coefficient_unit": "cm^2/s"}
    ).store()
    msd = Dict({"msd": [0.0, 0.1], "msd_unit": "angstrom^2"}).store()
    groups = [
        Group(label=label).store()
        for label in (
            "starting_structures",
            "concatenated_trajectories",
            "diffusion_coefficients",
            "msd",
        )
    ]
    for group, node in zip(groups, (structure, trajectory, diffusion, msd), strict=True):
        group.add_nodes([node])
    archive = create_archive(groups, tmp_path / "smoke.aiida")
    manager.unload_profile()

    result = inventory._query_path(archive, 5)

    assert result["structure_count"] == 1
    assert result["trajectory_count"] == 1
    assert result["temperature_metadata"]["distinct_temperature_count"] == 1
    assert result["capabilities"]["diffusion_reference_recoverable"]
    assert result["capabilities"]["msd_reference_recoverable"]

    source_sha = inventory._sha256(archive)
    direct_query = inventory._query_path

    def require_derived_copy(path, limit):
        if path == archive:
            raise RuntimeError("synthetic legacy incompatibility")
        return direct_query(path, limit)

    monkeypatch.setattr(inventory, "_query_path", require_derived_copy)
    migrated = inventory.query_archive_read_only(
        archive,
        environment={"environment_id": "smoke-environment"},
        settings={
            "example_limit": 5,
            "allow_derived_migration": True,
            "migration_root": str(tmp_path / "migrated"),
        },
    )

    assert migrated["queried_path_kind"] == "derived_migrated_copy"
    assert migrated["migration"]["source_sha256"] == source_sha
    assert inventory._sha256(archive) == source_sha
