"""Production run identity: what determines results, kept apart from how they ran.

``scientific_identity`` binds everything that can change a scientific number:
the Git commit and tree (and a clean worktree), the active protocol configuration,
the frozen parent protocol pin, the freeze-input configuration (datasets, G0
manifest, model registry, checkpoints, environment locks), the Study-1 and server
configurations, the scientific part of the production configuration (stage graph,
evidence bindings), the frozen ML-NEB settings and the pinned embedding evidence.
The run ID is its SHA-256 prefix.

``operational_identity`` records the host, the resource policy and the measured
allocations.  Paths, resource settings and stage prose are excluded from the run ID,
so editing a stage's ``next_task`` or a worker count never orphans a run.  It is written next to the scientific identity but does not enter
the run ID, so changing a worker count never orphans a run.  Threads per worker
can change CPU floating-point reductions.  They are bound per stage in the
allocation record and in every shard fingerprint, and a resume refuses a change.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUN_IDENTITY_SCHEMA = "e3sse.production.run-identity"
# Operational sections: paths, resource policy and prose.  They never change a number.
OPERATIONAL_SECTIONS = ("resource_policy", "resource_profiling", "runtime", "output", "evidence",
                        "preflight", "description")
STAGE_PROSE_FIELDS = ("public_name", "description", "next_task")
RUN_ID_LENGTH = 20


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repository: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "tree": run("rev-parse", "HEAD^{tree}"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(status),
            "dirty_paths": sorted(line[3:] for line in status.splitlines())}


def scientific_config_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """The production configuration without operational sections or stage prose."""

    view = {key: value for key, value in config.items() if key not in OPERATIONAL_SECTIONS}
    if "stages" in view:
        view["stages"] = [{k: v for k, v in stage.items() if k not in STAGE_PROSE_FIELDS}
                          for stage in view["stages"]]
    return view


def project_environment_sha256() -> str:
    """Installed distributions of the orchestrator's (project) interpreter.

    The project environment runs the statistics (bootstrap, IRLS, weighted conformal
    selection), so its package set is scientific identity.  ``uv.lock`` is not used
    because it is not part of the reviewed tree.
    """

    from importlib import metadata

    names = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}"
                    for d in metadata.distributions()})
    return canonical_sha256({"python": sys.version.split()[0], "distributions": names})


def build_scientific_identity(
    *,
    git: Mapping[str, Any],
    production_config: Mapping[str, Any],
    bound_files: Mapping[str, Path],
    active_protocol: Mapping[str, Any],
    freeze_config: Mapping[str, Any],
    ml_neb_settings_sha256: str,
    project_environment_sha256: str,
) -> dict[str, Any]:
    """Every input that can change a scientific number; see the module docstring."""

    from e3sse.production.model_runtimes import frozen_models

    registry = {
        name: {"weights_sha256": model["weights_sha256"],
               "package": f"{model['package_name']}=={model['package_version']}",
               "adapter_version": model["adapter_version"],
               "frozen_group": model["frozen_group"]}
        for name, model in frozen_models(freeze_config).items()
    }
    environments = freeze_config["inputs"]["production_evidence"]["runtime_environments"]
    embedding = active_protocol["amendments"]["representation"].get(
        "embedding_capability_evidence", {}).get("reports", {})
    return {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "git_commit": git["commit"],
        "git_tree": git["tree"],
        "git_dirty": git["dirty"],
        "bound_file_sha256": {name: file_sha256(path) for name, path in sorted(bound_files.items())},
        "production_config_scientific_sha256": canonical_sha256(
            scientific_config_view(production_config)),
        "parent_protocol": dict(active_protocol["parent_protocol"]),
        "governance": {key: active_protocol["governance"][key] for key in
                       ("governance_mode", "external_registration_required",
                        "preregistration_claimed")},
        "g0_input_manifest": dict(freeze_config["inputs"]["g0_input_manifest"]),
        "model_registry": dict(sorted(registry.items())),
        "environment_lock_sha256": {name: record["environment_lock"]["sha256"]
                                    for name, record in sorted(environments.items())},
        "embedding_evidence_sha256": {name: pin["sha256"] for name, pin in sorted(embedding.items())},
        "ml_neb_settings_sha256": ml_neb_settings_sha256,
        "project_environment_sha256": project_environment_sha256,
    }


def run_id(scientific_identity: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(scientific_identity))[:RUN_ID_LENGTH]


def operational_identity(*, host: Mapping[str, Any], resource_policy: Mapping[str, Any],
                         command: list[str]) -> dict[str, Any]:
    return {
        "classification": "operational_execution_metadata_not_scientific_identity",
        "host": dict(host),
        "hostname": platform.node(),
        "orchestrator_python": sys.version.split()[0],
        "orchestrator_executable": sys.executable,
        "resource_policy": dict(resource_policy),
        "command": list(command),
    }
