"""Shared inputs, gated outcome access, artifacts and sharded execution for runners.

Runs in the project environment and in the frozen model runtimes (Python 3.11), so
it imports model packages never and heavy libraries lazily.

Failure classes map to process exit codes that the orchestrator classifies:
``MissingEvidence`` (2: evidence that only the server can produce is absent) and
``RunnerValidationError`` (3: evidence contradicts a requirement).  Any other
exception is an infrastructure failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXIT_MISSING_EVIDENCE = 2
EXIT_VALIDATION = 3
STAGE_MANIFEST = "stage_manifest.json"
MODEL_MANIFEST = "model_manifest.json"
STAGE_MANIFEST_SCHEMA = "e3sse.production.stage-manifest"
PREPARE_MANIFEST = "prepare_manifest.json"


class MissingEvidence(RuntimeError):
    """Evidence the stage needs does not exist yet (server-only or upstream)."""


class RunnerValidationError(RuntimeError):
    """Evidence exists but contradicts a frozen requirement."""


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass
class RunnerContext:
    repository: Path
    config_path: Path
    run_dir: Path
    stage: str
    model: str | None = None
    workers: int = 1
    threads: int = 1
    memory_reserve_bytes: int | None = None
    resume: bool = True
    _cache: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- configuration
    @property
    def config(self) -> dict[str, Any]:
        if "config" not in self._cache:
            self._cache["config"] = load_json(self.config_path)
        return self._cache["config"]

    def binding(self, key: str) -> Path:
        return _resolve(self.config_path.parent, self.config["scientific_bindings"][key])

    @property
    def active_protocol(self) -> dict[str, Any]:
        if "active" not in self._cache:
            self._cache["active"] = load_json(self.binding("active_protocol_config"))
        return self._cache["active"]

    @property
    def freeze_config(self) -> dict[str, Any]:
        if "freeze_config" not in self._cache:
            self._cache["freeze_config"] = load_json(self.binding("protocol_freeze_config"))
        return self._cache["freeze_config"]

    @property
    def study1_config(self) -> dict[str, Any]:
        if "study1" not in self._cache:
            self._cache["study1"] = load_json(self.binding("study1_config"))
        return self._cache["study1"]

    @property
    def server(self) -> dict[str, Any]:
        if "server" not in self._cache:
            from e3sse.config import load_config

            self._cache["server"] = load_config(self.binding("server_config"))
        return self._cache["server"]

    @property
    def data_root(self) -> Path:
        return Path(self.server["paths"]["data_root"])

    @property
    def outputs_root(self) -> Path:
        return Path(self.server["paths"]["outputs_root"])

    def runner_settings(self, key: str) -> dict[str, Any]:
        return dict(self.config.get("runners", {}).get(key, {}))

    # ------------------------------------------------------------- frozen inputs
    @property
    def protocol_dir(self) -> Path:
        return Path(self.study1_config["frozen_protocol"]["directory"]).expanduser()

    @property
    def parent_protocol(self) -> dict[str, Any]:
        if "parent" not in self._cache:
            from e3sse.softening.study1_inputs import Study1InputError, load_frozen_protocol

            try:
                self._cache["parent"] = load_frozen_protocol(self.study1_config,
                                                             directory=self.protocol_dir)
            except Study1InputError as exc:
                raise MissingEvidence(f"frozen parent protocol unavailable: {exc}") from exc
        return self._cache["parent"]

    @property
    def membership(self) -> list[dict[str, Any]]:
        if "membership" not in self._cache:
            from e3sse.softening.study1_inputs import load_membership

            self._cache["membership"] = load_membership(self.protocol_dir, self.parent_protocol)
        return self._cache["membership"]

    @property
    def partition_manifest(self) -> dict[str, Any]:
        if "partition" not in self._cache:
            from e3sse.softening.study1_inputs import load_partition_manifest

            self._cache["partition"] = load_partition_manifest(self.protocol_dir,
                                                               self.parent_protocol)
        return self._cache["partition"]

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        from e3sse.production.model_runtimes import frozen_models

        return frozen_models(self.freeze_config)

    def checkpoint(self, model: str) -> Path:
        path = Path(self.config["runtime"]["checkpoint_map"]).expanduser()
        if not path.is_file():
            raise MissingEvidence(f"checkpoint map {path} absent")
        mapping = load_json(path)
        if model not in mapping:
            raise MissingEvidence(f"checkpoint map lacks {model}")
        return Path(mapping[model]).expanduser()

    # ------------------------------------------------------------- artifacts
    @property
    def stage_dir(self) -> Path:
        return self.run_dir / self.stage

    @property
    def work_dir(self) -> Path:
        """Per-model directory for per-model stages, else the stage directory."""

        return self.stage_dir / "models" / self.model if self.model else self.stage_dir

    def publish(self, relative: str, value: Any) -> Path:
        """Create-once JSON under the work directory; an existing file must be identical."""

        from e3sse.execution.sharded_execution import publish_json_once

        path = self.work_dir / relative
        if path.exists():
            if load_json(path) != json.loads(json.dumps(value, allow_nan=False)):
                raise RunnerValidationError(f"{path} exists with different content")
            return path
        publish_json_once(path, value, root=self.run_dir)
        return path

    def upstream(self, stage: str, name: str, *, model: str | None = None) -> Any:
        """A verified JSON output of an earlier stage of this run (fail closed)."""

        return load_json(self.upstream_path(stage, name, model=model))

    def upstream_path(self, stage: str, name: str, *, model: str | None = None,
                      phase: str | None = None) -> Path:
        """The verified path of an earlier stage's output (SHA-256 checked).

        ``phase="prepare"`` reads the stage's prepare-phase manifest (outputs a later
        phase of the same stage, or another stage, consumes)."""

        directory = self.run_dir / stage / ("models/" + model if model else "")
        manifest_path = directory / (MODEL_MANIFEST if model else STAGE_MANIFEST)
        if phase == "prepare":
            manifest_path = self.run_dir / stage / PREPARE_MANIFEST
        elif not manifest_path.is_file() and not model:
            # Script-driven stages have only the orchestrator's completion record.
            completion = directory / "stage_completion.json"
            if completion.is_file():
                manifest_path = completion
        if not manifest_path.is_file():
            raise MissingEvidence(f"{stage} has no manifest at {manifest_path}")
        manifest = load_json(manifest_path)
        output = manifest["outputs"].get(name)
        if output is None:
            raise MissingEvidence(f"{stage} produced no {name!r}")
        path = Path(output["path"])
        if not path.is_file() or file_sha256(path) != output["sha256"]:
            raise RunnerValidationError(f"{stage}/{name} differs from its manifest")
        return path

    def upstream_models(self, stage: str) -> list[str]:
        root = self.run_dir / stage / "models"
        return sorted(p.name for p in root.iterdir() if (p / MODEL_MANIFEST).is_file()) \
            if root.is_dir() else []

    # ------------------------------------------------------------- outcome access
    def internal_freeze(self) -> dict[str, Any]:
        if "internal" not in self._cache:
            from e3sse.protocol.internal_freeze import validate_internal_freeze_protocol

            path = self.run_dir / "freeze" / "internal_freeze_binding.json"
            if not path.is_file():
                raise MissingEvidence("no internal freeze binding in this run")
            binding = load_json(path)
            artifact = Path(binding["directory"]) / "internal_freeze_protocol.json"
            if file_sha256(artifact) != binding["file_sha256"]:
                raise RunnerValidationError("internal-freeze artifact differs from its binding")
            internal = load_json(artifact)
            validate_internal_freeze_protocol(internal, self.parent_protocol)
            self._cache["internal"] = internal
        return self._cache["internal"]

    def frozen_amendments(self) -> dict[str, Any]:
        """Post-freeze runners read every rule from the frozen artifact, never the config."""

        return self.internal_freeze()["amendments"]

    def require_outcome(self, *, role: str, outcome: str, outcome_kind: str,
                        unit_ids: Sequence[str] | None = None,
                        purpose: str = "prespecified_analysis",
                        development_manifest: Mapping[str, Any] | None = None) -> None:
        """The frozen contract decides; every granted access leaves a provenance record."""

        from e3sse.protocol.contracts import require_outcome_access
        from e3sse.protocol.internal_freeze import internal_freeze_provenance

        internal = self.internal_freeze()
        seal2: dict[str, Any] = {}
        if (self.run_dir / "pilot" / STAGE_MANIFEST).is_file():
            # Seal 2: the training-only pilot amendment bound to this internal freeze.
            # The contract demands it for every non-training read and for any
            # published-train nebDFT2k barrier read.
            seal2 = {"pilot_amendment": self.upstream("pilot", "pilot_amendment"),
                     "internal_pilot_record": self.upstream("pilot", "internal_pilot_record")}
        elif role != "train" or purpose == "prospective_model_selection":
            raise MissingEvidence("non-training outcomes need the pilot amendment of this run")
        require_outcome_access(
            role=role, outcome=outcome, outcome_kind=outcome_kind,
            frozen_protocol=self.parent_protocol, purpose=purpose,
            internal_freeze_protocol=internal,
            partition_records=self.membership if role == "train" else None,
            unit_ids=list(unit_ids) if unit_ids is not None else None,
            development_manifest=development_manifest, **seal2,
        )
        self.publish(f"outcome_access/{utc().replace(':', '').replace('-', '')}-{os.getpid()}.json", {
            "stage": self.stage, "model": self.model, "role": role, "outcome": outcome,
            "outcome_kind": outcome_kind, "purpose": purpose,
            "unit_count": None if unit_ids is None else len(unit_ids),
            "unit_ids_sha256": None if unit_ids is None else canonical_sha256(sorted(unit_ids)),
            **internal_freeze_provenance(internal), "accessed_at_utc": utc(),
        })

    # ------------------------------------------------------------- execution
    def code_sha256(self, files: Iterable[str]) -> dict[str, str]:
        return {name: file_sha256(self.repository / name) for name in sorted(set(files))}

    def execute(self, *, namespace: str, units: Sequence[tuple[Any, Any]],
                task: Callable[[Any, Any], Any], identity: Mapping[str, Any],
                setup: Callable[..., Any] | None = None, setup_args: tuple = (),
                configure_torch: bool = False) -> dict[str, Any]:
        """Run ``units`` (identity, payload) as resumable shards; results by shard ID.

        Workers and threads come from the orchestrator (central scheduler); the plan
        only verifies them against the observed envelope.
        """

        from e3sse.execution.resource_planning import (
            WorkloadPolicy,
            observe_host_resources,
            plan_workers,
        )
        from e3sse.execution.sharded_execution import ShardedRun, execute_shards, make_shard

        shards = [make_shard(namespace, unit_identity, payload) for unit_identity, payload in units]
        if not shards:
            return {}
        policy = WorkloadPolicy(name=f"{self.stage}:{namespace}",
                                threads_per_worker=self.threads,
                                configure_torch=configure_torch)
        plan = plan_workers(self.workers, policy, observe_host_resources(),
                            unit_count=len(shards))
        run = ShardedRun(output_dir=self.work_dir / "shards" / namespace,
                         scientific_identity={**identity, "stage": self.stage,
                                              "model": self.model},
                         task=task, protected_roots=[self.data_root],
                         worker_setup=setup, worker_setup_args=setup_args)
        result = execute_shards(run, shards, plan, resume=self.resume,
                                memory_floor_bytes=self.memory_reserve_bytes,
                                progress=lambda message: print(message, file=sys.stderr))
        return result.results
