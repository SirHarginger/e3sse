"""Route model work to each frozen model's own pinned interpreter; never install anything.

The six frozen models live in separate Python environments (registry:
``configs/protocol_freeze.json`` ``inputs.production_evidence.runtime_environments``).
Model work runs as ``<runtime_root>/<environment>/bin/python <repository script>``
with ``PYTHONPATH=<repository>/src`` and the environment's frozen variables, so the
model packages never enter the orchestrator's interpreter.  Thread variables are
set by the stage runners per workload (``e3sse.execution.thread_control``), not here.

Commands are built as argument lists and checked: no package manager is ever a
command, so a production run cannot change an environment.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_MANAGER_EXECUTABLES = frozenset({"pip", "pip3", "uv", "conda", "mamba", "micromamba",
                                         "poetry", "pipx", "easy_install"})
PACKAGE_MANAGER_ARGUMENTS = (("-m", "pip"), ("-m", "ensurepip"), ("-m", "uv"))


class RuntimeRoutingError(RuntimeError):
    """A model has no unique, present, frozen runtime."""


@dataclass(frozen=True)
class ModelRuntime:
    model: str
    environment: str
    python: Path
    python_version: str
    pinned_packages: dict[str, str]
    environment_variables: dict[str, str]
    environment_lock_path: Path
    environment_lock_sha256: str
    weights_sha256: str
    package_name: str
    package_version: str

    def as_json(self) -> dict[str, Any]:
        return {
            "model": self.model, "environment": self.environment, "python": str(self.python),
            "python_present": self.python.is_file() and os.access(self.python, os.X_OK),
            "python_version": self.python_version, "pinned_packages": self.pinned_packages,
            "environment_variables": self.environment_variables,
            "environment_lock_path": str(self.environment_lock_path),
            "environment_lock_sha256": self.environment_lock_sha256,
            "weights_sha256": self.weights_sha256,
            "package": f"{self.package_name}=={self.package_version}",
        }


def frozen_models(freeze_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The six frozen models (groups A, B1, B2), by the same rule Study 1 uses."""

    from e3sse.softening.study1_inputs import Study1InputError
    from e3sse.softening.study1_inputs import frozen_models as study1_frozen_models

    try:
        return dict(sorted(study1_frozen_models(freeze_config["protocol"]).items()))
    except Study1InputError as exc:
        raise RuntimeRoutingError(str(exc)) from exc


def model_runtimes(freeze_config: Mapping[str, Any], runtime_root: Path) -> dict[str, ModelRuntime]:
    environments = freeze_config["inputs"]["production_evidence"]["runtime_environments"]
    root = Path(runtime_root).expanduser()
    runtimes = {}
    for name, model in frozen_models(freeze_config).items():
        matches = [(env, record) for env, record in sorted(environments.items())
                   if name in record["models"]]
        if len(matches) != 1:
            raise RuntimeRoutingError(f"model {name} needs exactly one frozen runtime, "
                                      f"found {[env for env, _ in matches]}")
        environment, record = matches[0]
        runtimes[name] = ModelRuntime(
            model=name, environment=environment,
            python=root / environment / "bin" / "python",
            python_version=record["python_version"],
            pinned_packages=dict(record["pinned_packages"]),
            environment_variables=dict(record.get("environment_variables", {})),
            environment_lock_path=Path(record["environment_lock"]["path"]),
            environment_lock_sha256=record["environment_lock"]["sha256"],
            weights_sha256=model["weights_sha256"],
            package_name=model["package_name"], package_version=model["package_version"],
        )
    return runtimes


def require_present(runtime: ModelRuntime) -> None:
    if not (runtime.python.is_file() and os.access(runtime.python, os.X_OK)):
        raise RuntimeRoutingError(
            f"{runtime.model}: frozen interpreter {runtime.python} is missing; create the "
            f"{runtime.environment} environment from its lock "
            f"({runtime.environment_lock_path}); production never installs packages"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment(runtime: ModelRuntime | None, repository: Path,
                        base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Child environment: repository sources on the path plus frozen variables."""

    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = str(Path(repository) / "src")
    # Model packages must come from the runtime, never from a user site directory.
    env["PYTHONNOUSERSITE"] = "1"
    if runtime is not None:
        env.update(runtime.environment_variables)
    return env


def assert_no_package_installation(command: Sequence[str]) -> None:
    """Refuse a command that would run a package manager."""

    parts = [str(part) for part in command]
    if not parts:
        raise RuntimeRoutingError("empty command")
    for part in parts:
        if Path(part).name in PACKAGE_MANAGER_EXECUTABLES:
            raise RuntimeRoutingError(f"package manager in production command: {parts}")
    for first, second in PACKAGE_MANAGER_ARGUMENTS:
        for index in range(len(parts) - 1):
            if parts[index] == first and parts[index + 1] == second:
                raise RuntimeRoutingError(f"package manager in production command: {parts}")


def runtime_command(runtime: ModelRuntime | None, script: Path, arguments: Sequence[str],
                    *, project_python: Path) -> list[str]:
    """``<interpreter> <script> <arguments...>`` for a model runtime or the project."""

    python = runtime.python if runtime is not None else project_python
    command = [str(python), str(script), *[str(argument) for argument in arguments]]
    assert_no_package_installation(command)
    return command
