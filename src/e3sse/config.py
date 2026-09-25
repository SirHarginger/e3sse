"""Configuration loading and path-safety checks for E3-SSE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_PATHS = ("data_root", "outputs_root", "logs_root", "scratch_root")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON (or legacy YAML), resolve paths, and reject unsafe outputs."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            config = json.load(handle)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            config = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")

    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        # Malformed configuration content is a value error for callers, not a type error.
        raise ValueError("Configuration must contain a 'paths' object")  # noqa: TRY004
    paths = config["paths"]
    missing = [key for key in REQUIRED_PATHS if not paths.get(key)]
    if missing:
        raise ValueError(f"Missing required configuration paths: {', '.join(missing)}")

    base = (config_path.parent / paths.get("relative_to", ".")).resolve()
    for key in REQUIRED_PATHS:
        value = Path(paths[key]).expanduser()
        paths[key] = str((base / value).resolve() if not value.is_absolute() else value.resolve())

    data_root = Path(paths["data_root"])
    for key in ("outputs_root", "logs_root", "scratch_root"):
        target = Path(paths[key])
        if target == data_root or _is_within(target, data_root):
            raise ValueError(f"Unsafe {key}: generated files cannot be written below {data_root}")

    config["_config_path"] = str(config_path)
    return config
