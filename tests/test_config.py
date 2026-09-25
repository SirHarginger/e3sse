import json
from pathlib import Path

import pytest

from e3sse.config import load_config


def test_local_json_config_resolves_from_config_location():
    config = load_config("configs/local.json")

    assert config["project"]["name"] == "e3sse"
    assert Path(config["paths"]["data_root"]) == Path.cwd() / "data"


def test_legacy_yaml_keeps_repository_relative_paths():
    config = load_config("configs/local.yaml")

    assert Path(config["paths"]["data_root"]) == Path.cwd() / "data"


def test_config_rejects_outputs_below_raw_data(tmp_path):
    config_path = tmp_path / "unsafe.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {
                    "data_root": "data",
                    "outputs_root": "data/raw/outputs",
                    "logs_root": "logs",
                    "scratch_root": "scratch",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generated files"):
        load_config(config_path)
