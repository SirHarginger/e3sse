from e3sse.config import load_config


def test_local_config_loads():
    config = load_config("configs/local.yaml")

    assert config["project"]["name"] == "e3sse"
    assert config["paths"]["data_root"] == "data"
