from src.config import load_config, on_kaggle


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg["audio"]["sample_rate"] == 16000
    assert cfg["cv"]["n_splits"] == 5
    assert "data_dir" in cfg and "out_dir" in cfg
    if not on_kaggle():
        assert cfg["data_dir"].endswith("data/sample")
