from pathlib import Path

from cyberworld.config import load_config


def test_config_inheritance_and_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CYBERWORLD_DEVICE", "cpu")
    config = load_config(
        Path("configs/experiment/smoke.yaml"),
        repo_root=Path.cwd(),
    )
    assert config.experiment.name == "world_model_smoke"
    assert config.training.device == "cpu"
    assert config.training.learning_rate == 0.0003
    assert config.paths.data_root.is_absolute()
