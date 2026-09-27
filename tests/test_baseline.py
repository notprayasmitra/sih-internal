from __future__ import annotations

import json

import numpy as np
import torch

from cyberworld.baseline import compute_metrics, make_baseline_samples
from cyberworld.config import load_config
from cyberworld.data.trajectory import Trajectory
from cyberworld.models import WorldModel


def test_baseline_reuses_history_windows_and_excludes_invalid_labels() -> None:
    trajectory = Trajectory(
        trajectory_id="fixture",
        states=np.asarray(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
            dtype=np.float32,
        ),
        availability=np.asarray(
            [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        stages=np.asarray([0, 0, 1, 8, 2], dtype=np.int64),
        label_valid=np.asarray([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )
    samples = make_baseline_samples([trajectory], history_windows=2, forecast_windows=1)

    assert samples.features.shape == (2, 4)
    np.testing.assert_allclose(samples.features[0], [1.0, 10.0, 2.0, 0.0])
    np.testing.assert_array_equal(samples.malicious, [1, 1])
    np.testing.assert_array_equal(samples.forecast_steps, [0, 0])
    assert len(samples.features) == 2
    metrics = compute_metrics(samples.malicious, np.asarray([1, 1], dtype=np.int64))
    assert metrics["support"] == 2


def test_benchmark_writes_baseline_and_world_model_metrics(tmp_path, capsys) -> None:
    trajectory_dir = tmp_path / "trajectories"
    run_dir = tmp_path / "runs"
    checkpoint_dir = tmp_path / "checkpoints"
    trajectory_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment:
  name: benchmark_test
  seed: 3
  deterministic: true
paths:
  data_root: {tmp_path / 'data'}
  trajectory_dir: {trajectory_dir}
  artifact_root: {tmp_path / 'artifacts'}
  checkpoint_root: {checkpoint_dir}
  run_root: {run_dir}
data:
  feature_dim: 2
  num_stages: 9
  history_windows: 2
  forecast_windows: 1
  window_seconds: 10
  batch_size: 2
  num_workers: 0
  pin_memory: false
  synthetic_if_missing: false
model:
  architecture: probabilistic_lstm
  hidden_dim: 4
  projection_dim: 4
  latent_dim: 4
  num_layers: 1
  dropout: 0.0
  min_log_variance: -8.0
  max_log_variance: 5.0
training:
  device: cpu
  epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  early_stopping_patience: 1
  mixed_precision: false
  validation_fraction: 0.5
logging:
  log_every_steps: 1
  save_every_epochs: 1
""".strip(),
        encoding="utf-8",
    )
    trajectories = [
        Trajectory(
            trajectory_id="train",
            states=np.arange(12, dtype=np.float32).reshape(6, 2),
            availability=np.ones((6, 2), dtype=np.float32),
            stages=np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64),
            label_valid=np.ones(6, dtype=np.float32),
        ),
        Trajectory(
            trajectory_id="validation",
            states=np.arange(12, 24, dtype=np.float32).reshape(6, 2),
            availability=np.ones((6, 2), dtype=np.float32),
            stages=np.asarray([0, 1, 0, 1, 8, 2], dtype=np.int64),
            label_valid=np.asarray([1, 1, 1, 1, 0, 1], dtype=np.float32),
        ),
    ]
    for trajectory in trajectories:
        np.savez_compressed(
            trajectory_dir / f"{trajectory.trajectory_id}.npz",
            trajectory_id=trajectory.trajectory_id,
            states=trajectory.states,
            availability=trajectory.availability,
            stages=trajectory.stages,
            label_valid=trajectory.label_valid,
        )

    config = load_config(config_path, repo_root=tmp_path)
    model = WorldModel(
        feature_dim=2,
        num_stages=9,
        projection_dim=4,
        hidden_dim=4,
        num_layers=1,
        dropout=0.0,
        min_log_variance=-8.0,
        max_log_variance=5.0,
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"model_state": model.state_dict(), "config": config.model_dump(mode="json")},
        checkpoint,
    )

    from cyberworld.cli import build_parser

    args = build_parser().parse_args(
        [
            "benchmark",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--state-shift-report",
        ]
    )
    args.handler(args)
    output = json.loads((run_dir / "benchmark_test" / "benchmark.json").read_text())
    assert set(output) >= {"baseline", "world_model", "excluded_invalid_label_count"}
    assert output["excluded_invalid_label_count"] == 1
    assert set(output["state_shift_report"]["feature_groups"]) == {
        "shared",
        "cic_unobserved",
    }
    for result in (output["baseline"], output["world_model"]):
        for metric in ("f1", "precision", "recall", "fpr"):
            assert isinstance(result[metric], float)
            assert 0.0 <= result[metric] <= 1.0
    stdout = capsys.readouterr().out
    assert "model,precision,recall,f1,fpr" in stdout
    assert "baseline," in stdout
    assert "world_model," in stdout
