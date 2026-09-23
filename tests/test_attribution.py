import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cyberworld.config import AppConfig
from cyberworld.data.normalization import StandardNormalizer
from cyberworld.explain import attribute_prediction
from cyberworld.models import WorldModel
from cyberworld.preprocessing import STATE_FEATURES, build_trajectory


def _make_model_and_window() -> tuple[WorldModel, torch.Tensor, torch.Tensor]:
    model = WorldModel(
        feature_dim=len(STATE_FEATURES),
        num_stages=9,
        projection_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        min_log_variance=-8.0,
        max_log_variance=5.0,
    )
    history = torch.linspace(
        0.1, 1.0, steps=6 * len(STATE_FEATURES), dtype=torch.float32
    ).reshape(6, len(STATE_FEATURES))
    history_availability = torch.ones_like(history)
    history_availability[:, -2:] = 0.0
    history = history * history_availability
    return model, history, history_availability


def test_masked_features_get_near_zero_attribution() -> None:
    model, history, availability = _make_model_and_window()
    attr = attribute_prediction(
        model,
        history,
        availability,
        target_head="state",
        forecast_step=0,
        n_steps=25,
    )
    assert attr.shape == (len(STATE_FEATURES),)
    assert np.all(np.isfinite(attr))
    assert np.abs(attr[-2:]).sum() < 1e-6


def test_attribution_scores_are_sensible() -> None:
    model, history, availability = _make_model_and_window()
    attr = attribute_prediction(
        model,
        history,
        availability,
        target_head="stage_logits",
        forecast_step=0,
        n_steps=25,
    )
    assert attr.shape == (len(STATE_FEATURES),)
    assert np.all(np.isfinite(attr))
    assert np.abs(attr).sum() > 1e-6
    assert np.abs(attr).sum() < 1e6


def test_attribution_shape_matches_state_features() -> None:
    model, history, availability = _make_model_and_window()
    attr = attribute_prediction(
        model,
        history,
        availability,
        target_head="malicious_risk",
        forecast_step=0,
        n_steps=25,
    )
    assert attr.shape == (len(STATE_FEATURES),)


def test_cli_runs_on_smoke_test_trajectory(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:05Z",
                "2025-01-01T00:00:10Z",
                "2025-01-01T00:00:15Z",
            ]),
            "duration_s": [1.0, 1.5, 2.0, 1.8],
            "protocol": [6, 17, 6, 17],
            "src_ip": ["a", "a", "a", "a"],
            "dst_ip": ["b", "c", "d", "e"],
            "src_port": [1000, 1001, 1002, 1003],
            "dst_port": [80, 53, 22, 443],
            "fwd_packets": [5, 2, 4, 3],
            "bwd_packets": [1, 2, 3, 1],
            "total_packets": [6, 4, 7, 4],
            "fwd_bytes": [500, 200, 400, 300],
            "bwd_bytes": [100, 190, 350, 90],
            "total_bytes": [600, 390, 750, 390],
            "flow_iat_mean_s": [5.0, 5.0, 5.0, 5.0],
            "flow_iat_std_s": [1.0, 1.0, 1.0, 1.0],
            "flow_iat_max_s": [10.0, 10.0, 10.0, 10.0],
            "syn_count": [1, 0, 1, 0],
            "ack_count": [1, 1, 1, 1],
            "fin_count": [0, 0, 0, 0],
            "rst_count": [0, 0, 0, 0],
            "psh_count": [0, 0, 0, 0],
            "urg_count": [0, 0, 0, 0],
            "packet_length_mean": [100, 90, 95, 110],
            "packet_length_std": [10, 8, 9, 12],
            "packet_length_min": [50, 45, 40, 55],
            "packet_length_max": [150, 130, 140, 160],
            "tcp_window_fwd": [200, 210, 230, 240],
            "tcp_window_bwd": [100, 110, 120, 130],
            "ttl_fwd": [64, 60, 62, 63],
            "ttl_bwd": [64, 63, 62, 61],
            "source_label": ["benign", "benign", "benign", "benign"],
            "source_stage": ["", "", "", ""],
            "operational_stage": [0, 0, 0, 0],
            "label_valid": [1.0, 1.0, 1.0, 1.0],
        }
    )
    trajectory_path = tmp_path / "demo.npz"
    build_trajectory(
        frame, trajectory_id="demo:scenario", window_seconds=10, output_path=trajectory_path
    )

    model = WorldModel(
        feature_dim=len(STATE_FEATURES),
        num_stages=9,
        projection_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        min_log_variance=-8.0,
        max_log_variance=5.0,
    )
    config = AppConfig.model_validate({
        "experiment": {"name": "smoke_explain", "seed": 7, "deterministic": True},
        "paths": {
            "data_root": "data",
            "trajectory_dir": "data/processed/trajectories",
            "artifact_root": "artifacts",
            "checkpoint_root": "checkpoints",
            "run_root": "runs",
        },
        "data": {
            "feature_dim": len(STATE_FEATURES),
            "num_stages": 9,
            "history_windows": 2,
            "forecast_windows": 2,
            "window_seconds": 10,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "synthetic_if_missing": False,
            "synthetic_trajectories": 2,
            "synthetic_length": 8,
        },
        "model": {
            "architecture": "probabilistic_lstm",
            "hidden_dim": 16,
            "projection_dim": 16,
            "latent_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "min_log_variance": -8.0,
            "max_log_variance": 5.0,
        },
        "training": {
            "device": "cpu",
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "accumulate_steps": 1,
            "teacher_forcing_ratio": 0.5,
            "early_stopping_patience": 1,
            "mixed_precision": False,
            "compile": False,
            "validation_fraction": 0.2,
            "lambda_state": 1.0,
            "lambda_stage": 0.5,
            "lambda_malicious": 0.5,
            "lambda_compromise": 0.5,
            "lambda_jepa": 0.25,
            "vicreg_invariance": 25.0,
            "vicreg_variance": 25.0,
            "vicreg_covariance": 1.0,
        },
        "logging": {"log_every_steps": 1, "save_every_epochs": 1},
    })
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {"model_state": model.state_dict(), "config": config.model_dump(mode="json")},
        checkpoint_path,
    )
    StandardNormalizer(
        mean=np.full(len(STATE_FEATURES), 100.0, dtype=np.float32),
        scale=np.full(len(STATE_FEATURES), 2.0, dtype=np.float32),
    ).save(tmp_path / "normalizer.npz")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cyberworld.cli",
            "explain",
            "--checkpoint",
            str(checkpoint_path),
            "--trajectory",
            str(trajectory_path),
            "--window-index",
            "0",
            "--top-k",
            "3",
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload
    assert "forecast_step" in payload[0]
    assert "top_features" in payload[0]
    assert payload[0]["attribution_space"] == "normalized"
    raw_states = np.load(trajectory_path)["states"]
    for item in payload[0]["top_features"]:
        index = STATE_FEATURES.index(item["feature"])
        assert item["raw_value"] == float(raw_states[0, index])
