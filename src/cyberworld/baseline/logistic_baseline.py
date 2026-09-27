"""Non-temporal logistic baseline over the world model's history windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from cyberworld.data.trajectory import Trajectory, TrajectoryWindowDataset


@dataclass(frozen=True)
class BaselineSamples:
    """Flattened history rows aligned with valid future prediction targets."""

    features: np.ndarray
    malicious: np.ndarray
    compromise: np.ndarray
    window_indices: np.ndarray
    forecast_steps: np.ndarray
    histories: np.ndarray
    history_availabilities: np.ndarray
    future_states: np.ndarray
    future_availabilities: np.ndarray


def make_baseline_samples(
    trajectories: list[Trajectory],
    *,
    history_windows: int,
    forecast_windows: int,
) -> BaselineSamples:
    """Build samples by adapting the canonical trajectory window dataset.

    Each valid future step gets one row whose features are the complete observed
    history, flattened in time-major order. Availability masking is applied before
    flattening so unavailable values cannot become training signal.
    """
    dataset = TrajectoryWindowDataset(
        trajectories,
        history_windows=history_windows,
        forecast_windows=forecast_windows,
    )
    features: list[np.ndarray] = []
    malicious: list[int] = []
    compromise: list[int] = []
    window_indices: list[int] = []
    forecast_steps: list[int] = []
    histories: list[np.ndarray] = []
    history_availabilities: list[np.ndarray] = []
    future_states: list[np.ndarray] = []
    future_availabilities: list[np.ndarray] = []
    for window_index in range(len(dataset)):
        sample = dataset[window_index]
        valid_steps = np.flatnonzero(sample["future_risk_valid"].numpy() > 0)
        history = (
            sample["history"].numpy() * sample["history_availability"].numpy()
        ).reshape(-1)
        history_tensor = sample["history"].numpy()
        availability_tensor = sample["history_availability"].numpy()
        for forecast_step in valid_steps:
            features.append(history.astype(np.float32, copy=True))
            malicious.append(int(sample["future_risk_malicious"][forecast_step].item()))
            compromise.append(int(sample["future_risk_compromise"][forecast_step].item()))
            window_indices.append(window_index)
            forecast_steps.append(int(forecast_step))
            histories.append(history_tensor.copy())
            history_availabilities.append(availability_tensor.copy())
            future_states.append(sample["future_states"][forecast_step].numpy().copy())
            future_availabilities.append(
                sample["future_availability"][forecast_step].numpy().copy()
            )

    feature_dim = history_windows * trajectories[0].states.shape[1] if trajectories else 0
    return BaselineSamples(
        features=np.asarray(features, dtype=np.float32).reshape(-1, feature_dim),
        malicious=np.asarray(malicious, dtype=np.int64),
        compromise=np.asarray(compromise, dtype=np.int64),
        window_indices=np.asarray(window_indices, dtype=np.int64),
        forecast_steps=np.asarray(forecast_steps, dtype=np.int64),
        histories=np.asarray(histories, dtype=np.float32),
        history_availabilities=np.asarray(history_availabilities, dtype=np.float32),
        future_states=np.asarray(future_states, dtype=np.float32),
        future_availabilities=np.asarray(future_availabilities, dtype=np.float32),
    )


def train_logistic_baseline(
    samples: BaselineSamples,
    *,
    target: str = "malicious",
) -> LogisticRegression:
    """Fit logistic regression on valid flattened history-window samples."""
    if target not in {"malicious", "compromise"}:
        raise ValueError(f"Unsupported baseline target: {target}")
    labels = samples.malicious if target == "malicious" else samples.compromise
    if samples.features.shape[0] == 0 or np.unique(labels).size < 2:
        raise ValueError("Logistic baseline requires valid samples from both classes")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )
    model.fit(samples.features, labels)
    return model
