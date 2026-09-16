"""Boundary-safe window datasets built from independent trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from cyberworld.labels import AttackStage, is_compromise, is_reliably_malicious


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    states: np.ndarray
    availability: np.ndarray
    stages: np.ndarray
    label_valid: np.ndarray

    def validate(self, feature_dim: int, num_stages: int) -> None:
        if self.states.ndim != 2 or self.states.shape[1] != feature_dim:
            raise ValueError(f"{self.trajectory_id}: invalid state shape {self.states.shape}")
        if self.availability.shape != self.states.shape:
            raise ValueError(f"{self.trajectory_id}: availability shape mismatch")
        if self.stages.shape != (len(self.states),):
            raise ValueError(f"{self.trajectory_id}: stage shape mismatch")
        if self.label_valid.shape != (len(self.states),):
            raise ValueError(f"{self.trajectory_id}: label_valid shape mismatch")
        if not np.isfinite(self.states).all():
            raise ValueError(f"{self.trajectory_id}: non-finite state values")
        if np.any((self.availability < 0) | (self.availability > 1)):
            raise ValueError(f"{self.trajectory_id}: availability must be in [0, 1]")
        if np.any((self.stages < 0) | (self.stages >= num_stages)):
            raise ValueError(f"{self.trajectory_id}: stage outside [0, {num_stages})")


def load_trajectories(directory: Path, feature_dim: int, num_stages: int) -> list[Trajectory]:
    trajectories = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            trajectory = Trajectory(
                trajectory_id=str(payload.get("trajectory_id", path.stem)),
                states=np.asarray(payload["states"], dtype=np.float32),
                availability=np.asarray(
                    payload.get("availability", np.ones_like(payload["states"])),
                    dtype=np.float32,
                ),
                stages=np.asarray(payload["stages"], dtype=np.int64),
                label_valid=np.asarray(
                    payload.get("label_valid", np.ones(len(payload["states"]))),
                    dtype=np.float32,
                ),
            )
        trajectory.validate(feature_dim, num_stages)
        trajectories.append(trajectory)
    return trajectories


class TrajectoryWindowDataset(Dataset[dict[str, Tensor]]):
    """Create samples without ever crossing trajectory boundaries."""

    def __init__(
        self,
        trajectories: Iterable[Trajectory],
        *,
        history_windows: int,
        forecast_windows: int,
    ) -> None:
        self.trajectories = tuple(trajectories)
        self.history_windows = history_windows
        self.forecast_windows = forecast_windows
        self._indices: list[tuple[int, int]] = []
        span = history_windows + forecast_windows
        for trajectory_index, trajectory in enumerate(self.trajectories):
            for start in range(max(0, len(trajectory.states) - span + 1)):
                self._indices.append((trajectory_index, start))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        trajectory_index, start = self._indices[index]
        trajectory = self.trajectories[trajectory_index]
        split = start + self.history_windows
        end = split + self.forecast_windows
        stages = trajectory.stages[split:end]
        valid = trajectory.label_valid[split:end]
        malicious = np.asarray([is_reliably_malicious(x) for x in stages], dtype=np.float32)
        compromise = np.asarray([is_compromise(x) for x in stages], dtype=np.float32)
        valid = valid * (stages != int(AttackStage.UNKNOWN))
        return {
            "history": torch.from_numpy(trajectory.states[start:split]),
            "history_availability": torch.from_numpy(trajectory.availability[start:split]),
            "future_states": torch.from_numpy(trajectory.states[split:end]),
            "future_availability": torch.from_numpy(trajectory.availability[split:end]),
            "future_stages": torch.from_numpy(stages),
            "future_label_valid": torch.from_numpy(valid.astype(np.float32)),
            "future_malicious": torch.from_numpy(malicious),
            "future_compromise": torch.from_numpy(compromise),
            "trajectory_index": torch.tensor(trajectory_index, dtype=torch.int64),
        }
