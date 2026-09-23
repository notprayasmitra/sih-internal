"""Mask-aware training-only normalization for trajectory states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cyberworld.data.trajectory import Trajectory


@dataclass(frozen=True)
class StandardNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, trajectories: list[Trajectory]) -> StandardNormalizer:
        if not trajectories:
            raise ValueError("Cannot fit a normalizer without training trajectories")
        states = np.concatenate([item.states for item in trajectories], axis=0).astype(np.float64)
        masks = np.concatenate([item.availability for item in trajectories], axis=0).astype(
            np.float64
        )
        count = masks.sum(axis=0)
        safe_count = np.maximum(count, 1.0)
        mean = (states * masks).sum(axis=0) / safe_count
        variance = (((states - mean) * masks) ** 2).sum(axis=0) / safe_count
        scale = np.sqrt(variance)
        scale[(scale < 1e-6) | (count == 0)] = 1.0
        mean[count == 0] = 0.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, trajectory: Trajectory) -> Trajectory:
        states = ((trajectory.states - self.mean) / self.scale) * trajectory.availability
        return Trajectory(
            trajectory_id=trajectory.trajectory_id,
            states=states.astype(np.float32),
            availability=trajectory.availability,
            stages=trajectory.stages,
            label_valid=trajectory.label_valid,
            risk_malicious=trajectory.risk_malicious,
            risk_compromise=trajectory.risk_compromise,
            risk_valid=trajectory.risk_valid,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path: Path) -> StandardNormalizer:
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                mean=np.asarray(payload["mean"], dtype=np.float32),
                scale=np.asarray(payload["scale"], dtype=np.float32),
            )
