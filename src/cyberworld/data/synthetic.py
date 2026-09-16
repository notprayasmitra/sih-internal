"""Deterministic synthetic trajectories for smoke tests only."""

from __future__ import annotations

import numpy as np

from cyberworld.data.trajectory import Trajectory
from cyberworld.labels import AttackStage


def make_synthetic_trajectories(
    count: int,
    length: int,
    feature_dim: int,
    seed: int,
) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    result = []
    ordered_stages = np.asarray(
        [
            AttackStage.NORMAL,
            AttackStage.RECONNAISSANCE,
            AttackStage.INITIAL_ACCESS,
            AttackStage.LATERAL_MOVEMENT,
            AttackStage.COMMAND_AND_CONTROL,
            AttackStage.EXFILTRATION,
        ],
        dtype=np.int64,
    )
    for trajectory_index in range(count):
        states = np.zeros((length, feature_dim), dtype=np.float32)
        states[0] = rng.normal(0.0, 0.2, feature_dim)
        boundaries = np.linspace(0, length, len(ordered_stages) + 1, dtype=int)
        stages = np.zeros(length, dtype=np.int64)
        for stage_index, stage in enumerate(ordered_stages):
            stages[boundaries[stage_index] : boundaries[stage_index + 1]] = stage
        for time_index in range(1, length):
            stage_signal = float(stages[time_index]) / max(1, len(ordered_stages) - 1)
            drift = np.zeros(feature_dim, dtype=np.float32)
            drift[: min(4, feature_dim)] = stage_signal * 0.08
            states[time_index] = 0.92 * states[time_index - 1] + drift
            states[time_index] += rng.normal(0.0, 0.04, feature_dim)
        availability = np.ones_like(states, dtype=np.float32)
        if feature_dim > 4 and trajectory_index % 2:
            availability[:, -2:] = 0.0
            states[:, -2:] = 0.0
        result.append(
            Trajectory(
                trajectory_id=f"synthetic:{trajectory_index:03d}",
                states=states,
                availability=availability,
                stages=stages,
                label_valid=np.ones(length, dtype=np.float32),
            )
        )
    return result
