import numpy as np

from cyberworld.data.trajectory import Trajectory, TrajectoryWindowDataset


def make_trajectory(name: str, offset: float) -> Trajectory:
    states = np.arange(24, dtype=np.float32).reshape(6, 4) + offset
    return Trajectory(
        trajectory_id=name,
        states=states,
        availability=np.ones_like(states),
        stages=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        label_valid=np.ones(6, dtype=np.float32),
    )


def test_windows_never_cross_trajectory_boundaries() -> None:
    first = make_trajectory("first", 0)
    second = make_trajectory("second", 10_000)
    dataset = TrajectoryWindowDataset([first, second], history_windows=2, forecast_windows=2)
    assert len(dataset) == 6
    for sample in dataset:
        combined = np.concatenate((sample["history"].numpy(), sample["future_states"].numpy()))
        assert np.all(combined < 10_000) or np.all(combined >= 10_000)


def test_unknown_labels_are_masked() -> None:
    trajectory = make_trajectory("unknown", 0)
    stages = trajectory.stages.copy()
    stages[2] = 8
    trajectory = Trajectory(
        trajectory.trajectory_id,
        trajectory.states,
        trajectory.availability,
        stages,
        trajectory.label_valid,
    )
    sample = TrajectoryWindowDataset([trajectory], history_windows=2, forecast_windows=2)[0]
    assert sample["future_label_valid"][0].item() == 0.0
