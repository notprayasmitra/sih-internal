import numpy as np

from cyberworld.cli import _split_trajectories, _split_trajectories_explicit
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


def test_split_uses_disjoint_trajectory_ids_and_preserves_size_difference() -> None:
    trajectories = [
        make_trajectory("short", 0),
        make_trajectory("medium", 100),
        Trajectory(
            trajectory_id="long",
            states=np.arange(40, dtype=np.float32).reshape(10, 4) + 200,
            availability=np.ones((10, 4), dtype=np.float32),
            stages=np.asarray([0, 0, 1, 1, 2, 2, 0, 0, 1, 1], dtype=np.int64),
            label_valid=np.ones(10, dtype=np.float32),
        ),
    ]
    train, validation = _split_trajectories(trajectories, validation_fraction=0.34, seed=7)
    train_ids = {item.trajectory_id for item in train}
    validation_ids = {item.trajectory_id for item in validation}
    assert train_ids.isdisjoint(validation_ids)
    train_windows = len(TrajectoryWindowDataset(train, history_windows=2, forecast_windows=2))
    validation_windows = len(
        TrajectoryWindowDataset(validation, history_windows=2, forecast_windows=2)
    )
    assert train_windows != validation_windows


def test_explicit_split_resolves_exact_ids_in_configured_order() -> None:
    trajectories = [
        make_trajectory("alpha", 0),
        make_trajectory("beta", 100),
        make_trajectory("gamma", 200),
        make_trajectory("delta", 300),
    ]

    train, validation = _split_trajectories_explicit(
        trajectories,
        train_trajectory_ids=("gamma", "alpha"),
        validation_trajectory_ids=("delta", "beta"),
    )

    assert [item.trajectory_id for item in train] == ["gamma", "alpha"]
    assert [item.trajectory_id for item in validation] == ["delta", "beta"]
