import numpy as np

from cyberworld.data.normalization import StandardNormalizer
from cyberworld.data.trajectory import Trajectory


def test_normalizer_ignores_unavailable_values() -> None:
    trajectory = Trajectory(
        "x",
        np.asarray([[1.0, 999.0], [3.0, -999.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([0, 0]),
        np.ones(2, dtype=np.float32),
    )
    normalizer = StandardNormalizer.fit([trajectory])
    transformed = normalizer.transform(trajectory)
    np.testing.assert_allclose(transformed.states[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(transformed.states[:, 1], [0.0, 0.0])
