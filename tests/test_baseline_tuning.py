import math

from motions.baseline_tuning import tune_baselines
from motions.data import TrajectoryDataset


def test_baseline_tuning_returns_validation_grid_choices():
    validation = TrajectoryDataset(
        6,
        split="val",
        seed=13,
        obs_length=8,
        horizon=4,
        dt_jitter=0.4,
    )
    parameters, scores = tune_baselines(validation, max_examples=4)
    assert parameters.velocity_window in {2, 4, 8}
    assert parameters.acceleration_window in {3, 5, 8}
    assert parameters.kalman_measurement_variance in {1e-4, 1e-3, 1e-2, 1e-1}
    assert all(math.isfinite(value) for value in scores.values())
