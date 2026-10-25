"""Validation-only tuning for classical motion baselines."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from itertools import product

import numpy as np

from .baselines import constant_acceleration, constant_velocity, linear_kalman_filter
from .metrics import ade


@dataclass(frozen=True)
class BaselineParameters:
    velocity_window: int = 2
    velocity_ridge: float = 0.0
    acceleration_window: int = 3
    acceleration_ridge: float = 0.0
    kalman_measurement_variance: float = 0.01
    kalman_acceleration_variance: float = 0.1

    def as_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


def _items(dataset, max_examples: int) -> list[dict[str, object]]:
    if len(dataset) <= max_examples:
        indices = range(len(dataset))
    else:
        indices = np.linspace(0, len(dataset) - 1, max_examples, dtype=int)
    return [dataset[int(index)] for index in indices]


def _score(items, forecast) -> float:
    errors = []
    for item in items:
        prediction = forecast(
            item["observed"].tolist(),
            item["timestamps"].tolist(),
            item["future_timestamps"].tolist(),
        )
        means = prediction.means if hasattr(prediction, "means") else prediction
        errors.append(ade(means, item["future"].tolist()))
    return float(np.mean(errors))


def tune_baselines(
    dataset, *, max_examples: int = 256
) -> tuple[BaselineParameters, dict[str, float]]:
    """Select all hyperparameters on validation ADE, never on test data."""
    items = _items(dataset, max_examples)
    velocity_candidates = list(product((2, 4, 8), (0.0, 0.01, 0.1)))
    acceleration_candidates = list(product((3, 5, 8), (0.0, 0.01, 0.1)))
    kalman_candidates = list(product((1e-4, 1e-3, 1e-2, 1e-1), (1e-3, 1e-2, 1e-1, 1.0)))

    velocity_scores = {
        candidate: _score(
            items,
            lambda history, times, future, c=candidate: constant_velocity(
                history, times, future, window=c[0], ridge=c[1]
            ),
        )
        for candidate in velocity_candidates
    }
    acceleration_scores = {
        candidate: _score(
            items,
            lambda history, times, future, c=candidate: constant_acceleration(
                history, times, future, window=c[0], ridge=c[1]
            ),
        )
        for candidate in acceleration_candidates
    }
    kalman_scores = {
        candidate: _score(
            items,
            lambda history, times, future, c=candidate: linear_kalman_filter(
                history,
                times,
                future,
                measurement_variance=c[0],
                acceleration_variance=c[1],
            ),
        )
        for candidate in kalman_candidates
    }
    velocity = min(velocity_scores, key=velocity_scores.get)
    acceleration = min(acceleration_scores, key=acceleration_scores.get)
    kalman = min(kalman_scores, key=kalman_scores.get)
    parameters = BaselineParameters(
        velocity_window=velocity[0],
        velocity_ridge=velocity[1],
        acceleration_window=acceleration[0],
        acceleration_ridge=acceleration[1],
        kalman_measurement_variance=kalman[0],
        kalman_acceleration_variance=kalman[1],
    )
    scores = {
        "constant_velocity_ade": velocity_scores[velocity],
        "constant_acceleration_ade": acceleration_scores[acceleration],
        "kalman_ade": kalman_scores[kalman],
    }
    return parameters, scores


__all__ = ["BaselineParameters", "tune_baselines"]
