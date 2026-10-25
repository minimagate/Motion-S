"""Small, dependency-free trajectory prediction baselines.

Coordinates use ``[time, dimension]`` sequences. Timestamps may be irregular,
but must be strictly increasing. Forecasts are returned in the same coordinate
space as the input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

Vector = Sequence[float]
Matrix = Sequence[Vector]


def _validate(
    history: Matrix, timestamps: Sequence[float], future_timestamps: Sequence[float]
) -> tuple[list[list[float]], list[float], list[float]]:
    points = [[float(x) for x in row] for row in history]
    times = [float(x) for x in timestamps]
    future = [float(x) for x in future_timestamps]
    if not points or len(points) != len(times):
        raise ValueError("history and timestamps must have equal, non-zero length")
    dims = len(points[0])
    if dims == 0 or any(len(p) != dims for p in points):
        raise ValueError("history must have a consistent non-zero dimension")
    if any(not isfinite(v) for p in points for v in p) or any(
        not isfinite(t) for t in times + future
    ):
        raise ValueError("coordinates and timestamps must be finite")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("history timestamps must be strictly increasing")
    if any(t <= times[-1] for t in future) or any(b <= a for a, b in zip(future, future[1:])):
        raise ValueError("future timestamps must be increasing and after history")
    return points, times, future


def constant_velocity(
    history: Matrix,
    timestamps: Sequence[float],
    future_timestamps: Sequence[float],
    *,
    window: int = 2,
    ridge: float = 0.0,
) -> list[list[float]]:
    """Fit a linear trajectory over the latest validation-selected window."""
    points, times, future = _validate(history, timestamps, future_timestamps)
    if len(points) < 2:
        return [points[-1][:] for _ in future]
    return _polynomial_forecast(points, times, future, degree=1, window=window, ridge=ridge)


def constant_acceleration(
    history: Matrix,
    timestamps: Sequence[float],
    future_timestamps: Sequence[float],
    *,
    window: int = 3,
    ridge: float = 0.0,
) -> list[list[float]]:
    """Quadratic extrapolation from the latest two interval velocities.

    With fewer than three observations this falls back to constant velocity;
    a single observation produces a stationary forecast.
    """
    points, times, future = _validate(history, timestamps, future_timestamps)
    if len(points) < 3:
        return constant_velocity(points, times, future)
    return _polynomial_forecast(points, times, future, degree=2, window=window, ridge=ridge)


@dataclass(frozen=True)
class KalmanForecast:
    """Forecast position means and marginal covariance matrices per horizon."""

    means: list[list[float]]
    covariances: list[list[list[float]]]


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def _inverse(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    aug = [list(row) + ident for row, ident in zip(a, _identity(n))]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular covariance matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [x - factor * y for x, y in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def _polynomial_forecast(
    points: list[list[float]],
    times: list[float],
    future: list[float],
    *,
    degree: int,
    window: int,
    ridge: float,
) -> list[list[float]]:
    if window < degree + 1 or ridge < 0:
        raise ValueError("window must identify the polynomial and ridge must be non-negative")
    count = min(window, len(points))
    points, times = points[-count:], times[-count:]
    center = times[-1]
    design = [[(time - center) ** power for power in range(degree + 1)] for time in times]
    gram = _matmul(_transpose(design), design)
    for index in range(1, degree + 1):
        gram[index][index] += ridge
    inverse = _inverse(gram)
    coefficients = _matmul(_matmul(inverse, _transpose(design)), points)
    return [
        [
            sum(coefficients[power][axis] * (time - center) ** power for power in range(degree + 1))
            for axis in range(len(points[0]))
        ]
        for time in future
    ]


def linear_kalman_filter(
    history: Matrix,
    timestamps: Sequence[float],
    future_timestamps: Sequence[float],
    *,
    measurement_variance: float = 1.0,
    acceleration_variance: float = 0.1,
) -> KalmanForecast:
    """Constant-velocity linear Kalman filter with white-acceleration noise.

    State order is ``[positions..., velocities...]``. The initial velocity uses
    the last observed finite difference, or zero for a one-sample history.
    ``measurement_variance`` and ``acceleration_variance`` are scalar per-axis
    variances, not standard deviations.
    """
    points, times, future = _validate(history, timestamps, future_timestamps)
    if measurement_variance <= 0 or acceleration_variance < 0:
        raise ValueError(
            "measurement_variance must be positive and acceleration_variance non-negative"
        )
    d = len(points[0])
    n = 2 * d
    if len(points) > 1:
        dt = times[1] - times[0]
        velocity = [(b - a) / dt for a, b in zip(points[0], points[1])]
    else:
        velocity = [0.0] * d
    # Start at the first measurement and assimilate the remaining history in
    # chronological order.  Initializing at the final point and replaying the
    # history would leak the endpoint backward through the filter.
    x = points[0][:] + velocity
    p = [[0.0] * n for _ in range(n)]
    for i in range(d):
        p[i][i] = measurement_variance
        first_dt = times[1] - times[0] if len(points) > 1 else 1.0
        p[d + i][d + i] = (
            measurement_variance / max(first_dt**2, 1e-6) if len(points) > 1 else 100.0
        )
    # Filter observed positions using a linear constant-velocity model.
    for k in range(1, len(points)):
        dt = times[k] - times[k - 1]
        f = _identity(n)
        for i in range(d):
            f[i][d + i] = dt
        q = [[0.0] * n for _ in range(n)]
        for i in range(d):
            q[i][i] = acceleration_variance * dt**4 / 4
            q[i][d + i] = q[d + i][i] = acceleration_variance * dt**3 / 2
            q[d + i][d + i] = acceleration_variance * dt**2
        x = [sum(f[i][j] * x[j] for j in range(n)) for i in range(n)]
        p = [
            [v + q[i][j] for j, v in enumerate(row)]
            for i, row in enumerate(_matmul(_matmul(f, p), _transpose(f)))
        ]
        s = [
            [p[i][j] + (measurement_variance if i == j else 0.0) for j in range(d)]
            for i in range(d)
        ]
        # K = P H' S^-1, where H selects positions.
        k_gain = _matmul([row[:d] for row in p], _inverse(s))
        residual = [points[k][i] - x[i] for i in range(d)]
        x = [x[i] + sum(k_gain[i][j] * residual[j] for j in range(d)) for i in range(n)]
        kh = [[k_gain[i][j] if j < d else 0.0 for j in range(n)] for i in range(n)]
        p = _matmul([[(1.0 if i == j else 0.0) - kh[i][j] for j in range(n)] for i in range(n)], p)
    means: list[list[float]] = []
    covs: list[list[list[float]]] = []
    last_t = times[-1]
    for target_t in future:
        dt = target_t - last_t
        f = _identity(n)
        for i in range(d):
            f[i][d + i] = dt
        q = [[0.0] * n for _ in range(n)]
        for i in range(d):
            q[i][i] = acceleration_variance * dt**4 / 4
            q[i][d + i] = q[d + i][i] = acceleration_variance * dt**3 / 2
            q[d + i][d + i] = acceleration_variance * dt**2
        x = [sum(f[i][j] * x[j] for j in range(n)) for i in range(n)]
        p = [
            [v + q[i][j] for j, v in enumerate(row)]
            for i, row in enumerate(_matmul(_matmul(f, p), _transpose(f)))
        ]
        means.append(x[:d])
        covs.append([row[:d] for row in p[:d]])
        last_t = target_t
    return KalmanForecast(means, covs)


constant_velocity_forecast = constant_velocity
constant_acceleration_forecast = constant_acceleration
