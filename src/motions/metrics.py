"""Dependency-free trajectory and probabilistic forecast metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import exp, log, pi, sqrt
from time import perf_counter
from typing import Any


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("coordinate vectors must have equal, non-zero dimensions")
    return sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def ade(predictions: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> float:
    """Average displacement error across all forecast steps."""
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must have equal, non-zero length")
    return sum(_distance(p, t) for p, t in zip(predictions, targets)) / len(targets)


def fde(predictions: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> float:
    """Final displacement error."""
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must have equal, non-zero length")
    return _distance(predictions[-1], targets[-1])


def _inverse_and_logdet(matrix: Sequence[Sequence[float]]) -> tuple[list[list[float]], float]:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("covariances must be non-empty square matrices")
    a = [list(map(float, row)) for row in matrix]
    inv = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    logdet = 0.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if a[pivot][col] <= 0 and abs(a[pivot][col]) < 1e-12:
            raise ValueError("covariance must be positive definite")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            inv[col], inv[pivot] = inv[pivot], inv[col]
        diag = a[col][col]
        if diag <= 0:
            raise ValueError("covariance must be positive definite")
        logdet += log(diag)
        a[col] = [v / diag for v in a[col]]
        inv[col] = [v / diag for v in inv[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
                inv[row] = [x - factor * y for x, y in zip(inv[row], inv[col])]
    return inv, logdet


def gaussian_mixture_nll(
    target: Sequence[float],
    means: Sequence[Sequence[float]],
    covariances: Sequence[Sequence[Sequence[float]]],
    weights: Sequence[float],
) -> float:
    """Negative log likelihood of one target under a full-covariance Gaussian mixture."""
    if not means or len(means) != len(covariances) or len(means) != len(weights):
        raise ValueError("mixture means, covariances, and weights must have equal non-zero length")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("mixture weights must be non-negative with positive total")
    d = len(target)
    if not d or any(len(m) != d for m in means):
        raise ValueError("target and means dimensions must match")
    total = sum(weights)
    terms = []
    for mean, covariance, weight in zip(means, covariances, weights):
        if weight == 0:
            continue
        inv, logdet = _inverse_and_logdet(covariance)
        delta = [float(y) - float(x) for x, y in zip(mean, target)]
        mahal = sum(delta[i] * inv[i][j] * delta[j] for i in range(d) for j in range(d))
        terms.append(log(weight / total) - 0.5 * (d * log(2 * pi) + logdet + mahal))
    maximum = max(terms)
    return -(maximum + log(sum(exp(v - maximum) for v in terms)))


def gaussian_mixture_nll_over_time(
    targets: Sequence[Sequence[float]],
    means: Sequence[Sequence[Sequence[float]]],
    covariances: Sequence[Sequence[Sequence[Sequence[float]]]],
    weights: Sequence[Sequence[float]],
) -> float:
    """Mean per-step Gaussian mixture NLL. Arrays are time-major, then component-major."""
    if (
        not targets
        or len(targets) != len(means)
        or len(targets) != len(covariances)
        or len(targets) != len(weights)
    ):
        raise ValueError("time dimensions must be equal and non-zero")
    return sum(
        gaussian_mixture_nll(t, m, c, w) for t, m, c, w in zip(targets, means, covariances, weights)
    ) / len(targets)


def gaussian_mixture_calibration(
    target: Sequence[float],
    means: Sequence[Sequence[float]],
    covariances: Sequence[Sequence[Sequence[float]]],
    weights: Sequence[float],
    *,
    levels: Sequence[float] = (0.5, 0.9),
) -> dict[float, bool]:
    """Check empirical ellipsoidal coverage using the moment-matched mixture Gaussian.

    Returns inclusion booleans by requested nominal coverage. For dimension ``d``,
    the chi-square thresholds use a Wilson-Hilferty approximation.
    """
    if not means or len(means) != len(covariances) or len(means) != len(weights):
        raise ValueError("mixture inputs must have equal non-zero length")
    if any(level <= 0 or level >= 1 for level in levels):
        raise ValueError("coverage levels must lie strictly between 0 and 1")
    total = sum(weights)
    if total <= 0 or any(w < 0 for w in weights):
        raise ValueError("invalid mixture weights")
    d = len(target)
    norm_weights = [w / total for w in weights]
    mean = [sum(w * m[j] for w, m in zip(norm_weights, means)) for j in range(d)]
    covariance = [
        [
            sum(
                w * (c[i][j] + (m[i] - mean[i]) * (m[j] - mean[j]))
                for w, m, c in zip(norm_weights, means, covariances)
            )
            for j in range(d)
        ]
        for i in range(d)
    ]
    inv, _ = _inverse_and_logdet(covariance)
    delta = [float(y) - x for x, y in zip(mean, target)]
    mahal = sum(delta[i] * inv[i][j] * delta[j] for i in range(d) for j in range(d))

    # Wilson-Hilferty chi-square approximation with a compact inverse-normal
    # approximation for z.
    def zscore(p: float) -> float:
        # Acklam rational approximation
        a = [
            -39.6968302866538,
            220.946098424521,
            -275.928510447189,
            138.357751867269,
            -30.6647980661472,
            2.50662827745924,
        ]
        b = [
            -54.4760987982241,
            161.585836858041,
            -155.698979859887,
            66.8013118877197,
            -13.2806815528857,
        ]
        c = [
            -0.00778489400243029,
            -0.322396458041136,
            -2.40075827716184,
            -2.54973253934373,
            4.37466414146497,
            2.93816398269878,
        ]
        dd = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
        if p < 0.02425:
            q = sqrt(-2 * log(p))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
                (((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1
            )
        if p > 0.97575:
            return -zscore(1 - p)
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )

    return {
        level: mahal <= d * (1 - 2 / (9 * d) + zscore(level) * sqrt(2 / (9 * d))) ** 3
        for level in levels
    }


def inference_latency(
    call: Callable[..., Any], *args: Any, repeats: int = 100, **kwargs: Any
) -> dict[str, float]:
    """Measure mean wall-clock seconds per call and calls per second."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    start = perf_counter()
    for _ in range(repeats):
        call(*args, **kwargs)
    seconds = (perf_counter() - start) / repeats
    return {
        "seconds_per_call": seconds,
        "calls_per_second": 1.0 / seconds if seconds else float("inf"),
    }


def parameter_count(model: Any) -> int:
    """Count parameters on common NumPy/PyTorch-like models or nested containers."""
    if hasattr(model, "parameters") and callable(model.parameters):
        return sum(int(p.numel()) for p in model.parameters())
    if hasattr(model, "n_parameters"):
        return int(model.n_parameters)
    if isinstance(model, dict):
        return sum(parameter_count(v) for v in model.values())
    if isinstance(model, (list, tuple)):
        return sum(parameter_count(v) for v in model)
    if isinstance(model, (int, float, complex)):
        return 1
    if hasattr(model, "size") and isinstance(model.size, int):
        return model.size
    return 0


average_displacement_error = ade
final_displacement_error = fde
mixture_nll = gaussian_mixture_nll
