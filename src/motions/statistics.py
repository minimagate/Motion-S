"""Small deterministic uncertainty summaries for experiment metrics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import numpy as np


def mean_ci95(values: Sequence[float]) -> dict[str, float | int]:
    """Student-t interval across independent experiment seeds."""
    values = [float(value) for value in values]
    if not values:
        raise ValueError("at least one value is required")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"mean": mean, "low": mean, "high": mean, "n": 1}
    critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(len(values), 1.96)
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "low": mean - half_width, "high": mean + half_width, "n": len(values)}


def cluster_bootstrap_ci95(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int = 0,
    resamples: int = 2_000,
) -> dict[str, float | int]:
    """Bootstrap clusters, retaining within-actor correlation among windows."""
    if len(values) != len(clusters) or not values:
        raise ValueError("values and clusters must have equal non-zero length")
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters):
        grouped.setdefault(str(cluster), []).append(float(value))
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        sample = [value for key in chosen for value in grouped[str(key)]]
        draws[index] = np.mean(sample)
    return {
        "mean": float(np.mean(values)),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "clusters": len(keys),
        "resamples": resamples,
    }


__all__ = ["cluster_bootstrap_ci95", "mean_ci95"]
