from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .utils import select_device


@dataclass(frozen=True)
class Forecast:
    timestamps: np.ndarray
    mean: np.ndarray
    component_means: np.ndarray
    aleatoric_std: np.ndarray
    component_weights: np.ndarray
    measurement_noise_std: float
    samples: np.ndarray | None = None


class Forecaster:
    """Small inference API around a trained Kine-S checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        self.device = select_device(device)
        self.model, self.config, _ = load_checkpoint(checkpoint, self.device)

    @torch.inference_mode()
    def forecast(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray,
        timestamps: Sequence[float] | np.ndarray,
        *,
        horizon: int | None = None,
        future_timestamps: Sequence[float] | np.ndarray | None = None,
        num_samples: int = 0,
    ) -> Forecast:
        positions_array = np.asarray(positions, dtype=np.float32)
        times_array = np.asarray(timestamps, dtype=np.float32)
        if positions_array.ndim != 2 or positions_array.shape[1] != self.config.data.dimension:
            raise ValueError(f"positions must have shape [time, {self.config.data.dimension}]")
        if times_array.shape != (positions_array.shape[0],):
            raise ValueError("timestamps must have one value per position")
        if len(times_array) < 2 or np.any(np.diff(times_array) <= 0):
            raise ValueError("at least two strictly increasing timestamps are required")
        requested_horizon = horizon if horizon is not None else self.config.data.horizon
        if requested_horizon != self.config.data.horizon:
            raise ValueError(
                f"this checkpoint predicts {self.config.data.horizon} steps; "
                "train/load a checkpoint with the requested horizon"
            )
        positions_tensor = torch.as_tensor(positions_array, device=self.device)[None]
        times_tensor = torch.as_tensor(times_array, device=self.device)[None]
        mask = torch.ones_like(times_tensor, dtype=torch.bool)
        cadence = float(np.median(np.diff(times_array)))
        if future_timestamps is None:
            future_times = times_array[-1] + cadence * np.arange(1, requested_horizon + 1)
        else:
            future_times = np.asarray(future_timestamps, dtype=np.float32)
            if future_times.shape != (requested_horizon,) or np.any(
                future_times <= times_array[-1]
            ):
                raise ValueError("future_timestamps must match the horizon and follow observations")
        future_times = np.asarray(future_times, dtype=np.float32)
        future_tensor = torch.as_tensor(future_times, device=self.device)[None]
        distribution = self.model(positions_tensor, times_tensor, mask, future_tensor)
        draws = distribution.sample(num_samples)[0].cpu().numpy() if num_samples else None
        return Forecast(
            timestamps=future_times,
            mean=distribution.mean[0].cpu().numpy(),
            component_means=distribution.means[0].cpu().numpy(),
            aleatoric_std=distribution.scales[0].cpu().numpy(),
            component_weights=distribution.weights[0].cpu().numpy(),
            measurement_noise_std=float(distribution.measurement_noise_std[0].cpu()),
            samples=draws,
        )


__all__ = ["Forecast", "Forecaster"]
