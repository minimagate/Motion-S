from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from .model import TrajectoryTCN


def build_model(config: ExperimentConfig) -> TrajectoryTCN:
    return TrajectoryTCN(
        spatial_dim=config.data.dimension,
        horizon=config.data.horizon,
        components=config.model.mixtures,
        width=config.model.channels,
        depth=config.model.blocks,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
    )


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[TrajectoryTCN, ExperimentConfig, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = config_from_dict(payload["config"])
    model = build_model(config)
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    allowed_missing = {"future_time_projection.weight"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible checkpoint keys: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device).eval()
    return model, config, payload
