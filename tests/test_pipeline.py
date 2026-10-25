import dataclasses

import numpy as np
import torch

from motions.api import Forecaster
from motions.checkpoint import build_model
from motions.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from motions.data import TrajectoryDataset, collate_trajectories


def test_collate_and_checkpoint_forecasting(tmp_path):
    config = ExperimentConfig(
        data=DataConfig(dimension=2, horizon=4, min_observations=4, max_observations=6),
        model=ModelConfig(channels=32, blocks=2, mixtures=2, dropout=0.0),
        train=TrainConfig(),
    )
    model = build_model(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "config": dataclasses.asdict(config),
            "model_state": model.state_dict(),
            "epoch": 0,
        },
        checkpoint,
    )
    dataset = TrajectoryDataset(2, obs_length=(4, 6), horizon=4, seed=4)
    batch = collate_trajectories([dataset[0], dataset[1]])
    assert batch["observed"].shape[:2] == batch["mask"].shape

    forecaster = Forecaster(checkpoint, device="cpu")
    item = dataset[0]
    result = forecaster.forecast(item["observed"], item["timestamps"], num_samples=3)
    assert result.mean.shape == (4, 2)
    assert result.samples.shape == (3, 4, 2)
    assert np.isclose(result.component_weights.sum(), 1.0)
    assert result.measurement_noise_std > 0
