from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


@dataclass
class DataConfig:
    dimension: int = 2
    min_observations: int = 8
    max_observations: int = 32
    horizon: int = 20
    train_samples: int = 50_000
    val_samples: int = 2_000
    test_samples: int = 2_000
    seed: int = 7


@dataclass
class ModelConfig:
    channels: int = 256
    blocks: int = 6
    kernel_size: int = 3
    dropout: float = 0.1
    mixtures: int = 5


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    patience: int = 5
    grad_clip: float = 1.0
    noise_loss_weight: float = 0.05
    device: str = "auto"
    output_dir: str = "artifacts/default"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _update(instance: Any, values: dict[str, Any]) -> Any:
    valid = {field.name for field in dataclasses.fields(instance)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown {type(instance).__name__} keys: {sorted(unknown)}")
    return type(instance)(**{**dataclasses.asdict(instance), **values})


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return ExperimentConfig(
        data=_update(DataConfig(), raw.get("data", {})),
        model=_update(ModelConfig(), raw.get("model", {})),
        train=_update(TrainConfig(), raw.get("train", {})),
    )
