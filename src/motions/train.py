from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import build_model
from .config import ExperimentConfig, load_config
from .data import RESEARCH_PROFILE, collate_trajectories, make_splits
from .metrics import parameter_count
from .utils import atomic_torch_save, seed_everything, select_device, write_json


def _loader(dataset, config: ExperimentConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.data.seed + (1 if shuffle else 2))
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        collate_fn=collate_trajectories,
        generator=generator,
        persistent_workers=config.train.num_workers > 0,
    )


def _move(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def validate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_nll = total_ade = total_fde = total_noise_mae = 0.0
    total = 0
    for raw in loader:
        batch = _move(raw, device)
        distribution = model(batch["observed"], batch["timestamps"], batch["mask"])
        target = batch["future"]
        count = target.shape[0]
        errors = torch.linalg.vector_norm(distribution.mean - target, dim=-1)
        total_nll += float(distribution.nll(target)) * count
        total_ade += float(errors.mean()) * count
        total_fde += float(errors[:, -1].mean()) * count
        total_noise_mae += (
            float((distribution.measurement_noise_std - batch["noise_std"]).abs().mean()) * count
        )
        total += count
    if not total:
        raise ValueError("validation loader is empty")
    return {
        "nll": total_nll / total,
        "ade": total_ade / total,
        "fde": total_fde / total,
        "noise_mae": total_noise_mae / total,
    }


def train(config: ExperimentConfig) -> dict[str, object]:
    seed_everything(config.data.seed)
    device = select_device(config.train.device)
    splits = make_splits(
        train_size=config.data.train_samples,
        val_size=config.data.val_samples,
        test_size=config.data.test_samples,
        seed=config.data.seed,
        dim=config.data.dimension,
        obs_length=(config.data.min_observations, config.data.max_observations),
        horizon=config.data.horizon,
        profile=RESEARCH_PROFILE,
    )
    train_loader = _loader(splits["train"], config, shuffle=True)
    val_loader = _loader(splits["val"], config, shuffle=False)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_nll = float("inf")
    stale_epochs = 0
    started = time.perf_counter()
    print(f"device={device} parameters={parameter_count(model):,}")

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for raw in train_loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            distribution = model(batch["observed"], batch["timestamps"], batch["mask"])
            noise_loss = torch.nn.functional.mse_loss(
                distribution.measurement_noise_std.clamp_min(1e-4).log(),
                batch["noise_std"].clamp_min(1e-4).log(),
            )
            loss = distribution.nll(batch["future"]) + config.train.noise_loss_weight * noise_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            count = batch["future"].shape[0]
            train_loss += float(loss.detach()) * count
            seen += count
        metrics = validate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_nll": train_loss / seen,
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        history.append(record)
        print(
            " ".join(
                f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in record.items()
            )
        )

        payload = {
            "format_version": 1,
            "config": dataclasses.asdict(config),
            "training_protocol": {
                "held_out_families": list(RESEARCH_PROFILE.held_out_families),
                "unseen_combinations": RESEARCH_PROFILE.unseen_combinations,
                "split_unit": "trajectory seed before window generation",
            },
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "validation": metrics,
        }
        atomic_torch_save(payload, output_dir / "last.pt")
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            stale_epochs = 0
            atomic_torch_save(payload, output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= config.train.patience:
                print(f"early_stop epoch={epoch}")
                break

    result: dict[str, object] = {
        "device": str(device),
        "parameters": parameter_count(model),
        "elapsed_seconds": time.perf_counter() - started,
        "best_validation_nll": best_nll,
        "history": history,
    }
    write_json(result, output_dir / "train_metrics.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Motion-S probabilistic TCN")
    parser.add_argument("--config", default="configs/base.toml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
