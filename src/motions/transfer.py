"""Controlled synthetic-to-real transfer experiments for Kine-S."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .baseline_tuning import tune_baselines
from .checkpoint import build_model, load_checkpoint
from .config import ExperimentConfig, load_config
from .data import collate_trajectories
from .evaluate import _latency, evaluate_baselines, evaluate_model
from .metrics import parameter_count
from .real_data import RealTrajectoryDataset, load_eth, split_tracks
from .statistics import mean_ci95
from .train import train
from .utils import atomic_torch_save, seed_everything, select_device, write_json


def _loader(dataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_trajectories,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def _validation_nll(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        tensors = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = model(
            tensors["observed"],
            tensors["timestamps"],
            tensors["mask"],
            tensors["future_timestamps"],
        )
        size = tensors["future"].shape[0]
        total += float(output.nll(tensors["future"])) * size
        count += size
    return total / count


def fine_tune(
    model,
    config: ExperimentConfig,
    train_dataset,
    val_dataset,
    *,
    seed: int,
    mode: str,
    max_steps: int,
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    if mode not in {"partial", "full", "scratch"}:
        raise ValueError("mode must be partial, full, or scratch")
    seed_everything(seed)
    if mode == "partial":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    train_loader = _loader(train_dataset, config.train.batch_size, seed, shuffle=True)
    val_loader = _loader(val_dataset, config.train.batch_size, seed + 1, shuffle=False)
    best_nll = float("inf")
    best_state = None
    best_step = 0
    started = time.perf_counter()
    model.to(device)
    iterator = iter(train_loader)
    for step in range(1, max_steps + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        tensors = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        output_distribution = model(
            tensors["observed"],
            tensors["timestamps"],
            tensors["mask"],
            tensors["future_timestamps"],
        )
        loss = output_distribution.nll(tensors["future"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.train.grad_clip)
        optimizer.step()
        if step % 10 == 0 or step == max_steps:
            validation_nll = _validation_nll(model, val_loader, device)
            if validation_nll < best_nll:
                best_nll = validation_nll
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
    if best_state is None:
        raise RuntimeError("fine-tuning produced no checkpoint")
    model.load_state_dict(best_state)
    payload = {
        "format_version": 2,
        "config": dataclasses.asdict(config),
        "model_state": best_state,
        "epoch": best_step,
        "validation": {"nll": best_nll},
        "transfer": {
            "mode": mode,
            "seed": seed,
            "train_tracks": len(train_dataset.tracks),
            "optimizer_steps": max_steps,
            "selection_interval_steps": 10,
            "split_unit": "complete pedestrian track",
        },
    }
    atomic_torch_save(payload, output)
    return {
        "best_step": best_step,
        "best_validation_nll": best_nll,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_steps": max_steps,
    }


def _seed_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    return {
        metric: mean_ci95([run[metric] for run in runs])
        for metric in ("ade", "fde", "nll_per_coordinate")
    }


def run_transfer(
    config: ExperimentConfig,
    *,
    eth_root: str | Path,
    seeds: tuple[int, ...] = (7, 17, 27),
    track_counts: tuple[int, ...] = (16, 64, 0),
    fine_tune_steps: int = 60,
    scratch_steps: int | None = None,
) -> dict[str, object]:
    scratch_steps = scratch_steps or fine_tune_steps
    device = select_device(config.train.device)
    output_root = Path(config.train.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tracks = load_eth(eth_root)
    track_split = split_tracks(tracks, seed=2026)
    validation = RealTrajectoryDataset(
        track_split["val"], observations=8, horizon=config.data.horizon, stride=4, subset_seed=2026
    )
    test = RealTrajectoryDataset(
        track_split["test"], observations=8, horizon=config.data.horizon, stride=4, subset_seed=2026
    )
    baseline_parameters, baseline_validation = tune_baselines(validation)
    baselines = evaluate_baselines(test, baseline_parameters)

    seed_runs: list[dict[str, object]] = []
    maximum_tracks = len(
        RealTrajectoryDataset(
            track_split["train"],
            observations=8,
            horizon=config.data.horizon,
            stride=4,
            subset_seed=2026,
        ).tracks
    )
    resolved_counts = tuple(maximum_tracks if count == 0 else count for count in track_counts)

    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_metrics_path = seed_dir / "transfer_metrics.json"
        existing = None
        if seed_metrics_path.exists():
            existing = json.loads(seed_metrics_path.read_text())
            complete = all(
                str(count) in existing.get("sample_efficiency", {})
                and all(
                    existing["sample_efficiency"][str(count)]
                    .get(mode, {})
                    .get("training", {})
                    .get("optimizer_steps")
                    == (scratch_steps if mode == "scratch" else fine_tune_steps)
                    for mode in ("partial", "full", "scratch")
                )
                for count in resolved_counts
            )
            if complete:
                seed_runs.append(existing)
                continue
        pretrain_dir = seed_dir / "pretrain"
        pretrain_checkpoint = pretrain_dir / "best.pt"
        seed_config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, seed=seed),
            train=dataclasses.replace(config.train, output_dir=str(pretrain_dir)),
        )
        if not pretrain_checkpoint.exists():
            train(seed_config)
        pretrained, checkpoint_config, _ = load_checkpoint(pretrain_checkpoint, device)
        if existing is None:
            zero_shot = evaluate_model(pretrained, test, device, checkpoint_config.train.batch_size)
            seed_result: dict[str, object] = {
                "seed": seed,
                "pretrain_checkpoint": str(pretrain_checkpoint),
                "zero_shot": zero_shot,
                "sample_efficiency": {},
            }
        else:
            seed_result = existing
        for count in resolved_counts:
            real_train = RealTrajectoryDataset(
                track_split["train"],
                observations=8,
                horizon=config.data.horizon,
                stride=4,
                max_tracks=count,
                subset_seed=2026,
            )
            count_result: dict[str, object] = seed_result["sample_efficiency"].get(str(count), {})
            for mode in ("partial", "full", "scratch"):
                expected_steps = scratch_steps if mode == "scratch" else fine_tune_steps
                completed_steps = (
                    count_result.get(mode, {}).get("training", {}).get("optimizer_steps")
                )
                if completed_steps == expected_steps:
                    continue
                if mode == "scratch":
                    seed_everything(seed)
                    model = build_model(checkpoint_config)
                else:
                    model, _, _ = load_checkpoint(pretrain_checkpoint, device)
                checkpoint_path = seed_dir / f"real_{count}_{mode}.pt"
                training = fine_tune(
                    model,
                    checkpoint_config,
                    real_train,
                    validation,
                    seed=seed,
                    mode=mode,
                    max_steps=expected_steps,
                    output=checkpoint_path,
                    device=device,
                )
                metrics = evaluate_model(model, test, device, checkpoint_config.train.batch_size)
                count_result[mode] = {**metrics, "training": training}
            seed_result["sample_efficiency"][str(count)] = count_result
        seed_runs.append(seed_result)
        write_json(seed_result, seed_metrics_path)

    zero_shot_summary = _seed_summary([run["zero_shot"] for run in seed_runs])
    efficiency_summary: dict[str, object] = {}
    for count in resolved_counts:
        methods = {}
        for mode in ("partial", "full", "scratch"):
            runs = [run["sample_efficiency"][str(count)][mode] for run in seed_runs]
            methods[mode] = _seed_summary(runs)
        paired_gain = [
            run["sample_efficiency"][str(count)]["scratch"]["ade"]
            - run["sample_efficiency"][str(count)]["full"]["ade"]
            for run in seed_runs
        ]
        methods["pretraining_ade_gain_vs_scratch"] = mean_ci95(paired_gain)
        efficiency_summary[str(count)] = methods

    full_count = max(resolved_counts)
    cross_budget_contrasts = {
        str(count): mean_ci95(
            [
                run["sample_efficiency"][str(full_count)]["scratch"]["ade"]
                - run["sample_efficiency"][str(count)]["full"]["ade"]
                for run in seed_runs
            ]
        )
        for count in resolved_counts
    }

    latency_model, _, _ = load_checkpoint(seed_runs[0]["pretrain_checkpoint"], device)
    latency_batch = collate_trajectories([test[0]])
    latency_inputs = tuple(
        latency_batch[key].to(device)
        for key in ("observed", "timestamps", "mask", "future_timestamps")
    )
    with torch.no_grad():
        batch1_latency_ms = _latency(lambda: latency_model(*latency_inputs), device, repeats=100)

    result: dict[str, object] = {
        "device": str(device),
        "runtime": {
            "parameters": parameter_count(latency_model),
            "batch1_latency_ms": batch1_latency_ms,
            "latency_repeats": 100,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "seeds": list(seeds),
        "training_budget": {
            "pretrained_optimizer_steps": fine_tune_steps,
            "scratch_optimizer_steps": scratch_steps,
            "selection_interval_steps": 10,
        },
        "dataset": {
            "name": "ETH Walking Pedestrians (ETH + Hotel)",
            "source_tracks": len(tracks),
            "eligible_train_tracks": maximum_tracks,
            "validation_tracks": len(validation.tracks),
            "test_tracks": len(test.tracks),
            "validation_windows": len(validation),
            "test_windows": len(test),
            "split_seed": 2026,
            "split_unit": "complete actor track, stratified by scene",
        },
        "baseline_tuning": {
            "parameters": baseline_parameters.as_dict(),
            "validation_ade": baseline_validation,
        },
        "baselines": baselines,
        "zero_shot": zero_shot_summary,
        "sample_efficiency": efficiency_summary,
        "sample_efficiency_contrasts": {
            "definition": (
                f"ADE gain of full fine-tuning at N actors over scratch at {full_count} actors"
            ),
            "full_fine_tune_vs_full_data_scratch": cross_budget_contrasts,
        },
        "runs": seed_runs,
    }
    write_json(result, output_root / "transfer_report.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kine-S synthetic-to-real transfer study")
    parser.add_argument("--config", default="configs/transfer.toml")
    parser.add_argument("--eth-root", default="data/real/ewap_dataset")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--track-counts", default="16,64,0")
    parser.add_argument("--fine-tune-steps", type=int, default=60)
    parser.add_argument("--scratch-steps", type=int)
    args = parser.parse_args()
    run_transfer(
        load_config(args.config),
        eth_root=args.eth_root,
        seeds=tuple(int(value) for value in args.seeds.split(",")),
        track_counts=tuple(int(value) for value in args.track_counts.split(",")),
        fine_tune_steps=args.fine_tune_steps,
        scratch_steps=args.scratch_steps,
    )


if __name__ == "__main__":
    main()
