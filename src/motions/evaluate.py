from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .baselines import constant_acceleration, constant_velocity, linear_kalman_filter
from .checkpoint import load_checkpoint
from .data import FAMILIES, RESEARCH_PROFILE, TrajectoryDataset, collate_trajectories
from .metrics import ade, fde, gaussian_mixture_calibration, parameter_count
from .utils import select_device, write_json


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _latency(call: Callable[[], object], device: torch.device, repeats: int = 30) -> float:
    for _ in range(3):
        call()
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    _sync(device)
    return 1000 * (time.perf_counter() - start) / repeats


def _calibration(distribution, target: torch.Tensor) -> tuple[dict[str, float], float]:
    """Moment-match the mixture at each step and measure ellipsoidal coverage."""
    weights = distribution.weights
    mean = distribution.mean
    differences = distribution.means - mean[:, None]
    rotation = distribution.transform.rotation
    spatial_scale = distribution.transform.scale[:, 0, 0]
    local_variances = (
        distribution.local_scales.square() * spatial_scale[:, None, None, None].square()
    )
    component_cov = torch.einsum("bia,bkha,bja->bkhij", rotation, local_variances, rotation)
    between_cov = differences[..., :, None] * differences[..., None, :]
    covariance = (weights[:, :, None, None, None] * (component_cov + between_cov)).sum(dim=1)
    eye = torch.eye(target.shape[-1], device=target.device, dtype=target.dtype)
    covariance = covariance + 1e-6 * eye
    delta = target - mean
    mahalanobis = torch.einsum("bhi,bhij,bhj->bh", delta, torch.linalg.inv(covariance), delta)
    thresholds = {2: {"50": 1.386294, "90": 4.605170}, 3: {"50": 2.365974, "90": 6.251389}}
    coverage = {
        level: float((mahalanobis <= threshold).float().mean())
        for level, threshold in thresholds[target.shape[-1]].items()
    }
    error = (abs(coverage["50"] - 0.5) + abs(coverage["90"] - 0.9)) / 2
    return coverage, error


@torch.no_grad()
def evaluate_model(
    model, dataset: TrajectoryDataset, device: torch.device, batch_size: int
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_trajectories)
    totals = {"ade": 0.0, "fde": 0.0, "nll": 0.0, "coverage_50": 0.0, "coverage_90": 0.0}
    count = 0
    model.eval()
    for raw in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw.items()
        }
        output = model(batch["observed"], batch["timestamps"], batch["mask"])
        errors = torch.linalg.vector_norm(output.mean - batch["future"], dim=-1)
        coverage, _ = _calibration(output, batch["future"])
        n = batch["future"].shape[0]
        totals["ade"] += float(errors.mean()) * n
        totals["fde"] += float(errors[:, -1].mean()) * n
        totals["nll"] += float(output.nll(batch["future"])) * n
        totals["coverage_50"] += coverage["50"] * n
        totals["coverage_90"] += coverage["90"] * n
        count += n
    result = {key: value / count for key, value in totals.items()}
    result["nll_per_coordinate"] = result["nll"] / (dataset.horizon * dataset.dim)
    result["calibration_error"] = (
        abs(result["coverage_50"] - 0.5) + abs(result["coverage_90"] - 0.9)
    ) / 2
    return result


def evaluate_baselines(dataset: TrajectoryDataset) -> dict[str, dict[str, float]]:
    accumulators = {
        "constant_velocity": {"ade": 0.0, "fde": 0.0},
        "constant_acceleration": {"ade": 0.0, "fde": 0.0},
        "kalman": {"ade": 0.0, "fde": 0.0, "coverage_50": 0.0, "coverage_90": 0.0},
    }
    for index in range(len(dataset)):
        item = dataset[index]
        history = item["observed"].tolist()
        times = item["timestamps"].tolist()
        future_times = item["future_timestamps"].tolist()
        target = item["future"].tolist()
        cv = constant_velocity(history, times, future_times)
        ca = constant_acceleration(history, times, future_times)
        kalman = linear_kalman_filter(
            history,
            times,
            future_times,
            measurement_variance=max(float(item["noise_std"]) ** 2, 1e-5),
            acceleration_variance=0.1,
        )
        for name, prediction in (
            ("constant_velocity", cv),
            ("constant_acceleration", ca),
            ("kalman", kalman.means),
        ):
            accumulators[name]["ade"] += ade(prediction, target)
            accumulators[name]["fde"] += fde(prediction, target)
        for point, mean, covariance in zip(target, kalman.means, kalman.covariances):
            # A small floor keeps deterministic-looking filters numerically valid.
            floored = [row[:] for row in covariance]
            for axis in range(len(floored)):
                floored[axis][axis] += 1e-6
            covered = gaussian_mixture_calibration(point, [mean], [floored], [1.0])
            accumulators["kalman"]["coverage_50"] += float(covered[0.5]) / dataset.horizon
            accumulators["kalman"]["coverage_90"] += float(covered[0.9]) / dataset.horizon
    for metrics in accumulators.values():
        for key in metrics:
            metrics[key] /= len(dataset)
    kalman_metrics = accumulators["kalman"]
    kalman_metrics["calibration_error"] = (
        abs(kalman_metrics["coverage_50"] - 0.5) + abs(kalman_metrics["coverage_90"] - 0.9)
    ) / 2
    return accumulators


def _datasets(config, size: int) -> dict[str, TrajectoryDataset]:
    common = dict(
        size=size,
        seed=config.data.seed + 10_000,
        dim=config.data.dimension,
        obs_length=(config.data.min_observations, config.data.max_observations),
        horizon=config.data.horizon,
        families=FAMILIES[:-1],
    )
    return {
        # A fresh validation seed follows the same joint regimes as fitting data.
        "in_distribution": TrajectoryDataset(split="val", profile=RESEARCH_PROFILE, **common),
        "unseen_parameters": TrajectoryDataset(split="test", profile="unseen_parameters", **common),
        "unseen_combinations": TrajectoryDataset(
            split="test", profile="unseen_combinations", **common
        ),
        "held_out_family": TrajectoryDataset(
            split="test", profile="held_out_family", **{**common, "families": FAMILIES}
        ),
    }


def evaluate(
    checkpoint: str | Path, *, size: int | None = None, requested_device: str = "auto"
) -> dict[str, object]:
    device = select_device(requested_device)
    model, config, payload = load_checkpoint(checkpoint, device)
    datasets = _datasets(config, size or config.data.test_samples)
    suites: dict[str, object] = {}
    for name, dataset in datasets.items():
        print(f"evaluating={name} samples={len(dataset)}")
        suites[name] = {
            "neural": evaluate_model(model, dataset, device, config.train.batch_size),
            "baselines": evaluate_baselines(dataset),
        }

    item = collate_trajectories([datasets["in_distribution"][0]])
    model_inputs = tuple(item[key].to(device) for key in ("observed", "timestamps", "mask"))
    with torch.no_grad():
        neural_ms = _latency(lambda: model(*model_inputs), device)
    baseline_item = datasets["in_distribution"][0]
    args = (
        baseline_item["observed"].tolist(),
        baseline_item["timestamps"].tolist(),
        baseline_item["future_timestamps"].tolist(),
    )
    latency = {
        "neural_batch1_ms": neural_ms,
        "constant_velocity_ms": _latency(lambda: constant_velocity(*args), torch.device("cpu")),
        "constant_acceleration_ms": _latency(
            lambda: constant_acceleration(*args), torch.device("cpu")
        ),
        "kalman_ms": _latency(lambda: linear_kalman_filter(*args), torch.device("cpu")),
    }
    result: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "device": str(device),
        "parameters": parameter_count(model),
        "latency": latency,
        "suites": suites,
        "notes": {
            "selection": "checkpoint selected only by validation NLL",
            "calibration": "moment-matched per-step ellipsoidal coverage at 50% and 90%",
            "kalman": "process variance is fixed at 0.1; it was not tuned on test data",
        },
    }
    output = Path(checkpoint).parent / "evaluation.json"
    write_json(result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Motion-S and classical baselines")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, size=args.samples, requested_device=args.device)
    print(f"wrote={Path(args.checkpoint).parent / 'evaluation.json'}")
    for suite, values in result["suites"].items():
        neural = values["neural"]
        print(f"{suite}: neural_ADE={neural['ade']:.4f} neural_FDE={neural['fde']:.4f}")


if __name__ == "__main__":
    main()
