"""Forecast a held-out synthetic trajectory with a trained checkpoint."""

import argparse

from motions.api import Forecaster
from motions.data import TrajectoryDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/smoke/best.pt")
    args = parser.parse_args()
    item = TrajectoryDataset(1, split="test", families=("hybrid",), seed=123)[0]
    forecast = Forecaster(args.checkpoint).forecast(
        item["observed"], item["timestamps"], num_samples=10
    )
    print("mean shape:", forecast.mean.shape)
    print("mode weights:", forecast.component_weights)
    print("first predicted point:", forecast.mean[0])


if __name__ == "__main__":
    main()
