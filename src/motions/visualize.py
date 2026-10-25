from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from .api import Forecast, Forecaster
from .data import FAMILIES, TrajectoryDataset


def trajectory_figure(
    observed: np.ndarray,
    forecast: Forecast,
    ground_truth: np.ndarray | None = None,
):
    """Build an interactive Plotly figure for a 2D or 3D forecast."""
    import plotly.graph_objects as go

    dim = observed.shape[1]
    if dim not in (2, 3):
        raise ValueError("visualization supports 2D or 3D positions")
    figure = go.Figure()
    if dim == 2:
        figure.add_trace(
            go.Scatter(
                x=observed[:, 0],
                y=observed[:, 1],
                mode="lines+markers",
                name="observed",
                line=dict(width=4),
            )
        )
        if ground_truth is not None:
            figure.add_trace(
                go.Scatter(
                    x=ground_truth[:, 0],
                    y=ground_truth[:, 1],
                    mode="lines+markers",
                    name="ground truth",
                    line=dict(color="black"),
                )
            )
        for index, (component, weight) in enumerate(
            zip(forecast.component_means, forecast.component_weights)
        ):
            figure.add_trace(
                go.Scatter(
                    x=component[:, 0],
                    y=component[:, 1],
                    mode="lines",
                    name=f"mode {index + 1} ({weight:.2f})",
                    line=dict(dash="dot"),
                )
            )
        if forecast.samples is not None:
            for sample in forecast.samples:
                figure.add_trace(
                    go.Scatter(
                        x=sample[:, 0],
                        y=sample[:, 1],
                        mode="lines",
                        line=dict(color="rgba(120,80,220,0.09)"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        figure.add_trace(
            go.Scatter(
                x=forecast.mean[:, 0],
                y=forecast.mean[:, 1],
                mode="lines+markers",
                name="predictive mean",
                error_x=dict(
                    type="data",
                    array=(forecast.aleatoric_std * forecast.component_weights[:, None, None]).sum(
                        0
                    )[:, 0],
                    visible=True,
                ),
                error_y=dict(
                    type="data",
                    array=(forecast.aleatoric_std * forecast.component_weights[:, None, None]).sum(
                        0
                    )[:, 1],
                    visible=True,
                ),
                line=dict(width=4, color="#7b2cbf"),
            )
        )
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
    else:

        def trace(points, name, **kwargs):
            return go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines+markers",
                name=name,
                **kwargs,
            )

        figure.add_trace(trace(observed, "observed", line=dict(width=6)))
        if ground_truth is not None:
            figure.add_trace(trace(ground_truth, "ground truth", line=dict(color="black")))
        for index, (component, weight) in enumerate(
            zip(forecast.component_means, forecast.component_weights)
        ):
            figure.add_trace(
                trace(component, f"mode {index + 1} ({weight:.2f})", line=dict(dash="dot"))
            )
        if forecast.samples is not None:
            for sample in forecast.samples:
                figure.add_trace(
                    trace(sample, "uncertainty sample", showlegend=False, opacity=0.08)
                )
        figure.add_trace(
            trace(forecast.mean, "predictive mean", line=dict(width=7, color="#7b2cbf"))
        )
        figure.update_scenes(aspectmode="data")
    figure.update_layout(
        template="plotly_white", title="Motion-S probabilistic forecast", hovermode="closest"
    )
    return figure


def run_app(checkpoint: str) -> None:
    import streamlit as st

    st.set_page_config(page_title="Motion-S", layout="wide")
    st.title("Motion-S forecast inspector")
    forecaster = Forecaster(checkpoint)
    family = st.selectbox("Synthetic motion family", FAMILIES)
    seed = st.number_input("Trajectory seed", min_value=0, value=0, step=1)
    dataset = TrajectoryDataset(
        1,
        seed=int(seed),
        dim=forecaster.config.data.dimension,
        obs_length=(
            forecaster.config.data.min_observations,
            forecaster.config.data.max_observations,
        ),
        horizon=forecaster.config.data.horizon,
        families=(family,),
    )
    item = dataset[0]
    result = forecaster.forecast(
        item["observed"].numpy(), item["timestamps"].numpy(), num_samples=40
    )
    st.plotly_chart(
        trajectory_figure(item["observed"].numpy(), result, item["future"].numpy()),
        use_container_width=True,
    )
    st.caption(
        "Faint paths are full mixture samples; error bars show component-weighted "
        "aleatoric marginal standard deviations."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Motion-S interactive forecast viewer")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    app = Path(__file__).with_name("visualize_app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app), "--", "--checkpoint", args.checkpoint],
        check=True,
    )


if __name__ == "__main__":
    main()
