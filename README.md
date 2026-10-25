# Motion-S

Motion-S is a minimal research project for testing whether a tiny temporal network can learn
useful, general spatial motion priors from synthetic data. It consumes variable-length 2D or 3D
positions with timestamps and predicts a configurable fixed-step future distribution.

The default model is a 2.54M-parameter causal TCN. The included initial checkpoint uses a smaller
1.18M-parameter configuration and was trained entirely on CPU. The project does **not** assume the
neural model beats classical extrapolation; every evaluation runs constant-velocity,
constant-acceleration, and Kalman-filter baselines on the same trajectories.

## Design

- Deterministic lazy generation of linear, accelerated, circular, oscillatory, ballistic,
  stochastic, goal-directed, and hybrid trajectories.
- Random direction, origin, scale, speed, acceleration, phase, cadence, and observation noise.
- Per-history translation, rotation, spatial-scale, and time-span normalization.
- Right-padded batches with explicit masks for variable observation lengths.
- Modular `CausalTCNEncoder` interface; a Transformer can replace it without changing
  normalization or the output head.
- A trajectory-level diagonal Gaussian mixture: mode separation represents ambiguous futures,
  component scales represent within-mode forecast variance, and a separately supervised head
  estimates ordinary observation noise.
- Split assignment happens by deterministic trajectory seed before generation; no trajectory or
  sampled window is reused across fitting and evaluation.

See [docs/methodology.md](docs/methodology.md) for the protocol and caveats.

## Install

Python 3.10–3.13 is supported. From the repository root:

```bash
uv venv --python 3.11 .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -e '.[dev,viz]'
```

PyTorch selects CUDA, then MPS, then CPU when `device = "auto"`. MPS was built into the PyTorch
used for the included experiment but unavailable to that process, so the recorded results are CPU
results.

## Train and evaluate

```bash
.venv/bin/motion-train --config configs/initial.toml
.venv/bin/motion-eval --checkpoint artifacts/initial/best.pt --samples 512
.venv/bin/python examples/forecast.py --checkpoint artifacts/initial/best.pt
.venv/bin/motion-viz --checkpoint artifacts/initial/best.pt
```

`configs/smoke.toml` is a two-epoch pipeline check. `configs/base.toml` is the larger 2.54M-parameter
experiment definition and is intentionally not presented as having been trained here. Checkpoints
store the exact configuration, model and optimizer states, epoch, validation metrics, and split
protocol.

## Python API

```python
from motions.api import Forecaster

forecaster = Forecaster("artifacts/initial/best.pt")
forecast = forecaster.forecast(
    positions=[[0.0, 0.0], [0.1, 0.04], [0.22, 0.09]],
    timestamps=[0.0, 0.1, 0.2],
    num_samples=100,
)

print(forecast.mean)                   # [horizon, dimension]
print(forecast.component_means)        # ambiguous modes
print(forecast.aleatoric_std)          # within-mode marginal std
print(forecast.measurement_noise_std)  # estimated input measurement noise
```

The checkpoint fixes the output horizon. Future timestamps are produced by extending the median
observed cadence; train a checkpoint with another horizon rather than silently truncating or
extrapolating network outputs.

## Actual initial experiment

The checked-in metrics were produced on an Apple M4 Pro (12 cores), PyTorch 2.14.1, CPU, seed 7.
The 1,178,068-parameter model trained for 8 epochs on 8,192 procedural trajectories, with 1,024
validation trajectories, in 136.5 seconds. The best checkpoint was selected only by validation
NLL. Each evaluation suite below contains 512 unseen trajectory seeds. Lower is better.

| Suite | Model | ADE | FDE | Calibration error |
|---|---|---:|---:|---:|
| In distribution | TCN mixture | **0.619** | **1.156** | **0.109** |
|  | Constant velocity | 0.954 | 1.863 | — |
|  | Constant acceleration | 5.261 | 13.510 | — |
|  | Kalman | 0.740 | 1.377 | 0.418 |
| Unseen parameters | TCN mixture | **1.090** | **1.946** | **0.142** |
|  | Constant velocity | 2.112 | 3.964 | — |
|  | Constant acceleration | 12.671 | 32.410 | — |
|  | Kalman | 1.451 | 2.496 | 0.509 |
| Unseen speed/noise combinations | TCN mixture | **0.689** | **1.298** | **0.108** |
|  | Constant velocity | 0.979 | 1.931 | — |
|  | Constant acceleration | 5.004 | 12.849 | — |
|  | Kalman | 0.834 | 1.540 | 0.438 |
| Held-out hybrid family | TCN mixture | **0.724** | 1.530 | **0.040** |
|  | Constant velocity | 0.740 | **1.519** | — |
|  | Constant acceleration | 4.022 | 10.303 | — |
|  | Kalman | 0.834 | 1.704 | 0.647 |

Calibration error is the mean absolute deviation of empirical 50% and 90% moment-matched
ellipsoidal coverage from its nominal level. Only probabilistic methods receive this metric.
The Kalman process variance was fixed at 0.1 and was not tuned on test data.

Batch-1 CPU latency, averaged over 30 warmed calls on one example, was 1.76 ms for the TCN,
0.46 ms for Kalman, 0.006 ms for constant velocity, and 0.009 ms for constant acceleration. These
are local microbenchmarks, not portable hardware claims. Complete machine-readable results are in
[`artifacts/initial/evaluation.json`](artifacts/initial/evaluation.json).

## What the experiment establishes

On this synthetic protocol, a 1.18M-parameter TCN learns a useful motion prior and transfers to
withheld parameter ranges and combinations. It is slower than every baseline. Constant velocity is
still marginally better on held-out-family FDE, and the large constant-acceleration errors show its
sensitivity to noisy finite differences rather than proving that acceleration models are generally
poor. The experiment is evidence about this generator and split design, not about real-world
universal motion.

## Tests

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
```

The suite covers deterministic data generation, split isolation, every motion family, variable
lengths, normalization round trips, padding invariance, finite probabilistic gradients, parameter
budget, irregular-time baselines, calibration metrics, checkpoint loading, and public API output.

## Project layout

```text
configs/                 reproducible TOML experiments
examples/forecast.py     API example
src/motions/data.py      synthetic generators, split profiles, collation
src/motions/model.py     causal TCN and Gaussian-mixture output
src/motions/normalization.py
src/motions/baselines.py
src/motions/train.py     supervised fitting and checkpoints
src/motions/evaluate.py  OOD suites, baselines, latency, calibration
src/motions/api.py       inference API
src/motions/visualize.py interactive Plotly/Streamlit viewer
tests/                   unit and integration tests
artifacts/initial/       actual checkpoint and metrics from the reported run
```


