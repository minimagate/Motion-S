# Kine

Kine is a minimal research project for testing whether a tiny spatial dynamics model can learn
transferable motion representations from synthetic trajectories. Kine-S is a 1.18M-parameter
causal TCN that consumes variable-length 2D or 3D positions and timestamps, then predicts a
trajectory-level Gaussian mixture over a configurable number of future positions.

The current evidence is deliberately modest: synthetic pretraining gives a small full-data ADE
improvement on ETH pedestrians, but no statistically resolved low-data sample-efficiency gain, and
a tuned Kalman filter remains better. See the [transfer report](docs/transfer_report.md).

## Design

- Deterministic lazy generation of linear, accelerated, circular, oscillatory, ballistic,
  stochastic, goal-directed, piecewise, spiral, damped, and hybrid trajectories.
- Random direction, origin, scale, speed, acceleration, phase, cadence, and observation noise.
- Irregular sampling intervals with occasional gaps, supplied explicitly to every method.
- Per-history translation, rotation, spatial-scale, and time-span normalization.
- Right-padded batches with explicit masks for variable observation lengths.
- Modular `CausalTCNEncoder`; a Transformer can replace it without changing preprocessing or loss.
- A trajectory-level diagonal Gaussian mixture for ambiguous futures, plus a separately supervised
  synthetic measurement-noise head.
- Disjoint synthetic seed namespaces and real splits formed from complete actor tracks before
  windowing.
- Validation-only baseline tuning and checkpoint selection.
- Five-seed transfer summaries with Student-t intervals and actor-cluster bootstrap intervals.

See [docs/methodology.md](docs/methodology.md) for the protocol and audit findings.

## Install

Python 3.10–3.13 is supported. From the repository root:

```bash
uv venv --python 3.11 .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -e '.[dev,viz]'
```

PyTorch selects CUDA, then MPS, then CPU when `device = "auto"`.

## Train and evaluate

```bash
.venv/bin/kine-train --config configs/initial.toml
.venv/bin/kine-eval --checkpoint artifacts/initial/best.pt --samples 512
.venv/bin/python examples/forecast.py --checkpoint artifacts/initial/best.pt
.venv/bin/kine-viz --checkpoint artifacts/initial/best.pt
```

`configs/smoke.toml` is a two-epoch pipeline check. Checkpoints store the exact configuration,
model and optimizer states, epoch, validation metrics, and split protocol.

## Python API

```python
from motions.api import Forecaster

forecaster = Forecaster("artifacts/initial/best.pt")
forecast = forecaster.forecast(
    positions=[[0.0, 0.0], [0.1, 0.04], [0.22, 0.09]],
    timestamps=[0.0, 0.1, 0.2],
    num_samples=100,
)

print(forecast.mean)
print(forecast.component_means)
print(forecast.aleatoric_std)
print(forecast.measurement_noise_std)
```

The checkpoint fixes the number of output positions. Callers may supply explicit future timestamps;
otherwise the API extends the median observed cadence.

## Synthetic-to-real transfer study

Download the checksum-pinned official ETH annotations, then reproduce the experiment:

```bash
.venv/bin/kine-download-eth
.venv/bin/kine-transfer --config configs/transfer.toml \
  --seeds 7,17,27,37,47 --track-counts 16,64,0 \
  --fine-tune-steps 60 --scratch-steps 240
```

The run uses 228/82/83 eligible train/validation/test actor tracks from the ETH and Hotel scenes.
At all 228 training tracks, full fine-tuning reaches ADE 0.481 (95% CI 0.450–0.512), compared with
0.515 (0.464–0.566) from scratch. The paired gain is 0.034 (0.001–0.068). This does not establish
broad sample efficiency: the 16- and 64-track gain intervals include zero. The validation-tuned
Kalman baseline reaches 0.440 ADE (actor-bootstrap CI 0.352–0.537). Complete machine-readable
results are in [artifacts/transfer/transfer_report.json](artifacts/transfer/transfer_report.json).

The earlier single-seed synthetic experiment is retained as an engineering artifact, but its README
claims are superseded by the audited protocol because its baselines were not competitively tuned and
its confidence intervals were absent.

## Tests

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
```

The suite covers deterministic generation, actor-level split isolation, all motion families,
irregular intervals, normalization degeneracies, padding invariance, finite probabilistic gradients,
parameter budget, baselines, confidence intervals, checkpoint loading, and the public API.

## Project layout

```text
configs/                    reproducible TOML experiments
examples/forecast.py        API example
src/motions/data.py         synthetic generators, split profiles, collation
src/motions/real_data.py    leakage-safe real-data loading and windowing
src/motions/model.py        causal TCN and Gaussian-mixture output
src/motions/normalization.py
src/motions/baselines.py
src/motions/baseline_tuning.py
src/motions/train.py        supervised fitting and checkpoints
src/motions/evaluate.py     OOD suites, CIs, baselines, latency, calibration
src/motions/transfer.py     zero-shot, fine-tuned, scratch, sample-efficiency study
src/motions/api.py          inference API
src/motions/visualize.py    interactive Plotly/Streamlit viewer
tests/                      unit and integration tests
artifacts/initial/          original engineering checkpoint and metrics
artifacts/transfer/         per-seed and aggregate transfer metrics
```
