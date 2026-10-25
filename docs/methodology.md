# Methodology and limitations

## Architecture

The default encoder is a stack of dilated residual causal 1D convolutions with per-time-step layer
normalization. The initial experiment uses width 192, five residual blocks, kernel size 3, and
three mixture components. The head emits a joint future trajectory per component, not independent
mixtures at every time step, so sampled paths remain mode-consistent across the horizon.

The `CausalTCNEncoder` contract is deliberately small: `[batch, time, features]` plus a padding mask
maps to `[batch, time, hidden]`. A causal Transformer encoder can implement that contract while
reusing all preprocessing, likelihood, checkpoint, training, and evaluation code.

## Normalization

The final valid observation is translated to the origin. The most recent displacement defines the
local forward axis; degenerate histories fall back to the identity frame. Mean history distance
from the endpoint supplies an isotropic spatial scale, and observed time span supplies time scale.
The likelihood includes the spatial change-of-variables Jacobian when evaluated in world units.

This provides exact translation equivariance and analytic rotation/scale canonicalization. It does
not guarantee learned invariance: noise, short histories, and nearly stationary endpoints can make
the estimated frame unstable.

## Data and leakage controls

Every dataset item owns a stable seed derived from `(base seed, split, index)`. The item generates a
single complete history and future, so windows from one latent trajectory cannot cross splits.
Validation and test use different seed namespaces; the final evaluation additionally offsets the
base seed from training.

Fitting excludes the hybrid family. It also includes high-speed/low-noise and
low-speed/high-noise regimes but withholds their high/high and low/low pairings. The suites are:

- **In distribution:** new seeds, known seven families, fitted joint parameter regimes.
- **Unseen parameters:** disjoint high speed and acceleration ranges.
- **Unseen combinations:** known marginal speed/noise regimes in withheld pairings.
- **Held-out family:** hybrid piecewise accelerated/oscillatory motion only.

The generator randomizes origin, direction, spatial scale, cadence, velocity, acceleration, phase,
and measurement noise. Cadence is fixed within a trajectory; truly irregular sampling is accepted
by the model and baselines but is not yet represented during synthetic training.

## Uncertainty

Three outputs should not be conflated:

1. Separation among component means and their weights represents multimodal future ambiguity.
2. Per-component diagonal scales represent residual within-mode future uncertainty.
3. A scalar head estimates the observation measurement-noise standard deviation, supervised from
   the generator. Its validation MAE was 0.0264 for noise sampled between 0 and 0.08 world units.

These quantities are identifiable only because synthetic training supplies a measurement-noise
label. Within-mode variance still combines unmodeled dynamics and conditional process noise. A
single realized future per history provides weak supervision for genuinely multimodal branching;
explicit shared-prefix/multiple-future generation is the most important data improvement.

Calibration moment-matches the mixture at each future step and measures 50% and 90% ellipsoidal
coverage. Moment matching can hide mode-specific miscalibration, and the reported scalar error uses
only two coverage levels.

## Baselines

Constant velocity and acceleration use the latest irregular-time finite differences. The linear
Kalman filter has position/velocity state, per-example measurement variance from the known
synthetic noise level, and fixed white-acceleration process variance 0.1. No baseline hyperparameter
was tuned on test sets. That favors methodological cleanliness but does not constitute a maximally
competitive baseline study.

## Largest remaining limitations

- No real trajectory corpus is used, so simulator bias is unmeasured.
- The eight named families are simple equations and one lightweight hybrid, not a compositional
  physics engine with contacts, constraints, agents, or scene context.
- Training uses fixed cadence per trajectory and a fixed number of future steps per checkpoint.
- The 3D path is implemented and tested but the included trained checkpoint is 2D only.
- A single training seed and architecture configuration provides no confidence intervals or ablation
  evidence.
- The TCN is evaluated against untuned baselines; stronger smoothing, robust polynomial fits,
  interacting multiple models, and tuned Kalman variants may close or reverse gaps.
- The observation-noise estimator assumes the synthetic noise model is isotropic Gaussian.
- Batch-1 latency is a local CPU microbenchmark without thread-count or power controls.
- The initial run is deliberately small and had not clearly plateaued at epoch 8, so it is an
  engineering baseline rather than a converged scaling result.

## Recommended next experiments

1. Generate multiple futures from identical observed prefixes and score mode coverage explicitly.
2. Add irregular sampling, missing observations, outliers, stops, collisions, and constrained motion.
3. Run at least five seeds and report confidence intervals by family and forecast horizon.
4. Tune classical baselines only on validation data and add robust polynomial and switching-Kalman
   baselines.
5. Train the 3D configuration and evaluate rotation/scale stress tests directly.
6. Replace only `CausalTCNEncoder` with a parameter-matched Transformer for a controlled ablation.
7. Fine-tune on a real, domain-specific corpus while retaining synthetic OOD suites as regression
   tests.

