# Audited methodology

## Model and data

Kine-S is unchanged in scale: a 1,180,372-parameter causal TCN with width 192, five residual
blocks, and three trajectory-level diagonal Gaussian components. A zero-initialized projection of
the requested future sampling schedule was added so irregular future timestamps are explicit rather
than silently treated as uniform. The encoder interface remains replaceable by a Transformer.

Synthetic examples are generated lazily from stable per-item seeds. Eleven families cover linear,
accelerated, circular, oscillatory, ballistic, stochastic, goal-directed, piecewise, spiral,
damped, and hybrid motion. Each example randomizes physical parameters, orientation, origin, scale,
sampling cadence, interval jitter, occasional gaps, and measurement noise. The hybrid family and
joint parameter regimes remain available as synthetic out-of-distribution suites.

The real-data loader supports the official ETH Walking Pedestrians ETH and Hotel annotations. It
converts ground-plane coordinates to metres and frame numbers to seconds, splits complete actor
tracks 60/20/20 within each scene, and only then creates 8-observation/12-future windows. Gaps over
0.6 seconds break a track segment. The split seed is fixed at 2026.

## Leakage and fairness audit

The audit found and repaired these material issues:

1. **Degenerate normalization:** an exactly stationary observed history used the numerical epsilon
   as spatial scale, turning ordinary future motion into normalized coordinates above 200,000.
   Histories with unidentifiable scale now use a neutral unit scale; nondegenerate rotation and scale
   equivariance are unchanged.
2. **Oracle Kalman noise:** the earlier synthetic evaluator supplied each test example's generator
   noise to the Kalman filter. Evaluation now uses one globally validation-tuned measurement
   variance and process variance.
3. **Weak baselines:** velocity and acceleration previously used only the latest finite differences.
   They now fit validation-tuned local polynomial regressions. Window, ridge, Kalman measurement
   variance, and Kalman acceleration variance are selected solely by validation ADE.
4. **Window leakage risk:** real windows are never split independently. Actor identities are
   disjoint across train, validation, and test, and overlapping windows stay within one split.
5. **Unfair update counts:** epoch-based low-data comparisons gave smaller subsets fewer optimizer
   updates. Pretrained models now receive 60 updates at every data budget. Scratch controls receive
   240 updates, making the compute allocation conservative against pretraining. All checkpoints are
   considered every 10 steps and selected by validation NLL only.

No preprocessing statistics are fitted globally: translation, rotation, spatial scale, and time
scale are derived only from each observed history. Targets and test data never affect the transform.
The fixed actor split includes both scenes in every partition, so it measures transfer to unseen
actors, not leave-scene-out or cross-dataset generalization.

## Statistical protocol

Neural summaries use five complete synthetic-pretraining and downstream-training seeds
`7, 17, 27, 37, 47`. Reported 95% intervals are Student-t intervals across those five runs. The
pretraining gain is paired by seed before computing its interval. Within any one real test run,
ADE/FDE intervals resample complete actors (2,000 deterministic cluster-bootstrap replicates), not
overlapping windows. Deterministic classical baselines therefore report actor-bootstrap intervals.

The data budgets are nested deterministic subsets of 16, 64, and all 228 eligible training actors.
Validation contains 82 eligible actors/260 windows and test contains 83 actors/204 windows. Baseline
tuning examines at most 256 validation windows. The test split is evaluated only after selection.

## Uncertainty semantics

Mixture-component separation represents alternative future paths; component scales represent
within-mode residual uncertainty. A separate supervised scalar estimates synthetic measurement
noise. These are conceptually distinct, but a single realized future per synthetic prefix remains
weak supervision for genuine branching ambiguity.

Calibration is the average absolute error of empirical 50% and 90% moment-matched ellipsoidal
coverage. It is useful as a coarse diagnostic, not a complete mixture-calibration test. On ETH the
models are substantially over-dispersed: the full-data fine-tuned model averages 84.3% coverage for
the nominal 50% region and 96.9% for the nominal 90% region.

## Remaining validity limits

- ETH and Hotel are small pedestrian scenes from one corpus; scene context and interactions are
  ignored, and both scenes appear in every split.
- Five seeds leave wide intervals, especially at 16 actors.
- Some full-data scratch runs select the last tested step, so 240 updates may not fully converge.
- The synthetic generator has no contacts, obstacles, multi-agent coupling, missing detections, or
  explicit shared-prefix/multiple-future supervision.
- Fine-tuning selects NLL while the primary comparison is ADE; this is intentional for probabilistic
  training but can rank checkpoints differently.
- Moment-matched calibration can hide component-specific failures.
- 3D support is implemented and tested, but this transfer study is 2D only.
