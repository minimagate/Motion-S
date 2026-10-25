# Kine-S synthetic-to-real transfer report

## Result

**Assessment: share with caveats.** Synthetic pretraining transfers, but the evidence is narrow.
With all 228 real training actors it improves mean ADE over a longer-trained scratch control by
0.034 m, with a paired five-seed 95% CI of 0.001–0.068 m. The low-data intervals include zero, so
this experiment does **not** establish improved real-world sample efficiency. A validation-tuned
Kalman filter remains the strongest tested method.

All values below are from the completed CPU run in
[`artifacts/transfer/transfer_report.json`](../artifacts/transfer/transfer_report.json). Neural values
are means with Student-t 95% confidence intervals across seeds 7, 17, 27, 37, and 47. Classical
intervals are actor-cluster bootstraps on the fixed test set. Lower is better.

| Training condition | Real actors | ADE, m | FDE, m | NLL / coordinate |
|---|---:|---:|---:|---:|
| Kine-S zero-shot | 0 | 0.776 [0.548, 1.003] | 1.786 [1.252, 2.320] | 0.819 [0.663, 0.975] |
| Partial fine-tune | 16 | 0.578 [0.451, 0.704] | 1.159 [0.891, 1.427] | 0.310 [0.239, 0.380] |
| Full fine-tune | 16 | 0.584 [0.432, 0.736] | 1.198 [0.850, 1.546] | 0.276 [0.205, 0.347] |
| From scratch | 16 | 1.076 [0.541, 1.610] | 2.260 [0.844, 3.676] | 0.693 [0.169, 1.216] |
| Partial fine-tune | 64 | 0.544 [0.457, 0.631] | 1.154 [0.952, 1.357] | 0.198 [0.133, 0.262] |
| Full fine-tune | 64 | 0.493 [0.474, 0.512] | 1.027 [0.946, 1.108] | 0.109 [0.054, 0.163] |
| From scratch | 64 | 0.540 [0.397, 0.684] | 1.108 [0.866, 1.350] | 0.120 [0.050, 0.190] |
| Partial fine-tune | 228 | 0.481 [0.451, 0.511] | 0.983 [0.906, 1.059] | 0.216 [0.120, 0.312] |
| Full fine-tune | 228 | **0.481 [0.450, 0.512]** | 0.989 [0.940, 1.039] | 0.117 [0.061, 0.173] |
| From scratch | 228 | 0.515 [0.464, 0.566] | 1.033 [0.950, 1.117] | **0.101 [0.039, 0.163]** |

The paired full-fine-tune ADE gain over scratch is 0.492 m [-0.063, 1.046] at 16 actors,
0.047 m [-0.107, 0.201] at 64, and 0.034 m [0.001, 0.068] at 228. Only the full-data interval
excludes zero, and only barely. A 64-actor pretrained model is not significantly better than a
228-actor scratch model, so there is no supported sample-count reduction.

## Classical controls

Hyperparameters were selected by ADE on validation only. The selected velocity fit uses four
observations and ridge 0.01; acceleration uses eight and ridge 0.01; Kalman uses measurement
variance 0.001 and acceleration variance 0.01.

| Method | ADE, m | FDE, m |
|---|---:|---:|
| Constant velocity | 0.454 [0.366, 0.548] | 0.926 [0.745, 1.147] |
| Constant acceleration | 0.828 [0.686, 0.972] | 1.900 [1.564, 2.254] |
| Kalman filter | **0.440 [0.352, 0.537]** | **0.907 [0.700, 1.129]** |

Zero-shot Kine-S is worse than constant velocity and Kalman, though slightly better than constant
acceleration. Even after full fine-tuning, its mean ADE remains 9% above Kalman. This is the
decisive result for the present model, not a side note.

## Protocol and compute

The official ETH Walking Pedestrians light annotations contain 749 valid actor tracks across ETH and
Hotel after discarding one one-point track. Complete actor tracks are split within scene before
windowing; train/validation/test actor overlap is zero. The eligible partitions contain 228/82/83
actors and the fixed test set contains 204 windows. Synthetic pretraining uses 4,096 examples and
512 validation examples per seed for six epochs, with irregular sampling and eleven dynamics.

Pretrained models receive 60 real-data optimizer updates. Scratch controls receive 240 updates;
checkpoint selection occurs every ten updates using validation NLL. This intentionally favors the
scratch control on compute, though several full-data scratch runs still selected step 220 or 240.
Kine-S has 1,180,372 parameters. A warmed batch-1 forward pass measured 1.89 ms on the local CPU
(PyTorch 2.14.1, eight threads, 100 repeats); this microbenchmark is not portable.

Calibration is poor. Full-data fine-tuning has mean calibration error 0.206 and covers 84.3%/96.9%
of targets with nominal 50%/90% ellipsoids. From-scratch NLL is also slightly better than pretrained
NLL at full data, even though its ADE is worse. Pretraining therefore improves the mean path in this
condition, not every aspect of the forecast distribution.

## Limitations and recommendation

This is an unseen-actor test inside two overlapping scenes, not unseen-scene or cross-dataset
transfer. ETH pedestrians are one object class and one collection pipeline. Overlapping windows are
handled statistically by actor clustering, but only five model seeds remain. The generator lacks
interactions, obstacles, stop/start intent, missing detections, and multiple futures sharing an
identical observed prefix.

**Recommendation:** do not scale the network. The next experiment should add ETH/UCY
leave-one-scene-out evaluation and matched synthetic stop/start, piecewise-intent, and shared-prefix branching
dynamics. Keep the same 1.18M model, run at least ten seeds at the smallest data budgets, and train
scratch controls to a demonstrated validation plateau. Continue only if pretraining yields a stable
paired gain against tuned classical baselines on held-out scenes.
