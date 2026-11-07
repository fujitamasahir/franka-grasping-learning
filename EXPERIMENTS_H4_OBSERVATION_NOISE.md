# Observation Noise Robustness of Frozen H4 Replanning Policies

## 1. Research Question and Hypotheses

This focused experiment asks whether four-step action chunking (R4) preserves the success advantage previously observed over one-step replanning (R1) when object-position observations contain synthetic XY noise. It also asks whether the R4–R1 difference changes with noise and whether the pattern is directionally consistent across three frozen behavior-cloning (BC) training seeds. No directional effect under noise was assumed.

The primary estimands are the paired success-rate gap `R4 − R1` at each noise level and its interaction with noise, `Gap(σ) − Gap(0)`. Rescue and harmful transitions are reported alongside rates because the same aggregate rate can conceal different position-level changes.

## 2. Experimental Setup

Three frozen H4 BC checkpoints (training seeds 42, 123, and 456) were evaluated without updates. Their paths, SHA-256 digests, architecture, normalization, and provenance are recorded in the preregistered protocol and final audit. R1 replans each control step; R4 executes the existing four-action chunk. All other policy and environment settings were held fixed.

For each model, both intervals were evaluated at σ = 0, 0.005, and 0.010 m. At each control step, independent zero-mean Gaussian XY errors were added only to observed cube X/Y and the corresponding relative X/Y entries (observation indices 3, 4, 6, 7). The true simulator state, physics, reward, success rule, and other observation entries were unchanged; observations were not clipped. Noise arrays were precomputed by evaluation seed and absolute environment control step, making them independent of inference-call timing and paired across intervals and σ levels.

Each of the 18 cells contains 1,000 episodes on evaluation seeds 321000–321999, for 18,000 Final episodes. All cells shared identical initial positions per evaluation seed. A protocol SHA was committed before smoke and Final outcomes. The Final integrity audit verified exact seed sets, row counts, schemas, metadata, checkpoint hashes, shared positions, and base/applied noise schedule hashes. Existing outputs were not overwritten.

The zero-noise software regression check passed on the prior evaluation seeds, including outcome/control counters and the saved legacy action trace. The pre-Final smoke check comprised 360 episodes across 18 cells. A start-command schema-key mismatch was corrected before any smoke rollout; the correction is logged. No Final outcome was inspected to alter conditions.

## 3. Results

### Success and inference load

Each entry is successes / 1,000 (success rate); mean BC forward-pass calls per episode are shown in parentheses.

| Frozen BC seed | Interval | σ=0 mm | σ=5 mm | σ=10 mm |
|---:|:---:|---:|---:|---:|
| 42 | R1 | 824 (126.7) | 870 (122.7) | 890 (115.0) |
| 42 | R4 | 946 (30.4) | 941 (28.6) | 870 (28.2) |
| 123 | R1 | 940 (122.4) | 932 (120.4) | 925 (115.9) |
| 123 | R4 | 934 (30.9) | 905 (29.7) | 880 (28.6) |
| 456 | R1 | 929 (122.9) | 927 (121.0) | 890 (118.7) |
| 456 | R4 | 956 (30.8) | 931 (30.1) | 848 (29.4) |

### Paired R4 versus R1 outcomes

Beneficial means R1 failed and R4 succeeded; harmful means R1 succeeded and R4 failed. The gap equals (beneficial − harmful) / 1,000. CIs are paired position bootstrap percentile 95% intervals. P-values are exact two-sided McNemar tests; Holm correction is across the three BC seeds separately at each σ.

| BC seed | σ (mm) | R4−R1 (pp), 95% bootstrap CI | Beneficial | Harmful | Net | Exact p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0 | +12.2 [9.9, 14.5] | 137 | 15 | +122 | 7.85e−26 | 2.36e−25 |
| 42 | 5 | +7.1 [5.0, 9.3] | 101 | 30 | +71 | 3.50e−10 | 1.05e−9 |
| 42 | 10 | −2.0 [−4.3, 0.2] | 55 | 75 | −20 | 0.0953 | 0.0953 |
| 123 | 0 | −0.6 [−1.8, 0.6] | 17 | 23 | −6 | 0.4296 | 0.4296 |
| 123 | 5 | −2.7 [−4.4, −1.1] | 21 | 48 | −27 | 0.00155 | 0.00310 |
| 123 | 10 | −4.5 [−6.4, −2.6] | 25 | 70 | −45 | 4.27e−6 | 1.28e−5 |
| 456 | 0 | +2.7 [1.7, 3.8] | 29 | 2 | +27 | 4.63e−7 | 9.26e−7 |
| 456 | 5 | +0.4 [−1.4, 2.2] | 45 | 41 | +4 | 0.7465 | 0.7465 |
| 456 | 10 | −4.2 [−6.4, −2.0] | 41 | 83 | −42 | 0.000204 | 0.000408 |

The R4−R1 gap changed at 5 mm by −5.1, −2.1, and −2.3 pp for BC seeds 42, 123, and 456, respectively. At 10 mm, changes were −14.2, −3.9, and −6.9 pp. Paired bootstrap 95% intervals for those interactions were: seed 42, [−7.2, −3.1] and [−17.1, −11.4]; seed 123, [−4.0, −0.2] and [−6.0, −1.8]; seed 456, [−4.0, −0.6] and [−9.1, −4.7]. Each resample sampled evaluation-position indices jointly across all six interval-by-noise outcomes within the frozen model.

### Change from each interval's clean-observation result

| BC seed | Interval | 5 mm rate change (pp) | Paired transitions beneficial / harmful | Holm p | 10 mm rate change (pp) | Paired transitions beneficial / harmful | Holm p |
|---:|:---:|---:|:---:|---:|---:|---:|---:|
| 42 | R1 | +4.6 | 50 / 4 | 3.42e−10 | +6.6 | 77 / 11 | 2.64e−12 |
| 42 | R4 | −0.5 | 24 / 29 | 1.000 | −7.6 | 23 / 99 | 1.99e−11 |
| 123 | R1 | −0.8 | 10 / 18 | 0.555 | −1.5 | 15 / 30 | 0.143 |
| 123 | R4 | −2.9 | 24 / 53 | 0.00758 | −5.4 | 27 / 81 | 1.53e−6 |
| 456 | R1 | −0.2 | 15 / 17 | 1.000 | −3.9 | 18 / 57 | 5.07e−5 |
| 456 | R4 | −2.5 | 23 / 48 | 0.0203 | −10.8 | 19 / 127 | 1.01e−19 |

Holm correction covers the 12 noisy-versus-clean tests jointly. These mixed direction results show that small observation noise does not uniformly reduce R1 success, while 10 mm noise reduced success in all R4 models.

### Region and execution metrics

The machine-readable summary reports EN, ES, WN, and WS success and paired transition counts for each model, interval, and noise level. Regional R4−R1 patterns are heterogeneous. For example, at zero noise, R4 improved WN and WS success strongly for seed 42 (paired transitions 47/0 and 86/0), while seed 123 had mixed regional changes. At 10 mm, R4 regressions increased in several regions; seed 456 had 83 harmful versus 41 beneficial positions overall, with particularly many harmful transitions in ES (20) and WN (29). Region analyses are exploratory.

At σ=0, R4 used 28–31 mean inference calls versus 122–127 for R1 (about 75% fewer calls). At 10 mm, it used 28–29 versus 115–119 calls (about 75% fewer). Mean call latency was approximately 0.035 ms for R1 and 0.043–0.045 ms for R4 in this run. Episode wall time was dominated by simulation and is descriptive; it did not show a comparable 75% reduction. Per-cell mean control steps, successful-episode steps, close/lift steps, total inference time, and wall time are included in `bc_results/h4_observation_noise/summary.json` and `cell_summary.csv`.

Five paired figures are provided as SVG and PNG: success versus noise by model; R4−R1 gap with intervals; gap-noise interactions; beneficial/harmful transitions; and regional R4−R1 gaps.

## 4. Statistical Analysis

The evaluation seed/initial position is the paired unit within each fixed BC model. Exact two-sided McNemar tests compare discordant paired outcomes. Holm adjustment was applied as preregistered: across the three model-specific R4-vs-R1 tests within each noise level, and across all 12 noisy-vs-clean tests. Bootstrap confidence intervals use 10,000 paired resamples and a fixed RNG seed. They quantify position-sampling uncertainty conditional on these frozen models and this environment; they do not turn the three trained policies into a large policy sample.

At clean observation, R4 improved over R1 for seeds 42 and 456, but not seed 123. At 5 mm, R4 remained significantly better only for seed 42, was significantly worse for seed 123, and was statistically unresolved for seed 456 after the specified correction. At 10 mm, R4 was significantly worse for seeds 123 and 456; seed 42 had a negative but not corrected-significant gap. The interaction was negative in all three models at both nonzero noise levels, with each conditional position-bootstrap interval below zero. This indicates a consistent loss of the R4 relative advantage in this test set, not universal proof about training-seed populations.

## 5. Discussion

The results do not support a general claim that frozen R4 replanning robustly retains its clean-observation advantage under XY observation noise. Noise interacted negatively with the R4−R1 gap for each of the three models, and 10 mm noise made R4 worse than R1 in every model. R4 nevertheless retained materially lower inference-call counts. This is a compute-versus-robustness tradeoff: executing an action chunk without new inference is efficient, but it delays correction when the object-position observation is perturbed.

The experiment also illustrates why rate-only summaries are insufficient. Seed 42 R1 gained successes at moderate noise despite a different set of positions changing outcome, whereas seed 42 R4 lost little at 5 mm and substantially more at 10 mm. Rescue/harmful counts make these paired changes explicit. No conclusion should be drawn that noise is beneficial: the Gaussian perturbation can accidentally move a biased observation toward a useful estimate, and this effect was not consistent across models or intervals.

## 6. Limitations

- Only three frozen BC training seeds were tested; trained-policy variation is too sparsely sampled for broad generalization claims.
- The 1,000 evaluation positions are shared and paired across conditions. They are not independent policy replications, and position-level inference is conditional on these policies and sampled positions.
- The perturbation is synthetic, independent Gaussian noise on object XY observations, not camera rendering, occlusion, correlated estimator drift, or full perception failure.
- Only R1 and R4 were tested, with the specified environment and noise levels.
- Timing is sensitive to sequential simulator and hardware load and is descriptive.
- Regional analyses are exploratory, with variable regional sample sizes.

## 7. Next Research Direction

Measure and fit realistic perception-error processes from a camera or state estimator, including temporal correlation, bias, and occasional outliers. Then preregister a larger set of independent BC training seeds and compare R1/R4 plus a noise-aware adaptive replanning controller on paired positions. Such a controller should be evaluated for both recovery from observation errors and harmful corrections, with inference cost reported alongside success. The current results motivate adaptive feedback under uncertainty but do not establish that a particular gate or training modification will help.

## Reproduction and Artifacts

From the repository root, run `.venv/bin/python audit_h4_observation_noise.py`, then `MPLCONFIGDIR=/private/tmp/matplotlib-h4 .venv/bin/python analyze_h4_observation_noise.py`. The analysis refuses to run without the passed integrity audit. The protocol and its pre-outcome SHA are `bc_results/h4_observation_noise_protocol.json` and `.sha256`; all cell CSV/metadata files, summary CSV/JSON, audit, and charts are in `bc_results/h4_observation_noise/`. Evaluator: `evaluate_h4_observation_noise.py`.
