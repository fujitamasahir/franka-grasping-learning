# H4 Replanning-Frequency Ablation of Frozen Action-Chunked Behavior Cloning

**Study status:** Final evaluation completed with a transparent recovery run for one corrupted output. **Protocol SHA-256:** `682633025a7e80ec4f50ea03a905b28b2ff1d6d4cf9b2c312004ec32cfe25a8b`.

## 1. Abstract

We compared runtime replanning intervals of 1, 2, and 4 control steps using the same frozen H4 behavior-cloning (BC) policy for each of three BC training seeds. Across the three fixed checkpoints, mean success was 88.58% (R1), 93.43% (R2), and 94.32% (R4). R4 exceeded R1 by 5.73 percentage points for all three checkpoints, with seed-specific differences of 12.95, 0.20, and 4.05 pp. The effect was heterogeneous: the position-level Holm-adjusted exact McNemar test was significant for BC seeds 42 and 456, but not seed 123. R4 used about 75% fewer BC inference calls than R1, while mean episode wall time changed by only about 1.9%. These are conditional results for three fixed models and 2,000 shared positions; they do not establish population-level superiority.

## 2. Motivation

The earlier H1-versus-H4 comparison changed the prediction horizon, number of supervised future actions, output-head size, and runtime replanning rate together. It therefore could not isolate which factor explained the success-rate difference. This study keeps each H4 checkpoint and its normalization fixed and changes only how many predicted actions are executed before replanning.

## 3. Research Questions

- **RQ:** Does executing cached action chunks open-loop improve grasping success compared with replanning more frequently using the same trained action-chunked policy?
- **Primary contrast:** paired R4 − R1 success for each frozen BC model.
- **Secondary contrast:** paired R2 − R1; R4 − R2 is descriptive.

## 4. Preregistered Hypotheses

- **H1:** R4 may outperform R1 through temporally consistent action execution.
- **H2:** R1 may outperform R4 through more frequent feedback and state correction.
- **H3:** R2 may trade off temporal consistency and feedback frequency.
- **H4:** R4 should reduce BC inference calls; this need not proportionally reduce total episode time.

These hypotheses were treated as competing possibilities, not assumed conclusions.

## 5. Frozen Model and Dataset Provenance

The study used three existing, frozen, state-conditioned H4 MLP BC checkpoints. No new BC or SAC training, parameter updates, reward changes, or residual controller were used. Every condition retained prediction horizon 4 and output shape `(4, 4)`. The three checkpoints have identical observation-normalization arrays. Their SHA-256 hashes matched the existing training provenance summaries before and after evaluation.

| BC training seed | Checkpoint | SHA-256 | Parameters |
|---:|---|---|---:|
| 42 | `bc_models/action_chunking/H4_seed42.pt` | `41cda0af609e3064ba8e485a4ef81289c319472b08d1f710d96b30f6cb470557` | 20,240 |
| 123 | `bc_models/action_chunking/H4_seed123.pt` | `5dc02a9b9cce85b93beb18288c0681c90e5174181210cd4c0c23d518d653495c` | 20,240 |
| 456 | `bc_models/action_chunking/H4_seed456.pt` | `649e42de42f0a04a2dc180dfa513afbbc52aba2e78f4f6c28a3deb41eb3cd706` | 20,240 |

Observation-mean SHA-256: `d833cc7443412972194a8dedfee2bc5b29ad3ce854e8acd51735941a68bec62b`; observation-standard-deviation SHA-256: `8902f6e7f5947d2e4f8fc2860a7799aa600dfe9bc8476c7be58b5b32a127a3c3`. The model training position range in checkpoint metadata is ±0.12 m; evaluation used the preregistered ±0.20 m position range.

## 6. Experimental Conditions

| Condition | Actions executed from each H4 prediction | Replan interval |
|---|---|---:|
| R1 | Index 0 only; remaining predictions discarded | 1 step |
| R2 | Indices 0 and 1; indices 2–3 discarded | 2 steps |
| R4 | Indices 0–3 | 4 steps |

All conditions used the same checkpoint within each BC seed, checkpoint normalization, gripper convention (`>=0` open; `<0` close), FrankaAutonomousEnv dynamics, `max_move=0.15 m`, `max_time=16 s`, `control_dt=0.1 s`, and existing success rule (lift ≥0.10 m for at least five control steps). No SAC, learned gate, or reward modification was used.

## 7. Software Equivalence and R4 Regression Test

R4 was compared against existing H4 final rollout records on the predetermined software-test positions 290000–290019 for all three BC seeds (60 episodes total). Success/failure, episode steps, deterministic initial cube coordinates, first close/lift step, inference-call count, terminal chunk index, closed fraction, and return matched the saved records. The recorded seed-290042 trajectory also matched all 121 stepwise XYZ commands, gripper commands, and chunk indices within 1e-7. These old positions were used only for software verification and are excluded from the new hypothesis analysis.

## 8. Evaluation Protocol

A 20-position-per-cell smoke test used seeds 312000–312019 (180 episodes), followed by the fixed Final range 310000–311999. Each of the nine model-by-interval cells contains exactly 2,000 unique seeds. All nine cells share identical initial positions. Runs were sequential; per-episode CSV rows were flushed and resume-validated. The recovery output for H4 seed 456 / R1 is versioned as `final_H4_seed456_R1_recovery1.csv`; it follows the same preregistered conditions and seeds. The corrupt partial is excluded from analysis and retained separately.

Integrity audit: 18,000 valid episodes across 9 cells; exact seed sets, checkpoint hashes, protocol hashes, and cross-cell initial positions passed. Model weights were unchanged. See `bc_results/h4_replanning/final_integrity_audit.json`.

A post-run analysis QA check found that the first generated summary multiplied the mean per-call latency by 1,000 twice. That summary is preserved as `summary_initial_latency_scale_error.json`; `summary_latency_correction.json` records the correction. The reported per-call latency was recomputed from unchanged raw inference milliseconds and call counts. Success outcomes, inference-call counts, total inference time, and tests were unaffected.

### Recovery and data-integrity disclosure

The first H4 seed 456 / R1 partial file contained 2,062 rows for 2,000 unique seeds, with duplicated seeds 311938–311999. Duplicate records had matching non-timing fields but different inference and wall-clock measurements. The file was preserved byte-for-byte and excluded. With explicit user authorization, that entire 2,000-episode cell was rerun to a separate `recovery1` output; no rows were carried over. During diagnosis, two example duplicated records (311938 and 311939) were directly viewed and both were successes. This limited outcome exposure occurred before the recovery run; no aggregate condition comparison was computed and no condition or setting was selected or tuned. This deviation and exposure are retained in `recovery_authorization.json` and `corrupt_partial_seed456_R1_audit.json`.

## 9. Success Rate Results

| Frozen BC seed | R1 | R2 | R4 | R4 − R1 (pp) |
|---:|---:|---:|---:|---:|
| 42 | 1619/2000 (80.95%) | 1857/2000 (92.85%) | 1878/2000 (93.90%) | +12.95 |
| 123 | 1862/2000 (93.10%) | 1849/2000 (92.45%) | 1866/2000 (93.30%) | +0.20 |
| 456 | 1834/2000 (91.70%) | 1900/2000 (95.00%) | 1915/2000 (95.75%) | +4.05 |
| Mean across three fixed models | 88.58% | 93.43% | 94.32% | **+5.73** |

The three-model mean is descriptive and weights the three existing checkpoints equally. It is not a 6,000-policy estimate.

## 10. Paired Rescue and Regression Analysis

“Rescue” and “regression” below mean paired success transitions between replanning conditions, not SAC interventions.

| BC seed | R2 rescue / regression vs R1 | Net | R4 rescue / regression vs R1 | Net |
|---:|---:|---:|---:|---:|
| 42 | 253 / 15 | +238 | 287 / 28 | +259 |
| 123 | 7 / 20 | -13 | 45 / 41 | +4 |
| 456 | 68 / 2 | +66 | 85 / 4 | +81 |

For R4−R1, the per-seed success differences equal rescue minus regression: +259, +4, and +81 successes. Across the three paired model-position cells, the descriptive total is 417 rescue transitions and 73 regressions (net +344); because evaluation positions are shared across models, this aggregate is not an independent-sample test. R2−R1 yielded 328 rescues and 37 regressions (net +291). R4 exceeded R2 in all three seeds by +1.05, +0.85, and +0.75 pp; its paired transitions and arithmetic are in `summary.json`.

Regional success and transition counts are exploratory. Each fixed model has 493 EN, 512 ES, 503 WN, and 492 WS positions.

| BC seed | Region (n) | R1 success | R2 success | R4 success | R2 rescue / regression vs R1 | R4 rescue / regression vs R1 |
|---:|:---:|---:|---:|---:|---:|---:|
| 42 | EN (493) | 92.29% | 92.09% | 95.33% | 1 / 2 | 15 / 0 |
| 42 | ES (512) | 100.00% | 97.46% | 94.53% | 0 / 13 | 0 / 28 |
| 42 | WN (503) | 74.75% | 97.22% | 97.22% | 113 / 0 | 113 / 0 |
| 42 | WS (492) | 56.10% | 84.35% | 88.41% | 139 / 0 | 159 / 0 |
| 123 | EN (493) | 95.54% | 94.93% | 93.51% | 2 / 5 | 11 / 21 |
| 123 | ES (512) | 97.66% | 98.44% | 93.95% | 4 / 0 | 1 / 20 |
| 123 | WN (503) | 86.88% | 84.69% | 89.86% | 1 / 12 | 15 / 0 |
| 123 | WS (492) | 92.28% | 91.67% | 95.93% | 0 / 3 | 18 / 0 |
| 456 | EN (493) | 96.35% | 96.35% | 95.54% | 2 / 2 | 0 / 4 |
| 456 | ES (512) | 93.36% | 98.44% | 99.22% | 26 / 0 | 30 / 0 |
| 456 | WN (503) | 89.86% | 91.05% | 92.64% | 6 / 0 | 14 / 0 |
| 456 | WS (492) | 87.20% | 94.11% | 95.53% | 34 / 0 | 41 / 0 |

The largest R4−R1 regional changes occurred for seed 42 in WN and WS; seed 123 had small declines in EN/ES but gains in WN/WS; seed 456 gains were largest in ES/WS. These region-level patterns should not be generalized because the analysis is exploratory and each region contains about 500 positions per fixed model.

## 11. BC Inference Frequency

| BC seed | Interval | Calls / episode | Mean latency / call (ms) | Total inference / episode (ms) |
|---:|---:|---:|---:|---:|
| 42 | R1 | 127.18 | 0.0363 | 4.616 |
| 42 | R2 | 60.72 | 0.0399 | 2.424 |
| 42 | R4 | 30.51 | 0.0550 | 1.678 |
| 123 | R1 | 123.12 | 0.0359 | 4.425 |
| 123 | R2 | 61.93 | 0.0389 | 2.406 |
| 123 | R4 | 30.94 | 0.0385 | 1.192 |
| 456 | R1 | 123.42 | 0.0388 | 4.793 |
| 456 | R2 | 61.32 | 0.0547 | 3.354 |
| 456 | R4 | 30.85 | 0.0736 | 2.269 |

Across the three models, mean calls fell from 124.57 (R1) to 30.77 (R4), a 75.3% reduction. Mean measured inference time fell from 4.611 to 1.713 ms per episode (62.9%). Per-call latency fluctuated across sequential runs and is descriptive.

## 12. Execution Time

| BC seed | Interval | Mean control steps | Mean successful steps | Mean total episode wall time (s) | Mean first close step | Mean first lift step |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | R1 | 127.18 | 119.45 | 0.3707 | 69.52 | 114.48 |
| 42 | R2 | 120.98 | 117.98 | 0.3520 | 71.72 | 112.97 |
| 42 | R4 | 120.64 | 118.08 | 0.3568 | 71.09 | 113.06 |
| 123 | R1 | 123.12 | 120.39 | 0.3602 | 70.22 | 115.39 |
| 123 | R2 | 123.40 | 120.41 | 0.3592 | 68.23 | 115.41 |
| 123 | R4 | 122.54 | 119.85 | 0.3527 | 66.64 | 114.85 |
| 456 | R1 | 123.42 | 120.11 | 0.3669 | 70.82 | 115.11 |
| 456 | R2 | 122.17 | 120.18 | 0.3711 | 71.14 | 115.18 |
| 456 | R4 | 122.02 | 120.33 | 0.3669 | 70.71 | 115.33 |

Mean total episode wall time was 0.3659 s for R1 and 0.3588 s for R4, a descriptive 1.9% difference. Episode timing starts immediately before the initial policy inference and ends at the terminal transition; environment reset and model loading are excluded. The runtime figure separates full episode time from millisecond-scale model inference. The modest wall-time change is not evidence of a proportional end-to-end speedup; hardware load and serial run order affect timings.

## 13. Statistical Analysis

For each fixed model, paired success outcomes on the same 2,000 positions were compared with a two-sided exact McNemar test. Holm correction was applied across the three model-specific tests within each comparison family. A 95% percentile paired bootstrap interval uses 10,000 resamples of positions and fixed RNG seed 20261106. These inferential quantities are conditional on the evaluated model and position distribution.

| BC seed | R4−R1 (pp) | 95% paired bootstrap CI (pp) | Exact McNemar p | Holm p |
|---:|---:|---:|---:|---:|
| 42 | +12.95 | [+11.35, +14.65] | 2.83e-55 | 8.5e-55 |
| 123 | +0.20 | [-0.70, +1.10] | 0.747 | 0.747 |
| 456 | +4.05 | [+3.15, +4.95] | 8.27e-21 | 1.65e-20 |

The primary position-level contrast is significant after the specified correction for seeds 42 and 456, but not for seed 123. The three seed-level effects are +12.95, +0.20, and +4.05 pp (mean +5.73, median +4.05, sample SD 6.54, range +0.20 to +12.95). All three directions are positive, but three trained seeds are too few to support strong population-level generalization. The shared 2,000 positions are not independent policy replications.

R2−R1 exact paired tests are reported in `summary.json`; Holm-adjusted results are significant for each seed, with effects +11.90, −0.65, and +3.30 pp. These position-level tests do not establish that a replanning interval is generally superior across training runs or tasks.

## 14. Interpretation of R1/R2/R4

R4 yielded higher success than R1 for all three frozen checkpoints, and it exceeded R2 for all three by less than 1.1 pp. R1 did not outperform R4 in any of the three seed-level comparisons. The magnitude varied sharply: seed 42 accounted for the largest observed R4−R1 gain, while seed 123 was nearly unchanged. R2 was not a uniformly better compromise: it was below R1 for seed 123 and above R1 for seeds 42 and 456.

## 15. Implications for Action Chunking and Closed-Loop Feedback

For these H4 policies and this simulated grasping task, executing cached actions for longer blocks was associated with higher success than replanning every step. This is consistent with temporal consistency being useful here; it does not show that open-loop chunks are universally better. Frequent feedback did not yield higher mean success in this experiment. R4 reduced BC inference demand substantially, while measured end-to-end episode runtime changed little.

The earlier H1-versus-H4 result remains confounded by architecture, supervision count, output parameterization, and runtime frequency. This controlled H4 study shows that runtime interval alone can matter for some checkpoints, especially seed 42, but it cannot attribute the entire H1/H4 gap to runtime replanning. The R4 replication closely matches earlier H4 Final rates: new minus prior differences were −0.05, −0.15, and +0.05 pp for seeds 42, 123, and 456, respectively.

## 16. Limitations

- Only three independently trained BC seeds and one simulated task were studied; positions are shared within each seed across conditions.
- Position-level p-values and confidence intervals are conditional on fixed checkpoints and the chosen position distribution; they do not replace more independent training seeds.
- Sequential wall-clock timing is sensitive to machine load. Call reduction, inference-time reduction, and end-to-end runtime reduction are distinct quantities.
- The recovery run followed a documented duplicate-output incident. Two duplicated success outcomes were directly observed before the full-cell recovery; no tuning or interval selection followed, but this is a limited data-exposure deviation.
- The experiment isolates runtime replanning within H4 only. It does not isolate H4 multi-step supervision from policy architecture or compare additional chunk sizes.

## 17. Relationship to ResiP / ResFiT

[ResiP](https://arxiv.org/abs/2407.16677) combines frozen chunked behavior cloning with a closed-loop residual policy learned using reinforcement learning. [ResFiT](https://arxiv.org/abs/2509.19301) studies off-policy residual fine-tuning around a frozen BC base. Both motivate preserving a pretrained policy while adding corrections. The present experiment is a prerequisite-style runtime ablation: it trains neither a residual policy nor any RL controller, and evaluates only how a frozen H4 policy behaves when the cached chunk is consumed at different rates.

## 18. Next Research Direction

Before making broad claims, replicate the same preregistered R1/R2/R4 comparison with more independently trained H4 seeds and a separately held-out position distribution. A subsequent study can test whether the observed seed- and region-specific failures are corrected by a residual controller, but that should be a distinct experiment with its own training and selection protocol. A matched timing protocol or repeated interleaved run order is needed before making strong runtime claims.

## 19. Reproduction Instructions

From the repository root on branch `experiment/h4-replanning-ablation`:

```bash
./.venv/bin/python -m unittest discover -v
./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed 42 --interval 1
./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed 42 --interval 2
./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed 42 --interval 4
./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed 123 --interval 1
./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed 123 --interval 2
./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed 123 --interval 4
./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed 456 --interval 1 --run-label recovery1
./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed 456 --interval 2
./.venv/bin/python evaluate_h4_replanning.py --phase final --model-seed 456 --interval 4
MPLCONFIGDIR=/private/tmp/matplotlib-h4 ./.venv/bin/python analyze_h4_replanning.py
./.venv/bin/python write_h4_replanning_report.py
```

Completed output CSVs validate against the expected unique seed set and provenance hashes. Do not run duplicate processes for the same configuration. The `recovery1` label is specific to the documented seed-456/R1 recovery and keeps the preserved corrupt partial separate.
