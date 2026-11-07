# Uncertainty-Aware Adaptive Replanning for Frozen H4 BC

**Status: staged execution for transfer to a higher-performance Mac.** The estimator, smoke audit, preregistration, and resumable evaluator are in place. Calibration and holdout Final are intended to run sequentially. This report must be updated from the committed `calibration_summary.json` and final integrity/progress audit after execution; incomplete holdout results must not be analyzed as full-cell results.

## 1. Abstract

This controlled study tests a simple episode-level rule for switching a frozen H4 behavior-cloning policy between one-step replanning (R1) and four-step chunk execution (R4). During the first 20 control steps the controller executes R4, estimates XY observation noise from the sample variance of 20 noisy cube-position measurements, and selects R4 below a single globally calibrated threshold or R1 above it. No BC/SAC training or policy updates are performed. The preregistered calibration uses seeds 330000–330299; untouched Final positions use 331000–331999 (27,000 episodes). The Final set must remain unopened for outcome analysis until all 27 cells are complete and pass integrity checks.

## 2. Background and Motivation

The preceding H4 observation-noise study found that the relative R4 advantage declined as synthetic XY observation noise increased, with the direction and size depending on the frozen BC training seed. This motivates a transparent test of whether noisy observations can drive a coarse episode-level replanning choice. The present rule makes one decision after a 20-step R4 warm-up; it does not adapt continuously.

## 3. Research Questions and Hypotheses

The questions and hypotheses are preregistered in `bc_results/h4_adaptive_replanning_protocol.json`. They concern noise estimation from accessible observations, the performance of adaptive versus fixed R1/R4, preservation of clean-observation performance, reduction of noisy-condition harm, and heterogeneity across the three frozen models. No positive outcome is presumed.

## 4. Related Work

Action Chunking with Transformers (ACT) predicts action sequences for bimanual manipulation; Diffusion Policy learns a conditional action-sequence distribution and uses receding-horizon execution. ResiP augments a frozen chunked behavior-cloning policy with a learned, closed-loop RL residual. ResFiT is likewise an RL fine-tuning approach for behavior-cloned policies. Recent test-time chunk execution work such as PACE chooses execution boundaries from predicted action profiles, while RAIN predicts whether action proposals need refinement. This experiment does not claim adaptive replanning as a novel idea. Its contribution is a controlled, frozen-policy comparison of a simple observation-variance threshold, explicitly separating the R4 warm-up from the later choice and auditing paired trajectories.

## 5. Frozen Models and Environment

The exact checkpoints are `bc_models/action_chunking/H4_seed{42,123,456}.pt`; their SHA-256 digests and the source commit are frozen in the protocol. The existing Franka environment, H4 architecture, observation normalization, action ranges, gripper convention, physics, reward, and success rule are unchanged. There is no training, SAC, policy update, or checkpoint selection.

## 6. Observation Uncertainty Estimation

Independent, zero-mean Gaussian errors are applied only to observed cube XY and matching relative XY entries, using the existing noise generator at σ = 0, 5, or 10 mm. True state and physics remain unchanged. The estimator is

`sigma_hat = sqrt((sample_var(x, ddof=1) + sample_var(y, ddof=1))/2)`

over noisy cube XY at observation times 0–19. It receives no true σ, clean coordinates, error vector, environment hidden state, evaluation seed feature, or outcome label. The first 20 actions execute from R4 chunks; at the chunk boundary after step 20 the controller selects R4 if `sigma_hat <= threshold`, else R1 for the rest of the episode. If termination occurs before a decision, no switch is made.

Before Calibration, 720 smoke episodes on seeds 333000–333019 passed completeness, pairing, switch timing, and branch-equivalence checks. Across all models and noise levels, maximum ground-truth cube displacement in the first 20 steps was **1.90e−10 m**, below the preregistered 1 mm stationarity criterion. Ground truth was used only for this audit. Smoke `sigma_hat` medians were 0, 5.075, and 10.150 mm for true σ of 0, 5, and 10 mm (60 episodes per level). The smoke threshold was used only to exercise code; it did not inform Calibration selection.

## 7. Adaptive Replanning Controller

R1 always replans at every step. R4 consumes each four-action chunk. Adaptive executes R4 for 20 control steps, estimates uncertainty once, and then continues at one fixed interval for the remainder. This is episode-level switching, not per-step adaptation. Switching occurs on a four-action boundary, avoiding half-consumed chunks. Adaptive versus Fixed R1 is an end-to-end comparison because R1 does not share the R4 warm-up; Adaptive versus Fixed R4 shares the first 20-step trajectory.

An additional warm-up-matched branch executes R4 for 20 steps, then R1 regardless of uncertainty. Smoke verified for 180 model/noise/seed combinations that Adaptive's outcome and trajectory counters exactly matched the selected complete branch: Fixed R4 when it selected R4, or the separately executed warm-up-matched R1 branch when it selected R1. Calibration therefore derives each candidate's complete adaptive outcome from these verified branches; it never substitutes a full-episode Fixed R1 label for a switched trajectory.

## 8. Preregistration and Calibration

The protocol SHA-256 is recorded in `bc_results/h4_adaptive_replanning_protocol.sha256`. Calibration uses 300 shared positions per BC seed and noise level. One global threshold is selected from 5.0, 7.5, and 10.0 mm by equally weighted success across nine model-seed × noise cells. Ties are resolved by fewer inference calls, then the smaller threshold. Fixed R1/R4 are measured on the same seeds; the better fixed method is selected from Calibration only. The frozen threshold is stored before Final begins.

Calibration output and selected threshold: **pending completed Calibration**.

## 9. Final Evaluation Protocol

Final covers seeds 331000–331999 for each of three controllers, three BC models, and three noise conditions: 27,000 episodes. All conditions share initial positions and absolute-step-indexed noise. Each cell is written append-safely to a separate CSV with checkpoint/protocol/noise hashes. A clean interruption leaves `.partial.csv` files that the evaluator resumes after validating schema, order, uniqueness, and provenance. Do not edit conditions based on Final outcomes.

Final completion and integrity audit: **pending; inspect no success rates before all 27 cells pass**.

## 10–16. Results, Statistics, Costs, and Failure Analysis

Populate these sections only after the full Final seed sets pass audit. Report success equally weighted across the nine model × noise cells; paired differences and beneficial/harmful transitions per BC seed and noise; exact McNemar tests with Holm correction; paired bootstrap intervals jointly resampling position indices across controllers and noise; sigma-hat distribution and switching frequency; inference calls, inference time, wall time, control steps; and EN/ES/WN/WS summaries. Position-level intervals are conditional on the three fixed policies and sampled positions, not evidence of a broad population of trained policies. Failures must be separated into early termination, uncertainty misclassification, threshold choice, R1/R4 branch limitations, and timing/software defects where evidence permits.

## 17. Limitations

Only three independent BC training seeds are available. Evaluation positions and noise schedules are paired across conditions and do not create independent policy-training replications. The noise is synthetic Gaussian state-observation perturbation, not camera occlusion or realistic perception drift. The stationarity assumption is supported by the specific initial 20-step smoke audit, not guaranteed in other tasks or controller configurations. The decision is coarse and episode-level. Execution timing is hardware-load sensitive.

## 18. Implications for Reliable Robotic Manipulation

The study is designed to test whether a transparent uncertainty statistic can choose between reactive feedback and temporally extended execution while preserving a frozen policy. Any claim of improvement must be limited to this benchmark, these three models, and the specified synthetic noise process. A null or negative result remains informative for the SoP narrative.

## 19. Future Work

If warranted by the completed result, evaluate realistic temporally correlated perception errors and a preregistered larger sample of independently trained policies. Such work is outside this experiment's scope.

## 20. Reproduction Instructions

Run `audit_h4_adaptive_replanning_smoke.py` after the smoke cells are complete. Run `.venv/bin/python evaluate_h4_adaptive_replanning.py --phase calibration --model-seed 42` and likewise for seeds 123 and 456; analyze with `analyze_h4_adaptive_calibration.py`. Freeze the resulting threshold before running each Final model seed. Use `audit_h4_adaptive_replanning_final.py` and `analyze_h4_adaptive_replanning.py` only after all 27 Final cells pass. The global `.study_run.lock` prevents concurrent writers.

### References

- Zhao et al., [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) (ACT).
- Chi et al., [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137).
- Ankile et al., [From Imitation to Refinement—Residual RL for Precise Assembly](https://arxiv.org/abs/2407.16677) (ResiP).
- [Residual Off-Policy RL for Finetuning Behavior Cloning Policies](https://arxiv.org/abs/2509.19301) (ResFiT).
- Nie et al., [PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking](https://arxiv.org/abs/2606.00537).
- Liu et al., [Refine Actions If Needed: Adaptive Action Chunk Refinement for Robot Manipulation](https://arxiv.org/abs/2610.19765) (RAIN).
