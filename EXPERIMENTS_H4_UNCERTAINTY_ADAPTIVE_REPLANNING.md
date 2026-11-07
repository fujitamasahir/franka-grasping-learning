# Uncertainty-Aware Adaptive Replanning for Frozen H4 BC

**Status: Calibration complete; Final paused for hardware transfer.** The preregistered protocol, estimator, smoke audit, resumable evaluator, and calibration are complete. Final was stopped at a complete cell boundary for the planned transfer to a higher-performance Mac. The count-only audit verifies 21/27 cells (21,000/27,000 episodes), with no partial cell files. No Final outcomes or performance statistics have been inspected or summarized. Resume the remaining six cells and run the full audit and analysis only after all 27 cells pass.

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

| BC training seed | Checkpoint SHA-256 |
|---:|---|
| 42 | `41cda0af609e3064ba8e485a4ef81289c319472b08d1f710d96b30f6cb470557` |
| 123 | `5dc02a9b9cce85b93beb18288c0681c90e5174181210cd4c0c23d518d653495c` |
| 456 | `649e42de42f0a04a2dc180dfa513afbbc52aba2e78f4f6c28a3deb41eb3cd706` |

## 6. Observation Uncertainty Estimation

Independent, zero-mean Gaussian errors are applied only to observed cube XY and matching relative XY entries, using the existing noise generator at σ = 0, 5, or 10 mm. True state and physics remain unchanged. The estimator is

`sigma_hat = sqrt((sample_var(x, ddof=1) + sample_var(y, ddof=1))/2)`

over noisy cube XY at observation times 0–19. It receives no true σ, clean coordinates, error vector, environment hidden state, evaluation seed feature, or outcome label. The first 20 actions execute from R4 chunks; at the chunk boundary after step 20 the controller selects R4 if `sigma_hat <= threshold`, else R1 for the rest of the episode. If termination occurs before a decision, no switch is made.

Before Calibration, 720 smoke episodes on seeds 333000–333019 passed completeness, pairing, switch timing, and branch-equivalence checks. Across all models and noise levels, maximum ground-truth cube displacement in the first 20 steps was **1.90e−10 m**, below the preregistered 1 mm stationarity criterion. Ground truth was used only for this audit. Smoke `sigma_hat` medians were 0, 5.075, and 10.150 mm for true σ of 0, 5, and 10 mm (60 episodes per level). The smoke threshold was used only to exercise code; it did not inform Calibration selection.

The earlier 540-episode smoke run used an earlier protocol-file SHA. It is retained with that hash as preliminary software evidence and excluded from threshold selection; the official post-amendment smoke2 is the 720-episode run above. A pre-Final reporting correction makes `region` derive from initial reset position rather than terminal cube position. Calibration does not select thresholds by region, and the Final audit verifies each initial-position region.

## 7. Adaptive Replanning Controller

R1 always replans at every step. R4 consumes each four-action chunk. Adaptive executes R4 for 20 control steps, estimates uncertainty once, and then continues at one fixed interval for the remainder. This is episode-level switching, not per-step adaptation. Switching occurs on a four-action boundary, avoiding half-consumed chunks. Adaptive versus Fixed R1 is an end-to-end comparison because R1 does not share the R4 warm-up; Adaptive versus Fixed R4 shares the first 20-step trajectory.

An additional warm-up-matched branch executes R4 for 20 steps, then R1 regardless of uncertainty. Smoke verified for 180 model/noise/seed combinations that Adaptive's outcome and trajectory counters exactly matched the selected complete branch: Fixed R4 when it selected R4, or the separately executed warm-up-matched R1 branch when it selected R1. Calibration therefore derives each candidate's complete adaptive outcome from these verified branches; it never substitutes a full-episode Fixed R1 label for a switched trajectory.

## 8. Preregistration and Calibration

The protocol SHA-256 is recorded in `bc_results/h4_adaptive_replanning_protocol.sha256`. Calibration uses 300 shared positions per BC seed and noise level. One global threshold is selected from 5.0, 7.5, and 10.0 mm by equally weighted success across nine model-seed × noise cells. Ties are resolved by fewer inference calls, then the smaller threshold. Fixed R1/R4 are measured on the same seeds; the better fixed method is selected from Calibration only. The frozen threshold is stored before Final begins.

Calibration completed with 300 shared evaluation seeds for each of 3 BC seeds × 3 noise levels, yielding 2,700 episodes per executed branch and 8,100 complete rollouts across Fixed R1, Fixed R4, and Warmup-R1. The three candidate adaptive outcomes were derived from complete, smoke-verified paths: select the Fixed R4 trajectory when the R4-prefix estimate is under threshold, otherwise the separately executed Warmup-R1 trajectory. Fixed R1 labels were not used to stand in for a switched R1 trajectory.

| Global threshold | Equal-weighted Adaptive success | Mean inference calls/episode | R1 selection at 0 / 5 / 10 mm |
|---:|---:|---:|---:|
| 5.0 mm | 90.852% | 66.39 | 0% / 47.3% / 100% |
| **7.5 mm (selected)** | **91.037%** | **54.07** | **0% / 0% / 99.0% |
| 10.0 mm | 90.630% | 41.38 | 0% / 0% / 47.3% |

Fixed R1 achieved an equal-weighted 88.963% with 121.26 mean inference calls; Fixed R4 achieved 90.222% with 29.83 calls. Fixed R4 was the Calibration-selected fixed baseline. The threshold was selected solely by the preregistered highest-success rule (no tie-break was needed) and is frozen in `bc_results/h4_adaptive_replanning/frozen_threshold.json` before Final.

The estimator's Calibration mean/median estimates were approximately 0/0 mm for true 0 mm, 4.959/4.959 mm for true 5 mm, and 9.919/9.919 mm for true 10 mm (900 estimates per level). Nearest nominal-sigma bucket agreement was 100%, 100%, and 99.0%, respectively; this is a descriptive diagnostic, not an input to the controller.

## 9. Final Evaluation Protocol

Final covers seeds 331000–331999 for each of three controllers, three BC models, and three noise conditions: 27,000 episodes. All conditions share initial positions and absolute-step-indexed noise. Each cell is written append-safely to a separate CSV with checkpoint/protocol/noise hashes. A clean interruption leaves `.partial.csv` files that the evaluator resumes after validating schema, order, uniqueness, and provenance. Do not edit conditions based on Final outcomes.

Final progress/integrity audit: **21/27 cells complete (21,000/27,000 rows), all complete cells passed provenance, ordering, pairing, and initial-position region checks; 0 incomplete files.** Complete cells are all nine conditions for BC seeds 42 and 123, plus all three controllers at 0 mm for BC seed 456. The six remaining cells are seed 456 at 5 and 10 mm × R1/R4/Adaptive. The count-only audit is `bc_results/h4_adaptive_replanning/final_progress_audit.json`. Final outcomes were deliberately not opened for performance analysis. No incomplete Final performance claims are made.

## 10. Success Rate Results

Pending the complete, audited Final position set. Report all 27 cells, per-noise, per-model, and equal-weighted rates.

## 11. Comparison with Fixed R1/R4

Pending Final. Adaptive versus R1 is end-to-end because of the R4 warm-up; Adaptive versus R4 shares the initial trajectory.

## 12. Uncertainty Estimation Accuracy

Smoke estimates are summarized in Section 6. Final sigma-hat distributions and switching accuracy by actual σ remain pending.

## 13. Beneficial and Harmful Outcome Analysis

Pending Final paired transitions. Report exact rescue/harmful counts per model and noise; do not infer switched R1 outcomes from fixed R1 labels.

## 14. Inference Cost

Pending Final. Report inference calls and time, wall-clock time, and control steps separately.

## 15. Statistical Interpretation

Pending complete Final analysis. Use paired position bootstrap and exact McNemar tests with Holm adjustment. Intervals are conditional on three frozen policies and do not quantify generalization to arbitrary training seeds.

## 16. Failure Analysis

Pending complete Final audit. Distinguish episodes ending before a decision, noisy-estimate decisions, calibrated-threshold limitations, fixed-branch limitations, and software/timing defects when supported by evidence.

## 17. Limitations

Only three independent BC training seeds are available. Evaluation positions and noise schedules are paired across conditions and do not create independent policy-training replications. The noise is synthetic Gaussian state-observation perturbation, not camera occlusion or realistic perception drift. The stationarity assumption is supported by the specific initial 20-step smoke audit, not guaranteed in other tasks or controller configurations. The decision is coarse and episode-level. Execution timing is hardware-load sensitive.

## 18. Implications for Reliable Robotic Manipulation

The study is designed to test whether a transparent uncertainty statistic can choose between reactive feedback and temporally extended execution while preserving a frozen policy. Any claim of improvement must be limited to this benchmark, these three models, and the specified synthetic noise process. A null or negative result remains informative for the SoP narrative.

## 19. Future Work

If warranted by the completed result, evaluate realistic temporally correlated perception errors and a preregistered larger sample of independently trained policies. Such work is outside this experiment's scope.

## 20. Reproduction Instructions

Run `audit_h4_adaptive_replanning_smoke.py` after the smoke cells are complete. Calibration is already complete and frozen at threshold `0.0075` (7.5 mm) in `bc_results/h4_adaptive_replanning/frozen_threshold.json`. On the target Mac, first copy the two ignored model checkpoints noted below, verify hashes, then resume with `.venv/bin/python evaluate_h4_adaptive_replanning.py --phase final --model-seed 456 --threshold 0.0075`. The evaluator skips the six already complete seed-456/0 mm cells and continues the remaining six cells in fixed order; no completed CSV should be removed or edited. The evaluator resumes validated `.partial.csv` files. To stop cleanly at a complete cell boundary, create `bc_results/h4_adaptive_replanning/.STOP_AFTER_CELL`; after the process exits, remove that marker before resuming. Use `audit_h4_adaptive_replanning_final.py --allow-incomplete` for count-only progress or the same audit without the flag after all 27 cells pass. Run `MPLCONFIGDIR=/private/tmp/matplotlib-h4 .venv/bin/python analyze_h4_adaptive_replanning.py` only after the complete audit. The global `.study_run.lock` prevents concurrent writers.

Checkpoint transfer note: the seed-42 checkpoint is tracked in the base repository, but seed-123 and seed-456 `.pt` files are gitignored and are not pushed. Copy those two unchanged files from the current Mac into the same paths on the target Mac and verify all three hashes against the protocol before resuming. No model files were committed by this study.

### References

- Zhao et al., [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) (ACT).
- Chi et al., [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137).
- Ankile et al., [From Imitation to Refinement—Residual RL for Precise Assembly](https://arxiv.org/abs/2407.16677) (ResiP).
- [Residual Off-Policy RL for Finetuning Behavior Cloning Policies](https://arxiv.org/abs/2509.19301) (ResFiT).
- Nie et al., [PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking](https://arxiv.org/abs/2606.00537).
- Liu et al., [Refine Actions If Needed: Adaptive Action Chunk Refinement for Robot Manipulation](https://arxiv.org/abs/2610.19765) (RAIN).
