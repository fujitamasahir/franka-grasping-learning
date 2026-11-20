# Franka grasping experiments

This page is the index for the project's experiments and their results. The studies follow the project page's progression from behavior cloning (BC), to residual reinforcement learning (SAC), then to more reliable replanning under observation uncertainty.

## Where to start

- Read the study note for the question, protocol, conclusion and reproduction steps.
- Open the linked result folder for raw rollouts, audits, summaries and figures.
- Files named `protocol` and their `.sha256` files capture the preregistered design; `final` files are held-out results; `smoke`, `development`, `calibration`, and `validation` are supporting phases with distinct roles.
- Some studies are preparation only or stopped at a preregistered gate. Their notes say so explicitly; a folder or implementation does not imply a successful experiment.

## Project progression

| Stage | Experiment | Question / result | Study note | Results |
|---|---|---|---|---|
| 1. BC baseline | Behavior cloning and generalization | Compare expert and DAgger datasets, model seeds, and generalization to new grasp positions. | [BC and DAgger overview](experiments/behavior-cloning/README.md) | [`bc_results/baselines/`](bc_results/baselines/) |
| 2. Residual RL | Controlled residual SAC | Test uniform vs hard-case sampling and sparse vs shaped rewards for a residual controller on frozen BC. | [Controlled SAC](experiments/residual-sac/EXPERIMENTS_SAC.md) | [`bc_results/residual_sac/`](bc_results/residual_sac/) |
| 2. Residual RL | Selective residual gate | Test whether a gate can preserve BC successes while applying SAC selectively to likely failures. | [Selective gate](experiments/residual-sac/EXPERIMENTS_SELECTIVE_GATE.md) and [multi-seed follow-up](experiments/residual-sac/EXPERIMENTS_SELECTIVE_MULTISEED.md) | [`bc_results/selective_gate/`](bc_results/selective_gate/) |
| 2. Residual RL | Policy-specific gate | Calibrate and evaluate per-policy gate thresholds across independent SAC seeds. | [Policy-specific gate](experiments/residual-sac/EXPERIMENTS_POLICY_SPECIFIC_GATE.md) | [`bc_results/selective_gate/`](bc_results/selective_gate/) |
| 2. Residual RL | SAC training stability | Measure checkpoint and training-seed variability for residual SAC. | [Training stability](experiments/residual-sac/EXPERIMENTS_SAC_TRAINING_STABILITY.md) | [`bc_results/sac_training_stability/`](bc_results/sac_training_stability/) |
| 2. Residual RL | Regression control | Diagnose when SAC residuals regress successful BC episodes and test regularization controls. | [Regression control](experiments/residual-sac/EXPERIMENTS_SAC_REGRESSION_CONTROL.md) | [`bc_results/sac_regression_control/`](bc_results/sac_regression_control/) |
| 2. Residual RL | Regularization replication | Independently replicate the SAC regularization comparison with new training seeds. | [Replication](experiments/residual-sac/EXPERIMENTS_SAC_REGULARIZATION_REPLICATION.md) | [`bc_results/sac_regularization_replication/`](bc_results/sac_regularization_replication/) |
| 2. Residual RL | Action-chunked BC + SAC | Compare single-step and four-step BC, then evaluate stepwise residual SAC on each. | [Action chunking](experiments/residual-sac/EXPERIMENTS_ACTION_CHUNKED_BC_RESIDUAL_RL.md) | [`bc_results/action_chunking/`](bc_results/action_chunking/) |
| 3. Replanning | Fixed replanning ablation | Compare fixed replanning intervals for frozen H4 BC policies. | [Replanning ablation](experiments/h4-replanning/EXPERIMENTS_H4_REPLANNING_ABLATION.md) | [`bc_results/h4_replanning/`](bc_results/h4_replanning/) |
| 3. Replanning | Observation-noise robustness | Measure fixed replanning performance as observation noise increases. | [Observation noise](experiments/h4-replanning/EXPERIMENTS_H4_OBSERVATION_NOISE.md) | [`bc_results/h4_replanning/observation_noise/`](bc_results/h4_replanning/observation_noise/) |
| 3. Replanning | Adaptive replanning | Select a replanning interval from an online uncertainty estimate. | [Uncertainty-adaptive replanning](experiments/h4-replanning/EXPERIMENTS_H4_UNCERTAINTY_ADAPTIVE_REPLANNING.md) | [`bc_results/h4_replanning/adaptive/`](bc_results/h4_replanning/adaptive/) |
| 3. Replanning | Dynamic uncertainty | Test online switching under changing noise. This study stopped at its stationarity gate; see the report for interpretation. | [Dynamic uncertainty](experiments/h4-replanning/EXPERIMENTS_H4_DYNAMIC_UNCERTAINTY_REPLANNING.md) | [`bc_results/h4_replanning/dynamic/`](bc_results/h4_replanning/dynamic/) |
| 4. Noise estimation | Motion-aware estimator | Validate an estimator against movement-related confounds; later stages did not proceed when preregistered gates failed. | [Motion-aware estimator](experiments/h4-noise-estimation/EXPERIMENTS_H4_MOTION_AWARE_NOISE_ESTIMATION.md) | [`bc_results/h4_noise_estimation/motion_aware/`](bc_results/h4_noise_estimation/motion_aware/) |
| 4. Noise estimation | Model-based Kalman diagnostic | Implement and prepare an independent diagnostic. The model-based note labels work not yet run. | [Model-based estimator](experiments/h4-noise-estimation/README_H4_MODEL_BASED_NOISE_ESTIMATION.md) and [independent validation](experiments/h4-noise-estimation/README_H4_KALMAN_INDEPENDENT_VALIDATION.md) | [`bc_results/h4_noise_estimation/`](bc_results/h4_noise_estimation/) |
| 4. Noise estimation | Kalman online replanning | Prepare a frozen closed-loop study with online Kalman estimates; see the report for execution status. | [Kalman online study](experiments/h4-noise-estimation/README_H4_KALMAN_ONLINE_DYNAMIC.md) | [`bc_results/h4_noise_estimation/kalman_online/`](bc_results/h4_noise_estimation/kalman_online/) |

## Folders

`experiments/` holds the study notes grouped by topic. `bc_results/` holds the evidence grouped by the same experiment names. The root scripts remain at the repository root so existing commands and imports continue to work.

| Results folder | Contents |
|---|---|
| `bc_results/baselines/` | BC, DAgger, fair-dataset, and generalization rollouts and comparisons. |
| `bc_results/residual_sac/` | Early controlled SAC pilot outputs and residual-scale comparisons. |
| `bc_results/selective_gate/` | Selective and policy-specific gate development, paired Final rollouts, summaries, and manifests. |
| `bc_results/sac_training_stability/` | SAC stability validation/Final rollouts, checkpoint audits, summary tables and figures. |
| `bc_results/sac_regression_control/` | SAC regression diagnostics, regularization study and focused outcomes. |
| `bc_results/sac_regularization_replication/` | Independent-seed replication data and analysis. |
| `bc_results/action_chunking/` | Action-chunking protocol, BC/SAC runs, rollouts, audits, tables and figures. |
| `bc_results/h4_replanning/` | Fixed-interval ablation and observation-noise robustness. |
| `bc_results/h4_replanning/adaptive/` | Adaptive replanning evaluation and analysis. |
| `bc_results/h4_replanning/dynamic/` | Dynamic replanning smoke, calibration and gate-audit evidence. |
| `bc_results/h4_noise_estimation/` | Motion-aware, model-based Kalman, independent validation and online study evidence. |

Historical filenames are retained inside the topic folders. See each study note for the exact mapping of protocol, raw data, summaries, figures and audit files.
