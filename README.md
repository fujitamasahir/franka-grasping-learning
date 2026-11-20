# Franka Grasping Project

This repository contains the Franka grasping research code, experiment reports, and result artifacts. Use this page to find the implementation, report, and data for each experiment.

## Start here

- [Experiment index](EXPERIMENTS.md): a short overview of the research sequence and experiments.
- [Python source guide](PYTHON.md): source layout, run commands, and protocol-pinned compatibility files.
- [`experiments/`](experiments/): methods, result interpretation, and reproduction notes.
- [`bc_results/README.md`](bc_results/README.md): result data organization and links.
- [`scripts/README.md`](scripts/README.md): Python implementation folders.

## Experiment, code, report, and data map

Run commands from the repository root with `.venv/bin/python scripts/run.py <group>/<file.py> [arguments]`. For example: `.venv/bin/python scripts/run.py action_chunking/train_action_chunk_bc.py`. Python names in the table refer to implementations under [`scripts/`](scripts/) unless marked as root-level protocol-pinned sources. Some studies pin source filenames and SHA-256 hashes in their protocols, so those files remain at the repository root.

| Stage | Experiment | Python implementation | Report | Results |
|---|---|---|---|---|
| BC | BC / DAgger baseline: demonstration collection, BC training, fair-dataset comparison, and generalization evaluation. | `collect_demonstrations.py`, `collect_dagger.py`, `prepare_fair_datasets.py`, `train_behavior_cloning.py`, `evaluate_behavior_cloning.py`, `evaluate_generalization.py`, `analyze_fair_experiment.py` | [BC overview](experiments/behavior-cloning/README.md) | [baselines](bc_results/baselines/) |
| Residual SAC | Controlled residual SAC: compare sampling strategies and reward designs with a fixed BC policy. | `train_bc_residual_sac.py`, `evaluate_bc_residual_sac.py` | [EXPERIMENTS_SAC.md](experiments/residual-sac/EXPERIMENTS_SAC.md) | [residual_sac](bc_results/residual_sac/) |
| Residual SAC | Selective residual gate: evaluate applying SAC only when BC is likely to fail. | `analyze_selective_residual_gate.py`, `evaluate_selective_residual_sac.py` | [EXPERIMENTS_SELECTIVE_GATE.md](experiments/residual-sac/EXPERIMENTS_SELECTIVE_GATE.md) | [selective_gate](bc_results/selective_gate/) |
| Residual SAC | Selective gate multi-seed: reevaluate the selective gate across multiple BC/SAC seeds. | `analyze_selective_gate_multiseed.py` | [EXPERIMENTS_SELECTIVE_MULTISEED.md](experiments/residual-sac/EXPERIMENTS_SELECTIVE_MULTISEED.md) | [selective_gate](bc_results/selective_gate/) |
| Residual SAC | Policy-specific gate: calibrate a gate for each policy and validate it on independent seeds. | `analyze_policy_specific_gate.py`, `calibrate_policy_specific_gate.py`, `verify_policy_specific_gate_live.py` | [EXPERIMENTS_POLICY_SPECIFIC_GATE.md](experiments/residual-sac/EXPERIMENTS_POLICY_SPECIFIC_GATE.md) | [selective_gate](bc_results/selective_gate/) |
| Residual SAC | SAC training stability: measure performance variation across training seeds and checkpoints. | `run_sac_stability_training.py`, `run_sac_stability_evaluations.py`, `freeze_sac_checkpoint_selection.py`, `analyze_sac_stability.py`, `plot_sac_stability.py` | [EXPERIMENTS_SAC_TRAINING_STABILITY.md](experiments/residual-sac/EXPERIMENTS_SAC_TRAINING_STABILITY.md) | [sac_training_stability](bc_results/sac_training_stability/) |
| Residual SAC | SAC regression control: diagnose regressions from BC success after SAC and compare regularization. | `run_sac_regression_training.py`, `run_sac_regression_evaluations.py`, `freeze_sac_regression_selection.py`, `analyze_sac_regression_diagnostics.py`, `analyze_sac_regression_results.py`, `plot_sac_regression_results.py` | [EXPERIMENTS_SAC_REGRESSION_CONTROL.md](experiments/residual-sac/EXPERIMENTS_SAC_REGRESSION_CONTROL.md) | [sac_regression_control](bc_results/sac_regression_control/) |
| Residual SAC | SAC regularization replication: replicate the regularization comparison with independent training seeds. | `run_sac_regularization_replication_training.py`, `run_sac_regularization_replication_evaluations.py`, `freeze_sac_regularization_replication_selection.py`, `analyze_sac_regularization_replication.py` | [EXPERIMENTS_SAC_REGULARIZATION_REPLICATION.md](experiments/residual-sac/EXPERIMENTS_SAC_REGULARIZATION_REPLICATION.md) | [sac_regularization_replication](bc_results/sac_regularization_replication/) |
| Residual SAC | Action-chunked BC + SAC: compare H1/H4 BC with chunk-aware residual SAC. | `train_action_chunk_bc.py`, `train_action_chunk_sac.py`, `evaluate_action_chunking.py`, `freeze_action_chunk_selections.py`, `analyze_action_chunking.py`, `plot_action_chunking.py`, `write_action_chunking_report.py` | [EXPERIMENTS_ACTION_CHUNKED_BC_RESIDUAL_RL.md](experiments/residual-sac/EXPERIMENTS_ACTION_CHUNKED_BC_RESIDUAL_RL.md) | [action_chunking](bc_results/action_chunking/) |
| H4 replanning | Fixed H4 replanning ablation: compare performance and inference cost across fixed replanning intervals. | Root-pinned `evaluate_h4_replanning.py`; `analyze_h4_replanning.py`, `write_h4_replanning_report.py` | [EXPERIMENTS_H4_REPLANNING_ABLATION.md](experiments/h4-replanning/EXPERIMENTS_H4_REPLANNING_ABLATION.md) | [h4_replanning](bc_results/h4_replanning/) |
| H4 replanning | Observation-noise robustness: evaluate fixed-replanning policies under different observation-noise levels. | `evaluate_h4_observation_noise.py`, `audit_h4_observation_noise.py`, `analyze_h4_observation_noise.py`, `verify_h4_observation_noise_zero.py` | [EXPERIMENTS_H4_OBSERVATION_NOISE.md](experiments/h4-replanning/EXPERIMENTS_H4_OBSERVATION_NOISE.md) | [observation_noise](bc_results/h4_replanning/observation_noise/) |
| H4 replanning | Uncertainty-adaptive replanning: select R1 or R4 using observation-based uncertainty estimates. | `evaluate_h4_adaptive_replanning.py`, `analyze_h4_adaptive_calibration.py`, `analyze_h4_adaptive_replanning.py`, `audit_h4_adaptive_replanning_smoke.py`, `audit_h4_adaptive_replanning_final.py` | [EXPERIMENTS_H4_UNCERTAINTY_ADAPTIVE_REPLANNING.md](experiments/h4-replanning/EXPERIMENTS_H4_UNCERTAINTY_ADAPTIVE_REPLANNING.md) | [adaptive](bc_results/h4_replanning/adaptive/) |
| H4 replanning | Dynamic-noise replanning: evaluate online switching under changing noise. The study stopped at the stationarity gate documented in its report. | Root-pinned `evaluate_h4_dynamic_replanning.py`; `analyze_h4_dynamic_replanning.py`, `audit_h4_dynamic_replanning.py`, `diagnose_h4_dynamic_switches.py`, `validate_h4_dynamic_legacy.py` | [EXPERIMENTS_H4_DYNAMIC_UNCERTAINTY_REPLANNING.md](experiments/h4-replanning/EXPERIMENTS_H4_DYNAMIC_UNCERTAINTY_REPLANNING.md) | [dynamic](bc_results/h4_replanning/dynamic/) |
| H4 replanning | Motion-aware noise estimator: test separating motion-related variation from observation noise. The preregistered gate did not pass, so the study did not proceed to the next stage. | Root-pinned `run_h4_motion_estimator_stage1.py`, `analyze_h4_motion_estimator_stage1.py`; `audit_h4_motion_estimator_stage1.py` | [EXPERIMENTS_H4_MOTION_AWARE_NOISE_ESTIMATION.md](experiments/h4-noise-estimation/EXPERIMENTS_H4_MOTION_AWARE_NOISE_ESTIMATION.md) | [motion_aware](bc_results/h4_noise_estimation/motion_aware/) |
| H4 replanning | Model-based Kalman diagnostic: implementation and comparison preparation for a CV Kalman noise estimator. The report says no diagnostic Final was run for this work. | `compare_h4_noise_estimators.py`; shared estimator: root-pinned `kalman_noise_estimator.py` | [README_H4_MODEL_BASED_NOISE_ESTIMATION.md](experiments/h4-noise-estimation/README_H4_MODEL_BASED_NOISE_ESTIMATION.md) | No dedicated completed Final result. See the motion-aware folder for diagnostic data. |
| H4 replanning | Independent Kalman validation: generate, audit, and analyze estimator diagnostics on independent data. | `validate_h4_kalman_independent.py`; related audit and analysis use the `run_h4_motion_estimator_stage1.py` source family. | [README_H4_KALMAN_INDEPENDENT_VALIDATION.md](experiments/h4-noise-estimation/README_H4_KALMAN_INDEPENDENT_VALIDATION.md) | [independent_validation](bc_results/h4_noise_estimation/independent_validation/) |
| H4 replanning | Kalman online replanning: a closed-loop study using Kalman estimates for online replanning. | `kalman_online_experiment.py`; shared rules and controllers are root-pinned sources. | [README_H4_KALMAN_ONLINE_DYNAMIC.md](experiments/h4-noise-estimation/README_H4_KALMAN_ONLINE_DYNAMIC.md) | [kalman_online](bc_results/h4_noise_estimation/kalman_online/) |

## Folder guide

### Research documents, code, and data

| Path | Contents |
|---|---|
| [`experiments/behavior-cloning/`](experiments/behavior-cloning/) | BC/DAgger baseline guide. There is no dedicated long-form results report; the CSV files linked above contain the recorded results. |
| [`experiments/residual-sac/`](experiments/residual-sac/) | Reports for residual SAC, gates, stability, regression, regularization, and action chunking. |
| [`experiments/h4-replanning/`](experiments/h4-replanning/) | Reports for fixed, observation-noise, adaptive, and dynamic replanning. |
| [`experiments/h4-noise-estimation/`](experiments/h4-noise-estimation/) | Reports for motion-aware estimation, model-based diagnostics, independent Kalman validation, and online Kalman. |
| [`scripts/core/`](scripts/core/) | Shared environments, controllers, estimators, and reusable code. |
| [`scripts/behavior_cloning/`](scripts/behavior_cloning/) | Demonstration collection, BC/DAgger training, evaluation, and analysis. |
| [`scripts/residual_sac/`](scripts/residual_sac/) | Residual SAC, gates, evaluation, checkpoint selection, statistics, and plots. |
| [`scripts/action_chunking/`](scripts/action_chunking/) | H1/H4 chunked BC/SAC training, evaluation, and analysis. |
| [`scripts/h4_replanning/`](scripts/h4_replanning/) | H4 replanning evaluation, audits, analysis, and report generation. |
| [`scripts/h4_noise_estimation/`](scripts/h4_noise_estimation/) | Noise estimation, Kalman experiments, and validation. |
| [`scripts/other_tools/`](scripts/other_tools/) | Early Franka demos, diagnostics, visualization, and video conversion. |
| [`scripts/tests/`](scripts/tests/) | Former shared test folder. Current tests are located beside their implementations as `test_*.py`. |
| [`bc_results/baselines/`](bc_results/baselines/) | BC/DAgger, fair-dataset, and generalization CSVs and summaries. |
| [`bc_results/residual_sac/`](bc_results/residual_sac/) | Early residual SAC pilots and residual-scale comparisons. |
| [`bc_results/selective_gate/`](bc_results/selective_gate/) | Selective and policy-specific gate calibration, final results, and summaries. |
| [`bc_results/sac_training_stability/`](bc_results/sac_training_stability/) | SAC seed/checkpoint stability evaluations, audits, and plots. |
| [`bc_results/sac_regression_control/`](bc_results/sac_regression_control/) | Regression-control CSVs, summaries, and plots. |
| [`bc_results/sac_regularization_replication/`](bc_results/sac_regularization_replication/) | Regularization replication protocols, rollouts, summaries, and plots. |
| [`bc_results/action_chunking/`](bc_results/action_chunking/) | Chunking protocols, training runs, rollouts, analyses, and figures. |
| [`bc_results/h4_replanning/`](bc_results/h4_replanning/) | Fixed-interval H4 replanning rollouts, audits, and analyses. |
| [`bc_results/h4_replanning/observation_noise/`](bc_results/h4_replanning/observation_noise/) | Observation-noise robustness rollouts, summaries, and figures. |
| [`bc_results/h4_replanning/adaptive/`](bc_results/h4_replanning/adaptive/) | Adaptive replanning protocols and artifacts. |
| [`bc_results/h4_replanning/dynamic/`](bc_results/h4_replanning/dynamic/) | Dynamic replanning smoke runs, gate checks, and analyses. |
| [`bc_results/h4_noise_estimation/`](bc_results/h4_noise_estimation/) | Motion-aware and Kalman protocols and study folders. |
| [`bc_results/h4_noise_estimation/motion_aware/`](bc_results/h4_noise_estimation/motion_aware/) | Motion-aware Stage 1 data, audits, summaries, and figures. |
| [`bc_results/h4_noise_estimation/independent_validation/`](bc_results/h4_noise_estimation/independent_validation/) | Independent Kalman diagnostic outputs. |
| [`bc_results/h4_noise_estimation/kalman_online/`](bc_results/h4_noise_estimation/kalman_online/) | Online Kalman study protocol, audits, rollouts, and analyses. |

### Other top-level folders

| Path | Contents |
|---|---|
| `assets/` | Project images and static assets. |
| `demonstrations/` | Expert and DAgger training demonstrations. |
| `bc_models/` | BC checkpoints and model configurations. |
| `best_franka_model/`, `autonomous_best/`, `autonomous_v2_best/`, `bc_residual_sac_best/` | Local storage for selected trained models. |
| `autonomous_checkpoints/`, `bc_residual_sac_runs/` | Local training runs and checkpoints. |
| `autonomous_eval/`, `autonomous_v2_eval/`, `bc_residual_sac_eval/`, `eval_logs/` | Evaluation results and run logs. |
| `autonomous_logs/`, `autonomous_v2_logs/`, `bc_residual_sac_logs/`, `tensorboard_logs/` | Training and TensorBoard logs. |
| `mujoco_menagerie/` | MuJoCo Menagerie asset checkout; required by the local simulation environment. |

### Individual top-level files

| File | Purpose |
|---|---|
| [`action_chunking.py`](action_chunking.py) | Shared action-chunking code. |
| [`franka_ai_env.py`](franka_ai_env.py), [`franka_autonomous_env.py`](franka_autonomous_env.py) | Environments for the early Franka AI and autonomous-training code. |
| [`analyze_h4_motion_estimator_stage1.py`](analyze_h4_motion_estimator_stage1.py), [`run_h4_motion_estimator_stage1.py`](run_h4_motion_estimator_stage1.py) | Motion-aware estimator Stage 1 protocol-pinned sources. |
| [`evaluate_h4_replanning.py`](evaluate_h4_replanning.py), [`evaluate_h4_adaptive_replanning.py`](evaluate_h4_adaptive_replanning.py), [`evaluate_h4_observation_noise.py`](evaluate_h4_observation_noise.py), [`evaluate_h4_dynamic_replanning.py`](evaluate_h4_dynamic_replanning.py) | H4 evaluation compatibility entry points; some are also protocol-pinned sources. |
| [`h4_dynamic_controller.py`](h4_dynamic_controller.py), [`h4_kalman_online_rule.py`](h4_kalman_online_rule.py), [`kalman_noise_estimator.py`](kalman_noise_estimator.py), [`motion_aware_noise_estimator.py`](motion_aware_noise_estimator.py) | Shared H4/Kalman research sources referenced by study protocols. |
| [`bc_residual_sac_final.zip`](bc_residual_sac_final.zip), [`franka_autonomous_sac.zip`](franka_autonomous_sac.zip), [`franka_autonomous_v2_sac.zip`](franka_autonomous_v2_sac.zip), [`franka_sac.zip`](franka_sac.zip) | Compressed models from earlier training runs. |
| [`franka_ai_comparison.csv`](franka_ai_comparison.csv), [`grasp_results.csv`](grasp_results.csv) | Comparison and grasp-result CSVs from early Franka experiments. |

## Protocol-pinned source hashes

A protocol's `frozen_source_sha256` values are SHA-256 fingerprints of the exact source bytes expected by that study. SHA-256 changes even when a source edit only changes an output path. The two Kalman protocols were created against earlier source versions. A later organization change moved result folders and updated three path constants, so these current files no longer match the historical fingerprints:

| Protocol | Source | Recorded SHA-256 | Current SHA-256 | Change |
|---|---|---|---|---|
| Independent Kalman validation | `action_chunking.py` | `442217081eaa59a0560f9fcd8ddf56cf537f3a6b55a7e5c8a7f8f47c5ccb9d02` | `8aeaf7a3e50da77fb7f4993367772d5fd66074d8497b6298c0ab9bc660437a0f` | Updated result directory paths. |
| Independent Kalman validation | `run_h4_motion_estimator_stage1.py` | `f98772f9fc842326fd3a4e0014f3dfdb5ff2779cf89ca81d4d993334089869b2` | `3e2ee94c2fdfe5aed357817ae27b7f131810dda0bb7149698516a40cc69c58a3` | Updated result directory paths. |
| Kalman online replanning | `action_chunking.py` | `442217081eaa59a0560f9fcd8ddf56cf537f3a6b55a7e5c8a7f8f47c5ccb9d02` | `8aeaf7a3e50da77fb7f4993367772d5fd66074d8497b6298c0ab9bc660437a0f` | Updated result directory paths. |
| Kalman online replanning | `evaluate_h4_dynamic_replanning.py` | `8dec46589bf11d28ec5dc2126bc6228cdb05b504ee27526fb925c591a5a3231d` | `8eec880b3120ed0c4c6c560f9569cf5b62c62d164e45eae0c9e57e0263c39550` | Updated result directory paths. |

The historical hashes match the source files at the protocols' recorded base commits for these changed files. This is not evidence of damaged files; it means the current `main` versions are not byte-for-byte identical to the preregistered versions. Reproducing or auditing a study against those protocols requires using the pinned source versions. Do not replace the recorded hashes without explicitly amending the protocol and preserving its history.

## Top-level reference files

| File | Purpose |
|---|---|
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Concise experiment index and links to results. |
| [`PYTHON.md`](PYTHON.md) | Python source layout and run instructions. |
| [`scripts/run.py`](scripts/run.py) | Shared runner for Python files in role-specific folders. |

`.venv/`, `.git/`, and `__pycache__/` are environment, Git, and Python cache directories. Some training logs, checkpoints, videos, and MuJoCo assets are ignored by Git or depend on the local setup. Each `README.md` explains the naming and purpose of files in its folder.
