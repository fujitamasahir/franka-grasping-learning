# Action-Chunked Behavior Cloning and Residual SAC

## Abstract

This preregistered study compares single-step (H1) and four-step (H4) state-conditioned MLP behavior cloning, then applies an always-on, stepwise SAC XYZ residual to each frozen BC policy. Demonstrations, valid chunk starts, robot dynamics, evaluation positions, action limits, reward terms, and checkpoint selection were fixed before training. Final conclusions are intentionally generated only after the disjoint Final set was evaluated. The study tests a small deterministic-state MLP on one simulated Franka task; it is not a reproduction of ACT or Diffusion Policy.

## Motivation and Related Work

H4 predicts four future actions from the current 12D state and executes them open loop while physics still advances at every control step. This can lower BC replanning cost but lets state error accumulate within a chunk. SAC observes the current state, the exact next planned BC action, and chunk phase, and can add a bounded XYZ correction on every step. The design tests whether that closed-loop residual changes the reliability of H4 relative to H1 under matched starts.

Ankile et al. describe ResiP as residual RL for precise assembly, using a learned imitation policy with residual refinement ([arXiv:2407.16677](https://arxiv.org/abs/2407.16677)). Their setting motivates correcting a nominal policy during execution. Ankile et al. also study residual off-policy RL for fine-tuning BC ([arXiv:2509.19301](https://arxiv.org/abs/2509.19301)). This experiment differs by freezing the BC policy, learning only a small per-step XYZ residual, and explicitly exposing the current action-chunk state. It uses a lightweight state-conditioned chunked MLP, not ACT or Diffusion Policy.

## Research Questions

1. Does H4 action chunking change BC grasp success under matched training episodes and starting observations?
2. Does stepwise residual SAC improve H4 more reliably than H1 relative to each method's own BC baseline?
3. How do rescue, regression, and seed variability change with chunking?
4. Does H4 reduce BC replanning while SAC continues acting every control step?

## Hypotheses

Chunking may improve temporal consistency or reduce closed-loop responsiveness. Stepwise residual SAC may correct errors within an open-loop chunk. H4 plus SAC may have a different rescue/regression balance than H1 plus SAC. BC and SAC training seeds may interact. No method is assumed to win.

## Dataset Audit and Data Integrity

The study uses the original unsampled `demonstrations/franka_expert_train.npz` and `demonstrations/franka_expert_wide.npz`: 200 and 100 additional successful episodes, respectively, with 35,377 observations total. Observation/action row counts aligned; all 12D observations and 4D actions were finite; gripper labels were `+1=open`, `-1=closed`; `max_move=0.15 m`, `max_time=16 s`, and `control_dt=0.1 s` matched. Within every episode, rows were contiguous and normalized observation time advanced by 0.1 s within 1e-4 s. The original wide file had 100 additional trajectories and no exact duplicates of the narrow file. Source hashes and the full audit are in `bc_results/action_chunking_data_audit.json`.

The historical `fair_wide_expert.npz` is excluded from chunk targets because random row subsampling left 73 nonconsecutive control intervals. Episode IDs were remapped for the wide source, then 30 of the 300 episodes were held out with fixed NumPy seed 20261106. H1 and H4 use the same 31,029 training windows and 3,448 validation starts; windows never cross episode boundaries or temporal gaps. H1 supervises one action per start (31,029 targets); H4 supervises four (124,116 targets). This extra future-action supervision is a method difference. The split lists, hashes, and audit are committed in the preregistration.

## Single-Step vs Action-Chunked BC

Both policies take 12D observations and use a 128-128 ReLU MLP. H1 predicts one tanh XYZ action plus one gripper logit and replans each step. H4 predicts four XYZ actions and four gripper logits, caching and executing the sequence for four consecutive physics steps. Both use Adam, learning rate 3e-4, batch 256, maximum 100 epochs, the same train-only observation normalization, SmoothL1 XYZ loss plus 0.25 BCE gripper loss, and early stopping on held-out episodes (patience 15, minimum loss improvement 1e-6). Three seeds (42, 123, 456) were trained per horizon. H1 has 18,692 parameters and H4 has 20,240.

## Chunk-Aware Residual SAC

The SAC observation has 17 values: environment state (12), the exact planned action being corrected (4), and normalized chunk phase (1; always 0 for H1 and 0, 1/3, 2/3, 1 for H4). SAC supplies only XYZ residuals, clipped to ±5 mm per axis; BC retains gripper control and its parameters remain frozen. SAC acts every control step, including within H4 chunks. No spatial gate is used. Condition C was retained: 30,000 steps, checkpoints at 10k/20k/30k, 90% hard samples outside the ±12 cm inner square, shaped reward, residual penalty coefficient 0.005, and SAC seeds 1103/1117/1123. Stable-Baselines3 SAC uses CPU, MLP [256,256], learning rate 3e-4, replay buffer 50,000, learning starts 1,000, batch 128, gamma .99, train frequency 1, gradient steps 1, and automatic entropy coefficient. The 18 runs cross 2 horizons × 3 BC seeds × 3 SAC seeds; these are not 18 independent BC training replications.

## Experimental Protocol

BC architecture validation uses shared seeds 270000–270499. SAC checkpoint validation and its paired BC regression baseline use the same shared positions 280000–280499. A separate 40-position training diagnostic panel (300000–300039) does not select checkpoints. Selection is highest validation success, then fewest regressions against the same-position BC, shortest successful episode, then earliest checkpoint. Selection hashes were frozen before Final evaluation. Final uses shared seeds 290000–291999. BC Final rollouts are reused across associated SAC policies. Final fixed-30k and selected policies are evaluated separately, except identical checkpoints are evaluated once and reused. Episode outputs are append-resumable CSVs. Source code, test outputs, data provenance, model hashes, and selection hashes are recorded with the artifacts.

## BC Results

Final BC success, mean episode BC calls, inference latency, and episode execution time are summarized below. The ± term is the sample standard deviation across the three BC training seeds; it is descriptive, not a population-level estimate.

| Policy | Success mean ± SD | BC calls / episode | BC latency (ms/call) | Episode time (s) |
|---|---|---|---|---|
| H1 | 83.42% ± 7.30% | 127.2 | 0.034 | 0.374 |
| H4 | 94.37% ± 1.18% | 30.8 | 0.040 | 0.352 |

| Horizon | BC seed | Final success | Successes | Mean steps | Mean first close step |
|---|---|---|---|---|---|
| H1 | 42 | 88.00% | 1760 | 126.55 | 70.83 |
| H1 | 123 | 75.00% | 1500 | 129.91 | 71.87 |
| H1 | 456 | 87.25% | 1745 | 125.19 | 71.24 |
| H4 | 42 | 93.95% | 1879 | 120.55 | 70.91 |
| H4 | 123 | 93.45% | 1869 | 122.45 | 66.29 |
| H4 | 456 | 95.70% | 1914 | 121.92 | 70.71 |

The historical `fair_wide_expert_seed42.pt` is not used as a matched baseline: it was trained from a different, randomly subsampled dataset and does not provide temporally reliable H4 targets. Its prior results are contextual only.

During validation implementation QA, an initial partial set used the environment defaults (`max_move=0.025 m`, `max_time=12 s`) instead of checkpoint values (`0.15 m`, `16 s`). These rollouts were invalidated and excluded; all six corrected BC validations were rerun with the checkpoint settings. The correction log records hashes and episode counts. No Final seed or training/model-selection decision was exposed to these discarded rollouts.

The first SAC validation callback also exposed that a paired regression count requires BC outcomes on the same 280000–280499 positions. The callback stopped before computing a selection statistic; that single 10k training attempt and its rollouts were preserved and excluded. The protocol amendment adds the matched BC-only evaluation on those SAC validation positions. The 18-run crossed design and all other conditions remain unchanged, and no Final outcome was inspected.

## SAC Results

Selected-checkpoint absolute success, success gain relative to the matching BC policy, rescue, regression, and conditional rates for every crossed seed cell:

| H | BC seed | SAC seed | Selected step | Success | Gain (pp) | Rescue | Regression | Net | Reg./BC success | Rescue/BC fail |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | 42 | 1103 | 30000 | 88.90% | +0.90 | 101 | 83 | 18 | 0.05 | 0.42 |
| H1 | 42 | 1117 | 10000 | 88.80% | +0.80 | 27 | 11 | 16 | 0.01 | 0.11 |
| H1 | 42 | 1123 | 20000 | 90.80% | +2.80 | 93 | 37 | 56 | 0.02 | 0.39 |
| H1 | 123 | 1103 | 20000 | 88.75% | +13.75 | 283 | 8 | 275 | 0.01 | 0.57 |
| H1 | 123 | 1117 | 20000 | 81.25% | +6.25 | 235 | 110 | 125 | 0.07 | 0.47 |
| H1 | 123 | 1123 | 20000 | 89.90% | +14.90 | 301 | 3 | 298 | 0.00 | 0.60 |
| H1 | 456 | 1103 | 10000 | 88.05% | +0.80 | 16 | 0 | 16 | 0.00 | 0.06 |
| H1 | 456 | 1117 | 10000 | 87.05% | -0.20 | 8 | 12 | -4 | 0.01 | 0.03 |
| H1 | 456 | 1123 | 10000 | 88.05% | +0.80 | 17 | 1 | 16 | 0.00 | 0.07 |
| H4 | 42 | 1103 | 30000 | 96.35% | +2.40 | 57 | 9 | 48 | 0.00 | 0.47 |
| H4 | 42 | 1117 | 20000 | 96.15% | +2.20 | 48 | 4 | 44 | 0.00 | 0.40 |
| H4 | 42 | 1123 | 30000 | 95.90% | +1.95 | 39 | 0 | 39 | 0.00 | 0.32 |
| H4 | 123 | 1103 | 10000 | 93.55% | +0.10 | 5 | 3 | 2 | 0.00 | 0.04 |
| H4 | 123 | 1117 | 10000 | 93.35% | -0.10 | 7 | 9 | -2 | 0.00 | 0.05 |
| H4 | 123 | 1123 | 10000 | 94.05% | +0.60 | 15 | 3 | 12 | 0.00 | 0.11 |
| H4 | 456 | 1103 | 10000 | 95.75% | +0.05 | 4 | 3 | 1 | 0.00 | 0.05 |
| H4 | 456 | 1117 | 20000 | 95.30% | -0.40 | 13 | 21 | -8 | 0.01 | 0.15 |
| H4 | 456 | 1123 | 30000 | 97.55% | +1.85 | 40 | 3 | 37 | 0.00 | 0.47 |

Fixed 30k results are shown separately so validation selection effects remain visible:

| H | BC seed | SAC seed | Success | Gain (pp) | Rescue | Regression | Net |
|---|---|---|---|---|---|---|---|
| H1 | 42 | 1103 | 88.90% | +0.90 | 101 | 83 | 18 |
| H1 | 42 | 1117 | 90.20% | +2.20 | 106 | 62 | 44 |
| H1 | 42 | 1123 | 86.70% | -1.30 | 62 | 88 | -26 |
| H1 | 123 | 1103 | 84.70% | +9.70 | 255 | 61 | 194 |
| H1 | 123 | 1117 | 73.45% | -1.55 | 118 | 149 | -31 |
| H1 | 123 | 1123 | 86.20% | +11.20 | 245 | 21 | 224 |
| H1 | 456 | 1103 | 87.25% | +0.00 | 44 | 44 | 0 |
| H1 | 456 | 1117 | 86.55% | -0.70 | 30 | 44 | -14 |
| H1 | 456 | 1123 | 84.90% | -2.35 | 50 | 97 | -47 |
| H4 | 42 | 1103 | 96.35% | +2.40 | 57 | 9 | 48 |
| H4 | 42 | 1117 | 93.15% | -0.80 | 4 | 20 | -16 |
| H4 | 42 | 1123 | 95.90% | +1.95 | 39 | 0 | 39 |
| H4 | 123 | 1103 | 92.70% | -0.75 | 13 | 28 | -15 |
| H4 | 123 | 1117 | 84.30% | -9.15 | 1 | 184 | -183 |
| H4 | 123 | 1123 | 88.60% | -4.85 | 15 | 112 | -97 |
| H4 | 456 | 1103 | 94.50% | -1.20 | 10 | 34 | -24 |
| H4 | 456 | 1117 | 91.85% | -3.85 | 11 | 88 | -77 |
| H4 | 456 | 1123 | 97.55% | +1.85 | 40 | 3 | 37 |

The primary differential benefit is computed from each architecture's own BC baseline:

| Checkpoint family | BC seed | SAC seed | Gain H1 (pp) | Gain H4 (pp) | Difference-in-gains (pp) |
|---|---|---|---|---|---|
| fixed30k | 42 | 1103 | +0.90 | +2.40 | +1.50 |
| fixed30k | 42 | 1117 | +2.20 | -0.80 | -3.00 |
| fixed30k | 42 | 1123 | -1.30 | +1.95 | +3.25 |
| fixed30k | 123 | 1103 | +9.70 | -0.75 | -10.45 |
| fixed30k | 123 | 1117 | -1.55 | -9.15 | -7.60 |
| fixed30k | 123 | 1123 | +11.20 | -4.85 | -16.05 |
| fixed30k | 456 | 1103 | +0.00 | -1.20 | -1.20 |
| fixed30k | 456 | 1117 | -0.70 | -3.85 | -3.15 |
| fixed30k | 456 | 1123 | -2.35 | +1.85 | +4.20 |
| selected | 42 | 1103 | +0.90 | +2.40 | +1.50 |
| selected | 42 | 1117 | +0.80 | +2.20 | +1.40 |
| selected | 42 | 1123 | +2.80 | +1.95 | -0.85 |
| selected | 123 | 1103 | +13.75 | +0.10 | -13.65 |
| selected | 123 | 1117 | +6.25 | -0.10 | -6.35 |
| selected | 123 | 1123 | +14.90 | +0.60 | -14.30 |
| selected | 456 | 1103 | +0.80 | +0.05 | -0.75 |
| selected | 456 | 1117 | -0.20 | -0.40 | -0.20 |
| selected | 456 | 1123 | +0.80 | +1.85 | +1.05 |

| Horizon | Mean selected success | Mean gain (pp) | Mean rescue | Mean regression | BC calls / ep | SAC activation | SAC calls / ep | Residual norm (mm) | Saturation | Intervention |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | 87.95% | +4.53 | 120.1 | 29.4 | 124.3 | 100.0% | 124.3 | 2.70 | 5.31% | 100.00% |
| H4 | 95.33% | +0.96 | 25.3 | 6.1 | 30.5 | 100.0% | 120.6 | 2.45 | 2.84% | 100.00% |

## Rescue and Regression

For each rollout pair, rescue is a BC failure corrected to success, and regression is a BC success changed to failure. Their difference equals the paired success-count difference. Conditional rates use BC-failure and BC-success denominators respectively. H1 and H4 have different baseline failure sets, so all residual results are paired only with their own BC policy. `bc_results/action_chunking_analysis/final_cell_metrics.csv` contains exact per-cell denominators and counts.

Regional paired rescue/regression counts for selected checkpoints:

| Horizon | Region | Episodes across crossed cells | Rescue | Regression | Net |
|---|---|---|---|---|---|
| H1 | EN | 4617 | 291 | 28 | 263 |
| H1 | ES | 4257 | 201 | 60 | 141 |
| H1 | WN | 4653 | 278 | 75 | 203 |
| H1 | WS | 4473 | 311 | 102 | 209 |
| H4 | EN | 4617 | 108 | 3 | 105 |
| H4 | ES | 4257 | 31 | 27 | 4 |
| H4 | WN | 4653 | 46 | 12 | 34 |
| H4 | WS | 4473 | 43 | 13 | 30 |

## Training-Seed Variability

The selected-policy mean gains by BC or SAC seed, averaged over the crossed other factor, are listed below. The SD column is variation over the three values of that other factor, not an independent replication estimate.

| H | Factor | Seed | Mean gain (pp) | SD over crossed other factor |
|---|---|---|---|---|
| H1 | bc_seed | 42 | +1.50 | 1.13 |
| H1 | bc_seed | 123 | +11.63 | 4.70 |
| H1 | bc_seed | 456 | +0.47 | 0.58 |
| H1 | sac_seed | 1103 | +5.15 | 7.45 |
| H1 | sac_seed | 1117 | +2.28 | 3.47 |
| H1 | sac_seed | 1123 | +6.17 | 7.63 |
| H4 | bc_seed | 42 | +2.18 | 0.23 |
| H4 | bc_seed | 123 | +0.20 | 0.36 |
| H4 | bc_seed | 456 | +0.50 | 1.19 |
| H4 | sac_seed | 1103 | +0.85 | 1.34 |
| H4 | sac_seed | 1117 | +0.57 | 1.42 |
| H4 | sac_seed | 1123 | +1.47 | 0.75 |

Checkpoint selection chose steps 10000, 20000, 30000 across 18 policies. The mean final selected-minus-fixed30k success difference was +2.54 percentage points (range -1.40 to +9.05). See `checkpoint_selection_effect.csv` for all paired rows. Validation success and Final success are separate measurements; selection was frozen using validation only.

## Computational Cost

H1 replans BC every step. H4 replans once per four steps, so the measured BC call frequency should be interpreted with per-call latency and episode time. SAC remains always-on: its inference-call rate is reported, and this experiment makes no SAC-call-saving claim. Residual norm, saturation, intervention rate, gripper-close timing, lift timing, and chunk phase at events are in the cell CSV. The selected H4 trajectory plot is a single preregistered illustrative seed and is not representative.

| Horizon | BC calls / episode | BC latency (ms) | SAC latency (ms) | Residual episode time (s) | SAC calls / episode |
|---|---|---|---|---|---|
| H1 | 127.2 | 0.034 | 0.098 | 0.384 | 124.3 |
| H4 | 30.8 | 0.040 | 0.099 | 0.369 | 120.6 |

BC optimization time by training seed:

| Horizon | BC seed | Training seconds | Best epoch |
|---|---|---|---|
| H1 | 42 | 10.14 | 100 |
| H1 | 123 | 10.18 | 99 |
| H1 | 456 | 10.30 | 100 |
| H4 | 42 | 10.59 | 100 |
| H4 | 123 | 10.70 | 99 |
| H4 | 456 | 10.70 | 100 |

Total accumulated SAC training wall time across completed runs: **3.92 hours** (18 complete runs with runtime records). Per-run durations are stored in `training_runtime.json`.

## Statistical Analysis

Exact two-sided McNemar tests compare paired final positions for each BC-vs-SAC cell and H1-vs-H4 BC seed. Holm corrections are applied within the preregistered architecture/checkpoint families. 25 of 36 paired BC-vs-SAC tests were Holm-adjusted below .05; 3 of the 3 H1-vs-H4 BC comparisons were below .05 after Holm correction. These tests quantify paired position outcomes for these trained models, not variation over model training.

The primary differential SAC effect is `Gain_H4 − Gain_H1`, each gain measured against its own matched BC rollout. A 10,000-replicate hierarchical bootstrap resamples the 3 BC seed indices and 3 SAC seed indices independently and forms the crossed 3×3 cells. It preserves the crossed training structure; shared positions remain paired within cells.

| Checkpoint family | Mean difference-in-gains (pp) | Across-cell SD (pp) | Hierarchical 95% interval (pp) |
|---|---|---|---|
| fixed30k | -3.61 | 6.69 | [-10.42, 2.58] |
| selected | -3.57 | 6.35 | [-11.22, 0.83] |

## Discussion

The H1/H4 BC comparison uses identical demonstration episodes, chunk-valid start observations, and Final positions. If H4 improves BC success, the result is specific to this horizon-four deterministic state MLP and includes fourfold future-action supervision per start. If H4 underperforms, reduced closed-loop responsiveness, optimization or data limitations, or execution implementation should be considered; this does not refute action chunking generally.

Residual SAC is judged by both rescue and regression against each architecture's own BC baseline. A positive gain can still contain substantial regressions, while low regression can coexist with limited rescue. Checkpoint-selected results are selection-conditioned and should be read alongside fixed-30k outcomes and validation curves. With only three BC and three SAC seeds, the crossed cell variation is large relative to what can support broad generalization.

## Limitations

- One simulated Franka grasping task and one short horizon (H=4).
- State-conditioned MLP, not ACT or Diffusion Policy.
- Demonstrations are successful scripted-expert trajectories.
- H4 has four times as many future-action labels per valid start.
- Only three BC seeds and three SAC seeds; seed-population uncertainty is weakly identified.
- Thousands of positions are repeated across policies and do not create independent training replications.
- SAC is always active, and results are simulation-only.
- Checkpoint selection uses one shared 500-position validation set and can be noisy.

## Implications for ResiP / ResFiT

The design shares the residual-control idea with ResiP: preserve a nominal imitation policy and use feedback to refine execution. Its H4 arm makes an explicit four-action open-loop plan, with SAC correcting each next action. The study does not reproduce ResiP's assembly setup or claim comparable performance. Relative to ResFiT's residual off-policy BC fine-tuning direction, this study keeps BC frozen and isolates chunking under fixed SAC settings; it does not test residual policy parameterization or online BC fine-tuning. Any observed benefit is therefore evidence about this controlled implementation only.

## Next Research Questions

Use these results to decide whether a small ACT or diffusion-policy baseline is warranted. A follow-up should retain matched demonstration starts, include a properly capacity-matched sequence model and multiple horizons, and preserve paired residual comparisons. If H4 gains are weak or seed-sensitive, first test whether the limitation is chunk execution or model expressivity before scaling to a larger policy family. Include additional independent BC/SAC seeds before population-level claims.

## Reproduction Instructions

1. Use branch `experiment/action-chunked-bc-residual-rl` at source base `10d607fe7cacaae3004c8a291348c5022e41f4a9`.
2. Verify the protocol SHA in `bc_results/action_chunking_protocol.sha256` and the dataset audit hashes.
3. Run `.venv/bin/python -m unittest discover -p 'test_*.py' -v`.
4. Train BC: `.venv/bin/python train_action_chunk_bc.py`.
5. Generate BC architecture validation rollouts: `.venv/bin/python evaluate_action_chunking.py --phase bc-validation`; generate paired BC checkpoint-selection baselines: `.venv/bin/python evaluate_action_chunking.py --phase bc-sac-validation`.
6. Train the 18 crossed SAC runs: `.venv/bin/python train_action_chunk_sac.py`; resume an interrupted run with `--resume --arch H1 --bc-seed 42 --sac-seed 1103` (choose the interrupted tuple).
7. Freeze validation selections: `.venv/bin/python freeze_action_chunk_selections.py`.
8. Run BC Final and then SAC Final: `.venv/bin/python evaluate_action_chunking.py --phase bc-final`; `.venv/bin/python evaluate_action_chunking.py --phase sac-final`.
9. Analyze and create report figures: `.venv/bin/python analyze_action_chunking.py`; `.venv/bin/python plot_action_chunking.py`; `.venv/bin/python write_action_chunking_report.py`.

Protocol SHA-256: `90eec69f7852c34f2685cef291b0a03a5e8dcd8885279ba279ef61ad21a09638`. Selection manifest SHA-256 is recorded in `bc_results/action_chunking_checkpoint_selections.sha256`. Final tables and analysis live under `bc_results/action_chunking_analysis/`; SVG/PNG figures are under `bc_results/action_chunking_figures/`.
