# Selective Residual SAC Across Training Seeds

## 1. Research Question and Hypothesis

**Research question.** Does selective residual reinforcement learning consistently preserve successful imitation-learned behavior and improve grasping performance across different SAC training seeds?

The primary hypothesis was that the frozen Safety Gate (SAC only in EN) would retain BC successes while rescuing some BC failures for each SAC training seed. The secondary hypothesis was that the frozen Balanced Gate (SAC in EN and WS) would improve BC success more often than it regressed it across seeds. These hypotheses concern the fixed gates selected on calibration data; no gate was refit using this evaluation.

## 2. Experimental Setup

- Branch: `experiment/selective-residual-gate`.
- BC checkpoint: `bc_models/fair_wide_expert_seed42.pt` for every arm.
- SAC checkpoints: Condition C (hard sampling, shaped reward), frozen `best_model.zip` checkpoints for training seeds 42, 123, and 456.
- SAC checkpoints were present at `bc_residual_sac_runs/C_hard_shaped_seed{42,123,456}/best_model.zip`; their configs specify the same BC model, 0.20 m position range, 0.005 m residual bound, hard sampling, shaped reward, 30,000 steps, and the corresponding SAC training seed.
- Four methods per SAC seed: BC only, always BC+SAC, frozen Safety Gate, and frozen Balanced Gate.
- Each SAC checkpoint was evaluated on the same 2,000 initial positions, seeds 170000–171999. The existing paired evaluator ran both BC and BC+SAC for every position. Maximum residual correction was 5 mm per axis and the position range was ±20 cm.
- The existing `bc_results/selective_gate_exploratory1000_gate.json` was used unchanged: Safety = EN; Balanced = EN and WS. Its calibration remains the earlier Seed 456 data; it was not refit here.
- `test_selective_residual_gate.py`: 5 tests passed.
- The three evaluation files were checked to have identical seed lists, cube coordinates, and BC outcomes. The repeated positions are shared observations, not three independent samples.
- Exact two-sided McNemar tests compare paired binary outcomes. Holm adjustment is reported across the three SAC training seeds separately for each gate versus BC family and each gate versus Always SAC family. These tests are conditional on the tested checkpoints and positions; they do not establish a population-level effect over SAC training seeds.

## 3. Results

Counts are out of 2,000 episodes per SAC training seed. Rescue and regression counts are relative to BC on the same episode. Net improvement is rescue minus regression. Success differences are percentage points (pp).

| SAC train seed | Method | Success | Rescue | Regression | Net | SAC uses | Activation | Δ vs BC | Δ vs Always SAC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | BC only | 1850/2000 (92.50%) | 0 | 0 | 0 | 0 | 0.0% | 0.00 pp | +0.10 pp |
| 42 | Always SAC | 1848/2000 (92.40%) | 28 | 30 | -2 | 2000 | 100.0% | -0.10 pp | 0.00 pp |
| 42 | Safety Gate | 1846/2000 (92.30%) | 1 | 5 | -4 | 338 | 16.9% | -0.20 pp | -0.10 pp |
| 42 | Balanced Gate | 1869/2000 (93.45%) | 24 | 5 | +19 | 672 | 33.6% | +0.95 pp | +1.05 pp |
| 123 | BC only | 1850/2000 (92.50%) | 0 | 0 | 0 | 0 | 0.0% | 0.00 pp | +0.30 pp |
| 123 | Always SAC | 1844/2000 (92.20%) | 30 | 36 | -6 | 2000 | 100.0% | -0.30 pp | 0.00 pp |
| 123 | Safety Gate | 1850/2000 (92.50%) | 4 | 4 | 0 | 338 | 16.9% | 0.00 pp | +0.30 pp |
| 123 | Balanced Gate | 1830/2000 (91.50%) | 11 | 31 | -20 | 672 | 33.6% | -1.00 pp | -0.70 pp |
| 456 | BC only | 1850/2000 (92.50%) | 0 | 0 | 0 | 0 | 0.0% | 0.00 pp | -1.80 pp |
| 456 | Always SAC | 1886/2000 (94.30%) | 53 | 17 | +36 | 2000 | 100.0% | +1.80 pp | 0.00 pp |
| 456 | Safety Gate | 1882/2000 (94.10%) | 32 | 0 | +32 | 338 | 16.9% | +1.60 pp | -0.20 pp |
| 456 | Balanced Gate | 1894/2000 (94.70%) | 44 | 0 | +44 | 672 | 33.6% | +2.20 pp | +0.40 pp |

Both gates used SAC less often than Always SAC on all three checkpoints because their choices depend only on the same positions: Safety used SAC 338 times (83.1% fewer activations) and Balanced 672 times (66.4% fewer). Their performance, however, varied with the checkpoint.

## 4. Statistical Analysis

Exact two-sided McNemar tests; `wins/losses` are discordant episode outcomes for the first method listed versus the comparator.

| SAC train seed | Comparison | Wins / losses | Exact p | Holm p across 3 seeds |
|---:|---|---:|---:|---:|
| 42 | Always SAC vs BC | 28 / 30 | 0.8957 | — |
| 42 | Safety vs BC | 1 / 5 | 0.2188 | 0.4375 |
| 42 | Balanced vs BC | 24 / 5 | 0.000546 | 0.00109 |
| 42 | Safety vs Always SAC | 25 / 27 | 0.8899 | 1.000 |
| 42 | Balanced vs Always SAC | 25 / 4 | 0.000104 | 0.000311 |
| 123 | Always SAC vs BC | 30 / 36 | 0.5386 | — |
| 123 | Safety vs BC | 4 / 4 | 1.0000 | 1.000 |
| 123 | Balanced vs BC | 11 / 31 | 0.00289 | 0.00289 |
| 123 | Safety vs Always SAC | 32 / 26 | 0.5118 | 1.000 |
| 123 | Balanced vs Always SAC | 5 / 19 | 0.00661 | 0.0132 |
| 456 | Always SAC vs BC | 53 / 17 | 0.0000192 | — |
| 456 | Safety vs BC | 32 / 0 | 4.66e-10 | 1.40e-9 |
| 456 | Balanced vs BC | 44 / 0 | 1.14e-13 | 3.41e-13 |
| 456 | Safety vs Always SAC | 17 / 21 | 0.6271 | 1.000 |
| 456 | Balanced vs Always SAC | 17 / 9 | 0.1686 | 0.1686 |

After adjustment within each three-seed gate family, Safety was significantly better than BC only for SAC seed 456. Balanced differed from BC for all three checkpoints, but the direction was positive for 42 and 456 and negative for 123. Balanced exceeded Always SAC for seed 42, fell below it for seed 123, and had no statistically clear difference for seed 456. The episode-level p-values are conditional on the checkpoints and shared evaluation locations; the three seed-specific tests are not three independent location samples.

## 5. Discussion

### Gate behavior by outer workspace quadrant

Each cell gives `rescue/regression` relative to BC, restricted to positions outside the demonstration square. A zero pair for a gate indicates SAC was not activated in that quadrant.

| SAC seed | Method | EN | ES | WN | WS |
|---:|---|---:|---:|---:|---:|
| 42 | Always SAC | 1/5 | 1/20 | 3/5 | 23/0 |
| 42 | Safety Gate | 1/5 | 0/0 | 0/0 | 0/0 |
| 42 | Balanced Gate | 1/5 | 0/0 | 0/0 | 23/0 |
| 123 | Always SAC | 4/4 | 14/0 | 5/5 | 7/27 |
| 123 | Safety Gate | 4/4 | 0/0 | 0/0 | 0/0 |
| 123 | Balanced Gate | 4/4 | 0/0 | 0/0 | 7/27 |
| 456 | Always SAC | 32/0 | 4/15 | 5/2 | 12/0 |
| 456 | Safety Gate | 32/0 | 0/0 | 0/0 | 0/0 |
| 456 | Balanced Gate | 32/0 | 0/0 | 0/0 | 12/0 |

1. **Safety did not consistently suppress regressions.** It produced 5, 4, and 0 regressions for seeds 42, 123, and 456. The calibration rule's zero-regression criterion was met by EN on the calibration checkpoint, but it did not guarantee transfer to other checkpoints.
2. **Balanced did not improve BC for every SAC seed.** It yielded net +19 and +44 episodes for seeds 42 and 456, but net -20 for seed 123. The frozen WS activation captured 23 rescues without regressions for seed 42, but for seed 123 WS yielded 7 rescues and 27 regressions. Seed 456 had 12 WS rescues and no regressions.
3. **Gate effects vary substantially by SAC training seed.** Safety net changes ranged from -4 to +32 episodes; Balanced ranged from -20 to +44. The underlying Always SAC net changes ranged from -6 to +36. This is evidence that the residual policy's quadrant-specific benefit is checkpoint-dependent.
4. **The earlier seed 456 result recurred on new locations.** On the previous 150000–151999 evaluation, Safety had 25 rescues and 0 regressions, while Balanced had 44 rescues and 1 regression. On the untouched 170000–171999 locations, seed 456 again showed positive net changes: Safety 32/0 and Balanced 44/0. Exact rates changed (BC was 92.95% previously and 92.50% now), so this is a directional replication for seed 456, not evidence of robustness across training seeds.
5. **SAC use was reduced substantially.** Safety used SAC on 16.9% of positions and Balanced on 33.6%, versus 100% for Always SAC. In this fixed test, that reduced activation count by 1,662 and 1,328 episodes, respectively. Balanced's success was close to Always SAC only for seeds 42 and 456; it was lower for seed 123.

## 6. Limitations

- Only three SAC training seeds were evaluated. Seed-to-seed variability is descriptive and too sparsely sampled to characterize the training-seed population reliably.
- All three SAC models were evaluated on exactly the same initial positions, and the BC outcomes were identical. This pairing improves comparisons but makes the model evaluations correlated. We do not pool the three 2,000-episode sets as 6,000 independent observations.
- The Safety and Balanced quadrant choices were calibrated using Seed 456 data. Although the final positions were new, this design does not independently validate a gate calibrated without reference to any one of the evaluated training seeds.
- The runtime gate observes only initial cube position; it cannot identify individual BC failures. Regression-free performance on one checkpoint is not a guarantee for deployment or unseen SAC checkpoints.
- McNemar tests compare fixed methods on these deterministic paired rollouts. They do not account for the wider model-selection process or provide uncertainty over new training runs.

## 7. Next Research Direction

Evaluate substantially more independently trained SAC checkpoints using a training-seed split reserved before gate calibration. Calibrate the gate on separate checkpoints, then test it on untouched checkpoints and untouched position seeds. Add a larger, pre-specified set of training seeds and report checkpoint-level uncertainty, while preserving paired initial positions. In parallel, test a gate based on online observation history or estimated failure risk, with its own validation split and an untouched confirmatory test. The present fixed quadrant gate should not be described as consistently regression-free across SAC training seeds.

## Artifacts

- Per-checkpoint paired MuJoCo rollouts: `bc_results/sac_C_seed{42,123,456}_gate_multiseed_final2000.csv`.
- Per-checkpoint gated episodes, frozen gate copy, and summary: `bc_results/selective_gate_seed{42,123,456}_multiseed_final2000*.{csv,json}`.
- Reproducible multi-seed aggregation code: `analyze_selective_gate_multiseed.py`.
- Aggregated method metrics, quadrant metrics, and summary: `bc_results/selective_gate_multiseed_{methods,quadrants}.csv` and `bc_results/selective_gate_multiseed_summary.json`.
