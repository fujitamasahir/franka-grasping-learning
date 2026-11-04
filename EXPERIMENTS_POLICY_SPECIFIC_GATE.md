# Policy-Specific Gate Calibration for Residual SAC

## 1. Research Question

**Can policy-specific calibration identify where a residual SAC policy helps or hurts a fixed imitation-learned controller, and can that calibration procedure generalize to previously unseen SAC training seeds?**

The method considered here is supervised at the episode level by paired calibration rollouts: each gate is calibrated for one frozen SAC checkpoint, then uses only the initial cube position to select BC or BC+SAC for an entire episode.

## 2. Hypotheses

- **H1:** A separate gate for each SAC model produces more stable performance across training seeds than the fixed Seed 456 gate.
- **H2:** Policy-Specific Safety reduces regressions relative to Always SAC while preserving or improving BC performance.
- **H3:** Policy-Specific Balanced achieves higher success than BC with fewer SAC activations than Always SAC.

These are evaluated as written. Calibration thresholds and quadrant definitions were fixed before the final evaluation, and no result was used to alter a gate or select a checkpoint.

## 3. Experimental Design

- The branch was created from `experiment/selective-residual-gate` at commit `7c59faf` as `experiment/policy-specific-gate`.
- Fixed BC checkpoint: `bc_models/fair_wide_expert_seed42.pt`.
- Condition C SAC training: 30,000 steps; hard sampling with 90% of training positions outside the ±12 cm demonstration square; shaped reward; position range ±20 cm; residual XYZ bound ±5 mm per axis; unchanged SAC hyperparameters.
- New SAC training seeds: **7, 19, 73, 202, 314**. Their run directories, `config.json`, `best_model.zip`, and `final_model.zip` are separate. Best checkpoints were selected using only the existing uniform validation seeds 61000–61039 (40 episodes at 10k, 20k, and 30k steps). The selected best checkpoints were at 10k steps for 7, 19, 73, and 202, and at 30k for 314.
- All five models used paired calibration positions 180000–180499 (500 per model) and paired final evaluation positions 190000–191999 (2,000 per model). All models used the same initial positions and had identical BC outcomes on them.
- Final evaluation positions were not used for calibration or checkpoint selection. The final evaluation files were only analyzed after all five policy-specific gate JSON files were written and the freeze manifest recorded their hashes.
- Fixed Safety remained EN; Fixed Balanced remained EN/WS from the existing Seed 456 gate JSON. No existing CSV, JSON, gate, or model was overwritten.
- The 8 tests in `test_selective_residual_gate.py` and `test_policy_specific_gate.py` passed.

### Calibration Procedure

The BC demonstration region is the square within ±12 cm of `(x=0.55 m, y=0)`. Episodes inside it always use BC. Outside it, initial positions are assigned to EN, ES, WN, or WS. Rescue means BC fails and BC+SAC succeeds; regression means BC succeeds and BC+SAC fails.

For each individual checkpoint, the calibration procedure used its own 500 paired episodes and the original thresholds:

- **Safety:** at least 30 calibration episodes in a quadrant, at least 5 rescues, and zero regressions.
- **Balanced:** at least 30 episodes, at least 3 rescues, and more rescues than regressions.
- If no quadrant qualifies, the gate selects BC for all episodes.

The configuration JSON records the SAC training seed, checkpoint path and SHA-256, calibration path and SHA-256, all calibration seeds, per-quadrant calibration counts, and the selected quadrants. The five calibration CSVs were verified to use the same coordinates and BC outcomes; none overlaps 190000–191999.

## 4. Results

Each cell reports `successes/2000 (rate)`. Rescue and regression counts are relative to BC on the same position. `Net` is rescue minus regression. SAC activations are counts out of 2,000.

| SAC train seed | Method | Success | Rescue | Regression | Net | SAC uses | Δ vs BC | Δ vs Always SAC | Δ vs matching Fixed Gate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | BC Only | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | +0.75 pp | — |
| 7 | Always SAC | 1819 (90.95%) | 1 | 16 | -15 | 2000 | -0.75 pp | 0.00 pp | — |
| 7 | Fixed Safety | 1834 (91.70%) | 0 | 0 | 0 | 336 | 0.00 pp | +0.75 pp | — |
| 7 | Fixed Balanced | 1833 (91.65%) | 1 | 2 | -1 | 643 | -0.05 pp | +0.70 pp | — |
| 7 | Policy-Specific Safety | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | +0.75 pp | 0.00 pp |
| 7 | Policy-Specific Balanced | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | +0.75 pp | +0.05 pp |
| 19 | BC Only | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | +0.10 pp | — |
| 19 | Always SAC | 1832 (91.60%) | 3 | 5 | -2 | 2000 | -0.10 pp | 0.00 pp | — |
| 19 | Fixed Safety | 1833 (91.65%) | 0 | 1 | -1 | 336 | -0.05 pp | +0.05 pp | — |
| 19 | Fixed Balanced | 1836 (91.80%) | 3 | 1 | +2 | 643 | +0.10 pp | +0.20 pp | — |
| 19 | Policy-Specific Safety | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | +0.10 pp | +0.05 pp |
| 19 | Policy-Specific Balanced | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | +0.10 pp | -0.10 pp |
| 73 | BC Only | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | +0.95 pp | — |
| 73 | Always SAC | 1815 (90.75%) | 4 | 23 | -19 | 2000 | -0.95 pp | 0.00 pp | — |
| 73 | Fixed Safety | 1836 (91.80%) | 2 | 0 | +2 | 336 | +0.10 pp | +1.05 pp | — |
| 73 | Fixed Balanced | 1824 (91.20%) | 2 | 12 | -10 | 643 | -0.50 pp | +0.45 pp | — |
| 73 | Policy-Specific Safety | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | +0.95 pp | -0.10 pp |
| 73 | Policy-Specific Balanced | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | +0.95 pp | +0.50 pp |
| 202 | BC Only | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | -0.15 pp | — |
| 202 | Always SAC | 1837 (91.85%) | 9 | 6 | +3 | 2000 | +0.15 pp | 0.00 pp | — |
| 202 | Fixed Safety | 1836 (91.80%) | 2 | 0 | +2 | 336 | +0.10 pp | -0.05 pp | — |
| 202 | Fixed Balanced | 1836 (91.80%) | 6 | 4 | +2 | 643 | +0.10 pp | -0.05 pp | — |
| 202 | Policy-Specific Safety | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | -0.15 pp | -0.10 pp |
| 202 | Policy-Specific Balanced | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | -0.15 pp | -0.10 pp |
| 314 | BC Only | 1834 (91.70%) | 0 | 0 | 0 | 0 | 0.00 pp | -4.40 pp | — |
| 314 | Always SAC | 1922 (96.10%) | 91 | 3 | +88 | 2000 | +4.40 pp | 0.00 pp | — |
| 314 | Fixed Safety | 1855 (92.75%) | 21 | 0 | +21 | 336 | +1.05 pp | -3.35 pp | — |
| 314 | Fixed Balanced | 1879 (93.95%) | 45 | 0 | +45 | 643 | +2.25 pp | -2.15 pp | — |
| 314 | Policy-Specific Safety | 1894 (94.70%) | 60 | 0 | +60 | 692 | +3.00 pp | -1.40 pp | +1.95 pp |
| 314 | Policy-Specific Balanced | 1898 (94.90%) | 67 | 3 | +64 | 1028 | +3.20 pp | -1.20 pp | +0.95 pp |

## 5. Statistical Analysis

We used two-sided exact McNemar tests on paired episode outcomes. Each family below contains five pre-specified tests (one per new SAC training seed); the table gives Holm-adjusted p-values. The direction and success differences are in the results table. Raw p-values and discordant win/loss counts are retained in `bc_results/policy_specific_gate_final_summary.json` and `_methods.csv`.

| Comparison family | Seed 7 | Seed 19 | Seed 73 | Seed 202 | Seed 314 |
|---|---:|---:|---:|---:|---:|
| Policy-Specific Safety vs Fixed Safety | 1.000 | 1.000 | 1.000 | 1.000 | 1.82e-11 |
| Policy-Specific Balanced vs Fixed Balanced | 1.000 | 1.000 | 0.0647 | 1.000 | 0.1377 |
| Policy-Specific Safety vs BC | 1.000 | 1.000 | 1.000 | 1.000 | 8.67e-18 |
| Policy-Specific Balanced vs BC | 1.000 | 1.000 | 1.000 | 1.000 | 4.85e-16 |
| Policy-Specific Safety vs Always SAC | 0.00110 | 1.000 | 0.00110 | 1.000 | 3.83e-06 |
| Policy-Specific Balanced vs Always SAC | 0.00110 | 1.000 | 0.00110 | 1.000 | 5.96e-07 |

At a two-sided familywise 0.05 threshold, Policy-Specific Safety was statistically different from Fixed Safety, BC, and Always SAC only for seed 314. The differences from Always SAC for seeds 7 and 73 also survived correction for both gate variants. Policy-Specific Balanced was different from Fixed Balanced for no seed after adjustment; its seed 73 raw p-value was 0.0129 but Holm-adjusted p was 0.0647. It differed from BC only for seed 314, where it improved success. It differed from Always SAC for seeds 7, 73, and 314; the 314 difference favored Always SAC.

Cross-seed success-rate variation is summarized descriptively using sample standard deviation across the five independently trained checkpoints:

| Method | Mean success | Sample SD | Range | Models improved vs BC | Models worsened vs BC |
|---|---:|---:|---:|---:|---:|
| Always SAC | 92.25% | 2.20 pp | 90.75–96.10% | 2 | 3 |
| Fixed Safety | 91.94% | 0.46 pp | 91.65–92.75% | 3 | 1 |
| Fixed Balanced | 92.08% | 1.07 pp | 91.20–93.95% | 3 | 2 |
| Policy-Specific Safety | 92.30% | 1.34 pp | 91.70–94.70% | 1 | 0 |
| Policy-Specific Balanced | 92.34% | 1.43 pp | 91.70–94.90% | 1 | 0 |

The common BC rate was 91.70% for every seed. For both policy-specific gates, four models fell back to BC and one model improved over BC; none worsened BC. This floor effect should not be confused with consistent active SAC gains.

## 6. Cross-Seed Robustness

Each policy-specific gate selected these regions from its own 500 calibration episodes:

| SAC seed | Safety quadrants | Balanced quadrants | Calibration rescue/regression counts by EN, ES, WN, WS |
|---:|---|---|---|
| 7 | none | none | 0/0, 1/1, 0/0, 2/0 |
| 19 | none | none | 0/0, 1/0, 0/0, 2/0 |
| 73 | none | none | 2/0, 2/0, 0/2, 0/0 |
| 202 | none | none | 2/0, 1/0, 0/0, 2/0 |
| 314 | EN, WN | EN, ES, WN | 8/0, 6/1, 6/0, 2/0 |

The EN/ES/WN/WS rescue/regression outcomes under Always SAC on the final test were:

| SAC seed | EN | ES | WN | WS |
|---:|---:|---:|---:|---:|
| 7 | 0/0 | 0/12 | 0/2 | 1/2 |
| 19 | 0/1 | 0/4 | 0/0 | 3/0 |
| 73 | 2/0 | 2/0 | 0/11 | 0/12 |
| 202 | 2/0 | 0/1 | 3/1 | 4/4 |
| 314 | 21/0 | 7/3 | 39/0 | 24/0 |

With the fixed Seed 456 Gate, Safety used SAC in EN for 336/2,000 episodes (16.8%); Balanced used SAC in EN/WS for 643/2,000 (32.15%). Policy-Specific Safety used SAC 0, 0, 0, 0, and 692 times; Policy-Specific Balanced used SAC 0, 0, 0, 0, and 1,028 times. Thus policy-specific calibration abstained for four models because no quadrant met the unchanged thresholds. For seed 314 it selected quadrants that differed from the fixed gate and captured substantial additional benefit.

The per-quadrant final outcomes explain the seed variation. Seed 73's fixed Balanced Gate activated in WS, where SAC had 12 regressions and no rescues. Seed 314's policy-specific gate activated in WN, yielding 39 additional successes without regression; its Balanced Gate also activated in ES, yielding 7 rescues and 3 regressions. Calibration correctly rejected WN/WS for many models, but four of five gates rejected every region and therefore provided no active residual-SAC benefit.

## 7. Discussion

- **H1 was not supported as a stability claim.** The observed cross-seed standard deviation was higher for Policy-Specific Safety (1.34 pp) than Fixed Safety (0.46 pp), and higher for Policy-Specific Balanced (1.43 pp) than Fixed Balanced (1.07 pp). Policy-specific gates did not significantly outperform their matching Fixed Gate after the five-seed Holm correction in either family. The result is dominated by one checkpoint (Seed 314) receiving a non-empty gate.
- **H2 was supported on these five models as a safety-preservation tradeoff.** Policy-Specific Safety produced zero regressions on every seed, versus 16, 5, 23, 6, and 3 for Always SAC. It retained BC's 91.70% success on four models and improved the fifth by 3.00 pp. It reduced Always SAC's activations from 2,000 to zero on four models and to 692 on seed 314. For seed 314 it still trailed Always SAC by 1.40 pp, reflecting the cost of avoiding three regressions and additional beneficial SAC actions.
- **H3 was not supported consistently.** Policy-Specific Balanced exceeded BC only for seed 314 (94.90% vs 91.70%); the other four gates fell back to BC. It used fewer activations than Always SAC for every model, but on seed 314 required 1,028 activations (51.4%) and remained 1.20 pp below Always SAC. The mean +0.64 pp gain over BC is entirely attributable to one of five checkpoints.
- The previous fixed gate was calibrated on SAC seed 456. Its EN/EN+WS decisions were unchanged here. Policy-specific calibration required 500 new paired episodes for each model and often chose different regions; this reduced transfer risk at the cost of per-model calibration data and sometimes all benefit from SAC.
- Actual MuJoCo gate execution was checked on 60 calibration-position decisions across the five checkpoints. Both SAC-selected and BC-fallback cases were exercised where available; every live outcome and initial coordinate matched the corresponding paired rollout.

## 8. Limitations

- Five SAC training seeds remain a small sample of the training process. Cross-seed rates and sample deviations are descriptive, not a well-estimated population distribution.
- The same 2,000 initial positions and BC outcomes were reused across all five models. Paired comparisons are appropriate within a model, but the 10,000 model-position records are correlated and are not treated as 10,000 independent samples.
- Each final-evaluated SAC model also supplied its own 500-episode calibration set. This tests the calibration *procedure* on unseen training seeds, not zero-shot gate transfer; the added per-model calibration cost is 500 paired episodes (1,000 physical rollouts) plus computation/storage per SAC checkpoint.
- Four of five calibrated gates selected no region. This obeys the pre-specified criteria and was not tuned after seeing final results, but it means the procedure commonly falls back to BC rather than delivering a selective-SAC improvement.
- Checkpoints were selected using the common 40-episode validation set, which is small and reused across training runs as specified by the existing protocol.
- The episode-level gate only sees initial cube position and cannot identify an individual BC failure or guarantee safety outside the tested checkpoints and workspace distribution.
- Exact McNemar tests condition on these frozen models and these positions. Holm correction was applied within each pre-specified five-test comparison family. It does not eliminate uncertainty over newly trained SAC policies.

## 9. Next Research Direction

Proceed to **Adaptive Gate Learning** as a research prototype, not as a deployment-ready policy. The main opportunity is to avoid the all-or-nothing quadrant selection that left four models at BC performance. A candidate gate could estimate failure risk or expected residual benefit from initial position plus early observation history and include an explicit abstain option. It should be trained and validated across a larger set of SAC checkpoints, with model-level splits: calibration/training checkpoints, validation checkpoints for threshold choice, and untouched SAC seeds and position seeds for confirmation. The validation objective should explicitly penalize regressions and account for SAC activation cost. The current five-seed result, especially the seed-314 improvement and four empty gates, should remain the baseline.

## 10. Artifacts and Reproduction

- Training was run with `train_bc_residual_sac.py`; binary checkpoints remain local and are not included in Git.
- Calibration rollouts: `bc_results/sac_C_seed{7,19,73,202,314}_policy_gate_calibration500.csv`.
- Frozen policy-specific Gate JSON: `bc_results/policy_specific_gate_seed{7,19,73,202,314}.json`.
- Gate freeze manifest: `bc_results/policy_specific_gate_freeze_manifest.json`.
- Final paired MuJoCo rollouts: `bc_results/sac_C_seed{7,19,73,202,314}_policy_gate_final2000.csv`.
- Per-model live MuJoCo verification: `bc_results/policy_specific_gate_seed{7,19,73,202,314}_live_verify.csv`.
- Final method and quadrant metrics: `bc_results/policy_specific_gate_final_methods.csv`, `bc_results/policy_specific_gate_final_quadrants.csv`, and `bc_results/policy_specific_gate_final_summary.json`.
- Code: `calibrate_policy_specific_gate.py`, `analyze_policy_specific_gate.py`, and `verify_policy_specific_gate_live.py`.
- Tests: `python3 -m unittest test_selective_residual_gate.py test_policy_specific_gate.py`.
