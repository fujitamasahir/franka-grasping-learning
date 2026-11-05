# Selective residual SAC: preserve BC successes, rescue BC failures

## Research question

Can we **keep a strong imitation-learning policy unchanged when it is
already likely to succeed**, but activate residual reinforcement learning
where it is expected to help?

This is **not** a comparison of standalone IL against standalone RL.
It compares **BC only**, **BC + SAC always**, and **BC + SAC with a gate**.
The BC and SAC model weights remain frozen.

## What the gate can actually observe

The current MuJoCo task supplies the cube's initial (x, y) position.
We first test a simple **episode-level spatial gate**:

1. Inside the BC demonstration square (±12 cm about x=0.55 m, y=0),
   always use BC.
2. Outside the square, divide the workspace into four quadrants:
   EN (+x,+y), ES (+x,-y), WN (-x,+y), WS (-x,-y).
3. From **calibration-only** paired BC/SAC rollouts, count:
   - **Rescue**: BC failed, BC+SAC succeeded.
   - **Regression**: BC succeeded, BC+SAC failed.
4. Freeze two gate variants:
   - **Safety (primary)**: use SAC in a quadrant only if it has at least
     30 calibration episodes, 5 or more rescues, and **0 regressions**.
   - **Balanced (secondary)**: use SAC in a quadrant only if it has at
     least 30 episodes, 3 or more rescues, and more rescues than regressions.

The runtime decision uses **only the initial cube position**.
Success labels calibrate the quadrant choices, never the decision
for an individual test episode. This is not a learned prediction
of each BC failure, and cannot guarantee zero regressions.

## Data separation

| Role | Evaluation seeds | Use |
|---|---|---|
| SAC checkpoint selection | 61000–61039 | Previous training validation |
| Gate calibration | **120000–120499** | Fit quadrants; 500 paired episodes |
| Already inspected | **130000–130999** | Retrospective check, NOT final evidence |
| **New confirmatory test** | **150000–151999** | 2,000 new paired episodes |
| Optional live smoke check | 160000 onward | Verify actual gate execution |

SAC training seed 456 was selected after earlier comparisons, including
inspection of the 130000–130999 results. Therefore those 1,000 episodes
are **not** an independent final test for the new method.

Do not adjust thresholds or choose a different gate after seeing
the 2,000 new episodes. Safety is the primary safety-oriented
analysis; balanced is a secondary success/safety tradeoff.

## Existing-data feasibility check (exploratory, not final evidence)

Using the C / Seed 456 residual SAC model:

| Method | 500 calibration positions | Previously seen 1,000 positions |
|---|---:|---:|
| BC only | 456/500 (91.2%) | 939/1000 (93.9%) |
| Always SAC | 470/500 (94.0%) | 957/1000 (95.7%) |
| Safety gate (EN only) | 467/500 (93.4%) | 955/1000 (95.5%) |
| Balanced gate (EN, WS) | 472/500 (94.4%) | 960/1000 (96.0%) |

On the already-inspected 1,000 positions:
- Safety: 16 rescues, **0 regressions**, SAC used 169 times.
- Balanced: 22 rescues, 1 regression, SAC used 347 times.
- Always SAC: 22 rescues, 4 regressions, SAC used 1,000 times.

These are **hypothesis-generating observations**, not proof of
generalization. The next step is the new test.

## Run the experiment

Run from the repository root on branch
**experiment/selective-residual-gate**.

### Step 0 — Check the gate logic (no MuJoCo needed)

```bash
python3 -m unittest test_selective_residual_gate.py
```

### Step 1 — Calibrate and freeze the gate

Use only the old 500-position paired CSV to select quadrants. The
previously inspected 1,000-position file is used only for an
exploratory check, never for tuning.

```bash
python3 analyze_selective_residual_gate.py \
  --calibration bc_results/sac_C_hard_shaped_seed456_test500.csv \
  --evaluation bc_results/sac_C_seed456_final1000.csv \
  --stage exploratory \
  --output bc_results/selective_gate_exploratory1000.csv
```

This writes a per-position CSV, summary JSON, and the **frozen gate**
configuration: bc_results/selective_gate_exploratory1000_gate.json.

Expected choices: safety = EN; balanced = EN and WS.

### Step 2 — Collect 2,000 NEW paired evaluation positions

**No SAC retraining needed.** Keep the same frozen C / Seed 456 model.

```bash
mjpython evaluate_bc_residual_sac.py \
  --bc-model bc_models/fair_wide_expert_seed42.pt \
  --sac-model bc_residual_sac_runs/C_hard_shaped_seed456/best_model.zip \
  --position-range 0.20 \
  --residual-meters 0.005 \
  --episodes 2000 \
  --seed 150000 \
  --output bc_results/sac_C_seed456_gate_final2000.csv
```

The file contains both BC and always-SAC outcomes at each initial
position, paired by seed.

### Step 3 — Evaluate the FROZEN gate (do not refit)

```bash
python3 analyze_selective_residual_gate.py \
  --gate-config bc_results/selective_gate_exploratory1000_gate.json \
  --evaluation bc_results/sac_C_seed456_gate_final2000.csv \
  --stage confirmatory \
  --output bc_results/selective_gate_final2000.csv
```

The script prints success, rescue, regression, SAC activation counts,
and paired exact McNemar p-values against BC. Two gate variants are
examined, so treat p-values as descriptive unless correcting for
multiple comparisons. This study uses **one previously selected SAC
training seed**; multi-seed robustness remains an open question.

### Step 4 — Optional live verification in MuJoCo

Offline evaluation is valid when the paired deterministic rollouts
reproduce the chosen policy. Verify the first 20 new test positions:

```bash
mjpython evaluate_selective_residual_sac.py \
  --gate-config bc_results/selective_gate_exploratory1000_gate.json \
  --mode safety \
  --episodes 20 \
  --seed 150000 \
  --paired-reference bc_results/sac_C_seed456_gate_final2000.csv \
  --output bc_results/selective_gate_live_verify20.csv
```

You may also run the balanced mode. This verifies reproducibility;
it does not create independent additional test data.

### Step 5 — Commit CSV and JSON results

```bash
git add bc_results/sac_C_seed456_gate_final2000.csv \
        bc_results/selective_gate_exploratory1000.csv \
        bc_results/selective_gate_exploratory1000_gate.json \
        bc_results/selective_gate_exploratory1000_summary.json \
        bc_results/selective_gate_final2000.csv \
        bc_results/selective_gate_final2000_gate.json \
        bc_results/selective_gate_final2000_summary.json
git commit -m "Evaluate frozen selective residual SAC gate on 2000 new positions"
git push
```

Local .pt and .zip checkpoints are ignored by Git. Do not upload them.

## Interpretation

The ideal result has many rescues, few regressions, high total success,
and fewer SAC activations than always-SAC. Zero regressions on
calibration data **does not guarantee** zero regressions on new cases.

If the gate does not improve over BC or always-SAC, report it. A
subsequent project could learn failure risk from the robot's
observation history, but that requires a new validation split and
another untouched test.
