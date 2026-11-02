# Controlled residual SAC experiments

Compare whether hard-case sampling and reward shaping help a residual SAC
policy improve a **fixed Wide Expert BC** policy. No held-out final evaluation
seeds are used for training or checkpoint selection.

## Requirements

Run from the repository root with the same MuJoCo, Gymnasium, PyTorch and
Stable-Baselines3 environment used for the existing BC experiments. The
`mujoco_menagerie/franka_emika_panda/scene.xml` asset and the local BC
checkpoint `bc_models/fair_wide_expert_seed42.pt` must exist.

## Experimental arms (30k-step pilot)

All arms use the same BC checkpoint, 3D residual action bound of +/-5 mm
per axis, SAC hyperparameters and training seed 42.

```bash
mjpython train_bc_residual_sac.py --run-name A_uniform_sparse_seed42 --sampling-mode uniform --reward-mode sparse --seed 42
mjpython train_bc_residual_sac.py --run-name B_hard_sparse_seed42 --sampling-mode hard --reward-mode sparse --seed 42
mjpython train_bc_residual_sac.py --run-name C_hard_shaped_seed42 --sampling-mode hard --reward-mode shaped --seed 42
```

For hard sampling, 90% of training initial positions are uniformly sampled
outside the square +/-12 cm about (x=0.55 m, y=0) but inside +/-20 cm;
10% are uniformly sampled inside the +/-12 cm square. This is an
**outer-region curriculum**, not selection by observed BC failure.

Shaped reward adds clipped *step-to-step progress* toward the grasp point
and upward cube motion to the existing sparse task reward. The same success
criterion and action constraints are used for all arms.

Each run saves a separate `bc_residual_sac_runs/<run-name>/` directory
with `config.json`, `eval_history.csv`, `best_model.zip`, and
`final_model.zip`. Checkpoints are selected by deterministic success rate
on **the same 40 uniformly sampled validation seeds (61000-61039)**,
with shorter successful episodes as a tie-breaker.

## Final held-out comparison

Run each evaluation with the same 200 seeds (90000-90199), with uniform
sampling. Example:

```bash
mjpython evaluate_bc_residual_sac.py --bc-model bc_models/fair_wide_expert_seed42.pt --sac-model bc_residual_sac_runs/A_uniform_sparse_seed42/best_model.zip --position-range 0.20 --residual-meters 0.005 --episodes 200 --seed 90000 --output bc_results/sac_A_uniform_sparse_seed42.csv
mjpython evaluate_bc_residual_sac.py --bc-model bc_models/fair_wide_expert_seed42.pt --sac-model bc_residual_sac_runs/B_hard_sparse_seed42/best_model.zip --position-range 0.20 --residual-meters 0.005 --episodes 200 --seed 90000 --output bc_results/sac_B_hard_sparse_seed42.csv
mjpython evaluate_bc_residual_sac.py --bc-model bc_models/fair_wide_expert_seed42.pt --sac-model bc_residual_sac_runs/C_hard_shaped_seed42/best_model.zip --position-range 0.20 --residual-meters 0.005 --episodes 200 --seed 90000 --output bc_results/sac_C_hard_shaped_seed42.csv
```

Compare paired outcomes (BC-only vs SAC-only successes) and results
inside/outside the +/-12 cm BC demonstration range. The 30k-step, single-seed
runs are pilots, not conclusive evidence. Repeat with multiple independent
training seeds if promising, and avoid selecting hyperparameters using the
held-out test set.

## Notes

- Validation is always uniform even when training is hard-sampled.
- This is **residual RL**, not direct fine-tuning of the BC network weights.
- The base environment's own reward is ignored by the residual wrapper.
- Gripper control remains with BC; SAC controls XYZ residuals only.
- This code has not been run on the local MuJoCo assets by the authoring
  assistant; run the smoke test below before full training.

```bash
mjpython train_bc_residual_sac.py --run-name smoke_test --sampling-mode hard --reward-mode shaped --timesteps 100 --eval-freq 50 --eval-episodes 2
```
