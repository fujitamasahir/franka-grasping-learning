"""Run the fixed five-seed SAC stability training study sequentially."""

import subprocess
import sys
import json
from pathlib import Path


SEEDS = (811, 823, 827, 829, 839)
REQUIRED = (
    "checkpoint_10000.zip",
    "checkpoint_20000.zip",
    "checkpoint_30000.zip",
    "best_model.zip",
    "final_model.zip",
    "eval_history.csv",
    "config.json",
)


def main():
    root = Path("bc_residual_sac_runs")
    for seed in SEEDS:
        run_name = f"stability_C_seed{seed}"
        run_dir = root / run_name
        if run_dir.exists():
            missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
            if not missing:
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                expected = {
                    "run_name": run_name,
                    "bc_model": "bc_models/fair_wide_expert_seed42.pt",
                    "position_range": 0.20,
                    "residual_meters": 0.005,
                    "sampling_mode": "hard",
                    "hard_inner_range": 0.12,
                    "hard_fraction": 0.90,
                    "reward_mode": "shaped",
                    "timesteps": 30000,
                    "seed": seed,
                    "eval_freq": 10000,
                    "eval_episodes": 40,
                    "save_trajectory_checkpoints": True,
                }
                mismatches = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
                if mismatches:
                    raise RuntimeError(f"Completed run {run_name} has unexpected config: {mismatches}")
                print(f"Skipping complete run {run_name}", flush=True)
                continue
            raise RuntimeError(
                f"Incomplete run directory {run_dir}; preserving it and refusing "
                f"to overwrite or silently retrain. Missing: {', '.join(missing)}"
            )

        command = [
            sys.executable,
            "train_bc_residual_sac.py",
            "--run-name", run_name,
            "--bc-model", "bc_models/fair_wide_expert_seed42.pt",
            "--position-range", "0.20",
            "--residual-meters", "0.005",
            "--sampling-mode", "hard",
            "--hard-inner-range", "0.12",
            "--hard-fraction", "0.90",
            "--reward-mode", "shaped",
            "--timesteps", "30000",
            "--seed", str(seed),
            "--eval-freq", "10000",
            "--eval-episodes", "40",
            "--save-trajectory-checkpoints",
        ]
        print(f"Starting {run_name}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
        missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Training returned but {run_name} is incomplete: {missing}"
            )
        print(f"Completed {run_name}", flush=True)

    print("All five stability training runs are complete.", flush=True)


if __name__ == "__main__":
    main()
