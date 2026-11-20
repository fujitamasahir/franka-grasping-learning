"""Train paired-seed original and stronger-regularization SAC runs serially."""

import json
import subprocess
import sys
from pathlib import Path


SEEDS = (907, 911, 919, 929, 937)
ARMS = (("A", 0.005), ("B", 0.020))
COMMON = {
    "bc_model": "bc_models/fair_wide_expert_seed42.pt",
    "position_range": 0.20,
    "residual_meters": 0.005,
    "sampling_mode": "hard",
    "hard_inner_range": 0.12,
    "hard_fraction": 0.90,
    "reward_mode": "shaped",
    "timesteps": 30000,
    "eval_freq": 10000,
    "eval_seed": 61000,
    "eval_episodes": 40,
    "verbose": 0,
    "save_trajectory_checkpoints": True,
    "save_replay_checkpoints": True,
}
REQUIRED = (
    "checkpoint_10000.zip", "checkpoint_20000.zip", "checkpoint_30000.zip",
    "replay_buffer_10000.pkl", "replay_buffer_20000.pkl", "replay_buffer_30000.pkl",
    "resume_state_10000.pkl", "resume_state_20000.pkl", "resume_state_30000.pkl",
    "best_model.zip", "final_model.zip", "eval_history.csv", "config.json",
    "diagnostics_episode_metrics.csv",
)


def expected_config(run_name, seed, coefficient):
    config = {
        "run_name": run_name,
        **COMMON,
        "seed": seed,
        "residual_penalty_coef": coefficient,
    }
    config.pop("verbose", None)
    return config


def run_is_complete(run_dir, expected):
    if not all((run_dir / name).is_file() for name in REQUIRED):
        return False
    actual = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return actual == expected


def command_for(run_name, seed, coefficient, resume=False):
    values = {
        **COMMON,
        "run_name": run_name,
        "seed": seed,
        "residual_penalty_coef": coefficient,
    }
    args = [sys.executable, "train_bc_residual_sac.py"]
    bool_flags = {"save_trajectory_checkpoints", "save_replay_checkpoints"}
    for name, value in values.items():
        flag = "--" + name.replace("_", "-")
        if name in bool_flags:
            if value:
                args.append(flag)
        else:
            args.extend((flag, str(value)))
    if resume:
        args.append("--resume")
    return args


def main():
    root = Path("bc_residual_sac_runs")
    for seed in SEEDS:
        for arm, coefficient in ARMS:
            run_name = f"regdiag_{arm}_seed{seed}"
            run_dir = root / run_name
            expected = expected_config(run_name, seed, coefficient)
            if run_dir.exists():
                if run_is_complete(run_dir, expected):
                    print(f"Skipping complete run {run_name}", flush=True)
                    continue
                config_path = run_dir / "config.json"
                if config_path.exists():
                    actual = json.loads(config_path.read_text(encoding="utf-8"))
                    if actual != expected:
                        raise RuntimeError(f"Refusing mismatched run directory {run_dir}: {actual}")
                if not any(run_dir.glob("checkpoint_*.zip")):
                    raise RuntimeError(
                        f"Incomplete run {run_dir} has no resumable milestone; "+
                        "preserving it rather than silently restarting."
                    )
                resume = True
            else:
                resume = False
            command = command_for(run_name, seed, coefficient, resume=resume)
            print(f"Running {run_name} (coefficient={coefficient}, resume={resume})", flush=True)
            subprocess.run(command, check=True)
            if not run_is_complete(run_dir, expected):
                missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
                raise RuntimeError(f"Run {run_name} returned incomplete: {missing}")
            print(f"Completed {run_name}", flush=True)
    print("All 10 paired-seed runs completed.", flush=True)


if __name__ == "__main__":
    main()
