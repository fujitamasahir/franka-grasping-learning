"""Run the preregistered independent-seed replication serially and resumably."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROTOCOL = Path("bc_results/sac_regularization_replication_protocol.json")
PROTOCOL_HASH = Path("bc_results/sac_regularization_replication_protocol.sha256")
RUNS = Path("bc_residual_sac_runs")
SEEDS = (1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061)
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


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_protocol():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    declared = PROTOCOL_HASH.read_text(encoding="utf-8").split()[0]
    if sha256(PROTOCOL) != declared:
        raise RuntimeError("Preregistered protocol SHA-256 mismatch")
    if protocol["training"]["independent_training_seeds"] != list(SEEDS):
        raise RuntimeError("Training seeds differ from preregistered protocol")
    if protocol["training"]["arms"] != {
        "A": {"residual_penalty_coefficient": 0.005},
        "B": {"residual_penalty_coefficient": 0.02},
    }:
        raise RuntimeError("Arm settings differ from preregistered protocol")
    return protocol


def expected_config(run_name, seed, coefficient):
    config = {
        "run_name": run_name,
        **COMMON,
        "seed": seed,
        "residual_penalty_coef": coefficient,
    }
    config.pop("verbose")
    return config


def complete(run_dir, expected):
    if not all((run_dir / name).is_file() for name in REQUIRED):
        return False
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8")) == expected


def command(run_name, seed, coefficient, resume):
    values = {**COMMON, "run_name": run_name, "seed": seed, "residual_penalty_coef": coefficient}
    flags = {"save_trajectory_checkpoints", "save_replay_checkpoints"}
    args = [sys.executable, "train_bc_residual_sac.py"]
    for name, value in values.items():
        flag = "--" + name.replace("_", "-")
        if name in flags:
            if value:
                args.append(flag)
        else:
            args.extend((flag, str(value)))
    if resume:
        args.append("--resume")
    return args


def main():
    verify_protocol()
    RUNS.mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm, coefficient in ARMS:
            run_name = f"replication_{arm}_seed{seed}"
            run_dir = RUNS / run_name
            expected = expected_config(run_name, seed, coefficient)
            if run_dir.exists():
                config_path = run_dir / "config.json"
                if config_path.exists():
                    actual = json.loads(config_path.read_text(encoding="utf-8"))
                    if actual != expected:
                        raise RuntimeError(f"Refusing mismatched existing run directory: {run_dir}")
                if complete(run_dir, expected):
                    print(f"Skipping completed run {run_name}", flush=True)
                    continue
                if not any(run_dir.glob("checkpoint_*.zip")):
                    raise RuntimeError(
                        f"Incomplete run {run_dir} has no saved milestone; preserving it rather than restarting."
                    )
                resume = True
            else:
                resume = False
            print(f"Running {run_name}: penalty={coefficient}, resume={resume}", flush=True)
            subprocess.run(command(run_name, seed, coefficient, resume), check=True)
            if not complete(run_dir, expected):
                missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
                raise RuntimeError(f"Run {run_name} incomplete after trainer exit; missing={missing}")
            print(f"Completed {run_name}", flush=True)
    print("All 20 preregistered paired-seed training runs completed.", flush=True)


if __name__ == "__main__":
    main()
