"""Evaluate one deterministic residual SAC policy on fixed episode seeds.

BC-only rows can be computed once with evaluate_bc_residual_sac.py and reused
for many SAC checkpoints. The same run_episode helper and environment are used.
"""

import argparse
import csv
import os
from pathlib import Path

from stable_baselines3 import SAC

from evaluate_bc_residual_sac import run_episode
from franka_bc_residual_env import BCResidualEnv


def main(args):
    model = SAC.load(args.sac_model, device="cpu")
    env = BCResidualEnv(
        bc_model=args.bc_model,
        position_range=args.position_range,
        residual_meters=args.residual_meters,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(output.suffix + ".partial")
    fieldnames = None
    rows_written = 0
    if checkpoint.exists():
        with checkpoint.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if int(row["seed"]) != args.seed + rows_written:
                    raise RuntimeError("Partial evaluation has a non-contiguous seed sequence")
                rows_written += 1
        if rows_written > args.episodes:
            raise RuntimeError("Partial evaluation has more episodes than requested")
    mode = "a" if rows_written else "w"
    partial_file = checkpoint.open(mode, newline="", encoding="utf-8")
    writer = None
    if rows_written:
        writer = csv.DictWriter(partial_file, fieldnames=fieldnames, lineterminator="\n")
    else:
        # The first rollout determines the CSV schema.
        writer = None
    try:
        for episode in range(rows_written, args.episodes):
            row = run_episode(env, args.seed + episode, model=model)
            row["policy"] = "bc_plus_sac"
            if writer is None:
                fieldnames = list(row)
                writer = csv.DictWriter(partial_file, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
            writer.writerow(row)
            partial_file.flush()
            os.fsync(partial_file.fileno())
            print(f"{episode + 1}/{args.episodes}", flush=True)
    finally:
        partial_file.close()
        env.close()

    if rows_written < args.episodes:
        # The in-memory count starts at the resume point.
        pass
    with checkpoint.open(newline="", encoding="utf-8") as f:
        completed_rows = sum(1 for _ in csv.DictReader(f))
    if completed_rows != args.episodes:
        raise RuntimeError(f"Expected {args.episodes} episodes, found {completed_rows}")
    os.replace(checkpoint, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-model", required=True)
    parser.add_argument("--sac-model", required=True)
    parser.add_argument("--position-range", type=float, default=0.20)
    parser.add_argument("--residual-meters", type=float, default=0.005)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
