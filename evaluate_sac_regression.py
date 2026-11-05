"""Resumable deterministic BC-only or residual-SAC evaluation with diagnostics."""

import argparse
import csv
import os
from pathlib import Path

from stable_baselines3 import SAC

from evaluate_bc_residual_sac import run_episode
from franka_bc_residual_env import BCResidualEnv


def complete_rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(args):
    output = Path(args.output)
    partial = output.with_name(output.stem + ".partial.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rows = complete_rows(partial)
    expected_policy = "bc_only" if args.policy == "bc" else "bc_plus_sac"
    if len(rows) > args.episodes:
        raise RuntimeError(f"Partial file has too many rows: {partial}")
    for i, row in enumerate(rows):
        if int(row["seed"]) != args.seed + i or row["policy"] != expected_policy:
            raise RuntimeError(f"Partial evaluation is not a contiguous matching run: {partial}")

    model = None
    if args.policy == "sac":
        if not args.sac_model:
            raise ValueError("--sac-model is required for --policy sac")
        model = SAC.load(args.sac_model, device="cpu")
    elif args.sac_model:
        raise ValueError("--sac-model cannot be used with --policy bc")

    env = BCResidualEnv(
        bc_model=args.bc_model,
        position_range=args.position_range,
        residual_meters=args.residual_meters,
        residual_penalty_coef=args.residual_penalty_coef,
    )
    try:
        for episode in range(len(rows), args.episodes):
            row = run_episode(
                env, args.seed + episode, model=model, include_diagnostics=True
            )
            row["policy"] = expected_policy
            rows.append(row)
            if episode % 10 == 9 or episode + 1 == args.episodes:
                with partial.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=list(rows[0]), lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"{episode + 1}/{args.episodes}", flush=True)
    finally:
        env.close()

    if len(rows) != args.episodes:
        raise RuntimeError(f"Expected {args.episodes} episodes, got {len(rows)}")
    if not partial.exists():
        with partial.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    os.replace(partial, output)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("bc", "sac"), required=True)
    parser.add_argument("--bc-model", default="bc_models/fair_wide_expert_seed42.pt")
    parser.add_argument("--sac-model")
    parser.add_argument("--position-range", type=float, default=0.20)
    parser.add_argument("--residual-meters", type=float, default=0.005)
    parser.add_argument("--residual-penalty-coef", type=float, default=0.005)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
