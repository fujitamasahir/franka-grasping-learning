
import argparse
import csv
from pathlib import Path

import numpy as np

from franka_autonomous_env import FrankaAutonomousEnv
from collect_demonstrations import expert_action
from evaluate_behavior_cloning import (
    load_policy,
    predict_action,
)


def run_episode(
    env, policy_name, seed,
    model, obs_mean, obs_std,
):
    obs, _ = env.reset(seed=seed)

    initial_x = float(env.cube_x)
    initial_y = float(env.cube_y)

    steps = 0
    max_lift = 0.0

    while True:
        if policy_name == "expert":
            action = expert_action(env)
        else:
            action = predict_action(
                model, obs, obs_mean, obs_std
            )

        obs, _, terminated, truncated, info = (
            env.step(action)
        )

        steps += 1
        max_lift = max(
            max_lift,
            float(info["lift_height"]),
        )

        if terminated or truncated:
            break

    return {
        "policy": policy_name,
        "seed": seed,
        "cube_x": initial_x,
        "cube_y": initial_y,
        "success": int(info["is_success"]),
        "steps": steps,
        "max_lift": max_lift,
        "final_lift": float(info["lift_height"]),
    }


def main(args):
    model, obs_mean, obs_std, checkpoint = (
        load_policy(args.model)
    )

    results = []

    for position_range in args.ranges:
        print(
            f"\n===== POSITION RANGE: "
            f"±{position_range:.3f}m ====="
        )

        env = FrankaAutonomousEnv(
            position_range=position_range
        )

        # Match the configuration used for BC training.
        env.max_move = float(checkpoint["max_move"])
        env.max_time = float(checkpoint["max_time"])

        try:
            for episode in range(args.episodes):
                seed = args.seed + episode

                # Same seed and cube position for both policies.
                expert = run_episode(
                    env, "expert", seed,
                    model, obs_mean, obs_std,
                )

                bc = run_episode(
                    env, "bc", seed,
                    model, obs_mean, obs_std,
                )

                # Check that the paired initial states match.
                assert np.isclose(
                    expert["cube_x"], bc["cube_x"]
                )
                assert np.isclose(
                    expert["cube_y"], bc["cube_y"]
                )

                expert["position_range"] = position_range
                bc["position_range"] = position_range

                results.extend([expert, bc])

                if (episode + 1) % 50 == 0 or episode + 1 == episodes:
                    print(
                        f"Episode {episode + 1:3d}/"
                        f"{args.episodes} | "
                        f"Expert: {expert['success']} | "
                        f"BC: {bc['success']} | "
                        f"Cube: ({bc['cube_x']:.3f}, "
                        f"{bc['cube_y']:.3f})"
                    )

        finally:
            env.close()

        subset = [
            r for r in results
            if r["position_range"] == position_range
        ]

        for policy in ("expert", "bc"):
            selected = [
                r for r in subset
                if r["policy"] == policy
            ]

            successes = sum(
                r["success"] for r in selected
            )

            print(
                f"{policy.upper():6s}: "
                f"{successes}/{len(selected)} "
                f"({successes / len(selected):.1%})"
            )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== GENERALIZATION SUMMARY =====")

    print(
        f"{'Range':>8} | "
        f"{'Expert':>10} | "
        f"{'BC':>10}"
    )
    print("-" * 36)

    for position_range in args.ranges:
        row = []

        for policy in ("expert", "bc"):
            selected = [
                r for r in results
                if r["position_range"] == position_range
                and r["policy"] == policy
            ]

            successes = sum(
                r["success"] for r in selected
            )

            row.append(
                f"{successes}/{len(selected)}"
            )

        print(
            f"±{position_range:.2f}m | "
            f"{row[0]:>10} | "
            f"{row[1]:>10}"
        )

    print(f"\nSaved CSV: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="bc_models/franka_bc.pt",
    )

    parser.add_argument(
        "--ranges",
        type=float,
        nargs="+",
        default=[0.02, 0.04, 0.06, 0.08, 0.12],
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12000,
    )

    parser.add_argument(
        "--output",
        default="bc_results/generalization.csv",
    )

    main(parser.parse_args())
