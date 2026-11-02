
import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from franka_bc_residual_env import BCResidualEnv


def run_episode(env, seed, model=None):
    obs, _ = env.reset(seed=seed)

    cube_x = float(env.env.cube_x)
    cube_y = float(env.env.cube_y)

    steps = 0
    max_lift = 0.0
    residual_total = 0.0

    while True:
        if model is None:
            action = np.zeros(3, dtype=np.float32)
        else:
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

        obs, _, terminated, truncated, info = (
            env.step(action)
        )

        steps += 1
        max_lift = max(
            max_lift,
            float(info["lift_height"]),
        )
        residual_total += info["residual_norm"]

        if terminated or truncated:
            break

    return {
        "seed": seed,
        "cube_x": cube_x,
        "cube_y": cube_y,
        "success": int(info["is_success"]),
        "steps": steps,
        "max_lift": max_lift,
        "mean_residual_m": residual_total / steps,
    }


def main(args):
    sac_model = None

    if args.sac_model:
        sac_model = SAC.load(
            args.sac_model,
            device="cpu",
        )

    env = BCResidualEnv(
        bc_model=args.bc_model,
        position_range=args.position_range,
        residual_meters=args.residual_meters,
    )

    results = []

    try:
        for episode in range(args.episodes):
            seed = args.seed + episode

            bc = run_episode(
                env,
                seed,
                model=None,
            )
            bc["policy"] = "bc_only"
            results.append(bc)

            if sac_model is not None:
                sac = run_episode(
                    env,
                    seed,
                    model=sac_model,
                )
                sac["policy"] = "bc_plus_sac"
                results.append(sac)

                assert np.isclose(
                    bc["cube_x"], sac["cube_x"]
                )
                assert np.isclose(
                    bc["cube_y"], sac["cube_y"]
                )

                print(
                    f"{episode + 1:3d}/{args.episodes} | "
                    f"BC: {bc['success']} | "
                    f"BC+SAC: {sac['success']}"
                )
            else:
                print(
                    f"{episode + 1:3d}/{args.episodes} | "
                    f"BC: {bc['success']}"
                )

    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== COMPARISON RESULT =====")

    for policy in ("bc_only", "bc_plus_sac"):
        selected = [
            row for row in results
            if row["policy"] == policy
        ]

        if not selected:
            continue

        successes = sum(
            row["success"] for row in selected
        )

        successful_steps = [
            row["steps"] for row in selected
            if row["success"]
        ]

        print(
            f"{policy}: "
            f"{successes}/{len(selected)} "
            f"({successes / len(selected):.1%})"
        )

        if successful_steps:
            print(
                "  Mean steps on successful episodes: "
                f"{np.mean(successful_steps):.1f}"
            )

    print(f"Saved CSV: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bc-model",
        default="bc_models/franka_bc_dagger1.pt",
    )

    parser.add_argument(
        "--sac-model",
        default=None,
    )

    parser.add_argument(
        "--position-range",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--residual-meters",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=71000,
    )

    parser.add_argument(
        "--output",
        default="bc_results/residual_comparison.csv",
    )

    main(parser.parse_args())
