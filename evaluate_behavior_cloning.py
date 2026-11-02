
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from franka_autonomous_env import FrankaAutonomousEnv
from train_behavior_cloning import BCPolicy


def load_policy(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model = BCPolicy()
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    obs_mean = checkpoint["obs_mean"].numpy()
    obs_std = checkpoint["obs_std"].numpy()

    return model, obs_mean, obs_std, checkpoint


def predict_action(model, obs, obs_mean, obs_std):
    normalized = (obs - obs_mean) / obs_std

    tensor = torch.from_numpy(
        normalized.astype(np.float32)
    ).unsqueeze(0)

    with torch.no_grad():
        xyz, grip_logits = model(tensor)

    xyz = xyz.squeeze(0).numpy()

    # OPEN = +1, CLOSED = -1
    grip = 1.0 if grip_logits.item() >= 0 else -1.0

    return np.array(
        [xyz[0], xyz[1], xyz[2], grip],
        dtype=np.float32,
    )


def main(args):
    model, obs_mean, obs_std, checkpoint = load_policy(
        args.model
    )

    env = FrankaAutonomousEnv(
        position_range=float(checkpoint["position_range"])
    )

    # Match the teacher's control configuration.
    env.max_move = float(checkpoint["max_move"])
    env.max_time = float(checkpoint["max_time"])

    results = []

    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            obs, _ = env.reset(seed=seed)

            initial_x = float(env.cube_x)
            initial_y = float(env.cube_y)

            max_lift = 0.0
            steps = 0

            while True:
                action = predict_action(
                    model, obs, obs_mean, obs_std
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                steps += 1
                max_lift = max(
                    max_lift,
                    float(info["lift_height"]),
                )

                if terminated or truncated:
                    break

            success = bool(info["is_success"])

            results.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "cube_x": initial_x,
                    "cube_y": initial_y,
                    "success": int(success),
                    "steps": steps,
                    "max_lift": max_lift,
                    "final_lift": float(info["lift_height"]),
                }
            )

            print(
                f"Episode {episode + 1:3d}/{args.episodes} | "
                f"Success: {success!s:5s} | "
                f"Max lift: {max_lift:.3f}m | "
                f"Steps: {steps}"
            )

    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    success_rate = np.mean(
        [r["success"] for r in results]
    )

    print("\n===== BC EVALUATION RESULT =====")
    print(f"Episodes: {args.episodes}")
    print(f"Successful: {sum(r['success'] for r in results)}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Saved CSV: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="bc_models/franka_bc.pt",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument(
        "--output",
        default="bc_results/franka_bc_eval.csv",
    )

    main(parser.parse_args())
