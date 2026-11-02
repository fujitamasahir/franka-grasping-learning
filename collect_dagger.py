
import argparse
from pathlib import Path

import numpy as np

from franka_autonomous_env import FrankaAutonomousEnv
from collect_demonstrations import expert_action
from evaluate_behavior_cloning import (
    load_policy,
    predict_action,
)


def main(args):
    model, obs_mean, obs_std, checkpoint = load_policy(
        args.model
    )

    with np.load(args.original_data) as data:
        original_obs = data["observations"].copy()
        original_actions = data["actions"].copy()
        original_ids = data["episode_ids"].copy()

        original_max_move = float(data["max_move"])
        original_max_time = float(data["max_time"])

    max_move = float(checkpoint["max_move"])
    max_time = float(checkpoint["max_time"])

    if not np.isclose(original_max_move, max_move):
        raise ValueError("Teacher and BC max_move differ.")

    if not np.isclose(original_max_time, max_time):
        raise ValueError("Teacher and BC max_time differ.")

    env = FrankaAutonomousEnv(
        position_range=args.position_range
    )

    env.max_move = max_move
    env.max_time = max_time

    rng = np.random.default_rng(args.seed)

    dagger_obs = []
    dagger_actions = []
    dagger_ids = []

    rollout_successes = []
    expert_executed = 0
    total_steps = 0

    next_episode_id = int(original_ids.max()) + 1

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode

            obs, _ = env.reset(seed=episode_seed)

            episode_steps = 0

            while True:
                # The expert labels the CURRENT state.
                teacher_action = expert_action(env)

                # The student proposes its own action.
                student_action = predict_action(
                    model,
                    obs,
                    obs_mean,
                    obs_std,
                )

                # Save the student-visited observation
                # and the expert's correct action.
                dagger_obs.append(obs.copy())
                dagger_actions.append(
                    teacher_action.copy()
                )
                dagger_ids.append(
                    next_episode_id + episode
                )

                # Optional expert intervention.
                use_expert = rng.random() < args.beta

                if use_expert:
                    executed_action = teacher_action
                    expert_executed += 1
                else:
                    executed_action = student_action

                obs, _, terminated, truncated, info = (
                    env.step(executed_action)
                )

                total_steps += 1
                episode_steps += 1

                if terminated or truncated:
                    break

            success = bool(info["is_success"])
            rollout_successes.append(success)

            print(
                f"Episode {episode + 1:3d}/"
                f"{args.episodes} | "
                f"Rollout success: {success!s:5s} | "
                f"Steps: {episode_steps:3d} | "
                f"Lift: {info['lift_height']:.3f}m"
            )

    finally:
        env.close()

    dagger_obs = np.asarray(
        dagger_obs, dtype=np.float32
    )
    dagger_actions = np.asarray(
        dagger_actions, dtype=np.float32
    )
    dagger_ids = np.asarray(
        dagger_ids, dtype=np.int32
    )

    combined_obs = np.concatenate(
        [original_obs, dagger_obs],
        axis=0,
    )
    combined_actions = np.concatenate(
        [original_actions, dagger_actions],
        axis=0,
    )
    combined_ids = np.concatenate(
        [original_ids, dagger_ids],
        axis=0,
    )

    assert combined_obs.shape[1] == 12
    assert combined_actions.shape[1] == 4
    assert len(combined_obs) == len(combined_actions)
    assert np.isfinite(combined_obs).all()
    assert np.isfinite(combined_actions).all()

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output,
        observations=combined_obs,
        actions=combined_actions,
        episode_ids=combined_ids,
        max_move=np.float32(max_move),
        max_time=np.float32(max_time),
        position_range=np.float32(
            args.position_range
        ),
        control_dt=np.float32(env.control_dt),
        dagger_rollout_successes=np.asarray(
            rollout_successes,
            dtype=np.bool_,
        ),
        dagger_beta=np.float32(args.beta),
        dagger_seed=np.int64(args.seed),
    )

    print("\n===== DAGGER COLLECTION RESULT =====")
    print(f"Original samples: {len(original_obs)}")
    print(f"DAgger samples: {len(dagger_obs)}")
    print(f"Combined samples: {len(combined_obs)}")
    print(
        "Rollout success: "
        f"{sum(rollout_successes)}/{args.episodes}"
    )
    print(
        "Expert intervention: "
        f"{expert_executed}/{total_steps}"
    )
    print(f"Saved to: {output}")

    print(
        "\nNote: rollout success is from the "
        "student/expert mixture, not the final "
        "DAgger policy."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="bc_models/franka_bc.pt",
    )
    parser.add_argument(
        "--original-data",
        default="demonstrations/franka_expert_train.npz",
    )
    parser.add_argument(
        "--output",
        default="demonstrations/franka_dagger1.npz",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--position-range",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=30000,
    )

    main(parser.parse_args())
