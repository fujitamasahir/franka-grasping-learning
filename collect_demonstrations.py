
import argparse
from pathlib import Path

import numpy as np
import mujoco

from franka_autonomous_env import FrankaAutonomousEnv


# The configuration that succeeded in debug_expert.py
POSITION_RANGE = 0.02
MAX_MOVE = 0.15
MAX_TIME = 16.0


def make_env():
    env = FrankaAutonomousEnv(
        position_range=POSITION_RANGE
    )
    env.max_move = MAX_MOVE
    env.max_time = MAX_TIME
    return env


def expert_action(env):
    """Convert the rule-based target to a 4D action."""

    target, gripper = env._phase_target()

    mujoco.mj_forward(env.model, env.data)

    hand = env.data.site_xpos[env.site_id].copy()

    xyz_action = np.clip(
        (target - hand) / env.max_move,
        -1.0,
        1.0,
    )

    gripper_action = (
        1.0 if gripper == 255 else -1.0
    )

    return np.array(
        [*xyz_action, gripper_action],
        dtype=np.float32,
    )


def collect(num_episodes, seed, output):
    env = make_env()

    all_observations = []
    all_actions = []
    all_episode_ids = []

    episode_successes = []
    episode_lengths = []
    episode_seeds = []

    try:
        for episode in range(num_episodes):
            episode_seed = seed + episode

            obs, _ = env.reset(seed=episode_seed)

            episode_obs = []
            episode_actions = []

            while True:
                action = expert_action(env)

                # Store the observation BEFORE applying the action.
                episode_obs.append(obs.copy())
                episode_actions.append(action.copy())

                next_obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                obs = next_obs

                if terminated or truncated:
                    break

            success = bool(info["is_success"])

            episode_successes.append(success)
            episode_lengths.append(len(episode_obs))
            episode_seeds.append(episode_seed)

            # Only successful trajectories become BC data.
            if success:
                all_observations.extend(episode_obs)
                all_actions.extend(episode_actions)
                all_episode_ids.extend(
                    [episode] * len(episode_obs)
                )

            print(
                f"Episode {episode + 1:3d}/{num_episodes} | "
                f"Success: {success!s:5s} | "
                f"Steps: {len(episode_obs):3d} | "
                f"Lift: {info['lift_height']:.3f}m"
            )

    finally:
        env.close()

    successes = np.asarray(
        episode_successes,
        dtype=np.bool_,
    )

    print("\n===== COLLECTION RESULT =====")
    print(f"Episodes: {num_episodes}")
    print(f"Successful: {int(successes.sum())}")
    print(f"Success rate: {successes.mean():.1%}")
    print(f"Successful samples: {len(all_observations)}")

    if not all_observations:
        print(
            "\nNo successful demonstrations. "
            "Dataset was not saved."
        )
        return

    observations = np.asarray(
        all_observations,
        dtype=np.float32,
    )

    actions = np.asarray(
        all_actions,
        dtype=np.float32,
    )

    assert observations.ndim == 2
    assert observations.shape[1] == 12
    assert actions.shape == (len(observations), 4)
    assert np.isfinite(observations).all()
    assert np.isfinite(actions).all()
    assert np.max(np.abs(actions)) <= 1.00001

    output = Path(output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output,
        observations=observations,
        actions=actions,
        episode_ids=np.asarray(
            all_episode_ids,
            dtype=np.int32,
        ),
        successes=successes,
        episode_lengths=np.asarray(
            episode_lengths,
            dtype=np.int32,
        ),
        episode_seeds=np.asarray(
            episode_seeds,
            dtype=np.int64,
        ),
        max_move=np.float32(MAX_MOVE),
        max_time=np.float32(MAX_TIME),
        position_range=np.float32(POSITION_RANGE),
        control_dt=np.float32(0.1),
    )

    print(f"Observation shape: {observations.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demonstrations/franka_expert_test.npz",
    )

    args = parser.parse_args()

    collect(
        num_episodes=args.episodes,
        seed=args.seed,
        output=args.output,
    )
