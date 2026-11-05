import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from franka_autonomous_env import FrankaAutonomousEnv
from evaluate_behavior_cloning import load_policy, predict_action


class BCResidualEnv(gym.Wrapper):
    """BC policy plus SAC-controlled XYZ residuals.

    The gripper is always controlled by BC. Training can sample the
    outer position region more frequently; evaluation remains uniform.
    """

    def __init__(
        self,
        bc_model="bc_models/franka_bc_dagger1.pt",
        position_range=0.12,
        residual_meters=0.01,
        sampling_mode="uniform",
        hard_inner_range=0.12,
        hard_fraction=0.90,
        reward_mode="sparse",
    ):
        if sampling_mode not in ("uniform", "hard"):
            raise ValueError("sampling_mode must be 'uniform' or 'hard'")
        if reward_mode not in ("sparse", "shaped"):
            raise ValueError("reward_mode must be 'sparse' or 'shaped'")
        if not 0.0 <= hard_fraction <= 1.0:
            raise ValueError("hard_fraction must be between 0 and 1")
        if sampling_mode == "hard" and not 0 < hard_inner_range < position_range:
            raise ValueError("Require 0 < hard_inner_range < position_range")
        if not 0 < residual_meters <= 0.03:
            raise ValueError("residual_meters must be in (0, 0.03]")

        base_env = FrankaAutonomousEnv(position_range=position_range)
        super().__init__(base_env)

        (
            self.bc_policy,
            self.obs_mean,
            self.obs_std,
            checkpoint,
        ) = load_policy(bc_model)

        self.env.max_move = float(checkpoint["max_move"])
        self.env.max_time = float(checkpoint["max_time"])

        self.residual_meters = float(residual_meters)
        self.sampling_mode = sampling_mode
        self.hard_inner_range = float(hard_inner_range)
        self.hard_fraction = float(hard_fraction)
        self.reward_mode = reward_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.current_obs = None
        self.previous_distance = 0.0
        self.previous_lift = 0.0
        self.hard_sampled = False

    def _sample_position(self):
        """Sample 90% outside the BC square, 10% inside by default."""
        rng = self.env.np_random
        outer = self.env.position_range
        inner = self.hard_inner_range
        self.hard_sampled = bool(rng.random() < self.hard_fraction)

        if self.hard_sampled:
            # Rejection sampling is uniform over the outer square ring.
            while True:
                dx, dy = rng.uniform(-outer, outer, size=2)
                if abs(dx) > inner or abs(dy) > inner:
                    break
        else:
            dx, dy = rng.uniform(-inner, inner, size=2)

        return float(0.55 + dx), float(dy)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.hard_sampled = False

        if self.sampling_mode == "hard":
            x, y = self._sample_position()
            self.env.cube_x, self.env.cube_y = x, y
            start = self.env.cube_qpos
            self.env.data.qpos[start:start + 7] = [
                x, y, 0.425, 1, 0, 0, 0
            ]
            mujoco.mj_forward(self.env.model, self.env.data)
            self.env.initial_z = float(
                self.env.data.xpos[self.env.cube_id, 2]
            )
            self.env.previous_distance = self.env._distance_to_grasp()
            self.env.hold_steps = 0
            obs = self.env._observe()

        self.previous_distance = self.env._distance_to_grasp()
        self.previous_lift = 0.0
        self.current_obs = obs.copy()
        info = dict(info)
        info["hard_sampled"] = self.hard_sampled
        return obs, info

    def step(self, action):
        if self.current_obs is None:
            raise RuntimeError("Call reset() first")

        residual = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0
        )
        if residual.shape != (3,):
            raise ValueError("Expected action shape (3,)")

        bc_action = predict_action(
            self.bc_policy,
            self.current_obs,
            self.obs_mean,
            self.obs_std,
        )
        final_action = bc_action.copy()
        final_action[:3] = np.clip(
            bc_action[:3]
            + residual * self.residual_meters / self.env.max_move,
            -1.0,
            1.0,
        )
        # BC retains full control of the gripper.
        final_action[3] = bc_action[3]

        obs, _, terminated, truncated, info = self.env.step(final_action)
        self.current_obs = obs.copy()

        # Identical sparse task reward for all three experimental arms.
        reward = -0.01 - 0.005 * float(np.sum(residual ** 2))
        if info["is_success"]:
            reward += 20.0
        elif truncated:
            reward -= 5.0

        distance = float(info["distance"])
        lift = float(info["lift_height"])
        distance_progress = float(np.clip(
            self.previous_distance - distance, -0.03, 0.03
        ))
        lift_progress = float(np.clip(
            lift - self.previous_lift, -0.03, 0.03
        ))

        shaping_reward = 0.0
        if self.reward_mode == "shaped":
            # Reward progress, not absolute proximity: avoids rewarding
            # staying still near the cube at every time step.
            shaping_reward = (
                10.0 * distance_progress
                + 15.0 * lift_progress
            )
            reward += shaping_reward

        self.previous_distance = distance
        self.previous_lift = lift

        info = dict(info)
        info.update(
            hard_sampled=self.hard_sampled,
            shaping_reward=float(shaping_reward),
            distance_progress=distance_progress,
            lift_progress=lift_progress,
            residual_norm=float(
                np.linalg.norm(residual * self.residual_meters)
            ),
            bc_action=bc_action.copy(),
            executed_action=final_action.copy(),
        )
        return obs, float(reward), terminated, truncated, info
