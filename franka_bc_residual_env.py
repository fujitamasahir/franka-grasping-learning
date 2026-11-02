
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from franka_autonomous_env import FrankaAutonomousEnv
from evaluate_behavior_cloning import (
    load_policy,
    predict_action,
)


class BCResidualEnv(gym.Wrapper):
    def __init__(
        self,
        bc_model="bc_models/franka_bc_dagger1.pt",
        position_range=0.12,
        residual_meters=0.01,
    ):
        base_env = FrankaAutonomousEnv(
            position_range=position_range
        )

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

        if not 0 < self.residual_meters <= 0.03:
            raise ValueError(
                "residual_meters must be in (0, 0.03]."
            )

        # SAC controls only XYZ residuals.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.current_obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(
            seed=seed,
            options=options,
        )

        self.current_obs = obs.copy()
        return obs, info

    def step(self, action):
        if self.current_obs is None:
            raise RuntimeError("Call reset() first.")

        residual = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        if residual.shape != (3,):
            raise ValueError("Expected action shape (3,).")

        bc_action = predict_action(
            self.bc_policy,
            self.current_obs,
            self.obs_mean,
            self.obs_std,
        )

        final_action = bc_action.copy()

        # Convert physical correction in meters into
        # the normalized action used by the base env.
        final_action[:3] = np.clip(
            bc_action[:3]
            + residual
            * self.residual_meters
            / self.env.max_move,
            -1.0,
            1.0,
        )

        # Gripper command remains entirely controlled by BC.
        final_action[3] = bc_action[3]

        obs, _, terminated, truncated, info = (
            self.env.step(final_action)
        )

        self.current_obs = obs.copy()

        # Sparse task reward, not the old proximity reward.
        reward = -0.01
        reward -= 0.005 * float(
            np.sum(residual ** 2)
        )

        if info["is_success"]:
            reward += 20.0
        elif truncated:
            reward -= 5.0

        info = dict(info)
        info["residual_norm"] = float(
            np.linalg.norm(
                residual * self.residual_meters
            )
        )
        info["bc_action"] = bc_action.copy()
        info["executed_action"] = final_action.copy()

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )
