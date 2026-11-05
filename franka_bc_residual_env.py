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
        residual_penalty_coefficient=0.005,
        residual_penalty_coef=None,
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
        # Keep the old abbreviated keyword as a backward-compatible alias.
        if residual_penalty_coef is not None:
            if (
                residual_penalty_coefficient != 0.005
                and residual_penalty_coefficient != residual_penalty_coef
            ):
                raise ValueError("Specify only one residual penalty coefficient")
            residual_penalty_coefficient = residual_penalty_coef
        if residual_penalty_coefficient < 0:
            raise ValueError("residual_penalty_coefficient must be >= 0")
        self.residual_penalty_coefficient = float(residual_penalty_coefficient)
        self.residual_penalty_coef = self.residual_penalty_coefficient

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.current_obs = None
        self.previous_distance = 0.0
        self.previous_lift = 0.0
        self.hard_sampled = False

    def training_state_dict(self):
        """Capture state needed to resume at a saved SAC step boundary."""
        base = self.env
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration_state = np.empty(
            mujoco.mj_stateSize(base.model, state_spec), dtype=np.float64
        )
        mujoco.mj_getState(base.model, base.data, integration_state, state_spec)
        return {
            "integration_state": integration_state,
            "ctrl": base.data.ctrl.copy(),
            "cube_x": base.cube_x,
            "cube_y": base.cube_y,
            "initial_z": base.initial_z,
            "base_ctrl": base.ctrl.copy(),
            "gripper_command": base.gripper_command,
            "base_previous_distance": base.previous_distance,
            "hold_steps": base.hold_steps,
            "base_rng": base.np_random.bit_generator.state,
            "current_obs": None if self.current_obs is None else self.current_obs.copy(),
            "previous_distance": self.previous_distance,
            "previous_lift": self.previous_lift,
            "hard_sampled": self.hard_sampled,
        }

    def load_training_state_dict(self, state):
        base = self.env
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(
            base.model, base.data, state["integration_state"], state_spec
        )
        base.data.ctrl[:] = state["ctrl"]
        mujoco.mj_forward(base.model, base.data)
        base.cube_x = state["cube_x"]
        base.cube_y = state["cube_y"]
        base.initial_z = state["initial_z"]
        base.ctrl = state["base_ctrl"].copy()
        base.gripper_command = state["gripper_command"]
        base.previous_distance = state["base_previous_distance"]
        base.hold_steps = state["hold_steps"]
        base.np_random.bit_generator.state = state["base_rng"]
        self.current_obs = (
            None if state["current_obs"] is None else state["current_obs"].copy()
        )
        self.previous_distance = state["previous_distance"]
        self.previous_lift = state["previous_lift"]
        self.hard_sampled = state["hard_sampled"]

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
        commanded_residual_m = residual * self.residual_meters
        applied_residual_m = (
            final_action[:3] - bc_action[:3]
        ) * self.env.max_move
        residual_clip_difference_m = commanded_residual_m - applied_residual_m

        obs, _, terminated, truncated, info = self.env.step(final_action)
        self.current_obs = obs.copy()

        # Keep task, regularization, and shaping terms separately observable.
        task_reward = -0.01
        if info["is_success"]:
            task_reward += 20.0
        elif truncated:
            task_reward -= 5.0
        residual_action_penalty = -self.residual_penalty_coefficient * float(
            np.sum(residual ** 2)
        )

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
        reward = task_reward + residual_action_penalty + shaping_reward

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
            residual_action_norm=float(np.linalg.norm(residual)),
            residual_action_max_abs=float(np.max(np.abs(residual))),
            action_saturated=bool(np.any(np.abs(residual) >= 0.95)),
            commanded_residual_xyz_m=commanded_residual_m.copy(),
            applied_residual_xyz_m=applied_residual_m.copy(),
            residual_clip_difference_xyz_m=residual_clip_difference_m.copy(),
            applied_residual_norm_m=float(np.linalg.norm(applied_residual_m)),
            residual_clip_difference_norm_m=float(
                np.linalg.norm(residual_clip_difference_m)
            ),
            task_reward=float(task_reward),
            residual_action_penalty=float(residual_action_penalty),
            bc_action=bc_action.copy(),
            executed_action=final_action.copy(),
        )
        return obs, float(reward), terminated, truncated, info
