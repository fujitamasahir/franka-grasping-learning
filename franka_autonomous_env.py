
import numpy as np
import mujoco
from gymnasium import spaces

from franka_ai_env import FrankaGraspEnv


class FrankaAutonomousEnv(FrankaGraspEnv):
    def __init__(self, position_range=0.02):
        super().__init__()

        self.position_range = position_range

        # XYZ movement + gripper
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Hand XYZ (3), cube XYZ (3),
        # relative XYZ (3), time (1),
        # lift height (1), gripper command (1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        self.max_time = 12.0
        self.control_dt = 0.1
        self.max_move = 0.025

        self.gripper_command = 255.0
        self.previous_distance = 0.0
        self.hold_steps = 0

    def _observe(self):
        mujoco.mj_forward(self.model, self.data)

        hand = self.data.site_xpos[
            self.site_id
        ].copy()

        cube = self.data.xpos[
            self.cube_id
        ].copy()

        return np.concatenate([
            hand,
            cube,
            cube - hand,
            [self.data.time / self.max_time],
            [cube[2] - self.initial_z],
            [self.gripper_command / 255.0],
        ]).astype(np.float32)

    def _distance_to_grasp(self):
        hand = self.data.site_xpos[
            self.site_id
        ].copy()

        cube = self.data.xpos[
            self.cube_id
        ].copy()

        # Approximate grasp-point target
        target = cube + np.array([0, 0, 0.045])

        return float(np.linalg.norm(hand - target))

    def reset(self, seed=None, options=None):
        # Use the original environment reset
        super().reset(seed=seed)

        # Start with a small position range
        self.cube_x = float(
            self.np_random.uniform(
                0.55 - self.position_range,
                0.55 + self.position_range,
            )
        )
        self.cube_y = float(
            self.np_random.uniform(
                -self.position_range,
                self.position_range,
            )
        )

        self.data.qpos[
            self.cube_qpos:self.cube_qpos + 7
        ] = [
            self.cube_x,
            self.cube_y,
            0.425,
            1, 0, 0, 0,
        ]

        mujoco.mj_forward(self.model, self.data)

        self.initial_z = float(
            self.data.xpos[self.cube_id, 2]
        )

        self.gripper_command = 255.0
        self.ctrl = self.data.ctrl.copy()
        self.ctrl[self.gripper_id] = 255.0
        self.data.ctrl[:] = self.ctrl

        self.previous_distance = (
            self._distance_to_grasp()
        )
        self.hold_steps = 0

        return self._observe(), {}

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        # AI chooses a relative XYZ movement
        hand = self.data.site_xpos[
            self.site_id
        ].copy()

        target = hand + action[:3] * self.max_move

        # Restrict the workspace for safety
        target = np.clip(
            target,
            [0.30, -0.25, 0.43],
            [0.80, 0.25, 0.90],
        )

        # AI controls the gripper
        # action[3] >= 0: open
        # action[3] < 0: close
        self.gripper_command = (
            255.0 if action[3] >= 0 else 0.0
        )
        self.ctrl[self.gripper_id] = (
            self.gripper_command
        )

        steps = max(
            1,
            round(
                self.control_dt /
                self.model.opt.timestep
            ),
        )

        for _ in range(steps):
            self._move_toward(target)
            mujoco.mj_step(
                self.model, self.data
            )

        obs = self._observe()

        distance = self._distance_to_grasp()

        cube_z = float(
            self.data.xpos[self.cube_id, 2]
        )
        lift = cube_z - self.initial_z

        # Reward for getting closer
        progress = (
            self.previous_distance - distance
        )
        self.previous_distance = distance

        
        reward = 20.0 * progress

        # 1. Reward for staying near the grasp target
        near_reward = np.exp(
            -distance / 0.025
        )
        reward += 0.05 * near_reward

        # 2. Encourage closing the gripper near the cube
        gripper_closed = self.gripper_command == 0.0

        if gripper_closed and distance < 0.025:
            reward += 0.10

        # 3. Discourage closing too far away
        if gripper_closed and distance > 0.06:
            reward -= 0.03

        # 4. Reward actual lifting
        lift_reward = np.clip(
            lift / 0.15, 0.0, 1.0
        )
        reward += 0.5 * lift_reward

        # 5. Small movement penalty
        reward -= 0.001 * float(
            np.sum(action[:3] ** 2)
        )

        # 6. Require sustained lifting
        if lift >= 0.10:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        success = self.hold_steps >= 5

        terminated = bool(success)
        truncated = bool(
            self.data.time >= self.max_time
        )

        if success:
            reward += 30.0
        elif truncated:
            reward -= 2.0


        info = {
            "is_success": bool(success),
            "lift_height": float(lift),
            "distance": float(distance),
            "cube_x": self.cube_x,
            "cube_y": self.cube_y,
            "gripper_command": self.gripper_command,
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )
