
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "mujoco_menagerie/franka_emika_panda/scene.xml"

class FrankaGraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        spec = mujoco.MjSpec.from_file(str(PATH))

        table = spec.worldbody.add_body(
            name="experiment_table",
            pos=[0.50, 0, 0.375],
        )
        table.add_geom(
            name="table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.30, 0.35, 0.025],
            rgba=[0.55, 0.65, 0.75, 1],
        )

        cube = spec.worldbody.add_body(
            name="target_cube",
            pos=[0.55, 0, 0.425],
        )
        cube.add_freejoint()
        cube.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025, 0.025, 0.025],
            mass=0.10,
            rgba=[0.9, 0.15, 0.1, 1],
            friction=[1.2, 0.005, 0.0001],
        )

        hand = spec.body("hand")
        hand.add_site(
            name="grasp_point",
            pos=[0, 0, 0.105],
            size=[0.008, 0.008, 0.008],
            rgba=[0, 1, 0, 1],
        )

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.site_id = self.model.site("grasp_point").id
        self.cube_id = self.model.body("target_cube").id
        self.gripper_id = self.model.actuator("actuator8").id

        joint_ids = [
            self.model.joint(f"joint{i}").id
            for i in range(1, 8)
        ]
        self.qpos_ids = np.array([
            self.model.jnt_qposadr[j] for j in joint_ids
        ])
        self.dof_ids = np.array([
            self.model.jnt_dofadr[j] for j in joint_ids
        ])
        self.joint_ids = joint_ids

        cube_joint = self.model.body_jntadr[self.cube_id]
        self.cube_qpos = self.model.jnt_qposadr[cube_joint]

        # AI controls a residual XYZ correction
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        # Hand XYZ, cube XYZ, relative XYZ,
        # normalized phase, lift height
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )

        self.control_dt = 0.1
        self.correction_scale = 0.03
        self.max_time = 18.0
        self.cube_x = 0.55
        self.cube_y = 0.0
        self.initial_z = 0.425
        self.ctrl = np.zeros(self.model.nu)

    def _observe(self):
        mujoco.mj_forward(self.model, self.data)

        hand = self.data.site_xpos[self.site_id].copy()
        cube = self.data.xpos[self.cube_id].copy()

        return np.concatenate([
            hand,
            cube,
            cube - hand,
            [self.data.time / self.max_time],
            [cube[2] - self.initial_z],
        ]).astype(np.float32)

    def _phase_target(self):
        t = self.data.time
        x, y = self.cube_x, self.cube_y

        if t < 3:
            return np.array([x, y, 0.70]), 255
        if t < 7:
            return np.array([x, y, 0.47]), 255
        if t < 10:
            return np.array([x, y, 0.47]), 0
        return np.array([x, y, 0.75]), 0

    def _move_toward(self, target):
        mujoco.mj_forward(self.model, self.data)

        error = (
            target - self.data.site_xpos[self.site_id]
        )

        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        mujoco.mj_jacSite(
            self.model, self.data,
            jac_pos, jac_rot, self.site_id,
        )

        J = jac_pos[:, self.dof_ids]
        damping = 0.08

        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3),
            2.0 * error,
        )

        target_q = (
            self.data.qpos[self.qpos_ids]
            + dq * 0.05
        )

        for i, jid in enumerate(self.joint_ids):
            if self.model.jnt_limited[jid]:
                target_q[i] = np.clip(
                    target_q[i],
                    *self.model.jnt_range[jid],
                )

        self.ctrl[:7] = target_q
        self.data.ctrl[:] = self.ctrl

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(
            self.model, self.data, 0
        )

        # Broader positions than the previous experiment
        self.cube_x = self.np_random.uniform(0.49, 0.61)
        self.cube_y = self.np_random.uniform(-0.06, 0.06)

        self.data.qpos[
            self.cube_qpos:self.cube_qpos + 7
        ] = [
            self.cube_x, self.cube_y, 0.425,
            1, 0, 0, 0,
        ]

        mujoco.mj_forward(self.model, self.data)
        self.initial_z = self.data.xpos[self.cube_id, 2]
        self.ctrl = self.data.ctrl.copy()

        return self._observe(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)

        # AI chooses up to +/- 3 cm XYZ correction
        correction = action * self.correction_scale

        target, grip = self._phase_target()
        target = target + correction
        self.ctrl[self.gripper_id] = grip

        # Simulate 0.1 seconds
        steps = max(
            1,
            round(self.control_dt / self.model.opt.timestep)
        )

        for _ in range(steps):
            self._move_toward(target)
            mujoco.mj_step(self.model, self.data)

        obs = self._observe()

        cube_z = self.data.xpos[self.cube_id, 2]
        lift = max(0.0, cube_z - self.initial_z)

        # Dense reward + small action penalty
        reward = (
            5.0 * min(lift, 0.30)
            - 0.002 * float(np.sum(action**2))
        )

        terminated = False
        truncated = self.data.time >= self.max_time

        success = lift >= 0.10

        if truncated:
            reward += 10.0 if success else -2.0

        info = {
            "is_success": bool(success and truncated),
            "lift_height": float(lift),
            "cube_x": self.cube_x,
            "cube_y": self.cube_y,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        pass
