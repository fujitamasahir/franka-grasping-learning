import time, mujoco, mujoco.viewer, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "mujoco_menagerie/franka_emika_panda/scene.xml"

# ----- Create the scene -----
spec = mujoco.MjSpec.from_file(str(PATH))

# Table: top surface is at z = 0.40
table = spec.worldbody.add_body(
    name="experiment_table", pos=[0.50, 0, 0.375]
)
table.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.30, 0.35, 0.025],
    rgba=[0.55, 0.65, 0.75, 1],
)

# Cube: bottom surface is at z = 0.40
cube = spec.worldbody.add_body(
    name="target_cube", pos=[0.55, 0, 0.425]
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

# Add a reference point to the gripper
hand = spec.body("hand")
hand.add_site(
    name="grasp_point",
    pos=[0, 0, 0.105],
    size=[0.008, 0.008, 0.008],
    rgba=[0, 1, 0, 1],
)

model = spec.compile()
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

site_id = model.site("grasp_point").id
cube_id = model.body("target_cube").id

joint_ids = [
    model.joint(f"joint{i}").id
    for i in range(1, 8)
]
qpos_ids = np.array([
    model.jnt_qposadr[j] for j in joint_ids
])
dof_ids = np.array([
    model.jnt_dofadr[j] for j in joint_ids
])

home = data.qpos[qpos_ids].copy()
ctrl = data.ctrl.copy()

# Gripper actuator: 255=open, 0=closed
GRIPPER = model.actuator("actuator8").id
CUBE_X = 0.55
CUBE_Y = 0.0

def move_toward(target, gain=2.0):
    """Position-only damped least-squares IK."""
    mujoco.mj_forward(model, data)

    current = data.site_xpos[site_id].copy()
    error = target - current

    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model, data, jac_pos, jac_rot, site_id
    )

    J = jac_pos[:, dof_ids]
    damping = 0.08

    dq = J.T @ np.linalg.solve(
        J @ J.T + damping**2 * np.eye(3),
        gain * error,
    )

    # Integrate a small joint-space command
    target_q = data.qpos[qpos_ids] + dq * 0.05

    for i, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            target_q[i] = np.clip(
                target_q[i],
                *model.jnt_range[jid]
            )

    ctrl[:7] = target_q
    data.ctrl[:] = ctrl

def phase_target(t):
    if t < 3:
        return "Approach", np.array([CUBE_X, CUBE_Y, 0.70]), 255
    if t < 7:
        return "Descend", np.array([CUBE_X, CUBE_Y, 0.47]), 255
    if t < 10:
        return "Grasp", np.array([CUBE_X, CUBE_Y, 0.47]), 0
    if t < 16:
        return "Lift", np.array([CUBE_X, CUBE_Y, 0.75]), 0
    return "Hold", np.array([CUBE_X, CUBE_Y, 0.75]), 0

initial_cube_z = data.xpos[cube_id, 2]
previous_phase = None

print("Starting Pick-and-Lift experiment")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.monotonic()

    while viewer.is_running():
        wall_start = time.monotonic()
        t = wall_start - start

        phase, target, grip = phase_target(t)

        if phase == "Lift" and int(t * 2) != int((t - model.opt.timestep) * 2):
            mujoco.mj_forward(model, data)

            current_hand = data.site_xpos[site_id].copy()
            cube_position = data.xpos[cube_id].copy()

            print(
                f"Time: {t:.1f}s | "
                f"Hand Z: {current_hand[2]:.3f} | "
                f"Cube Z: {cube_position[2]:.3f} | "
                f"Target Z: {target[2]:.3f}"
            )

        ctrl[GRIPPER] = grip
        move_toward(target)

        mujoco.mj_step(model, data)
        viewer.sync()

        if t > 13:
            cube_z = data.xpos[cube_id, 2]
            lifted = cube_z > initial_cube_z + 0.10

            print(f"Cube height: {cube_z:.3f} m")
            print("Lift detected:", lifted)
            break

        elapsed = time.monotonic() - wall_start
        time.sleep(max(0, model.opt.timestep - elapsed))
