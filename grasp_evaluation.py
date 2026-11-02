
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "mujoco_menagerie/franka_emika_panda/scene.xml"

# ===== Experiment settings =====
NUM_TRIALS = 20
RANDOM_SEED = 42

# Keep the random positions close to the successful grasp
X_RANGE = (0.53, 0.57)
Y_RANGE = (-0.02, 0.02)

TABLE_Z = 0.40
CUBE_HALF_SIZE = 0.025
CUBE_Z = TABLE_Z + CUBE_HALF_SIZE

LIFT_THRESHOLD = 0.10
END_TIME = 18.0

# Display only the first trial
VISUALIZE_FIRST_TRIAL = True

rng = np.random.default_rng(RANDOM_SEED)

# ===== Build scene =====
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
    pos=[0.55, 0, CUBE_Z],
)
cube.add_freejoint()
cube.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[CUBE_HALF_SIZE] * 3,
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

model = spec.compile()
data = mujoco.MjData(model)

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

GRIPPER = model.actuator("actuator8").id

cube_joint_id = model.body_jntadr[cube_id]
cube_qpos_id = model.jnt_qposadr[cube_joint_id]

# ===== Controller =====
def move_toward(target, ctrl):
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
        2.0 * error,
    )

    target_q = data.qpos[qpos_ids] + dq * 0.05

    for i, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            target_q[i] = np.clip(
                target_q[i],
                *model.jnt_range[jid]
            )

    ctrl[:7] = target_q
    data.ctrl[:] = ctrl


def phase_target(t, cube_x, cube_y):
    if t < 3:
        return "Approach", np.array(
            [cube_x, cube_y, 0.70]
        ), 255

    if t < 7:
        return "Descend", np.array(
            [cube_x, cube_y, 0.47]
        ), 255

    if t < 10:
        return "Grasp", np.array(
            [cube_x, cube_y, 0.47]
        ), 0

    if t < 16:
        return "Lift", np.array(
            [cube_x, cube_y, 0.75]
        ), 0

    return "Hold", np.array(
        [cube_x, cube_y, 0.75]
    ), 0


# ===== Reset environment =====
def reset_environment(cube_x, cube_y):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    # Free joint: position (3) + quaternion (4)
    data.qpos[cube_qpos_id:cube_qpos_id + 7] = [
        cube_x, cube_y, CUBE_Z,
        1, 0, 0, 0
    ]

    mujoco.mj_forward(model, data)
    return data.ctrl.copy()


# ===== Single trial =====
def run_trial(trial_number, cube_x, cube_y, viewer=None):
    ctrl = reset_environment(cube_x, cube_y)

    initial_z = data.xpos[cube_id, 2]
    max_z = initial_z
    last_phase = None

    # Use simulation time, not wall-clock time
    while data.time < END_TIME:
        t = data.time
        phase, target, grip = phase_target(
            t, cube_x, cube_y
        )

        if phase != last_phase:
            if viewer is not None:
                print(f"  {t:.1f}s: {phase}")
            last_phase = phase

        ctrl[GRIPPER] = grip
        move_toward(target, ctrl)

        mujoco.mj_step(model, data)

        cube_z = data.xpos[cube_id, 2]
        max_z = max(max_z, cube_z)

        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)

            if not viewer.is_running():
                raise KeyboardInterrupt(
                    "Viewer closed by user"
                )

    final_z = data.xpos[cube_id, 2]
    lift_height = max_z - initial_z

    # Success requires the cube to remain lifted
    success = (
        final_z - initial_z >= LIFT_THRESHOLD
    )

    return {
        "trial": trial_number,
        "cube_x": round(cube_x, 4),
        "cube_y": round(cube_y, 4),
        "max_lift_m": round(lift_height, 4),
        "final_lift_m": round(final_z - initial_z, 4),
        "success": int(success),
    }


# ===== Run experiments =====
results = []

print(f"Starting {NUM_TRIALS} grasp trials")

for trial in range(1, NUM_TRIALS + 1):
    cube_x = rng.uniform(*X_RANGE)
    cube_y = rng.uniform(*Y_RANGE)

    print(
        f"\nTrial {trial}/{NUM_TRIALS}: "
        f"x={cube_x:.3f}, y={cube_y:.3f}"
    )

    if trial == 1 and VISUALIZE_FIRST_TRIAL:
        with mujoco.viewer.launch_passive(
            model, data
        ) as viewer:
            result = run_trial(
                trial, cube_x, cube_y, viewer
            )
    else:
        result = run_trial(
            trial, cube_x, cube_y
        )

    results.append(result)

    status = "SUCCESS" if result["success"] else "FAILED"
    print(
        f"{status} | "
        f"Final lift: {result['final_lift_m']:.3f} m"
    )

# ===== Save results =====
output = ROOT / "grasp_results.csv"

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

successes = sum(r["success"] for r in results)
success_rate = 100 * successes / NUM_TRIALS

print("\n===== EXPERIMENT RESULTS =====")
print(f"Trials: {NUM_TRIALS}")
print(f"Successful grasps: {successes}")
print(f"Success rate: {success_rate:.1f}%")
print(f"Results saved to: {output}")
