
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    ROOT / "mujoco_menagerie"
    / "franka_emika_panda" / "scene.xml"
)

# Load the existing Franka Panda scene
spec = mujoco.MjSpec.from_file(str(MODEL_PATH))

# Add a table as a static box
table = spec.worldbody.add_body(
    name="experiment_table",
    pos=[0.55, 0.0, 0.35]
)
table.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.30, 0.35, 0.025],
    rgba=[0.55, 0.65, 0.75, 1.0]
)

# Add a cube that can move freely
cube = spec.worldbody.add_body(
    name="target_cube",
    pos=[0.55, 0.0, 0.405]
)
cube.add_freejoint()
cube.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.025, 0.025, 0.025],
    mass=0.1,
    rgba=[0.9, 0.15, 0.1, 1.0],
    friction=[1.0, 0.005, 0.0001]
)

model = spec.compile()
data = mujoco.MjData(model)

# Start from the robot's home configuration
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.mj_forward(model, data)

print("Model loaded.")
print("Actuators:", model.nu)
print("Bodies:", model.nbody)

# Print actuator names and control ranges
for i in range(model.nu):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
    )
    print(
        i, name,
        model.actuator_ctrlrange[i]
    )

# Print sites that can be used as end effectors
print("\nAvailable sites:")
for i in range(model.nsite):
    print(
        i,
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SITE, i
        )
    )

# Display the scene
mujoco.viewer.launch(model, data)
