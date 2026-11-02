
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Load Franka Panda
model_path = (
    Path(__file__).resolve().parent
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Initialize the robot's default pose
mujoco.mj_resetDataKeyframe(model, data, 0)

print(f"Number of actuators: {model.nu}")

# Print actuator names
for i in range(model.nu):
    print(i, mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
    ))

# Save the default control values
initial_ctrl = data.ctrl.copy()

# Move the first arm actuator slowly
actuator_id = 0

# Keep the command within the actuator's limits
low, high = model.actuator_ctrlrange[actuator_id]
amplitude = min(
    0.15,
    high - initial_ctrl[actuator_id],
    initial_ctrl[actuator_id] - low
)

amplitude = max(0.0, amplitude)

# Open the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.monotonic()

    while viewer.is_running():
        step_start = time.monotonic()
        t = step_start - start

        # Start from the default pose
        data.ctrl[:] = initial_ctrl

        # Oscillate the first actuator
        data.ctrl[actuator_id] = (
            initial_ctrl[actuator_id]
            + amplitude * np.sin(t)
        )

        # Advance physics
        mujoco.mj_step(model, data)

        # Update visualization
        viewer.sync()

        # Run approximately in real time
        elapsed = time.monotonic() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
