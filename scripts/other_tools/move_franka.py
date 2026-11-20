import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
model_path = Path(__file__).resolve().parents[2] / 'mujoco_menagerie' / 'franka_emika_panda' / 'scene.xml'
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
print(f'Number of actuators: {model.nu}')
for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
initial_ctrl = data.ctrl.copy()
actuator_id = 0
low, high = model.actuator_ctrlrange[actuator_id]
amplitude = min(0.15, high - initial_ctrl[actuator_id], initial_ctrl[actuator_id] - low)
amplitude = max(0.0, amplitude)
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.monotonic()
    while viewer.is_running():
        step_start = time.monotonic()
        t = step_start - start
        data.ctrl[:] = initial_ctrl
        data.ctrl[actuator_id] = initial_ctrl[actuator_id] + amplitude * np.sin(t)
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.monotonic() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
