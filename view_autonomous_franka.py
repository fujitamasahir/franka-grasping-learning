
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import SAC

from franka_autonomous_env import FrankaAutonomousEnv

ROOT = Path(__file__).resolve().parent

BEST_MODEL = ROOT / "autonomous_v2_best/best_model.zip"
FINAL_MODEL = ROOT / "franka_autonomous_v2_sac.zip"

if BEST_MODEL.exists():
    model_path = BEST_MODEL
elif FINAL_MODEL.exists():
    model_path = FINAL_MODEL
else:
    raise FileNotFoundError(
        "No trained model found."
    )

print(f"Loading model: {model_path}")

ai = SAC.load(str(model_path))

env = FrankaAutonomousEnv(position_range=0.02)

obs, info = env.reset(seed=123)

print("\nStarting autonomous grasp experiment")
print(f"Cube position: X={env.cube_x:.3f}, Y={env.cube_y:.3f}")

with mujoco.viewer.launch_passive(
    env.model, env.data
) as viewer:

    step_count = 0

    while viewer.is_running():
        start = time.monotonic()

        action, _ = ai.predict(
            obs, deterministic=True
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        viewer.sync()

        step_count += 1

        if step_count % 10 == 0:
            gripper = (
                "OPEN" if action[3] >= 0
                else "CLOSED"
            )

            print(
                f"Time: {env.data.time:.1f}s | "
                f"Distance: {info['distance']:.3f}m | "
                f"Lift: {info['lift_height']:.3f}m | "
                f"Gripper: {gripper}"
            )

        if terminated or truncated:
            break

        elapsed = time.monotonic() - start
        time.sleep(
            max(0, env.control_dt - elapsed)
        )

    print("\n===== RESULT =====")
    print(f"Success: {info['is_success']}")
    print(f"Lift height: {info['lift_height']:.3f} m")
    print(f"Final distance: {info['distance']:.3f} m")
    print(f"Steps: {step_count}")

    if viewer.is_running():
        time.sleep(3)

env.close()
