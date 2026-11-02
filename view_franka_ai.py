
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import SAC

from franka_ai_env import FrankaGraspEnv

ROOT = Path(__file__).resolve().parent

# Prefer the best model saved during evaluation
best_path = ROOT / "best_franka_model" / "best_model.zip"
final_path = ROOT / "franka_sac.zip"

if best_path.exists():
    model_path = best_path
elif final_path.exists():
    model_path = final_path
else:
    raise FileNotFoundError(
        "No trained model found. "
        "Finish train_franka_ai.py first."
    )

print(f"Loading AI model: {model_path}")

ai = SAC.load(str(model_path))

env = FrankaGraspEnv()

# Use a reproducible test episode
obs, info = env.reset(seed=123)

print("Starting AI-controlled grasp")

with mujoco.viewer.launch_passive(
    env.model, env.data
) as viewer:

    previous_time = time.monotonic()

    while viewer.is_running():
        if env.data.time >= env.max_time:
            break

        # AI decides the next XYZ correction
        action, _ = ai.predict(
            obs,
            deterministic=True,
        )

        # Apply action to MuJoCo
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        # Display the updated simulation
        viewer.sync()

        # Each AI action advances 0.1 s in simulation
        now = time.monotonic()
        elapsed = now - previous_time
        time.sleep(max(0, env.control_dt - elapsed))
        previous_time = time.monotonic()

        if terminated or truncated:
            break

    print("\n===== AI GRASP RESULT =====")
    print(f"Cube X: {info['cube_x']:.3f}")
    print(f"Cube Y: {info['cube_y']:.3f}")
    print(f"Lift height: {info['lift_height']:.3f} m")
    print(f"Success: {info['is_success']}")

    # Keep the final scene visible briefly
    if viewer.is_running():
        time.sleep(3)

env.close()
