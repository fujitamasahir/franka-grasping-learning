
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import SAC

from franka_ai_env import FrankaGraspEnv

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_franka_model/best_model.zip"

if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "franka_sac.zip"

ai = SAC.load(str(MODEL_PATH))
env = FrankaGraspEnv()

failed_positions = [
    (0.4412, -0.0816),
    (0.4703, -0.0889),
    (0.4934, -0.1037),
    (0.4848, -0.0995),
    (0.5014, -0.0770),
]

for x, y in failed_positions:
    obs, _ = env.reset(seed=42)

    env.cube_x = x
    env.cube_y = y

    env.data.qpos[
        env.cube_qpos:env.cube_qpos + 7
    ] = [x, y, 0.425, 1, 0, 0, 0]

    mujoco.mj_forward(env.model, env.data)
    obs = env._observe()

    print(f"\n===== Cube ({x:.4f}, {y:.4f}) =====")

    previous_phase = None
    actions_by_phase = {}

    done = False

    while not done:
        t = env.data.time

        if t < 3:
            phase = "Approach"
        elif t < 7:
            phase = "Descend"
        elif t < 10:
            phase = "Grasp"
        elif t < 16:
            phase = "Lift"
        else:
            phase = "Hold"

        action, _ = ai.predict(
            obs, deterministic=True
        )

        actions_by_phase.setdefault(
            phase, []
        ).append(action.copy())

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

    for phase, actions in actions_by_phase.items():
        mean_action = np.mean(actions, axis=0)
        correction_cm = mean_action * 3.0

        print(
            f"{phase:8s} | "
            f"Mean correction (cm): "
            f"X={correction_cm[0]:+.2f}, "
            f"Y={correction_cm[1]:+.2f}, "
            f"Z={correction_cm[2]:+.2f}"
        )

    print(
        f"Result: {'SUCCESS' if info['is_success'] else 'FAILED'}"
    )
    print(f"Lift height: {info['lift_height']:.3f} m")

env.close()
