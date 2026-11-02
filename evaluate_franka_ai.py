
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from franka_ai_env import FrankaGraspEnv

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "best_franka_model/best_model.zip"

if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "franka_sac.zip"

ai = SAC.load(str(MODEL_PATH))

NUM_TRIALS = 50
rng = np.random.default_rng(12345)

# Wider than the training range
X_RANGE = (0.42, 0.68)
Y_RANGE = (-0.12, 0.12)

env = FrankaGraspEnv()
results = []

# Generate test positions once so both controllers
# are evaluated at exactly the same positions.
positions = [
    (
        rng.uniform(*X_RANGE),
        rng.uniform(*Y_RANGE),
    )
    for _ in range(NUM_TRIALS)
]

def run_trial(x, y, use_ai, correction_scale=0.03):
    obs, _ = env.reset()

    env.cube_x = float(x)
    env.cube_y = float(y)

    env.data.qpos[
        env.cube_qpos:env.cube_qpos + 7
    ] = [x, y, 0.425, 1, 0, 0, 0]

    import mujoco
    mujoco.mj_forward(env.model, env.data)

    env.correction_scale = correction_scale
    obs = env._observe()

    done = False
    info = {}

    while not done:
        if use_ai:
            action, _ = ai.predict(
                obs, deterministic=True
            )
        else:
            action = np.zeros(3, dtype=np.float32)

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

    return info

for i, (x, y) in enumerate(positions, start=1):
    print(f"Trial {i}/{NUM_TRIALS}")

    methods = [
        ("Rule-based", False, 0.0),
        ("SAC 3cm", True, 0.03),
        ("SAC 1cm", True, 0.01),
        ("SAC 0.5cm", True, 0.005),
    ]

    for method, use_ai, scale in methods:
        info = run_trial(
            x, y, use_ai, correction_scale=scale
        )

        results.append({
            "trial": i,
            "method": method,
            "cube_x": round(x, 4),
            "cube_y": round(y, 4),
            "success": int(info["is_success"]),
            "lift_height": round(
                info["lift_height"], 4
            ),
        })

df = pd.DataFrame(results)

print("\n===== RESULTS =====")
print(
    df.groupby("method")["success"].agg(
        ["sum", "count", "mean"]
    )
)

df.to_csv(
    ROOT / "franka_ai_comparison.csv",
    index=False
)

print("\nSaved: franka_ai_comparison.csv")

env.close()
