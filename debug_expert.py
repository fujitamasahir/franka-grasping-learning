
import numpy as np
import mujoco

from franka_autonomous_env import FrankaAutonomousEnv


def expert_action(env):
    target, gripper = env._phase_target()

    mujoco.mj_forward(env.model, env.data)

    hand = env.data.site_xpos[env.site_id].copy()

    xyz = np.clip(
        (target - hand) / env.max_move,
        -1.0,
        1.0,
    )

    grip = 1.0 if gripper == 255 else -1.0

    return np.array([*xyz, grip], dtype=np.float32)


env = FrankaAutonomousEnv(position_range=0.02)
env.max_move = 0.15

# Diagnostic only: allow the original lift phase to finish.
env.max_time = 16.0

obs, _ = env.reset(seed=1000)

print("===== EXPERT DEBUG =====")
print(f"Cube initial X: {env.cube_x:.3f}")
print(f"Cube initial Y: {env.cube_y:.3f}")
print(f"Max time: {env.max_time:.1f}s")

try:
    step_count = 0

    while True:
        action = expert_action(env)
        target, _ = env._phase_target()

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        step_count += 1

        # Print approximately every second.
        if step_count % 10 == 0 or terminated or truncated:
            hand = env.data.site_xpos[env.site_id].copy()
            cube = env.data.xpos[env.cube_id].copy()

            t = env.data.time

            if t < 3:
                phase = "APPROACH"
            elif t < 7:
                phase = "DESCEND"
            elif t < 10:
                phase = "GRASP"
            else:
                phase = "LIFT"

            grip = (
                "OPEN"
                if env.gripper_command == 255.0
                else "CLOSED"
            )

            print(
                f"t={t:5.1f}s | "
                f"phase={phase:8s} | "
                f"hand={np.round(hand, 3)} | "
                f"cube={np.round(cube, 3)} | "
                f"target={np.round(target, 3)} | "
                f"grip={grip:6s} | "
                f"lift={info['lift_height']:.3f}m | "
                f"dist={info['distance']:.3f}m"
            )

        if terminated or truncated:
            print("\n===== RESULT =====")
            print(f"Success: {info['is_success']}")
            print(f"Lift height: {info['lift_height']:.3f}m")
            print(f"Steps: {step_count}")
            break

finally:
    env.close()
