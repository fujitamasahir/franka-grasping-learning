
from stable_baselines3.common.env_checker import check_env
from franka_autonomous_env import FrankaAutonomousEnv

env = FrankaAutonomousEnv()

check_env(env, warn=True)

print("Environment check passed!")

obs, info = env.reset(seed=42)

print("Observation shape:", obs.shape)
print("Action shape:", env.action_space.shape)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = (
        env.step(action)
    )
    print(
        f"Step {i+1}: "
        f"reward={reward:.3f}, "
        f"distance={info['distance']:.3f}"
    )

    if terminated or truncated:
        obs, info = env.reset()

env.close()
