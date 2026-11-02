
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)

from franka_autonomous_env import FrankaAutonomousEnv

train_env = Monitor(
    FrankaAutonomousEnv(position_range=0.02)
)

eval_env = Monitor(
    FrankaAutonomousEnv(position_range=0.02)
)

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=256,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./autonomous_v2_logs/",
    seed=42,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./autonomous_v2_best/",
    log_path="./autonomous_v2_eval/",
    eval_freq=5_000,
    n_eval_episodes=10,
    deterministic=True,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./autonomous_checkpoints/",
    name_prefix="franka_autonomous",
)

model.learn(
    total_timesteps=100_000,
    callback=[
        eval_callback,
        checkpoint_callback,
    ],
    progress_bar=True,
)

model.save("franka_autonomous_v2_sac")

train_env.close()
eval_env.close()

print("Autonomous SAC training finished!")
