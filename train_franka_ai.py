
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from franka_ai_env import FrankaGraspEnv

train_env = Monitor(FrankaGraspEnv())
eval_env = Monitor(FrankaGraspEnv())

model = SAC(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=256,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    seed=42,
)

callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_franka_model/",
    log_path="./eval_logs/",
    eval_freq=5_000,
    n_eval_episodes=10,
    deterministic=True,
)

model.learn(
    total_timesteps=50_000,
    callback=callback,
    progress_bar=True,
)

model.save("franka_sac")
train_env.close()
eval_env.close()

print("AI training finished!")
