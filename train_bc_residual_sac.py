
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from franka_bc_residual_env import BCResidualEnv


def make_env(args, position_range):
    return Monitor(
        BCResidualEnv(
            bc_model=args.bc_model,
            position_range=position_range,
            residual_meters=args.residual_meters,
        )
    )


def main(args):
    np.random.seed(args.seed)

    train_env = make_env(
        args,
        args.position_range,
    )

    eval_env = make_env(
        args,
        args.position_range,
    )

    # Independent evaluation random-number stream.
    eval_env.reset(seed=61000)

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        device="cpu",
        seed=args.seed,
        verbose=1,
        tensorboard_log="bc_residual_sac_logs",
    )

    callback = EvalCallback(
        eval_env,
        best_model_save_path="bc_residual_sac_best",
        log_path="bc_residual_sac_eval",
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    print("===== BC + RESIDUAL SAC TRAINING =====")
    print(f"BC model: {args.bc_model}")
    print(f"Training range: ±{args.position_range}m")
    print(f"Residual limit: ±{args.residual_meters}m")
    print(f"Timesteps: {args.timesteps}")
    print("Device: CPU")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=False,
        )

        output = Path("bc_residual_sac_final")
        model.save(str(output))

        print("\n===== TRAINING FINISHED =====")
        print("Final model: bc_residual_sac_final.zip")
        print(
            "Best model: "
            "bc_residual_sac_best/best_model.zip"
        )

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bc-model",
        default="bc_models/franka_bc_dagger1.pt",
    )

    parser.add_argument(
        "--position-range",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--residual-meters",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    main(parser.parse_args())
