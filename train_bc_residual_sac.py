import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from franka_bc_residual_env import BCResidualEnv


class FixedSeedSuccessEval(BaseCallback):
    """Select best checkpoints by success on fixed, uniform validation seeds."""

    def __init__(self, eval_env, run_dir, eval_freq=10000,
                 n_eval_episodes=40, eval_seed=61000,
                 save_trajectory_checkpoints=False):
        super().__init__()
        self.eval_env = eval_env
        self.run_dir = Path(run_dir)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self.save_trajectory_checkpoints = save_trajectory_checkpoints
        self.best_rate = -1.0
        self.best_steps = float("inf")
        self.history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        # Optional snapshots at validation boundaries. This is separate from
        # best_model selection and never changes its success/step tie-break.
        if self.save_trajectory_checkpoints:
            checkpoint_path = (
                self.run_dir / f"checkpoint_{self.num_timesteps}.zip"
            )
            if checkpoint_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite {checkpoint_path}"
                )
            self.model.save(str(checkpoint_path.with_suffix("")))

        successes = 0
        successful_steps = []
        returns = []

        for episode in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=self.eval_seed + episode)
            total_reward = 0.0
            steps = 0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = (
                    self.eval_env.step(action)
                )
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break

            returns.append(total_reward)
            if info["is_success"]:
                successes += 1
                successful_steps.append(steps)

        rate = successes / self.n_eval_episodes
        mean_steps = (
            float(np.mean(successful_steps))
            if successful_steps else float("inf")
        )
        row = {
            "timesteps": self.num_timesteps,
            "successes": successes,
            "episodes": self.n_eval_episodes,
            "success_rate": rate,
            "mean_steps_success": (
                mean_steps if np.isfinite(mean_steps) else ""
            ),
            "mean_episode_reward": float(np.mean(returns)),
        }
        self.history.append(row)
        with (self.run_dir / "eval_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(self.history)

        # Same validation positions for every checkpoint and every arm.
        # Use mean successful steps only to break success-rate ties.
        if rate > self.best_rate or (
            rate == self.best_rate and mean_steps < self.best_steps
        ):
            self.best_rate = rate
            self.best_steps = mean_steps
            self.model.save(str(self.run_dir / "best_model"))
            print("New best model saved.")

        print(
            f"Eval @ {self.num_timesteps}: "
            f"{successes}/{self.n_eval_episodes} "
            f"({rate:.1%}), best={self.best_rate:.1%}"
        )
        return True


def make_env(args, sampling_mode):
    return BCResidualEnv(
        bc_model=args.bc_model,
        position_range=args.position_range,
        residual_meters=args.residual_meters,
        sampling_mode=sampling_mode,
        hard_inner_range=args.hard_inner_range,
        hard_fraction=args.hard_fraction,
        reward_mode=args.reward_mode,
    )


def main(args):
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run-name may contain only letters, numbers, _ and -")
    if args.timesteps <= 0 or args.eval_freq <= 0 or args.eval_episodes <= 0:
        raise ValueError("Timesteps, eval-freq and eval-episodes must be > 0")

    run_dir = Path("bc_residual_sac_runs") / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} already contains files; choose another --run-name"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    np.random.seed(args.seed)
    train_env = Monitor(make_env(args, args.sampling_mode))
    # Validation is ALWAYS uniform, regardless of training sampling.
    eval_env = make_env(args, "uniform")
    callback = FixedSeedSuccessEval(
        eval_env=eval_env,
        run_dir=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        save_trajectory_checkpoints=args.save_trajectory_checkpoints,
    )

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
        tensorboard_log=str(Path("bc_residual_sac_logs") / args.run_name),
    )

    print("===== BC + RESIDUAL SAC CONTROLLED EXPERIMENT =====")
    print(f"Run: {args.run_name}")
    print(f"Sampling: {args.sampling_mode} | Reward: {args.reward_mode}")
    print(f"BC: {args.bc_model}")
    print(f"Training range: +/-{args.position_range} m")
    print(f"Residual: +/-{args.residual_meters} m per axis")
    print(f"Timesteps: {args.timesteps} | Seed: {args.seed}")
    print("Validation: uniform, fixed seeds 61000 onward")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=False,
        )
        model.save(str(run_dir / "final_model"))
        print(f"Final model: {run_dir / 'final_model.zip'}")
        print(f"Best model: {run_dir / 'best_model.zip'}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--bc-model", default="bc_models/fair_wide_expert_seed42.pt"
    )
    parser.add_argument("--position-range", type=float, default=0.20)
    parser.add_argument("--residual-meters", type=float, default=0.005)
    parser.add_argument(
        "--sampling-mode", choices=["uniform", "hard"], default="uniform"
    )
    parser.add_argument("--hard-inner-range", type=float, default=0.12)
    parser.add_argument("--hard-fraction", type=float, default=0.90)
    parser.add_argument(
        "--reward-mode", choices=["sparse", "shaped"], default="sparse"
    )
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument(
        "--save-trajectory-checkpoints",
        action="store_true",
        help="Save a separate SAC snapshot at every evaluation boundary.",
    )
    main(parser.parse_args())
