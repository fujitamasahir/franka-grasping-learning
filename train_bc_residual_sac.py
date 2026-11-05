import argparse
import csv
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from franka_bc_residual_env import BCResidualEnv


class FixedSeedSuccessEval(BaseCallback):
    """Select best checkpoints by success on fixed, uniform validation seeds."""

    def __init__(self, eval_env, run_dir, eval_freq=10000,
                 n_eval_episodes=40, eval_seed=61000,
                 save_trajectory_checkpoints=False,
                 save_replay_checkpoints=False):
        super().__init__()
        self.eval_env = eval_env
        self.run_dir = Path(run_dir)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self.save_trajectory_checkpoints = save_trajectory_checkpoints
        self.save_replay_checkpoints = save_replay_checkpoints
        self.best_rate = -1.0
        self.best_steps = float("inf")
        history_path = self.run_dir / "eval_history.csv"
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as f:
                self.history = list(csv.DictReader(f))
            if self.history:
                best = max(float(row["success_rate"]) for row in self.history)
                best_rows = [row for row in self.history if float(row["success_rate"]) == best]
                self.best_rate = best
                self.best_steps = min(
                    (float(row["mean_steps_success"]) if row["mean_steps_success"] else float("inf"))
                    for row in best_rows
                )
        else:
            self.history = []

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

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
        # Save only after this milestone's validation is complete. The replay
        # buffer and exact training environment/RNG state enable safe resume.
        if self.save_trajectory_checkpoints:
            checkpoint_path = self.run_dir / f"checkpoint_{self.num_timesteps}.zip"
            if checkpoint_path.exists():
                raise FileExistsError(f"Refusing to overwrite {checkpoint_path}")
            self.model.save(str(checkpoint_path.with_suffix("")))
            if self.save_replay_checkpoints:
                self._save_resume_state()
        return True

    def _save_resume_state(self):
        vec_env = self.training_env
        monitor_env = vec_env.envs[0]
        base_env = monitor_env.env
        state = {
            "timesteps": self.num_timesteps,
            # SB3 invokes this callback before updating these two fields.
            "last_obs": self.locals["new_obs"],
            "last_episode_starts": self.locals["dones"],
            "environment": base_env.training_state_dict(),
            "monitor_state": {
                "rewards": list(monitor_env.rewards),
                "needs_reset": monitor_env.needs_reset,
                "episode_returns": list(monitor_env.episode_returns),
                "episode_lengths": list(monitor_env.episode_lengths),
                "episode_times": list(monitor_env.episode_times),
                "total_steps": monitor_env.total_steps,
                "current_reset_info": dict(monitor_env.current_reset_info),
            },
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
        }
        replay_path = self.run_dir / f"replay_buffer_{self.num_timesteps}.pkl"
        state_path = self.run_dir / f"resume_state_{self.num_timesteps}.pkl"
        for path in (replay_path, state_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
        replay_temp = replay_path.with_name(
            replay_path.stem + ".tmp.pkl"
        )
        state_temp = state_path.with_suffix(state_path.suffix + ".tmp")
        self.model.save_replay_buffer(str(replay_temp))
        with state_temp.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(replay_temp, replay_path)
        os.replace(state_temp, state_path)


class TrainingDiagnosticsCallback(BaseCallback):
    """Collect per-episode diagnostics from existing rollout info only."""

    FIELDS = (
        "task_reward", "shaping_reward", "residual_action_penalty",
        "residual_action_norm", "residual_norm", "applied_residual_norm_m",
        "residual_clip_difference_norm_m", "action_saturated",
    )

    def __init__(self, run_dir):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.buffers = {name: [] for name in self.FIELDS}
        self.episode_steps = 0
        self.episode_index = 0

    def _on_training_start(self):
        path = self.run_dir / "diagnostics_episode_metrics.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                self.episode_index = sum(1 for _ in csv.DictReader(f))

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if not infos:
            return True
        info = infos[0]
        self.episode_steps += 1
        for name in self.FIELDS:
            self.buffers[name].append(float(info[name]))
        if dones and dones[0]:
            self.episode_index += 1
            row = {
                "training_step": self.num_timesteps,
                "episode_index": self.episode_index,
                "episode_length": self.episode_steps,
            }
            for name, values in self.buffers.items():
                row[f"mean_{name}"] = float(np.mean(values)) if values else ""
                row[f"max_{name}"] = float(np.max(values)) if values else ""
                if name in ("task_reward", "shaping_reward", "residual_action_penalty"):
                    row[f"total_{name}"] = float(np.sum(values)) if values else ""
            monitor_info = info.get("episode", {})
            row["mean_episode_reward"] = float(monitor_info.get("r", 0.0))
            row["mean_episode_length"] = int(monitor_info.get("l", self.episode_steps))
            path = self.run_dir / "diagnostics_episode_metrics.csv"
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
            for name in self.FIELDS:
                self.logger.record(f"diagnostics/{name}_episode_mean", row[f"mean_{name}"])
            self.logger.record("diagnostics/episode_length", row["episode_length"])
            self.buffers = {name: [] for name in self.FIELDS}
            self.episode_steps = 0
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
        residual_penalty_coef=args.residual_penalty_coef,
    )


def main(args):
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run-name may contain only letters, numbers, _ and -")
    if args.timesteps <= 0 or args.eval_freq <= 0 or args.eval_episodes <= 0:
        raise ValueError("Timesteps, eval-freq and eval-episodes must be > 0")

    run_dir = Path("bc_residual_sac_runs") / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{run_dir} already contains files; choose another --run-name"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.pop("resume", None)
    config.pop("verbose", None)  # console verbosity is not an experiment condition
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"Resume configuration mismatch for {run_dir}")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    np.random.seed(args.seed)
    train_env = Monitor(make_env(args, args.sampling_mode))
    # Validation is ALWAYS uniform, regardless of training sampling.
    eval_env = make_env(args, "uniform")
    callback = FixedSeedSuccessEval(
        eval_env=eval_env,
        run_dir=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        save_trajectory_checkpoints=args.save_trajectory_checkpoints,
        save_replay_checkpoints=args.save_replay_checkpoints,
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
        verbose=args.verbose,
        tensorboard_log=str(Path("bc_residual_sac_logs") / args.run_name),
    )

    if args.resume:
        snapshots = []
        for path in run_dir.glob("checkpoint_*.zip"):
            step = int(path.stem.split("_")[-1])
            replay = run_dir / f"replay_buffer_{step}.pkl"
            resume_state = run_dir / f"resume_state_{step}.pkl"
            if replay.is_file() and resume_state.is_file():
                snapshots.append((step, path, replay, resume_state))
        if not snapshots:
            raise FileNotFoundError(f"No complete resumable checkpoint found in {run_dir}")
        step, checkpoint, replay, resume_state_path = max(snapshots)
        model = SAC.load(str(checkpoint), env=train_env, device="cpu")
        model.load_replay_buffer(str(replay))
        with resume_state_path.open("rb") as f:
            state = pickle.load(f)
        train_env.env.load_training_state_dict(state["environment"])
        monitor_state = state.get("monitor_state")
        if monitor_state is None:
            # Compatible with milestones saved before Monitor state was added.
            monitor_state = {
                "rewards": state.get("monitor_rewards", []),
                "needs_reset": False,
                "episode_returns": [],
                "episode_lengths": [],
                "episode_times": [],
                "total_steps": step,
                "current_reset_info": {},
            }
        train_env.rewards = list(monitor_state["rewards"])
        train_env.needs_reset = monitor_state["needs_reset"]
        train_env.episode_returns = list(monitor_state["episode_returns"])
        train_env.episode_lengths = list(monitor_state["episode_lengths"])
        train_env.episode_times = list(monitor_state["episode_times"])
        train_env.total_steps = monitor_state["total_steps"]
        train_env.current_reset_info = dict(monitor_state["current_reset_info"])
        model._last_obs = state["last_obs"]
        model._last_episode_starts = state["last_episode_starts"]
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        print(f"Resuming {args.run_name} from {step} steps", flush=True)
    else:
        step = 0

    print("===== BC + RESIDUAL SAC CONTROLLED EXPERIMENT =====")
    print(f"Run: {args.run_name}")
    print(f"Sampling: {args.sampling_mode} | Reward: {args.reward_mode}")
    print(f"BC: {args.bc_model}")
    print(f"Training range: +/-{args.position_range} m")
    print(f"Residual: +/-{args.residual_meters} m per axis")
    print(f"Residual penalty coefficient: {args.residual_penalty_coef}")
    print(f"Timesteps: {args.timesteps} | Seed: {args.seed}")
    print("Validation: uniform, fixed seeds 61000 onward")

    try:
        model.learn(
            total_timesteps=max(0, args.timesteps - step),
            callback=CallbackList([TrainingDiagnosticsCallback(run_dir), callback]),
            progress_bar=False,
            reset_num_timesteps=not args.resume,
        )
        model.save(str(run_dir / "final_model"))
        if args.save_trajectory_checkpoints and not (run_dir / f"checkpoint_{args.timesteps}.zip").is_file():
            model.save(str(run_dir / f"checkpoint_{args.timesteps}"))
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
    parser.add_argument("--residual-penalty-coef", type=float, default=0.005)
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
    parser.add_argument("--eval-seed", type=int, default=61000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--save-trajectory-checkpoints",
        action="store_true",
        help="Save a separate SAC snapshot at every evaluation boundary.",
    )
    parser.add_argument("--save-replay-checkpoints", action="store_true")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
