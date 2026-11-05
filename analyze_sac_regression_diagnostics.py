"""Extract existing SAC event logs and join them descriptively to outcomes."""

import csv
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SEEDS = (811, 823, 827, 829, 839)
TAG_MAP = {
    "actor_loss": "train/actor_loss",
    "critic_loss": "train/critic_loss",
    "entropy_coefficient": "train/ent_coef",
    "entropy_coefficient_loss": "train/ent_coef_loss",
    "training_mean_episode_reward": "rollout/ep_rew_mean",
    "training_mean_episode_length": "rollout/ep_len_mean",
    "training_rollout_success_rate": "rollout/success_rate",
}
UNAVAILABLE = (
    "mean_residual_action_norm", "max_residual_action_norm",
    "residual_action_norm_mm", "action_saturation_frequency",
    "effective_residual_correction", "commanded_applied_residual_difference",
    "task_reward_component", "shaping_reward_component",
    "residual_action_penalty_component", "TD_error", "Q1_Q2_disagreement",
)
OUTPUT = Path("bc_results")


def write_csv(path, rows, fields=None):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    scalar_rows = []
    by_seed_tag = {}
    availability_rows = []
    for seed in SEEDS:
        event_files = sorted(Path(f"bc_residual_sac_logs/stability_C_seed{seed}").rglob("events.out.tfevents*"))
        if len(event_files) != 1:
            raise RuntimeError(f"Expected one TensorBoard event file for seed {seed}, found {len(event_files)}")
        accumulator = EventAccumulator(str(event_files[0]))
        accumulator.Reload()
        present_tags = set(accumulator.Tags().get("scalars", []))
        for metric, tag in TAG_MAP.items():
            if tag not in present_tags:
                availability_rows.append({"sac_seed": seed, "metric": metric, "status": "Not Available", "source": str(event_files[0])})
                continue
            events = accumulator.Scalars(tag)
            by_seed_tag[(seed, metric)] = [(int(e.step), float(e.value)) for e in events]
            availability_rows.append({"sac_seed": seed, "metric": metric, "status": "available", "source": str(event_files[0])})
            scalar_rows.extend({
                "sac_seed": seed,
                "metric": metric,
                "tensorboard_tag": tag,
                "step": int(event.step),
                "wall_time": float(event.wall_time),
                "value": float(event.value),
                "status": "available",
            } for event in events)
        for metric in UNAVAILABLE:
            availability_rows.append({"sac_seed": seed, "metric": metric, "status": "Not Available", "source": "not logged during original training"})

    write_csv(OUTPUT / "sac_regression_existing_tensorboard_metrics.csv", scalar_rows)
    write_csv(OUTPUT / "sac_regression_existing_metric_availability.csv", availability_rows)

    outcome_path = OUTPUT / "sac_stability_checkpoint_metrics.csv"
    outcomes = read_csv(outcome_path)
    history_by_seed = {}
    for seed in SEEDS:
        run = Path(f"bc_residual_sac_runs/stability_C_seed{seed}")
        history_by_seed[seed] = read_csv(run / "eval_history.csv")
    step_outcomes = {}
    history_rows = []
    for seed in SEEDS:
        for row in history_by_seed[seed]:
            history_rows.append({
                "sac_seed": seed,
                "timesteps": int(row["timesteps"]),
                "validation_successes": int(row["successes"]),
                "validation_episodes": int(row["episodes"]),
                "validation_success_rate": float(row["success_rate"]),
                "mean_successful_episode_steps": row["mean_steps_success"],
                "validation_mean_episode_reward": row["mean_episode_reward"],
            })
        for row in outcomes:
            if int(row["sac_seed"]) == seed:
                step_outcomes[(seed, int(row["timesteps"]))] = row
    write_csv(OUTPUT / "sac_regression_existing_validation_and_paired_outcomes.csv", history_rows)

    focus = ((811, 20_000, 30_000), (827, 10_000, 20_000), (829, 20_000, 30_000))
    comparisons = []
    for seed, start, end in focus:
        before = step_outcomes[(seed, start)]
        after = step_outcomes[(seed, end)]
        row = {
            "sac_seed": seed,
            "transition": f"{start}_to_{end}",
            "validation_success_rate_before": float(before["validation_a_rate"]),
            "validation_success_rate_after": float(after["validation_a_rate"]),
            "final_success_rate_before": float(before["final_success_rate"]),
            "final_success_rate_after": float(after["final_success_rate"]),
            "rescue_before": int(before["rescue_count"]),
            "rescue_after": int(after["rescue_count"]),
            "regression_before": int(before["regression_count"]),
            "regression_after": int(after["regression_count"]),
            "net_improvement_before": int(before["net_improvement"]),
            "net_improvement_after": int(after["net_improvement"]),
        }
        for metric in TAG_MAP:
            events = by_seed_tag.get((seed, metric), [])
            def window_mean(step):
                values = [value for event_step, value in events if step - 1000 < event_step <= step]
                return sum(values) / len(values) if values else None
            a, b = window_mean(start), window_mean(end)
            row[f"{metric}_window_mean_before"] = a if a is not None else "Not Available"
            row[f"{metric}_window_mean_after"] = b if b is not None else "Not Available"
            row[f"{metric}_difference"] = (b - a) if a is not None and b is not None else "Not Available"
        comparisons.append(row)
    write_csv(OUTPUT / "sac_regression_focused_transition_diagnostics.csv", comparisons)

    summary = {
        "study": "SAC Regression Diagnostics: historical observational phase",
        "seeds": list(SEEDS),
        "tensorboard_sources": {str(seed): str(next(Path(f"bc_residual_sac_logs/stability_C_seed{seed}").rglob("events.out.tfevents*"))) for seed in SEEDS},
        "available_metrics": list(TAG_MAP),
        "not_available_metrics": list(UNAVAILABLE),
        "critic_loss_is_not_td_error": True,
        "q_value_disagreement_measured": False,
        "focus_transitions": [[seed, start, end] for seed, start, end in focus],
        "interpretation": "Exploratory associations only; training metrics were not manipulated and repeated checkpoints share models and evaluation positions.",
    }
    target = OUTPUT / "sac_regression_existing_diagnostics_summary.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event_rows": len(scalar_rows), "availability_rows": len(availability_rows), "focus": comparisons}, indent=2))


if __name__ == "__main__":
    main()
