"""Analyze the frozen two-arm residual-regularization experiment."""

import csv
import json
import math
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from analyze_sac_stability import exact_mcnemar, holm_adjust


SEEDS = (907, 911, 919, 929, 937)
ARMS = ("A", "B")
STEPS = (10_000, 20_000, 30_000)
VAL_RANGE = range(230_000, 230_500)
FINAL_RANGE = range(240_000, 242_000)
FORBIDDEN = (
    set(range(150_000, 152_000)) | set(range(170_000, 172_000))
    | set(range(190_000, 192_000)) | set(range(220_000, 222_000))
)
OUT = Path("bc_results")
RUNS = Path("bc_residual_sac_runs")
TAGS = {
    "actor_loss": "train/actor_loss",
    "critic_loss": "train/critic_loss",
    "entropy_coefficient": "train/ent_coef",
    "entropy_coefficient_loss": "train/ent_coef_loss",
    "training_mean_episode_reward": "rollout/ep_rew_mean",
    "training_mean_episode_length": "rollout/ep_len_mean",
    "training_rollout_success_rate": "rollout/success_rate",
}
DIAG_COLUMNS = (
    "mean_residual_action_norm", "max_residual_action_norm",
    "mean_residual_norm_mm", "max_residual_norm_mm",
    "mean_applied_residual_norm_mm", "mean_clip_difference_norm_mm",
    "action_saturation_frequency", "mean_task_reward_per_step",
    "mean_shaping_reward_per_step", "mean_residual_action_penalty_per_step",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not records:
        raise ValueError(f"Refusing to write empty {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def grouped_mean(values):
    return float(np.mean(values)) if values else "Not Available"


def paired_rows(path, expected_count, seed_range):
    rows = read_csv(path)
    if len(rows) != expected_count * 2:
        raise AssertionError(f"Expected {expected_count * 2} rows in {path}")
    bc = {int(r["seed"]): r for r in rows if r["policy"] == "bc_only"}
    sac = {int(r["seed"]): r for r in rows if r["policy"] == "bc_plus_sac"}
    expected = set(seed_range)
    if set(bc) != expected or set(sac) != expected:
        raise AssertionError(f"Wrong policy/seed set in {path}")
    for seed in expected:
        b, s = bc[seed], sac[seed]
        if (b["cube_x"], b["cube_y"]) != (s["cube_x"], s["cube_y"]):
            raise AssertionError(f"Initial position mismatch at {seed} in {path}")
    return bc, sac


def metric_summary(bc, sac):
    rescues = regressions = 0
    successful_steps = []
    for seed in bc:
        b, s = int(bc[seed]["success"]), int(sac[seed]["success"])
        rescues += int(not b and s)
        regressions += int(b and not s)
        if s:
            successful_steps.append(int(sac[seed]["steps"]))
    total = len(bc)
    successes = sum(int(row["success"]) for row in sac.values())
    bc_successes = sum(int(row["success"]) for row in bc.values())
    result = {
        "episodes": total,
        "bc_successes": bc_successes,
        "bc_success_rate": bc_successes / total,
        "successes": successes,
        "success_rate": successes / total,
        "rescue_count": rescues,
        "regression_count": regressions,
        "net_improvement": rescues - regressions,
        "bc_delta_pp": 100 * (successes - bc_successes) / total,
        "mean_successful_episode_steps": float(np.mean(successful_steps)) if successful_steps else "Not Available",
    }
    for column in DIAG_COLUMNS:
        result[column] = grouped_mean([float(row[column]) for row in sac.values()])
    return result


def position_region(row):
    dx, y = float(row["cube_x"]) - 0.55, float(row["cube_y"])
    if abs(dx) <= 0.12 and abs(y) <= 0.12:
        return "Inside ±12 cm"
    return ("E" if dx > 0 else "W") + ("N" if y > 0 else "S")


def outcome_compare(a_rows, b_rows, bc_rows):
    if set(a_rows) != set(b_rows) or set(a_rows) != set(bc_rows):
        raise AssertionError("Paired model seed sets differ")
    win = loss = 0
    reg_reduced = reg_increased = rescue_gained = rescue_lost = 0
    a_reg = b_reg = a_rescue = b_rescue = 0
    bc_success_positions = bc_failure_positions = 0
    for seed in a_rows:
        a, b, bc = int(a_rows[seed]["success"]), int(b_rows[seed]["success"]), int(bc_rows[seed]["success"])
        win += int(a == 0 and b == 1)
        loss += int(a == 1 and b == 0)
        if bc:
            bc_success_positions += 1
            a_reg += int(not a)
            b_reg += int(not b)
            reg_reduced += int(not a and b)
            reg_increased += int(a and not b)
        else:
            bc_failure_positions += 1
            a_rescue += int(a)
            b_rescue += int(b)
            rescue_gained += int(not a and b)
            rescue_lost += int(a and not b)
    return {
        "b_successes_over_a": win,
        "a_successes_over_b": loss,
        "overall_mcnemar_p": exact_mcnemar(win, loss),
        "bc_success_positions": bc_success_positions,
        "a_regressions": a_reg,
        "b_regressions": b_reg,
        "regression_difference_b_minus_a": b_reg - a_reg,
        "regression_reduced_by_b": reg_reduced,
        "regression_increased_by_b": reg_increased,
        "regression_mcnemar_p": exact_mcnemar(reg_reduced, reg_increased),
        "bc_failure_positions": bc_failure_positions,
        "a_rescues": a_rescue,
        "b_rescues": b_rescue,
        "rescue_difference_b_minus_a": b_rescue - a_rescue,
        "rescue_gained_by_b": rescue_gained,
        "rescue_lost_by_b": rescue_lost,
        "rescue_mcnemar_p": exact_mcnemar(rescue_gained, rescue_lost),
    }


def training_metric_rows():
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            run = RUNS / f"regdiag_{arm}_seed{seed}"
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            if config["seed"] != seed or config["residual_penalty_coef"] != (0.005 if arm == "A" else 0.020):
                raise AssertionError(f"Unexpected training config: {run}")
            for name in ("checkpoint_10000.zip", "checkpoint_20000.zip", "checkpoint_30000.zip"):
                if not (run / name).is_file():
                    raise FileNotFoundError(run / name)
            event_files = sorted(Path("bc_residual_sac_logs") .joinpath(f"regdiag_{arm}_seed{seed}").rglob("events.out.tfevents*"))
            if len(event_files) != 1:
                raise RuntimeError(f"Expected one TensorBoard file in {run}")
            accumulator = EventAccumulator(str(event_files[0]))
            accumulator.Reload()
            series = {}
            for metric, tag in TAGS.items():
                if tag in accumulator.Tags().get("scalars", []):
                    series[metric] = [(int(e.step), float(e.value)) for e in accumulator.Scalars(tag)]
                else:
                    series[metric] = []
            episode_rows = read_csv(run / "diagnostics_episode_metrics.csv")
            validation = {int(r["timesteps"]): r for r in read_csv(run / "eval_history.csv")}
            for step in STEPS:
                out = {"sac_seed": seed, "arm": arm, "residual_penalty_coef": config["residual_penalty_coef"], "timesteps": step}
                for metric, events in series.items():
                    vals = [v for t, v in events if step - 1000 < t <= step]
                    out[metric] = grouped_mean(vals)
                    out[f"{metric}_event_count"] = len(vals)
                ep = [r for r in episode_rows if step - 1000 < int(r["training_step"]) <= step]
                diag_map = {
                    "mean_residual_action_norm": "mean_residual_action_norm",
                    "max_residual_action_norm": "max_residual_action_norm",
                    "mean_residual_norm_mm": "mean_residual_norm",
                    "max_residual_norm_mm": "max_residual_norm",
                    "mean_applied_residual_norm_mm": "mean_applied_residual_norm_m",
                    "mean_clip_difference_norm_mm": "mean_residual_clip_difference_norm_m",
                    "action_saturation_frequency": "mean_action_saturated",
                    "mean_task_reward_per_step": "mean_task_reward",
                    "mean_shaping_reward_per_step": "mean_shaping_reward",
                    "mean_residual_action_penalty_per_step": "mean_residual_action_penalty",
                    "mean_total_training_reward": "mean_episode_reward",
                    "mean_diagnostic_episode_length": "mean_episode_length",
                }
                for output_name, source in diag_map.items():
                    values = [float(r[source]) for r in ep if r[source] != ""]
                    mean_val = grouped_mean(values)
                    if output_name.endswith("_norm_mm") and mean_val != "Not Available":
                        mean_val *= 1000.0
                    if output_name == "mean_applied_residual_norm_mm" and mean_val != "Not Available":
                        mean_val *= 1000.0
                    out[output_name] = mean_val
                val = validation.get(step)
                out["validation_a_successes"] = int(val["successes"]) if val else "Not Available"
                out["validation_a_episodes"] = int(val["episodes"]) if val else "Not Available"
                out["validation_a_success_rate"] = float(val["success_rate"]) if val else "Not Available"
                out["episode_diagnostic_rows_in_window"] = len(ep)
                rows.append(out)
    return rows


def main():
    freeze_path = OUT / "sac_regdiag_checkpoint_selection_freeze.json"
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    if manifest["validation_seed_range"] != [230_000, 230_499] or manifest["final_seed_range"] != [240_000, 241_999]:
        raise AssertionError("Evaluation ranges do not match frozen manifest")
    val_set, final_set = set(VAL_RANGE), set(FINAL_RANGE)
    if val_set & final_set or (val_set | final_set) & FORBIDDEN:
        raise AssertionError("Evaluation ranges overlap excluded positions")

    train_rows = training_metric_rows()
    write_csv(OUT / "sac_regdiag_training_diagnostics.csv", train_rows)

    final_baseline = {int(r["seed"]): r for r in read_csv(OUT / "sac_regdiag_bc_final2000.csv")}
    if set(final_baseline) != final_set:
        raise AssertionError("Final BC baseline seed mismatch")
    checkpoint_metrics = []
    final_cache = {}
    selection_map = {(s["sac_seed"], s["arm"]): s for s in manifest["selections"]}
    for seed in SEEDS:
        for arm in ARMS:
            selected = int(selection_map[(seed, arm)]["selected_step"])
            for step in sorted({30_000, selected}):
                name = f"seed{seed}_arm{arm}_step{step}"
                path = OUT / f"sac_regdiag_{name}_final2000.csv"
                bc, sac = paired_rows(path, 2000, FINAL_RANGE)
                signature = [(k, bc[k]["cube_x"], bc[k]["cube_y"], bc[k]["success"]) for k in sorted(bc)]
                if signature != [(k, final_baseline[k]["cube_x"], final_baseline[k]["cube_y"], final_baseline[k]["success"]) for k in sorted(final_baseline)]:
                    raise AssertionError(f"BC trajectory differs from shared baseline for {name}")
                final_cache[(seed, arm, step)] = (bc, sac)
                checkpoint_metrics.append({"sac_seed": seed, "arm": arm, "timesteps": step, "selection_role": "30k_and/or_selected", **metric_summary(bc, sac)})
    write_csv(OUT / "sac_regdiag_final_checkpoint_metrics.csv", checkpoint_metrics)

    primary_rows, selected_rows, test_rows = [], [], []
    test_families = {}
    for seed in SEEDS:
        a30bc, a30 = final_cache[(seed, "A", 30_000)]
        b30bc, b30 = final_cache[(seed, "B", 30_000)]
        if [(r["cube_x"], r["cube_y"]) for r in a30.values()] != [(r["cube_x"], r["cube_y"]) for r in b30.values()]:
            raise AssertionError(f"Arm A/B positions are not identically ordered for seed {seed}")
        a_metric, b_metric = metric_summary(a30bc, a30), metric_summary(b30bc, b30)
        paired = outcome_compare(a30, b30, a30bc)
        a_selected_step = int(selection_map[(seed, "A")]["selected_step"])
        b_selected_step = int(selection_map[(seed, "B")]["selected_step"])
        abcs, asel = final_cache[(seed, "A", a_selected_step)]
        bbcs, bsel = final_cache[(seed, "B", b_selected_step)]
        selected_paired = outcome_compare(asel, bsel, abcs)
        primary_rows.append({
            "sac_seed": seed,
            **{f"arm_a_{key}": value for key, value in a_metric.items()},
            **{f"arm_b_{key}": value for key, value in b_metric.items()},
            "success_rate_difference_b_minus_a_pp": 100 * (b_metric["success_rate"] - a_metric["success_rate"]),
            "rescue_difference_b_minus_a": b_metric["rescue_count"] - a_metric["rescue_count"],
            "regression_difference_b_minus_a": b_metric["regression_count"] - a_metric["regression_count"],
            **paired,
        })
        selected_rows.append({
            "sac_seed": seed,
            "arm_a_selected_step": a_selected_step,
            "arm_b_selected_step": b_selected_step,
            **{f"arm_a_{key}": value for key, value in metric_summary(abcs, asel).items()},
            **{f"arm_b_{key}": value for key, value in metric_summary(bbcs, bsel).items()},
            "success_rate_difference_b_minus_a_pp": 100 * (metric_summary(bbcs, bsel)["success_rate"] - metric_summary(abcs, asel)["success_rate"]),
            "rescue_difference_b_minus_a": metric_summary(bbcs, bsel)["rescue_count"] - metric_summary(abcs, asel)["rescue_count"],
            "regression_difference_b_minus_a": metric_summary(bbcs, bsel)["regression_count"] - metric_summary(abcs, asel)["regression_count"],
            **{f"selected_{k}": v for k, v in selected_paired.items()},
        })
        for family, p_key, win_key, loss_key in (
            ("primary_overall_success", "overall_mcnemar_p", "b_successes_over_a", "a_successes_over_b"),
            ("primary_regression_given_bc_success", "regression_mcnemar_p", "regression_reduced_by_b", "regression_increased_by_b"),
            ("primary_rescue_given_bc_failure", "rescue_mcnemar_p", "rescue_gained_by_b", "rescue_lost_by_b"),
        ):
            key = f"{family}_{seed}"
            p = paired[p_key]
            test_families.setdefault(family, []).append((key, p))
            test_rows.append({"family": family, "sac_seed": seed, "comparison": "Arm A 30k vs Arm B 30k", "b_favorable_discordances": paired[win_key], "b_unfavorable_discordances": paired[loss_key], "p_exact": p, "p_holm": None})
        for family, p_key, win_key, loss_key in (
            ("selected_overall_success", "overall_mcnemar_p", "b_successes_over_a", "a_successes_over_b"),
            ("selected_regression_given_bc_success", "regression_mcnemar_p", "regression_reduced_by_b", "regression_increased_by_b"),
            ("selected_rescue_given_bc_failure", "rescue_mcnemar_p", "rescue_gained_by_b", "rescue_lost_by_b"),
        ):
            key = f"{family}_{seed}"
            p = selected_paired[p_key]
            test_families.setdefault(family, []).append((key, p))
            test_rows.append({"family": family, "sac_seed": seed, "comparison": f"Arm A selected {a_selected_step} vs Arm B selected {b_selected_step}", "b_favorable_discordances": selected_paired[win_key], "b_unfavorable_discordances": selected_paired[loss_key], "p_exact": p, "p_holm": None})

    for family, values in test_families.items():
        adjusted = holm_adjust(values)
        for row in test_rows:
            if row["family"] == family:
                row["p_holm"] = adjusted[f"{family}_{row['sac_seed']}"]
    write_csv(OUT / "sac_regdiag_primary_seed_comparisons.csv", primary_rows)
    write_csv(OUT / "sac_regdiag_selected_seed_comparisons.csv", selected_rows)
    write_csv(OUT / "sac_regdiag_exact_mcnemar_tests.csv", test_rows)

    region_rows = []
    for seed in SEEDS:
        pairs = [
            ("A30k", final_cache[(seed, "A", 30_000)]),
            ("B30k", final_cache[(seed, "B", 30_000)]),
            ("A_selected", final_cache[(seed, "A", int(selection_map[(seed, "A")]["selected_step"]))]),
            ("B_selected", final_cache[(seed, "B", int(selection_map[(seed, "B")]["selected_step"]))]),
        ]
        for label, (bc, sac) in pairs:
            buckets = {}
            for episode_seed in bc:
                region = position_region(bc[episode_seed])
                bucket = buckets.setdefault(region, {"episodes": 0, "bc_successes": 0, "successes": 0, "rescues": 0, "regressions": 0})
                b, s = int(bc[episode_seed]["success"]), int(sac[episode_seed]["success"])
                bucket["episodes"] += 1
                bucket["bc_successes"] += b
                bucket["successes"] += s
                bucket["rescues"] += int(not b and s)
                bucket["regressions"] += int(b and not s)
            for region, metrics in buckets.items():
                region_rows.append({"sac_seed": seed, "arm_checkpoint": label, "region": region, **metrics,
                                    "success_rate": metrics["successes"] / metrics["episodes"]})
    write_csv(OUT / "sac_regdiag_regional_outcomes.csv", region_rows)

    deltas = {
        "success_rate_difference_pp": [r["success_rate_difference_b_minus_a_pp"] for r in primary_rows],
        "regression_difference": [r["regression_difference_b_minus_a"] for r in primary_rows],
        "rescue_difference": [r["rescue_difference_b_minus_a"] for r in primary_rows],
    }
    delta_summary = {}
    for name, vals in deltas.items():
        delta_summary[name] = {"mean": float(np.mean(vals)), "sample_sd": float(np.std(vals, ddof=1)), "minimum": min(vals), "maximum": max(vals), "per_seed": vals}
    criteria = manifest["exploratory_decision_criteria"]
    regression_better_count = sum(v < 0 for v in deltas["regression_difference"])
    mean_success_delta = float(np.mean(deltas["success_rate_difference_pp"]))
    mean_rescue_delta = float(np.mean(deltas["rescue_difference"]))
    summary = {
        "study": "SAC Regression Diagnostics & Controlled Regularization Experiment",
        "base_commit": manifest["base_commit"],
        "seeds": list(SEEDS),
        "arms": {"A": 0.005, "B": 0.020},
        "validation_seed_range": manifest["validation_seed_range"],
        "final_seed_range": manifest["final_seed_range"],
        "excluded_prior_final_ranges": manifest["excluded_prior_final_ranges"],
        "shared_final_bc_successes": sum(int(r["success"]) for r in final_baseline.values()),
        "shared_final_bc_episodes": len(final_baseline),
        "primary_arm_b_minus_arm_a_seed_level_deltas": delta_summary,
        "decision_criteria": criteria,
        "decision_criteria_result": {
            "regression_decreases_in_at_least_4_of_5": regression_better_count >= 4,
            "regression_decrease_count": regression_better_count,
            "mean_success_drop_within_0_5_pp": mean_success_delta >= -0.5,
            "mean_success_rate_difference_pp": mean_success_delta,
            "rescue_preservation_within_tolerance": mean_rescue_delta >= -10,
            "mean_rescue_difference": mean_rescue_delta,
            "all_criteria_met": regression_better_count >= 4 and mean_success_delta >= -0.5 and mean_rescue_delta >= -10,
        },
        "test_families": {key: "Holm correction across five training seeds" for key in test_families},
        "cross_model_position_rows_treated_as_independent": False,
        "td_error_directly_measured": False,
        "causal_claim_from_training_metric_correlation": False,
    }
    target = OUT / "sac_regdiag_analysis_summary.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": summary["decision_criteria_result"], "primary_deltas": delta_summary}, indent=2))


if __name__ == "__main__":
    main()
