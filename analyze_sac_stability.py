"""Validate and summarize the SAC training stability experiment."""

import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import gymnasium
import mujoco
import stable_baselines3
import torch


ROOT = Path("bc_residual_sac_runs")
RESULTS = Path("bc_results")
BC_CHECKPOINT = Path("bc_models/fair_wide_expert_seed42.pt")
NEW_SEEDS = (811, 823, 827, 829, 839)
OLD_SEEDS = (7, 19, 73, 202, 314)
STEPS = (10_000, 20_000, 30_000)
VAL_A_SEEDS = set(range(61_000, 61_040))
VAL_B_SEEDS = set(range(210_000, 210_500))
FINAL_SEEDS = set(range(220_000, 222_000))
FORBIDDEN_SEEDS = (
    set(range(150_000, 152_000))
    | set(range(170_000, 172_000))
    | set(range(190_000, 192_000))
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json_idempotent(path, payload):
    """Write a generated analysis artifact once, or accept identical reruns."""
    path = Path(path)
    content = json.dumps(payload, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite differing artifact {path}")
        return
    path.write_text(content, encoding="utf-8")


def exact_mcnemar(rescue, regression):
    """Two-sided exact binomial McNemar p-value for discordant pairs."""
    rescue, regression = int(rescue), int(regression)
    n = rescue + regression
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(rescue, regression) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def holm_adjust(items):
    """Return Holm-adjusted p-values for [(key, p), ...]."""
    ordered = sorted(items, key=lambda item: item[1])
    m = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def summarize_paired(bc_rows, sac_rows):
    bc = {int(row["seed"]): row for row in bc_rows}
    sac = {int(row["seed"]): row for row in sac_rows}
    if set(bc) != set(sac):
        raise AssertionError("BC/SAC paired episode seeds differ")
    rescues = regressions = 0
    successful_steps = []
    for seed in sorted(bc):
        b, s = bc[seed], sac[seed]
        if not (
            math.isclose(float(b["cube_x"]), float(s["cube_x"]), abs_tol=1e-10)
            and math.isclose(float(b["cube_y"]), float(s["cube_y"]), abs_tol=1e-10)
        ):
            raise AssertionError(f"Position mismatch at evaluation seed {seed}")
        b_success, s_success = int(b["success"]), int(s["success"])
        rescues += int(not b_success and s_success)
        regressions += int(b_success and not s_success)
        if s_success:
            successful_steps.append(int(s["steps"]))
    n = len(bc)
    successes = sum(int(row["success"]) for row in sac.values())
    bc_successes = sum(int(row["success"]) for row in bc.values())
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n,
        "bc_successes": bc_successes,
        "bc_success_rate": bc_successes / n,
        "rescue_count": rescues,
        "regression_count": regressions,
        "net_improvement": rescues - regressions,
        "mean_successful_steps": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "bc_delta_pp": 100.0 * (successes - bc_successes) / n,
    }


def compare_policies(reference_rows, comparison_rows):
    """Compare two policies on identical positions using paired discordances."""
    reference = {int(row["seed"]): row for row in reference_rows}
    comparison = {int(row["seed"]): row for row in comparison_rows}
    if set(reference) != set(comparison):
        raise AssertionError("Policy episode seeds differ")
    wins = losses = 0
    for seed in reference:
        a, b = reference[seed], comparison[seed]
        if not (
            math.isclose(float(a["cube_x"]), float(b["cube_x"]), abs_tol=1e-10)
            and math.isclose(float(a["cube_y"]), float(b["cube_y"]), abs_tol=1e-10)
        ):
            raise AssertionError(f"Position mismatch at evaluation seed {seed}")
        wins += int(int(a["success"]) == 0 and int(b["success"]) == 1)
        losses += int(int(a["success"]) == 1 and int(b["success"]) == 0)
    return {"rescue_count": wins, "regression_count": losses}


def raw_eval(name, label, expected_episodes):
    path = RESULTS / f"sac_stability_{name}_{label}.csv"
    rows = read_csv(path)
    if len(rows) != 2 * expected_episodes:
        raise AssertionError(f"Wrong row count in {path}: {len(rows)}")
    bc_rows = [row for row in rows if row["policy"] == "bc_only"]
    sac_rows = [row for row in rows if row["policy"] == "bc_plus_sac"]
    if len(bc_rows) != expected_episodes or len(sac_rows) != expected_episodes:
        raise AssertionError(f"Unexpected policy row counts in {path}")
    return bc_rows, sac_rows


def val_a_stats(seed, step):
    path = ROOT / f"stability_C_seed{seed}" / "eval_history.csv"
    rows = read_csv(path)
    matches = [row for row in rows if int(row["timesteps"]) == step]
    if len(matches) != 1:
        raise AssertionError(f"Missing unique Validation A result: seed {seed}, {step}")
    row = matches[0]
    return {
        "validation_a_successes": int(row["successes"]),
        "validation_a_episodes": int(row["episodes"]),
        "validation_a_rate": float(row["success_rate"]),
        "validation_a_mean_successful_steps": (
            float(row["mean_steps_success"]) if row["mean_steps_success"] else None
        ),
    }


def val_b_stats(name):
    bc, sac = raw_eval(name, "validationB500", 500)
    out = summarize_paired(bc, sac)
    out["validation_b_mean_successful_steps"] = out.pop("mean_successful_steps")
    out["validation_b_rate"] = out.pop("success_rate")
    out["validation_b_successes"] = out.pop("successes")
    return out


def select_checkpoint(records, rate_key, steps_key):
    # Original callback traverses checkpoints in chronological order and only
    # replaces on higher rate or strictly fewer successful steps.
    best = None
    for row in sorted(records, key=lambda r: int(r["timesteps"])):
        rate = float(row[rate_key])
        steps = row[steps_key]
        steps = float(steps) if steps is not None else float("inf")
        if best is None or rate > best[0] or (rate == best[0] and steps < best[1]):
            best = (rate, steps, row)
    return best[2]


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        avg = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = avg
        start = end
    return ranks


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def validate_seed_ranges():
    assert VAL_A_SEEDS.isdisjoint(VAL_B_SEEDS)
    assert VAL_A_SEEDS.isdisjoint(FINAL_SEEDS)
    assert VAL_B_SEEDS.isdisjoint(FINAL_SEEDS)
    assert not (VAL_A_SEEDS | VAL_B_SEEDS | FINAL_SEEDS) & FORBIDDEN_SEEDS


def audit_checkpoints():
    audit = {
        "bc_checkpoint": str(BC_CHECKPOINT),
        "bc_sha256": sha256(BC_CHECKPOINT),
        "software": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "validation_a_seeds": [61000, 61039],
        "validation_b_seed_range": [210000, 210499],
        "final_seed_range": [220000, 221999],
        "forbidden_prior_final_ranges": [[150000, 151999], [170000, 171999], [190000, 191999]],
        "validation_sets_disjoint": True,
        "old_models": [],
        "new_models": [],
    }
    for seed in OLD_SEEDS:
        run = ROOT / f"C_hard_shaped_seed{seed}"
        config = json.loads((run / "config.json").read_text())
        history = read_csv(run / "eval_history.csv")
        entry = {"seed": seed, "config": config, "eval_history": history,
                 "checkpoints": {}}
        for kind in ("best_model", "final_model"):
            path = run / f"{kind}.zip"
            if not path.is_file():
                raise FileNotFoundError(path)
            entry["checkpoints"][kind] = {
                "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
            }
        entry["best_checkpoint_steps"] = next(
            int(row["timesteps"]) for row in history
            if int(row["successes"]) == max(int(x["successes"]) for x in history)
            and float(row["mean_steps_success"] or "inf") == min(
                float(x["mean_steps_success"] or "inf") for x in history
                if int(x["successes"]) == max(int(y["successes"]) for y in history)
            )
        )
        audit["old_models"].append(entry)

    expected_config = {
        "bc_model": str(BC_CHECKPOINT), "position_range": 0.2,
        "residual_meters": 0.005, "sampling_mode": "hard",
        "hard_inner_range": 0.12, "hard_fraction": 0.9,
        "reward_mode": "shaped", "timesteps": 30000,
        "eval_freq": 10000, "eval_episodes": 40,
    }
    for seed in NEW_SEEDS:
        run = ROOT / f"stability_C_seed{seed}"
        config = json.loads((run / "config.json").read_text())
        for key, value in {**expected_config, "seed": seed}.items():
            if config.get(key) != value:
                raise AssertionError(f"Unexpected config {key} for new seed {seed}")
        entry = {"seed": seed, "config": config, "checkpoints": {}}
        for step in STEPS:
            path = run / f"checkpoint_{step}.zip"
            if not path.is_file():
                raise FileNotFoundError(path)
            entry["checkpoints"][str(step)] = {
                "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
            }
        for kind in ("best_model", "final_model"):
            path = run / f"{kind}.zip"
            entry["checkpoints"][kind] = {
                "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
            }
        audit["new_models"].append(entry)
    return audit


def main():
    selection_rows = []
    RESULTS.mkdir(exist_ok=True)
    validate_seed_ranges()
    selection_freeze_path = RESULTS / "sac_stability_selection_freeze.json"
    if not selection_freeze_path.is_file():
        raise RuntimeError("Selection freeze manifest is required before analysis")
    frozen_selection = json.loads(selection_freeze_path.read_text(encoding="utf-8"))
    audit = audit_checkpoints()
    model_lookup = {item["seed"]: item for item in audit["new_models"]}
    for selection in frozen_selection["selections"]:
        seed = selection["sac_seed"]
        for candidate in selection["candidates"]:
            step = str(candidate["timesteps"])
            actual_checkpoint_hash = model_lookup[seed]["checkpoints"][step]["sha256"]
            if actual_checkpoint_hash != candidate["checkpoint_sha256"]:
                raise AssertionError(f"Checkpoint hash changed after freeze: seed {seed}, {step}")
            val_b_path = RESULTS / f"sac_stability_seed{seed}_step{step}_validationB500.csv"
            if sha256(val_b_path) != candidate["validation_b_csv_sha256"]:
                raise AssertionError(f"Validation B file changed after freeze: {val_b_path}")
            history_path = ROOT / f"stability_C_seed{seed}" / "eval_history.csv"
            if sha256(history_path) != candidate["validation_a_history_sha256"]:
                raise AssertionError(f"Validation A history changed after freeze: {history_path}")
    audit_path = RESULTS / "sac_stability_checkpoint_audit.json"
    write_json_idempotent(audit_path, audit)

    checkpoint_rows = []
    raw_cache = {}
    for seed in NEW_SEEDS:
        by_step = []
        for step in STEPS:
            name = f"seed{seed}_step{step}"
            a = val_a_stats(seed, step)
            b = val_b_stats(name)
            bc, sac = raw_eval(name, "final2000", 2000)
            final = summarize_paired(bc, sac)
            raw_cache[name] = (bc, sac)
            row = {
                "sac_seed": seed, "timesteps": step,
                **a,
                "validation_b_successes": b["validation_b_successes"],
                "validation_b_episodes": 500,
                "validation_b_rate": b["validation_b_rate"],
                "validation_b_mean_successful_steps": b["validation_b_mean_successful_steps"],
                "final_successes": final["successes"],
                "final_episodes": final["episodes"],
                "final_success_rate": final["success_rate"],
                "bc_successes": final["bc_successes"],
                "bc_success_rate": final["bc_success_rate"],
                "rescue_count": final["rescue_count"],
                "regression_count": final["regression_count"],
                "net_improvement": final["net_improvement"],
                "mean_successful_steps": final["mean_successful_steps"],
                "bc_delta_pp": final["bc_delta_pp"],
            }
            checkpoint_rows.append(row)
            by_step.append({
                "timesteps": step,
                "validation_a_rate": a["validation_a_rate"],
                "validation_a_mean_successful_steps": a["validation_a_mean_successful_steps"],
                "validation_b_rate": b["validation_b_rate"],
                "validation_b_mean_successful_steps": b["validation_b_mean_successful_steps"],
            })

        selected_a = select_checkpoint(by_step, "validation_a_rate", "validation_a_mean_successful_steps")
        selected_b = select_checkpoint(by_step, "validation_b_rate", "validation_b_mean_successful_steps")
        frozen = next(item for item in frozen_selection["selections"] if item["sac_seed"] == seed)
        if (selected_a["timesteps"] != frozen["selection_a_step"]
                or selected_b["timesteps"] != frozen["selection_b_step"]):
            raise AssertionError(f"Analysis selection differs from pre-test freeze for seed {seed}")
        selection_rows.append({
            "sac_seed": seed,
            "selection_a_step": selected_a["timesteps"],
            "selection_a_validation_rate": selected_a["validation_a_rate"],
            "selection_b_step": selected_b["timesteps"],
            "selection_b_validation_rate": selected_b["validation_b_rate"],
            "selected_same_checkpoint": selected_a["timesteps"] == selected_b["timesteps"],
        })

    # Validate identical initial coordinates and BC outcomes across every
    # checkpoint file for each shared evaluation set.
    for label, episodes in (("validationB500", 500), ("final2000", 2000)):
        reference = None
        for seed in NEW_SEEDS:
            for step in STEPS:
                name = f"seed{seed}_step{step}"
                bc_rows, _ = raw_eval(name, label, episodes)
                signature = [(int(r["seed"]), float(r["cube_x"]), float(r["cube_y"]), int(r["success"])) for r in bc_rows]
                if reference is None:
                    reference = signature
                elif signature != reference:
                    raise AssertionError(f"BC positions/outcomes mismatch for {name} {label}")
        seed_values = {r[0] for r in reference}
        expected = VAL_B_SEEDS if label == "validationB500" else FINAL_SEEDS
        if seed_values != expected:
            raise AssertionError(f"Unexpected seed range in {label}")

    # Include existing Best and Final checkpoints on the new final-position set
    # only when both were distinct. These results are kept separate from new
    # three-checkpoint trajectory selection.
    existing_rows = []
    common_final_signature = None
    for seed in NEW_SEEDS:
        for step in STEPS:
            bc_rows, _ = raw_eval(f"seed{seed}_step{step}", "final2000", 2000)
            signature = [(int(r["seed"]), float(r["cube_x"]), float(r["cube_y"]), int(r["success"])) for r in bc_rows]
            if common_final_signature is None:
                common_final_signature = signature
            elif signature != common_final_signature:
                raise AssertionError("Final-evaluation BC outcomes differ across new checkpoints")
    for seed in OLD_SEEDS:
        for kind in ("best", "final"):
            name = f"existing_seed{seed}_{kind}"
            bc, sac = raw_eval(name, "final2000", 2000)
            signature = [(int(r["seed"]), float(r["cube_x"]), float(r["cube_y"]), int(r["success"])) for r in bc]
            if signature != common_final_signature:
                raise AssertionError(f"Final-evaluation BC positions/outcomes differ for {name}")
            metrics = summarize_paired(bc, sac)
            existing_rows.append({"sac_seed": seed, "checkpoint": kind, **metrics})

    final_baseline_path = RESULTS / "sac_stability_final_bc_baseline.csv"
    final_baseline_rows = read_csv(final_baseline_path)
    baseline_signature = [
        (int(r["seed"]), float(r["cube_x"]), float(r["cube_y"]), int(r["success"]))
        for r in final_baseline_rows
    ]
    if baseline_signature != common_final_signature:
        raise AssertionError("Shared final BC baseline differs from paired-evaluation BC rows")

    # Paired tests are reported per pre-specified family, across five seeds.
    test_rows = []
    families = {}
    for step in STEPS:
        fam = f"step_{step}_vs_bc"
        families[fam] = []
        for seed in NEW_SEEDS:
            name = f"seed{seed}_step{step}"
            bc, sac = raw_cache[name]
            metric = summarize_paired(bc, sac)
            p = exact_mcnemar(metric["rescue_count"], metric["regression_count"])
            key = f"{fam}_{seed}"
            families[fam].append((key, p))
            test_rows.append({"family": fam, "sac_seed": seed, "comparison": f"{step} vs BC",
                              "rescue": metric["rescue_count"], "regression": metric["regression_count"],
                              "p_exact": p, "p_holm": None})

    for selection in selection_rows:
        seed = selection["sac_seed"]
        a_name = f"seed{seed}_step{selection['selection_a_step']}"
        b_name = f"seed{seed}_step{selection['selection_b_step']}"
        bc_a, sac_a = raw_cache[a_name]
        bc_b, sac_b = raw_cache[b_name]
        if [(r["seed"],r["cube_x"],r["cube_y"]) for r in bc_a] != [(r["seed"],r["cube_x"],r["cube_y"]) for r in bc_b]:
            raise AssertionError("Selected checkpoint pair does not use identical final positions")
        m = compare_policies(sac_a, sac_b)
        # Treat Selection A as baseline: B successes after A failures are wins.
        p = exact_mcnemar(m["rescue_count"], m["regression_count"])
        fam = "selection_A_vs_B"
        families.setdefault(fam, []).append((f"{fam}_{seed}", p))
        test_rows.append({"family": fam, "sac_seed": seed, "comparison": "Selection A vs Selection B",
                          "rescue": m["rescue_count"], "regression": m["regression_count"],
                          "p_exact": p, "p_holm": None})

    for fam, entries in families.items():
        adjusted = holm_adjust(entries)
        for row in test_rows:
            if row["family"] == fam:
                row["p_holm"] = adjusted[f"{fam}_{row['sac_seed']}"]

    # Add final-evaluation outcomes for each selection rule.
    for row in selection_rows:
        seed = row["sac_seed"]
        for key, step_key in (("selection_a", "selection_a_step"), ("selection_b", "selection_b_step")):
            name = f"seed{seed}_step{row[step_key]}"
            bc, sac = raw_cache[name]
            metrics = summarize_paired(bc, sac)
            for metric_key in ("successes", "success_rate", "rescue_count", "regression_count", "net_improvement", "mean_successful_steps", "bc_delta_pp"):
                row[f"{key}_{metric_key}"] = metrics[metric_key]

    step_summary = []
    for step in STEPS:
        values = [r["final_success_rate"] for r in checkpoint_rows if r["timesteps"] == step]
        step_summary.append({
            "timesteps": step,
            "mean_success_rate": float(np.mean(values)),
            "sample_sd_pp": float(np.std(values, ddof=1) * 100.0),
            "minimum_rate": min(values), "maximum_rate": max(values),
            "range_pp": 100.0 * (max(values) - min(values)),
        })
    within_seed_changes = []
    within_seed_ranges = []
    for seed in NEW_SEEDS:
        values = [r["final_success_rate"] for r in checkpoint_rows if r["sac_seed"] == seed]
        within_seed_changes.extend(abs(values[i + 1] - values[i]) for i in range(len(values) - 1))
        within_seed_ranges.append(max(values) - min(values))

    progress = []
    for seed in NEW_SEEDS:
        vals = [next(r["final_success_rate"] for r in checkpoint_rows if r["sac_seed"] == seed and r["timesteps"] == step) for step in STEPS]
        if max(vals) - min(vals) <= 0.01:
            pattern = "little change (range <= 1 pp)"
        elif vals[0] < vals[1] < vals[2]:
            pattern = "continued improvement"
        elif vals.index(max(vals)) < 2 and vals[-1] < max(vals):
            pattern = "peaked before 30k, then declined"
        elif vals[0] < vals[1] and vals[1] == vals[2]:
            pattern = "improved, then plateaued"
        else:
            pattern = "non-monotonic"
        progress.append({"sac_seed": seed, "rates": vals, "range_pp": 100 * (max(vals)-min(vals)), "pattern": pattern})

    summary = {
        "study": "SAC Training Stability Analysis",
        "branch_base_commit": "f49b94f565b3e385fa9f5c8a4ccefc5bbb2e6efa",
        "new_training_seeds": list(NEW_SEEDS),
        "selection_freeze_sha256": sha256(selection_freeze_path),
        "validation_a_range": [61000, 61039],
        "validation_b_range": [210000, 210499],
        "final_evaluation_range": [220000, 221999],
        "shared_final_bc_baseline": {
            "path": str(final_baseline_path),
            "sha256": sha256(final_baseline_path),
            "rows": len(final_baseline_rows),
            "reused_for_every_checkpoint": True,
        },
        "old_model_final_evaluation": existing_rows,
        "checkpoint_rows": checkpoint_rows,
        "selection_rows": selection_rows,
        "learning_progress": progress,
        "step_summary": step_summary,
        "variability_decomposition_descriptive": {
            "mean_absolute_within_seed_adjacent_step_change_pp": float(np.mean(within_seed_changes) * 100),
            "mean_within_seed_10k_to_30k_range_pp": float(np.mean(within_seed_ranges) * 100),
            "mean_cross_seed_range_pp_across_steps": float(np.mean([row["range_pp"] for row in step_summary])),
            "note": "Descriptive only: the same models and held-out positions recur across checkpoints.",
        },
        "validation_final_correlations_descriptive": {
            "unit": "15 checkpoint observations; descriptive only, nested within five models and shared position seeds",
            "validation_a_40_vs_final_2000_pearson": correlation(
                [r["validation_a_rate"] for r in checkpoint_rows],
                [r["final_success_rate"] for r in checkpoint_rows],
            ),
            "validation_a_40_vs_final_2000_spearman": correlation(
                average_ranks([r["validation_a_rate"] for r in checkpoint_rows]),
                average_ranks([r["final_success_rate"] for r in checkpoint_rows]),
            ),
            "validation_b_500_vs_final_2000_pearson": correlation(
                [r["validation_b_rate"] for r in checkpoint_rows],
                [r["final_success_rate"] for r in checkpoint_rows],
            ),
            "validation_b_500_vs_final_2000_spearman": correlation(
                average_ranks([r["validation_b_rate"] for r in checkpoint_rows]),
                average_ranks([r["final_success_rate"] for r in checkpoint_rows]),
            ),
        },
        "selection_comparison_summary": {
            "same_checkpoint_count": sum(r["selected_same_checkpoint"] for r in selection_rows),
            "selection_a_higher_final_count": sum(r["selection_a_success_rate"] > r["selection_b_success_rate"] for r in selection_rows),
            "selection_b_higher_final_count": sum(r["selection_b_success_rate"] > r["selection_a_success_rate"] for r in selection_rows),
            "tied_final_count": sum(r["selection_a_success_rate"] == r["selection_b_success_rate"] for r in selection_rows),
            "mean_selection_a_final_rate": float(np.mean([r["selection_a_success_rate"] for r in selection_rows])),
            "mean_selection_b_final_rate": float(np.mean([r["selection_b_success_rate"] for r in selection_rows])),
        },
        "mcnemar_tests": test_rows,
        "multiple_comparison_families": {name: "Holm correction across five SAC training seeds" for name in families},
        "shared_positions_across_models": True,
        "cross_model_rows_treated_as_independent": False,
    }

    save_csv(RESULTS / "sac_stability_checkpoint_metrics.csv", checkpoint_rows)
    save_csv(RESULTS / "sac_stability_selection_comparison.csv", selection_rows)
    save_csv(RESULTS / "sac_stability_cross_seed_step_summary.csv", step_summary)
    save_csv(RESULTS / "sac_stability_mcnemar_tests.csv", test_rows)
    save_csv(RESULTS / "sac_stability_existing_checkpoint_metrics.csv", existing_rows)
    summary_path = RESULTS / "sac_stability_analysis_summary.json"
    write_json_idempotent(summary_path, summary)
    print(json.dumps({"step_summary": step_summary, "progress": progress, "selections": selection_rows}, indent=2))


if __name__ == "__main__":
    main()
