"""Freeze 40-episode and 500-episode checkpoint choices before final testing."""

import csv
import hashlib
import json
from pathlib import Path


RESULTS = Path("bc_results")
RUNS = Path("bc_residual_sac_runs")
SEEDS = (811, 823, 827, 829, 839)
STEPS = (10_000, 20_000, 30_000)
VAL_B_SEEDS = set(range(210_000, 210_500))
FINAL_SEEDS = set(range(220_000, 222_000))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_validation(path, expected_episodes):
    rows = read_rows(path)
    if len(rows) != expected_episodes * 2:
        raise AssertionError(f"Expected {expected_episodes * 2} rows in {path}")
    sac_rows = [r for r in rows if r["policy"] == "bc_plus_sac"]
    bc_rows = [r for r in rows if r["policy"] == "bc_only"]
    if len(sac_rows) != expected_episodes or len(bc_rows) != expected_episodes:
        raise AssertionError(f"Expected {expected_episodes} SAC rows in {path}")
    if {int(r["seed"]) for r in bc_rows} != VAL_B_SEEDS or {int(r["seed"]) for r in sac_rows} != VAL_B_SEEDS:
        raise AssertionError(f"Unexpected Validation B seed range in {path}")
    bc_map = {int(r["seed"]): r for r in bc_rows}
    sac_map = {int(r["seed"]): r for r in sac_rows}
    for seed, b in bc_map.items():
        s = sac_map[seed]
        if (float(b["cube_x"]) != float(s["cube_x"])
                or float(b["cube_y"]) != float(s["cube_y"])):
            raise AssertionError(f"Unpaired Validation B position in {path}, seed {seed}")
    successes = sum(int(r["success"]) for r in sac_rows)
    steps = [int(r["steps"]) for r in sac_rows if int(r["success"])]
    return {
        "episodes": expected_episodes,
        "successes": successes,
        "success_rate": successes / expected_episodes,
        "mean_successful_steps": sum(steps) / len(steps) if steps else None,
    }


def choose(rows, rate_key, steps_key):
    ordered = sorted(rows, key=lambda r: int(r["timesteps"]))
    winner = None
    for row in ordered:
        rate = row[rate_key]
        mean_steps = row[steps_key]
        mean_steps = float(mean_steps) if mean_steps is not None else float("inf")
        if winner is None or rate > winner[0] or (rate == winner[0] and mean_steps < winner[1]):
            winner = (rate, mean_steps, row)
    return winner[2]


def main():
    if VAL_B_SEEDS & FINAL_SEEDS:
        raise AssertionError("Validation B and final position seeds overlap")
    choices = []
    common_bc_signature = None
    for seed in SEEDS:
        records = []
        for step in STEPS:
            name = f"seed{seed}_step{step}"
            run_dir = RUNS / f"stability_C_seed{seed}"
            history_path = run_dir / "eval_history.csv"
            history = read_rows(history_path)
            a = [row for row in history if int(row["timesteps"]) == step]
            if len(a) != 1 or int(a[0]["episodes"]) != 40:
                raise AssertionError(f"Missing 40-episode Validation A record: {name}")
            val_b_path = RESULTS / f"sac_stability_{name}_validationB500.csv"
            b = summarize_validation(val_b_path, 500)
            validation_rows = read_rows(val_b_path)
            signature = [
                (int(row["seed"]), float(row["cube_x"]), float(row["cube_y"]), int(row["success"]))
                for row in validation_rows if row["policy"] == "bc_only"
            ]
            if common_bc_signature is None:
                common_bc_signature = signature
            elif signature != common_bc_signature:
                raise AssertionError(f"Validation B BC positions/outcomes differ for {name}")
            checkpoint_path = run_dir / f"checkpoint_{step}.zip"
            records.append({
                "timesteps": step,
                "validation_a_rate": float(a[0]["success_rate"]),
                "validation_a_mean_successful_steps": (
                    float(a[0]["mean_steps_success"]) if a[0]["mean_steps_success"] else None
                ),
                "validation_b_rate": b["success_rate"],
                "validation_b_mean_successful_steps": b["mean_successful_steps"],
                "validation_b_successes": b["successes"],
                "validation_b_csv_sha256": sha256(val_b_path),
                "validation_a_history_sha256": sha256(history_path),
                "checkpoint_sha256": sha256(checkpoint_path),
            })
        selected_a = choose(records, "validation_a_rate", "validation_a_mean_successful_steps")
        selected_b = choose(records, "validation_b_rate", "validation_b_mean_successful_steps")
        choices.append({
            "sac_seed": seed,
            "selection_a_step": selected_a["timesteps"],
            "selection_b_step": selected_b["timesteps"],
            "selected_same_checkpoint": selected_a["timesteps"] == selected_b["timesteps"],
            "candidates": records,
        })

    output = RESULTS / "sac_stability_selection_freeze.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = {
        "purpose": "Freeze checkpoint selection before opening final evaluation outcomes",
        "selection_a": "max validation success on fixed 40 episodes; tie by fewer mean successful steps; chronological tie remains earliest",
        "selection_b": "max validation success on fixed 500 episodes; same tie-break",
        "validation_a_seeds": [61000, 61039],
        "validation_b_seeds": [210000, 210499],
        "final_evaluation_seeds": [220000, 221999],
        "candidates": [10000, 20000, 30000],
        "selections": choices,
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("sac_seed", "selection_a_step", "selection_b_step", "selected_same_checkpoint")} for r in choices], indent=2))


if __name__ == "__main__":
    main()
