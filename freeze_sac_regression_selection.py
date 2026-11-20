"""Select each arm/seed checkpoint on Validation B and freeze hashes."""

import csv
import hashlib
import json
import math
from pathlib import Path


SEEDS = (907, 911, 919, 929, 937)
ARMS = ("A", "B")
STEPS = (10_000, 20_000, 30_000)
VAL_SEEDS = set(range(230_000, 230_500))
FINAL_SEEDS = set(range(240_000, 242_000))
BC_PATH = Path("bc_models/fair_wide_expert_seed42.pt")
OUT = Path("bc_results")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(path):
    rows = read(path)
    if len(rows) != 1000:
        raise AssertionError(f"Expected 1000 rows in {path}, got {len(rows)}")
    bc = {int(r["seed"]): r for r in rows if r["policy"] == "bc_only"}
    sac = {int(r["seed"]): r for r in rows if r["policy"] == "bc_plus_sac"}
    if set(bc) != VAL_SEEDS or set(sac) != VAL_SEEDS:
        raise AssertionError(f"Unexpected validation seeds/policies in {path}")
    rescue = regression = 0
    successful_steps = []
    for seed in sorted(VAL_SEEDS):
        b, s = bc[seed], sac[seed]
        if (b["cube_x"], b["cube_y"]) != (s["cube_x"], s["cube_y"]):
            raise AssertionError(f"Initial position mismatch at {seed} in {path}")
        b_success, s_success = int(b["success"]), int(s["success"])
        rescue += int(not b_success and s_success)
        regression += int(b_success and not s_success)
        if s_success:
            successful_steps.append(int(s["steps"]))
    successes = sum(int(r["success"]) for r in sac.values())
    return {
        "episodes": 500,
        "successes": successes,
        "success_rate": successes / 500,
        "rescue_count": rescue,
        "regression_count": regression,
        "net_improvement": rescue - regression,
        "mean_successful_episode_steps": sum(successful_steps) / len(successful_steps) if successful_steps else None,
        "mean_residual_norm_mm": sum(float(r["mean_residual_norm_mm"]) for r in sac.values()) / 500,
        "action_saturation_frequency": sum(float(r["action_saturation_frequency"]) for r in sac.values()) / 500,
    }


def main():
    if VAL_SEEDS & FINAL_SEEDS:
        raise AssertionError("Validation and final position seeds overlap")
    candidates_out, selections = [], []
    for seed in SEEDS:
        for arm in ARMS:
            candidates = []
            for step in STEPS:
                run = Path("bc_residual_sac_runs") / f"regdiag_{arm}_seed{seed}"
                checkpoint = run / f"checkpoint_{step}.zip"
                validation = OUT / f"sac_regdiag_seed{seed}_arm{arm}_step{step}_validationB500.csv"
                if not checkpoint.is_file() or not validation.is_file():
                    raise FileNotFoundError(checkpoint if not checkpoint.is_file() else validation)
                metrics = summarize(validation)
                candidate = {
                    "timesteps": step,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "validation_csv": str(validation),
                    "validation_csv_sha256": sha256(validation),
                    **metrics,
                }
                candidates.append(candidate)
                candidates_out.append({"sac_seed": seed, "arm": arm, **candidate})
            chosen = sorted(
                candidates,
                key=lambda x: (
                    -x["success_rate"],
                    x["regression_count"],
                    x["mean_successful_episode_steps"] if x["mean_successful_episode_steps"] is not None else math.inf,
                    x["timesteps"],
                ),
            )[0]
            selections.append({
                "sac_seed": seed,
                "arm": arm,
                "selected_step": chosen["timesteps"],
                "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
                "candidates": candidates,
            })
    freeze_path = OUT / "sac_regdiag_checkpoint_selection_freeze.json"
    metrics_path = OUT / "sac_regdiag_validation_metrics.csv"
    if freeze_path.exists() or metrics_path.exists():
        raise FileExistsError("Refusing to overwrite frozen selection artifacts")
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates_out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidates_out)
    manifest = {
        "purpose": "Freeze checkpoint choices before final outcomes are evaluated",
        "base_commit": "3037449bb64eb188627ac7b0a7804b34b40e5c54",
        "bc_checkpoint": str(BC_PATH),
        "bc_checkpoint_sha256": sha256(BC_PATH),
        "validation_seed_range": [230_000, 230_499],
        "final_seed_range": [240_000, 241_999],
        "excluded_prior_final_ranges": [[150_000, 151_999], [170_000, 171_999], [190_000, 191_999], [220_000, 221_999]],
        "checkpoint_selection_rule": [
            "highest Validation B success rate",
            "fewest paired regressions vs BC Only",
            "shortest mean successful episode steps",
            "earliest training checkpoint",
        ],
        "exploratory_decision_criteria": {
            "regression_decreases_in_at_least_n_of_5_seeds": 4,
            "mean_final_success_rate_drop_maximum_percentage_points": 0.5,
            "operational_rescue_preservation_tolerance": "Mean decrease no greater than 10 rescue episodes per 2,000 positions; this operationalizes the otherwise qualitative phrase 'no large rescue decrease'.",
        },
        "candidates": list(STEPS),
        "selections": selections,
    }
    freeze_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: x[k] for k in ("sac_seed", "arm", "selected_step")} for x in selections], indent=2))


if __name__ == "__main__":
    main()
