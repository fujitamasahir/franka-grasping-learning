"""Run Validation B and frozen Final evaluations for regularization arms."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


SEEDS = (907, 911, 919, 929, 937)
ARMS = ("A", "B")
STEPS = (10_000, 20_000, 30_000)
COEFFICIENTS = {"A": 0.005, "B": 0.020}
BC = "bc_models/fair_wide_expert_seed42.pt"
VALIDATION_SEED, VALIDATION_EPISODES = 230_000, 500
FINAL_SEED, FINAL_EPISODES = 240_000, 2_000
FORBIDDEN = (
    set(range(150_000, 152_000)) | set(range(170_000, 172_000))
    | set(range(190_000, 192_000)) | set(range(220_000, 222_000))
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def compose_paired(baseline_path, sac_path, output_path, policy_rows):
    if Path(output_path).exists():
        paired = rows(output_path)
        if len(paired) == policy_rows * 2:
            return
        raise RuntimeError(f"Existing incomplete paired output: {output_path}")
    bc_rows, sac_rows = rows(baseline_path), rows(sac_path)
    if len(bc_rows) != policy_rows or len(sac_rows) != policy_rows:
        raise AssertionError(f"Unexpected rollout row count for {output_path}")
    bc_by_seed = {int(r["seed"]): r for r in bc_rows}
    sac_by_seed = {int(r["seed"]): r for r in sac_rows}
    if set(bc_by_seed) != set(sac_by_seed):
        raise AssertionError(f"BC/SAC episode seeds differ in {output_path}")
    for seed in bc_by_seed:
        b, s = bc_by_seed[seed], sac_by_seed[seed]
        if (b["cube_x"], b["cube_y"]) != (s["cube_x"], s["cube_y"]):
            raise AssertionError(f"Paired initial position differs at seed {seed}")
    write_csv(output_path, bc_rows + sac_rows)
    Path(sac_path).unlink()


def eval_command(policy, seed, episodes, output, checkpoint=None, coefficient=0.005):
    cmd = [
        sys.executable, "evaluate_sac_regression.py", "--policy", policy,
        "--bc-model", BC, "--position-range", "0.20",
        "--residual-meters", "0.005", "--residual-penalty-coef", str(coefficient),
        "--episodes", str(episodes), "--seed", str(seed), "--output", str(output),
    ]
    if checkpoint is not None:
        cmd.extend(("--sac-model", str(checkpoint)))
    return cmd


def ensure_baseline(label, seed, episodes):
    baseline = Path("bc_results") / f"sac_regdiag_bc_{label}.csv"
    if baseline.exists():
        if len(rows(baseline)) != episodes:
            raise RuntimeError(f"Refusing incomplete baseline {baseline}")
        return baseline
    subprocess.run(eval_command("bc", seed, episodes, baseline), check=True)
    return baseline


def validation_candidates():
    for seed in SEEDS:
        for arm in ARMS:
            for step in STEPS:
                yield seed, arm, step


def run_validation():
    results = Path("bc_results")
    results.mkdir(exist_ok=True)
    val_seeds = set(range(VALIDATION_SEED, VALIDATION_SEED + VALIDATION_EPISODES))
    final_seeds = set(range(FINAL_SEED, FINAL_SEED + FINAL_EPISODES))
    if val_seeds & final_seeds or (val_seeds | final_seeds) & FORBIDDEN:
        raise AssertionError("Validation/final seed ranges overlap or reuse prior final seeds")
    baseline = ensure_baseline("validationB500", VALIDATION_SEED, VALIDATION_EPISODES)
    for seed, arm, step in validation_candidates():
        run_dir = Path("bc_residual_sac_runs") / f"regdiag_{arm}_seed{seed}"
        checkpoint = run_dir / f"checkpoint_{step}.zip"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        key = f"seed{seed}_arm{arm}_step{step}"
        paired = results / f"sac_regdiag_{key}_validationB500.csv"
        raw = results / f"sac_regdiag_{key}_validationB500_sac.csv"
        if paired.exists():
            if len(rows(paired)) != 2 * VALIDATION_EPISODES:
                raise RuntimeError(f"Incomplete paired result {paired}")
            continue
        if not raw.exists():
            subprocess.run(eval_command(
                "sac", VALIDATION_SEED, VALIDATION_EPISODES, raw, checkpoint,
                COEFFICIENTS[arm],
            ), check=True)
        compose_paired(baseline, raw, paired, VALIDATION_EPISODES)
        print(f"Saved {paired}", flush=True)


def run_final():
    results = Path("bc_results")
    freeze_path = results / "sac_regdiag_checkpoint_selection_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("Final evaluation requires frozen Validation B selection")
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    if manifest["validation_seed_range"] != [230_000, 230_499] or manifest["final_seed_range"] != [240_000, 241_999]:
        raise AssertionError("Unexpected frozen evaluation ranges")
    for selection in manifest["selections"]:
        for candidate in selection["candidates"]:
            model_path = Path(candidate["checkpoint_path"])
            val_path = Path(candidate["validation_csv"])
            if sha256(model_path) != candidate["checkpoint_sha256"]:
                raise AssertionError(f"Checkpoint changed after freeze: {model_path}")
            if sha256(val_path) != candidate["validation_csv_sha256"]:
                raise AssertionError(f"Validation data changed after freeze: {val_path}")
    baseline = ensure_baseline("final2000", FINAL_SEED, FINAL_EPISODES)
    for selection in manifest["selections"]:
        seed, arm = selection["sac_seed"], selection["arm"]
        steps = sorted({30_000, int(selection["selected_step"])})
        for step in steps:
            run_dir = Path("bc_residual_sac_runs") / f"regdiag_{arm}_seed{seed}"
            checkpoint = run_dir / f"checkpoint_{step}.zip"
            key = f"seed{seed}_arm{arm}_step{step}"
            paired = results / f"sac_regdiag_{key}_final2000.csv"
            raw = results / f"sac_regdiag_{key}_final2000_sac.csv"
            if paired.exists():
                if len(rows(paired)) != 2 * FINAL_EPISODES:
                    raise RuntimeError(f"Incomplete paired result {paired}")
                continue
            if not raw.exists():
                subprocess.run(eval_command(
                    "sac", FINAL_SEED, FINAL_EPISODES, raw, checkpoint,
                    COEFFICIENTS[arm],
                ), check=True)
            compose_paired(baseline, raw, paired, FINAL_EPISODES)
            print(f"Saved {paired}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("validation", "final"):
        raise SystemExit("Usage: run_sac_regression_evaluations.py {validation|final}")
    if sys.argv[1] == "validation":
        run_validation()
    else:
        run_final()
