"""Generate frozen Validation B and final paired MuJoCo rollout CSVs."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


BC = "bc_models/fair_wide_expert_seed42.pt"
NEW_SEEDS = (811, 823, 827, 829, 839)
OLD_SEEDS = (7, 19, 73, 202, 314)
STEPS = (10_000, 20_000, 30_000)
FINAL_SEED = 220_000
FINAL_EPISODES = 2_000
VALIDATION_B_SEED = 210_000
VALIDATION_B_EPISODES = 500
FORBIDDEN_SEEDS = (
    set(range(150_000, 152_000))
    | set(range(170_000, 172_000))
    | set(range(190_000, 192_000))
)


def python_runner():
    """Use the current interpreter; MuJoCo is run headlessly in evaluation."""
    return sys.executable


def validate_seed_ranges():
    val_b = set(range(VALIDATION_B_SEED, VALIDATION_B_SEED + VALIDATION_B_EPISODES))
    final = set(range(FINAL_SEED, FINAL_SEED + FINAL_EPISODES))
    if val_b & final or (val_b | final) & FORBIDDEN_SEEDS:
        raise AssertionError("Validation/final positions overlap a forbidden seed range")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_entries(include_old=False):
    root = Path("bc_residual_sac_runs")
    entries = []
    for seed in NEW_SEEDS:
        run = root / f"stability_C_seed{seed}"
        for step in STEPS:
            entries.append((f"seed{seed}_step{step}", run / f"checkpoint_{step}.zip"))
    if include_old:
        for seed in OLD_SEEDS:
            run = root / f"C_hard_shaped_seed{seed}"
            entries.extend((
                (f"existing_seed{seed}_best", run / "best_model.zip"),
                (f"existing_seed{seed}_final", run / "final_model.zip"),
            ))
    return entries


def complete_csv(path, expected_rows):
    if not path.is_file():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = sum(1 for _ in csv.DictReader(f))
        return rows == expected_rows
    except (OSError, csv.Error):
        return False


def run_group(entries, label, seed, episodes):
    out_dir = Path("bc_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, checkpoint in entries:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = out_dir / f"sac_stability_{name}_{label}.csv"
        expected_rows = episodes * 2
        if output.exists():
            if complete_csv(output, expected_rows):
                output.write_bytes(output.read_bytes().replace(b"\r\n", b"\n"))
                print(f"Skipping complete evaluation {output}", flush=True)
                continue
            raise RuntimeError(
                f"Existing incomplete output will not be overwritten: {output}"
            )
        command = [
            python_runner(),
            "evaluate_bc_residual_sac.py",
            "--bc-model", BC,
            "--sac-model", str(checkpoint),
            "--position-range", "0.20",
            "--residual-meters", "0.005",
            "--episodes", str(episodes),
            "--seed", str(seed),
            "--output", str(output),
        ]
        print(f"Evaluating {name} ({label})", flush=True)
        # Episode-level progress is redundant with the paired CSV and would
        # otherwise flood the console for 2,000 episodes per checkpoint.
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        output.write_bytes(output.read_bytes().replace(b"\r\n", b"\n"))
        if not complete_csv(output, expected_rows):
            raise RuntimeError(f"Evaluation output is incomplete: {output}")
        print(f"Saved {output}", flush=True)


def run_final_group(entries):
    """Reuse one BC-only baseline across paired final checkpoint evaluations."""
    out_dir = Path("bc_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / "sac_stability_final_bc_baseline.csv"
    if baseline_path.exists():
        if not complete_csv(baseline_path, FINAL_EPISODES):
            raise RuntimeError(f"Existing incomplete output will not be overwritten: {baseline_path}")
        baseline_path.write_bytes(baseline_path.read_bytes().replace(b"\r\n", b"\n"))
    else:
        command = [
            python_runner(), "evaluate_bc_residual_sac.py",
            "--bc-model", BC,
            "--position-range", "0.20",
            "--residual-meters", "0.005",
            "--episodes", str(FINAL_EPISODES),
            "--seed", str(FINAL_SEED),
            "--output", str(baseline_path),
        ]
        print("Evaluating shared final BC-only baseline", flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        baseline_path.write_bytes(baseline_path.read_bytes().replace(b"\r\n", b"\n"))
        if not complete_csv(baseline_path, FINAL_EPISODES):
            raise RuntimeError(f"BC baseline output is incomplete: {baseline_path}")

    with baseline_path.open(newline="", encoding="utf-8") as f:
        baseline_rows = list(csv.DictReader(f))
    if len(baseline_rows) != FINAL_EPISODES or any(r["policy"] != "bc_only" for r in baseline_rows):
        raise AssertionError("Shared final baseline has unexpected policy rows")
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    if set(baseline_by_seed) != set(range(FINAL_SEED, FINAL_SEED + FINAL_EPISODES)):
        raise AssertionError("Shared final baseline has unexpected position seeds")

    for name, checkpoint in entries:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output = out_dir / f"sac_stability_{name}_final2000.csv"
        if output.exists():
            if complete_csv(output, FINAL_EPISODES * 2):
                output.write_bytes(output.read_bytes().replace(b"\r\n", b"\n"))
                print(f"Skipping complete evaluation {output}", flush=True)
                continue
            raise RuntimeError(f"Existing incomplete output will not be overwritten: {output}")

        sac_path = out_dir / f"sac_stability_{name}_sac_only_partial.csv"
        command = [
            python_runner(), "-u", "evaluate_sac_residual_only.py",
            "--bc-model", BC,
            "--sac-model", str(checkpoint),
            "--position-range", "0.20",
            "--residual-meters", "0.005",
            "--episodes", str(FINAL_EPISODES),
            "--seed", str(FINAL_SEED),
            "--output", str(sac_path),
        ]
        print(f"Evaluating {name} (final2000 SAC; shared BC baseline)", flush=True)
        subprocess.run(command, check=True)
        if not complete_csv(sac_path, FINAL_EPISODES):
            raise RuntimeError(f"SAC-only output is incomplete for {name}")
        with sac_path.open(newline="", encoding="utf-8") as f:
            sac_rows = list(csv.DictReader(f))

        if any(row["policy"] != "bc_plus_sac" for row in sac_rows):
            raise AssertionError(f"Unexpected policy in SAC-only result for {name}")
        if set(int(row["seed"]) for row in sac_rows) != set(baseline_by_seed):
            raise AssertionError(f"Unpaired final seed set for {name}")
        for row in sac_rows:
            baseline = baseline_by_seed[int(row["seed"])]
            if (float(row["cube_x"]) != float(baseline["cube_x"])
                    or float(row["cube_y"]) != float(baseline["cube_y"])):
                raise AssertionError(f"Initial position mismatch for {name}, seed {row['seed']}")

        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(baseline_rows)
            writer.writerows(sac_rows)
        if not complete_csv(output, FINAL_EPISODES * 2):
            raise RuntimeError(f"Final paired result is incomplete: {output}")
        sac_path.unlink()
        print(f"Saved {output}", flush=True)


def main():
    validate_seed_ranges()
    mode = sys.argv[1] if len(sys.argv) == 2 else None
    if mode == "validation-b":
        run_group(
            checkpoint_entries(), "validationB500",
            VALIDATION_B_SEED, VALIDATION_B_EPISODES,
        )
    elif mode == "final":
        freeze = Path("bc_results/sac_stability_selection_freeze.json")
        if not freeze.is_file():
            raise RuntimeError(
                "Final evaluation is blocked until all Selection A/B choices "
                "are frozen in bc_results/sac_stability_selection_freeze.json"
            )
        manifest = json.loads(freeze.read_text(encoding="utf-8"))
        if [row["sac_seed"] for row in manifest["selections"]] != [811, 823, 827, 829, 839]:
            raise AssertionError("Checkpoint freeze manifest has unexpected seeds")
        for selection in manifest["selections"]:
            seed = selection["sac_seed"]
            for candidate in selection["candidates"]:
                step = candidate["timesteps"]
                checkpoint = Path("bc_residual_sac_runs") / f"stability_C_seed{seed}" / f"checkpoint_{step}.zip"
                validation = Path("bc_results") / f"sac_stability_seed{seed}_step{step}_validationB500.csv"
                history = Path("bc_residual_sac_runs") / f"stability_C_seed{seed}" / "eval_history.csv"
                if sha256(checkpoint) != candidate["checkpoint_sha256"]:
                    raise AssertionError(f"Checkpoint changed after selection freeze: {checkpoint}")
                if sha256(validation) != candidate["validation_b_csv_sha256"]:
                    raise AssertionError(f"Validation B CSV changed after selection freeze: {validation}")
                if sha256(history) != candidate["validation_a_history_sha256"]:
                    raise AssertionError(f"Validation A history changed after selection freeze: {history}")
        run_final_group(checkpoint_entries(include_old=True))
    else:
        raise SystemExit(
            "Usage: run_sac_stability_evaluations.py {validation-b|final}"
        )


if __name__ == "__main__":
    main()
