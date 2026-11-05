"""Run preregistered 500-position validation and frozen 2,000-position tests."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


OUT = Path("bc_results")
RUNS = Path("bc_residual_sac_runs")
PROTOCOL = OUT / "sac_regularization_replication_protocol.json"
PROTOCOL_HASH_FILE = OUT / "sac_regularization_replication_protocol.sha256"
FREEZE = OUT / "sac_regularization_replication_checkpoint_freeze.json"
SEEDS = (1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061)
ARMS = ("A", "B")
STEPS = (10_000, 20_000, 30_000)
COEFFICIENTS = {"A": 0.005, "B": 0.020}
BC = "bc_models/fair_wide_expert_seed42.pt"
VAL_START, VAL_N = 250_000, 500
FINAL_START, FINAL_N = 260_000, 2_000
OLD_RANGES = (
    (61_000, 61_039), (150_000, 151_999), (170_000, 171_999),
    (190_000, 191_999), (210_000, 210_499), (220_000, 221_999),
    (230_000, 230_499), (240_000, 241_999),
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_new_csv(path, records):
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def verify_ranges_and_protocol():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_hash = PROTOCOL_HASH_FILE.read_text(encoding="utf-8").split()[0]
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("Protocol hash does not match committed preregistration")
    val = set(range(VAL_START, VAL_START + VAL_N))
    final = set(range(FINAL_START, FINAL_START + FINAL_N))
    if val & final:
        raise AssertionError("Validation and Final ranges overlap")
    for start, end in OLD_RANGES:
        if val.intersection(range(start, end + 1)) or final.intersection(range(start, end + 1)):
            raise AssertionError(f"Evaluation range reuses historical seeds {start}-{end}")
    if protocol["evaluation"]["validation_seed_range_inclusive"] != [VAL_START, VAL_START + VAL_N - 1]:
        raise RuntimeError("Validation range differs from protocol")
    if protocol["evaluation"]["final_seed_range_inclusive"] != [FINAL_START, FINAL_START + FINAL_N - 1]:
        raise RuntimeError("Final range differs from protocol")
    return protocol_hash


def run_command(policy, start, count, output, checkpoint=None, coefficient=0.005):
    cmd = [
        sys.executable, "evaluate_sac_regression.py", "--policy", policy,
        "--bc-model", BC, "--position-range", "0.20", "--residual-meters", "0.005",
        "--residual-penalty-coef", str(coefficient), "--episodes", str(count),
        "--seed", str(start), "--output", str(output),
    ]
    if checkpoint:
        cmd.extend(("--sac-model", str(checkpoint)))
    return cmd


def ensure_bc_baseline(label, start, count):
    path = OUT / f"sac_replication_bc_{label}.csv"
    if path.exists():
        rows = read(path)
        expected_seeds = set(range(start, start + count))
        if len(rows) != count or {int(r["seed"]) for r in rows} != expected_seeds:
            raise RuntimeError(f"Existing BC baseline is incomplete or mismatched: {path}")
        return path
    subprocess.run(run_command("bc", start, count, path), check=True)
    return path


def compose_pair(bc_path, sac_path, paired_path, count):
    if paired_path.exists():
        existing = read(paired_path)
        if len(existing) != 2 * count:
            raise RuntimeError(f"Existing paired result is incomplete: {paired_path}")
        return
    bc_rows, sac_rows = read(bc_path), read(sac_path)
    if len(bc_rows) != count or len(sac_rows) != count:
        raise AssertionError(f"Wrong rollout count for {paired_path}")
    bmap = {int(r["seed"]): r for r in bc_rows}
    smap = {int(r["seed"]): r for r in sac_rows}
    if set(bmap) != set(smap):
        raise AssertionError(f"Paired policy seeds differ in {paired_path}")
    for seed in bmap:
        if (bmap[seed]["cube_x"], bmap[seed]["cube_y"]) != (smap[seed]["cube_x"], smap[seed]["cube_y"]):
            raise AssertionError(f"Initial positions differ at seed {seed}")
    write_new_csv(paired_path, bc_rows + sac_rows)
    Path(sac_path).unlink()


def run_validation(protocol_hash):
    OUT.mkdir(exist_ok=True)
    ensure_bc_baseline("validation500", VAL_START, VAL_N)
    for seed in SEEDS:
        for arm in ARMS:
            for step in STEPS:
                checkpoint = RUNS / f"replication_{arm}_seed{seed}" / f"checkpoint_{step}.zip"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                stem = f"seed{seed}_arm{arm}_step{step}_validation500"
                paired = OUT / f"sac_replication_{stem}.csv"
                raw = OUT / f"sac_replication_{stem}_sac.csv"
                if paired.exists():
                    if len(read(paired)) != 2 * VAL_N:
                        raise RuntimeError(f"Incomplete paired validation: {paired}")
                    continue
                if not raw.exists():
                    subprocess.run(run_command("sac", VAL_START, VAL_N, raw, checkpoint, COEFFICIENTS[arm]), check=True)
                compose_pair(OUT / "sac_replication_bc_validation500.csv", raw, paired, VAL_N)
                print(f"Validation complete: {seed} {arm} {step}", flush=True)
    print(f"All 60 validation candidates complete (protocol sha256 {protocol_hash}).", flush=True)


def run_final(protocol_hash):
    if not FREEZE.is_file():
        raise RuntimeError("Final evaluation is forbidden until the Validation selection is frozen")
    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != protocol_hash:
        raise RuntimeError("Frozen selection refers to a different preregistered protocol")
    if manifest["validation_seed_range"] != [VAL_START, VAL_START + VAL_N - 1]:
        raise AssertionError("Unexpected Validation range in frozen manifest")
    if manifest["final_seed_range"] != [FINAL_START, FINAL_START + FINAL_N - 1]:
        raise AssertionError("Unexpected Final range in frozen manifest")
    for selection in manifest["selections"]:
        for candidate in selection["candidates"]:
            if sha256(candidate["checkpoint_path"]) != candidate["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint changed after selection freeze: {candidate['checkpoint_path']}")
            if sha256(candidate["validation_csv"]) != candidate["validation_csv_sha256"]:
                raise RuntimeError(f"Validation data changed after freeze: {candidate['validation_csv']}")
    baseline = ensure_bc_baseline("final2000", FINAL_START, FINAL_N)
    for selection in manifest["selections"]:
        seed, arm = selection["sac_seed"], selection["arm"]
        selected = int(selection["selected_step"])
        for step in sorted({30_000, selected}):
            checkpoint = RUNS / f"replication_{arm}_seed{seed}" / f"checkpoint_{step}.zip"
            stem = f"seed{seed}_arm{arm}_step{step}_final2000"
            paired, raw = OUT / f"sac_replication_{stem}.csv", OUT / f"sac_replication_{stem}_sac.csv"
            if paired.exists():
                if len(read(paired)) != 2 * FINAL_N:
                    raise RuntimeError(f"Incomplete paired Final result: {paired}")
                continue
            if not raw.exists():
                subprocess.run(run_command("sac", FINAL_START, FINAL_N, raw, checkpoint, COEFFICIENTS[arm]), check=True)
            compose_pair(baseline, raw, paired, FINAL_N)
            print(f"Final complete: {seed} {arm} {step}", flush=True)
    print("All frozen Final checkpoints complete.", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("validation", "final"):
        raise SystemExit("Usage: run_sac_regularization_replication_evaluations.py {validation|final}")
    protocol_hash = verify_ranges_and_protocol()
    if sys.argv[1] == "validation":
        run_validation(protocol_hash)
    else:
        run_final(protocol_hash)


if __name__ == "__main__":
    main()
