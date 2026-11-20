"""Select all replication checkpoints on Validation and freeze all hashes."""

import csv
import hashlib
import json
import math
from pathlib import Path


OUT = Path("bc_results")
RUNS = Path("bc_residual_sac_runs")
PROTOCOL = OUT / "sac_regularization_replication_protocol.json"
PROTOCOL_HASH_FILE = OUT / "sac_regularization_replication_protocol.sha256"
BC = Path("bc_models/fair_wide_expert_seed42.pt")
SEEDS = (1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061)
ARMS = ("A", "B")
STEPS = (10_000, 20_000, 30_000)
VAL_SET = set(range(250_000, 250_500))
FINAL_SET = set(range(260_000, 262_000))
FREEZE = OUT / "sac_regularization_replication_checkpoint_freeze.json"
METRICS = OUT / "sac_regularization_replication_validation_metrics.csv"


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
        raise AssertionError(f"Expected 1000 paired rows in {path}; got {len(rows)}")
    bc = {int(r["seed"]): r for r in rows if r["policy"] == "bc_only"}
    sac = {int(r["seed"]): r for r in rows if r["policy"] == "bc_plus_sac"}
    if set(bc) != VAL_SET or set(sac) != VAL_SET:
        raise AssertionError(f"Unexpected validation seed set in {path}")
    rescues = regressions = 0
    success_steps = []
    successes = 0
    for seed in sorted(VAL_SET):
        b, s = bc[seed], sac[seed]
        if (b["cube_x"], b["cube_y"]) != (s["cube_x"], s["cube_y"]):
            raise AssertionError(f"BC/SAC position mismatch in {path} at {seed}")
        bs, ss = int(b["success"]), int(s["success"])
        successes += ss
        rescues += int(not bs and ss)
        regressions += int(bs and not ss)
        if ss:
            success_steps.append(int(s["steps"]))
    return {
        "episodes": len(VAL_SET),
        "successes": successes,
        "success_rate": successes / len(VAL_SET),
        "rescue_count": rescues,
        "regression_count": regressions,
        "net_improvement": rescues - regressions,
        "mean_successful_episode_steps": float(sum(success_steps) / len(success_steps)) if success_steps else None,
        "mean_residual_norm_mm": sum(float(r["mean_residual_norm_mm"]) for r in sac.values()) / len(VAL_SET),
        "action_saturation_frequency": sum(float(r["action_saturation_frequency"]) for r in sac.values()) / len(VAL_SET),
    }


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_hash = PROTOCOL_HASH_FILE.read_text(encoding="utf-8").split()[0]
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("Protocol SHA-256 mismatch")
    if VAL_SET & FINAL_SET:
        raise AssertionError("Validation and Final sets overlap")
    for start, end in protocol["evaluation"]["historical_evaluation_ranges_excluded"]:
        if VAL_SET.intersection(range(start, end + 1)) or FINAL_SET.intersection(range(start, end + 1)):
            raise AssertionError(f"New evaluation set overlaps old range {start}-{end}")
    if FREEZE.exists() or METRICS.exists():
        raise FileExistsError("Refusing to overwrite an existing selection freeze")
    candidates_out, selections = [], []
    checkpoint_audit = []
    for seed in SEEDS:
        for arm in ARMS:
            candidates = []
            run = RUNS / f"replication_{arm}_seed{seed}"
            config_path = run / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected_coef = 0.005 if arm == "A" else 0.020
            if config["seed"] != seed or config["residual_penalty_coef"] != expected_coef:
                raise AssertionError(f"Run config mismatch: {config_path}")
            if config["bc_model"] != "bc_models/fair_wide_expert_seed42.pt":
                raise AssertionError(f"BC model mismatch: {config_path}")
            for step in STEPS:
                checkpoint = run / f"checkpoint_{step}.zip"
                validation = OUT / f"sac_replication_seed{seed}_arm{arm}_step{step}_validation500.csv"
                if not checkpoint.is_file() or not validation.is_file():
                    raise FileNotFoundError(checkpoint if not checkpoint.is_file() else validation)
                digest = sha256(checkpoint)
                checkpoint_audit.append({
                    "sac_seed": seed, "arm": arm, "timesteps": step,
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest,
                    "run_config_sha256": sha256(config_path),
                })
                metrics = summarize(validation)
                candidate = {
                    "timesteps": step,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "run_config_sha256": sha256(config_path),
                    "validation_csv": str(validation),
                    "validation_csv_sha256": sha256(validation),
                    **metrics,
                }
                candidates.append(candidate)
                candidates_out.append({"sac_seed": seed, "arm": arm, **candidate})
            chosen = sorted(candidates, key=lambda x: (
                -x["success_rate"],
                x["regression_count"],
                x["mean_successful_episode_steps"] if x["mean_successful_episode_steps"] is not None else math.inf,
                x["timesteps"],
            ))[0]
            selections.append({
                "sac_seed": seed,
                "arm": arm,
                "selected_step": chosen["timesteps"],
                "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
                "candidates": candidates,
            })
    with METRICS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates_out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidates_out)
    manifest = {
        "purpose": "Freeze all checkpoint selections and hashes before Final outcomes are evaluated",
        "source_commit_sha": protocol["source_commit_sha"],
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": protocol_hash,
        "bc_checkpoint": str(BC),
        "bc_checkpoint_sha256": sha256(BC),
        "validation_seed_range": [250000, 250499],
        "final_seed_range": [260000, 261999],
        "selection_rule": protocol["checkpoint_selection"]["rule_order"],
        "decision_criteria": protocol["decision_criteria"],
        "all_60_checkpoint_hashes": checkpoint_audit,
        "selections": selections,
    }
    FREEZE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([
        {"seed": x["sac_seed"], "arm": x["arm"], "selected_step": x["selected_step"]}
        for x in selections
    ], indent=2))


if __name__ == "__main__":
    main()
