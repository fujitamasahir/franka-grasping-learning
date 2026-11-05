"""Calibrate an episode-level BC/SAC gate from paired rollout CSVs.

The gate observes only the cube's initial (x, y) position. It NEVER
uses a test episode's success label to choose a controller. Because the
BC-only and BC+SAC rollouts are deterministic and paired by seed, a
gated episode's outcome can be evaluated offline by selecting the
corresponding recorded outcome. Validate that assumption with
evaluate_selective_residual_sac.py on the actual simulator.

No MuJoCo, PyTorch, or scikit-learn dependency is needed here.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


QUADRANTS = ("EN", "ES", "WN", "WS")


def load_pairs(path):
    """Read and validate paired BC-only / BC+SAC evaluation rows."""
    pairs = defaultdict(dict)
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"seed", "cube_x", "cube_y", "success", "policy"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path}: missing columns {required - set(reader.fieldnames or [])}")

        for row in reader:
            seed = int(row["seed"])
            policy = row["policy"]
            if policy not in ("bc_only", "bc_plus_sac"):
                raise ValueError(f"Unexpected policy: {policy}")
            if policy in pairs[seed]:
                raise ValueError(f"Duplicate seed/policy: {seed}/{policy}")
            success = int(row["success"])
            if success not in (0, 1):
                raise ValueError(f"Invalid success label: {success}")
            pairs[seed][policy] = {
                "cube_x": float(row["cube_x"]),
                "cube_y": float(row["cube_y"]),
                "success": success,
            }

    if not pairs:
        raise ValueError(f"No episodes found in {path}")

    result = {}
    for seed, pair in pairs.items():
        if set(pair) != {"bc_only", "bc_plus_sac"}:
            raise ValueError(f"Missing paired policy for seed {seed}")
        bc, sac = pair["bc_only"], pair["bc_plus_sac"]
        for axis in ("cube_x", "cube_y"):
            if not math.isclose(bc[axis], sac[axis], abs_tol=1e-9, rel_tol=0):
                raise ValueError(f"Position mismatch for seed {seed}, axis {axis}")
        result[seed] = {
            "seed": seed,
            "cube_x": bc["cube_x"],
            "cube_y": bc["cube_y"],
            "bc": bc["success"],
            "sac": sac["success"],
        }
    return result


def quadrant(x, y, center_x=0.55, center_y=0.0):
    """EN = +x/+y; ES = +x/-y; WN = -x/+y; WS = -x/-y."""
    return ("E" if x >= center_x else "W") + ("N" if y >= center_y else "S")


def outside_demonstrations(x, y, center_x=0.55, center_y=0.0, inner_range=0.12):
    """The square complement, not Euclidean distance from the center."""
    return max(abs(x - center_x), abs(y - center_y)) > inner_range


def fit_gate(calibration, calibration_path, center_x=0.55, center_y=0.0,
             inner_range=0.12, min_quadrant_episodes=30):
    """Fit two prespecified, interpretable gate variants.

    Primary 'safety': at least 5 rescues and ZERO regressions in a
    calibration quadrant. Secondary 'balanced': at least 3 rescues
    and strictly more rescues than regressions. Both require >=30
    outer-region calibration episodes per quadrant.
    """
    stats = {q: {"episodes": 0, "rescues": 0, "regressions": 0} for q in QUADRANTS}
    for row in calibration.values():
        x, y = row["cube_x"], row["cube_y"]
        if not outside_demonstrations(x, y, center_x, center_y, inner_range):
            continue
        s = stats[quadrant(x, y, center_x, center_y)]
        s["episodes"] += 1
        s["rescues"] += int(row["bc"] == 0 and row["sac"] == 1)
        s["regressions"] += int(row["bc"] == 1 and row["sac"] == 0)

    eligible = {
        q: s for q, s in stats.items()
        if s["episodes"] >= min_quadrant_episodes
    }
    safety = sorted(
        q for q, s in eligible.items()
        if s["rescues"] >= 5 and s["regressions"] == 0
    )
    balanced = sorted(
        q for q, s in eligible.items()
        if s["rescues"] >= 3 and s["rescues"] > s["regressions"]
    )
    return {
        "schema_version": 1,
        "gate_type": "initial_cube_position_quadrant",
        "center_x": center_x,
        "center_y": center_y,
        "inner_range": inner_range,
        "min_quadrant_episodes": min_quadrant_episodes,
        "calibration_path": str(calibration_path),
        "calibration_sha256": hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
        "calibration_seeds": sorted(calibration),
        "calibration_stats": stats,
        "modes": {"safety": safety, "balanced": balanced},
        "selection_rules": {
            "safety": ">=5 rescues, 0 regressions, >=30 episodes per outer quadrant",
            "balanced": ">=3 rescues, rescues>regressions, >=30 episodes per outer quadrant",
        },
    }


def gate_uses_sac(config, mode, cube_x, cube_y):
    """Decide BEFORE a rollout using position only (never a success label)."""
    if mode not in ("safety", "balanced"):
        raise ValueError(f"Unknown gate mode: {mode}")
    if not outside_demonstrations(
        cube_x, cube_y, config["center_x"], config["center_y"], config["inner_range"]
    ):
        return False
    return quadrant(
        cube_x, cube_y, config["center_x"], config["center_y"]
    ) in config["modes"][mode]


def exact_mcnemar_p(rescues, regressions):
    """Two-sided exact binomial McNemar p-value for paired binary outcomes."""
    n = rescues + regressions
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(rescues, regressions) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def summarize(pairs, config):
    """Evaluate BC, always-SAC, and both gates on the SAME paired seeds."""
    n = len(pairs)
    totals = {
        "bc_only": {"successes": 0, "rescues": 0, "regressions": 0, "sac_episodes": 0},
        "always_sac": {"successes": 0, "rescues": 0, "regressions": 0, "sac_episodes": n},
        "safety": {"successes": 0, "rescues": 0, "regressions": 0, "sac_episodes": 0},
        "balanced": {"successes": 0, "rescues": 0, "regressions": 0, "sac_episodes": 0},
    }
    details = []
    for seed in sorted(pairs):
        row = pairs[seed]
        bc, sac = row["bc"], row["sac"]
        x, y = row["cube_x"], row["cube_y"]
        item = {
            "seed": seed, "cube_x": x, "cube_y": y,
            "region": "outside" if outside_demonstrations(
                x, y, config["center_x"], config["center_y"], config["inner_range"]
            ) else "inside",
            "quadrant": quadrant(x, y, config["center_x"], config["center_y"]),
            "bc_success": bc, "sac_success": sac,
        }
        totals["bc_only"]["successes"] += bc
        for mode in ("always_sac", "safety", "balanced"):
            use = (mode == "always_sac") or gate_uses_sac(config, mode, x, y)
            result = sac if use else bc
            totals[mode]["successes"] += result
            totals[mode]["rescues"] += int(bc == 0 and result == 1)
            totals[mode]["regressions"] += int(bc == 1 and result == 0)
            if mode != "always_sac":
                totals[mode]["sac_episodes"] += int(use)
                item[f"{mode}_uses_sac"] = int(use)
                item[f"{mode}_success"] = result
        details.append(item)

    for mode, s in totals.items():
        s["episodes"] = n
        s["success_rate"] = s["successes"] / n
        s["delta_vs_bc_pp"] = 100.0 * (
            s["successes"] - totals["bc_only"]["successes"]
        ) / n
        s["mcnemar_p_vs_bc"] = exact_mcnemar_p(s["rescues"], s["regressions"])
    return totals, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--calibration", help="Paired CSV used to FIT a new gate")
    source.add_argument("--gate-config", help="Frozen gate JSON; do not refit on final test")
    parser.add_argument("--evaluation", required=True, help="Paired evaluation CSV")
    parser.add_argument("--output", required=True, help="Per-episode gated results CSV")
    parser.add_argument("--stage", choices=("exploratory", "confirmatory"),
                        default="exploratory")
    args = parser.parse_args()
    if args.stage == "confirmatory" and not args.gate_config:
        parser.error("Confirmatory evaluation requires --gate-config (frozen gate)")

    if args.calibration:
        calibration = load_pairs(args.calibration)
        config = fit_gate(calibration, args.calibration)
    else:
        with Path(args.gate_config).open(encoding="utf-8") as file:
            config = json.load(file)
        if config.get("schema_version") != 1:
            raise ValueError("Unsupported gate configuration schema")

    evaluation = load_pairs(args.evaluation)
    overlap = set(config["calibration_seeds"]) & set(evaluation)
    if overlap:
        raise ValueError(f"Calibration/evaluation seed leakage: {sorted(overlap)[:10]}")

    totals, details = summarize(evaluation, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)

    config_path = output.with_name(output.stem + "_gate.json")
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    summary_path = output.with_name(output.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump({
            "stage": args.stage,
            "evaluation_csv": args.evaluation,
            "evaluation_seeds": [min(evaluation), max(evaluation)],
            "calibration_sha256": config["calibration_sha256"],
            "selected_quadrants": config["modes"],
            "methods": totals,
            "note": (
                "Exploratory: previous evaluation results have already been inspected."
                if args.stage == "exploratory"
                else "Confirmatory only if these seeds were untouched before freezing the gate."
            ),
        }, file, indent=2)

    print(f"Gate calibrated on {len(config['calibration_seeds'])} episodes")
    print(f"Selected quadrants: {config['modes']}")
    print(f"Evaluation: {len(evaluation)} matched positions ({args.stage})")
    print(f"{'Policy':<14} {'Success':>10} {'Rescue':>8} {'Regress':>9} {'Use SAC':>9} {'p vs BC':>11}")
    for mode, s in totals.items():
        print(f"{mode:<14} {s['successes']:>4}/{s['episodes']:<5} "
              f"{s['rescues']:>8} {s['regressions']:>9} "
              f"{s['sac_episodes']:>9} {s['mcnemar_p_vs_bc']:>11.4g}")
    print(f"Saved: {output}, {config_path}, {summary_path}")


if __name__ == "__main__":
    main()
