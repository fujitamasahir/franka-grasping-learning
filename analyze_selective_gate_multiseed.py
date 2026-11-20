"""Summarize fixed-gate outcomes across pre-trained SAC training seeds."""

import argparse
import csv
import json
from pathlib import Path

from analyze_selective_residual_gate import (
    exact_mcnemar_p,
    gate_uses_sac,
    load_pairs,
    outside_demonstrations,
    quadrant,
    summarize,
)


TRAINING_SEEDS = (42, 123, 456)
METHODS = ("bc_only", "always_sac", "safety", "balanced")
GATES = ("safety", "balanced")
QUADRANTS = ("EN", "ES", "WN", "WS")


def outcomes(pairs, config):
    result = {name: {} for name in METHODS}
    for seed, row in pairs.items():
        result["bc_only"][seed] = row["bc"]
        result["always_sac"][seed] = row["sac"]
        for gate in GATES:
            use_sac = gate_uses_sac(config, gate, row["cube_x"], row["cube_y"])
            result[gate][seed] = row["sac"] if use_sac else row["bc"]
    return result


def paired_comparison(a, b):
    wins = sum(a[s] == 1 and b[s] == 0 for s in a)
    losses = sum(a[s] == 0 and b[s] == 1 for s in a)
    return {"wins": wins, "losses": losses,
            "net": wins - losses,
            "mcnemar_p": exact_mcnemar_p(wins, losses)}


def holm_adjust(items):
    """Return Holm adjusted p values keyed by the original item key."""
    ordered = sorted(items, key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, (key, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * pvalue))
        adjusted[key] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-config", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--input", action="append", required=True,
                        help="TRAINING_SEED=paired_eval.csv; supply seeds 42,123,456")
    args = parser.parse_args()

    config = json.loads(Path(args.gate_config).read_text(encoding="utf-8"))
    if config.get("modes") != {"safety": ["EN"], "balanced": ["EN", "WS"]}:
        raise ValueError(f"Unexpected frozen gate choices: {config.get('modes')}")

    paths = {}
    for item in args.input:
        seed_text, path = item.split("=", 1)
        train_seed = int(seed_text)
        if train_seed in paths:
            raise ValueError(f"Duplicate training seed: {train_seed}")
        paths[train_seed] = Path(path)
    if tuple(sorted(paths)) != TRAINING_SEEDS:
        raise ValueError(f"Expected training seeds {TRAINING_SEEDS}, got {tuple(sorted(paths))}")

    results = {}
    shared_bc = None
    shared_positions = None
    expected_eval_seeds = set(range(170000, 172000))
    for train_seed in TRAINING_SEEDS:
        pairs = load_pairs(paths[train_seed])
        if set(pairs) != expected_eval_seeds:
            raise ValueError(f"Training seed {train_seed}: evaluation seed range/count mismatch")
        bc = {s: row["bc"] for s, row in pairs.items()}
        positions = {s: (row["cube_x"], row["cube_y"]) for s, row in pairs.items()}
        if shared_bc is None:
            shared_bc, shared_positions = bc, positions
        elif bc != shared_bc or positions != shared_positions:
            raise ValueError("BC outcomes or initial positions differ across SAC seeds")

        method_outcomes = outcomes(pairs, config)
        totals, _ = summarize(pairs, config)
        metrics = {}
        quadrant_rows = []
        quadrant_counts = {
            method: {q: {"episodes": 0, "rescues": 0, "regressions": 0,
                         "sac_episodes": 0} for q in QUADRANTS}
            for method in METHODS
        }
        for method in METHODS:
            s = totals[method]
            metrics[method] = {
                **s,
                "rescue_count": s["rescues"],
                "regression_count": s["regressions"],
                "net_improvement": s["rescues"] - s["regressions"],
                "sac_activation_rate": s["sac_episodes"] / len(pairs),
                "success_rate_vs_always_pp": 100 * (
                    s["successes"] - totals["always_sac"]["successes"]
                ) / len(pairs),
                "sac_episodes_saved_vs_always": len(pairs) - s["sac_episodes"],
                "sac_usage_reduction_vs_always": 1 - s["sac_episodes"] / len(pairs),
                "mcnemar_vs_bc": paired_comparison(method_outcomes[method], method_outcomes["bc_only"]),
                "mcnemar_vs_always_sac": paired_comparison(method_outcomes[method], method_outcomes["always_sac"]),
            }

        for seed, row in pairs.items():
            x, y = row["cube_x"], row["cube_y"]
            if not outside_demonstrations(x, y, config["center_x"],
                                          config["center_y"], config["inner_range"]):
                continue
            q = quadrant(x, y, config["center_x"], config["center_y"])
            bc, sac = row["bc"], row["sac"]
            for method in METHODS:
                c = quadrant_counts[method][q]
                c["episodes"] += 1
                use = method == "always_sac" or (
                    method in GATES and gate_uses_sac(config, method, x, y)
                )
                result = sac if use else bc
                c["rescues"] += int(bc == 0 and result == 1)
                c["regressions"] += int(bc == 1 and result == 0)
                c["sac_episodes"] += int(use)
        for method in METHODS:
            for q in QUADRANTS:
                quadrant_rows.append({
                    "training_seed": train_seed,
                    "method": method,
                    "quadrant": q,
                    **quadrant_counts[method][q],
                })
        results[train_seed] = {
            "evaluation_csv": str(paths[train_seed]),
            "evaluation_seeds": [170000, 171999],
            "methods": metrics,
            "outer_quadrants": quadrant_counts,
        }

    p_families = {}
    for gate in GATES:
        p_families[gate] = {
            seed: results[seed]["methods"][gate]["mcnemar_vs_bc"]["mcnemar_p"]
            for seed in TRAINING_SEEDS
        }
    p_families["safety_vs_always"] = {
        seed: results[seed]["methods"]["safety"]["mcnemar_vs_always_sac"]["mcnemar_p"]
        for seed in TRAINING_SEEDS
    }
    p_families["balanced_vs_always"] = {
        seed: results[seed]["methods"]["balanced"]["mcnemar_vs_always_sac"]["mcnemar_p"]
        for seed in TRAINING_SEEDS
    }
    holm = {
        family: holm_adjust([(str(seed), p) for seed, p in values.items()])
        for family, values in p_families.items()
    }
    for seed in TRAINING_SEEDS:
        for family, gate in (("safety", "safety"), ("balanced", "balanced")):
            results[seed]["methods"][gate]["mcnemar_vs_bc"]["holm_p_across_training_seeds"] = holm[family][str(seed)]
        for family, gate in (("safety_vs_always", "safety"), ("balanced_vs_always", "balanced")):
            results[seed]["methods"][gate]["mcnemar_vs_always_sac"]["holm_p_across_training_seeds"] = holm[family][str(seed)]

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps({
        "evaluation_seeds": [170000, 171999],
        "episodes_per_training_seed": 2000,
        "sac_training_seeds": list(TRAINING_SEEDS),
        "fixed_gate_config": args.gate_config,
        "selected_quadrants": config["modes"],
        "shared_initial_positions_and_bc_outcomes_verified": True,
        "per_seed": results,
        "holm_adjusted_p_values_across_training_seeds": holm,
        "interpretation_note": (
            "Episode-level exact McNemar tests are conditional on each frozen model and "
            "shared evaluation positions. They do not make the three SAC training seeds "
            "independent replicates of the initial positions; cross-training-seed "
            "generalization is descriptive with only three model seeds."
        ),
    }, indent=2), encoding="utf-8")

    metric_path = prefix.with_name(prefix.name + "_methods.csv")
    fields = ["training_seed", "method", "successes", "episodes", "success_rate",
              "rescues", "regressions", "net_improvement", "sac_episodes",
              "sac_activation_rate", "delta_vs_bc_pp", "success_rate_vs_always_pp",
              "sac_episodes_saved_vs_always", "sac_usage_reduction_vs_always",
              "mcnemar_vs_bc_wins", "mcnemar_vs_bc_losses", "mcnemar_p_vs_bc",
              "mcnemar_p_vs_bc_holm", "mcnemar_vs_always_wins",
              "mcnemar_vs_always_losses", "mcnemar_p_vs_always",
              "mcnemar_p_vs_always_holm"]
    with metric_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for train_seed in TRAINING_SEEDS:
            for method in METHODS:
                m = results[train_seed]["methods"][method]
                bc_test = m["mcnemar_vs_bc"]
                al_test = m["mcnemar_vs_always_sac"]
                writer.writerow({
                    "training_seed": train_seed, "method": method,
                    "successes": m["successes"], "episodes": m["episodes"],
                    "success_rate": m["success_rate"], "rescues": m["rescues"],
                    "regressions": m["regressions"], "net_improvement": m["net_improvement"],
                    "sac_episodes": m["sac_episodes"], "sac_activation_rate": m["sac_activation_rate"],
                    "delta_vs_bc_pp": m["delta_vs_bc_pp"],
                    "success_rate_vs_always_pp": m["success_rate_vs_always_pp"],
                    "sac_episodes_saved_vs_always": m["sac_episodes_saved_vs_always"],
                    "sac_usage_reduction_vs_always": m["sac_usage_reduction_vs_always"],
                    "mcnemar_vs_bc_wins": bc_test["wins"], "mcnemar_vs_bc_losses": bc_test["losses"],
                    "mcnemar_p_vs_bc": bc_test["mcnemar_p"],
                    "mcnemar_p_vs_bc_holm": bc_test.get("holm_p_across_training_seeds", ""),
                    "mcnemar_vs_always_wins": al_test["wins"], "mcnemar_vs_always_losses": al_test["losses"],
                    "mcnemar_p_vs_always": al_test["mcnemar_p"],
                    "mcnemar_p_vs_always_holm": al_test.get("holm_p_across_training_seeds", ""),
                })

    quadrant_path = prefix.with_name(prefix.name + "_quadrants.csv")
    with quadrant_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["training_seed", "method", "quadrant",
                                                "episodes", "rescues", "regressions", "sac_episodes"])
        writer.writeheader()
        for train_seed in TRAINING_SEEDS:
            for method in METHODS:
                for q in QUADRANTS:
                    writer.writerow({"training_seed": train_seed, "method": method,
                                     "quadrant": q,
                                     **results[train_seed]["outer_quadrants"][method][q]})
    print(f"Saved: {metric_path}, {quadrant_path}, {summary_path}")


if __name__ == "__main__":
    main()
