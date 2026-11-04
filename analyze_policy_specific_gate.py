"""Analyze policy-specific calibration against the frozen Seed-456 gate."""

import argparse
import csv
import json
import math
from pathlib import Path

from analyze_selective_residual_gate import (
    exact_mcnemar_p,
    gate_uses_sac,
    load_pairs,
    outside_demonstrations,
    quadrant,
    summarize,
)

TRAIN_SEEDS = (7, 19, 73, 202, 314)
METHODS = ("bc_only", "always_sac", "fixed_safety", "fixed_balanced",
           "policy_safety", "policy_balanced")
QUADRANTS = ("EN", "ES", "WN", "WS")
COMPARISONS = {
    "policy_safety_vs_fixed_safety": ("policy_safety", "fixed_safety"),
    "policy_balanced_vs_fixed_balanced": ("policy_balanced", "fixed_balanced"),
    "policy_safety_vs_bc": ("policy_safety", "bc_only"),
    "policy_balanced_vs_bc": ("policy_balanced", "bc_only"),
    "policy_safety_vs_always": ("policy_safety", "always_sac"),
    "policy_balanced_vs_always": ("policy_balanced", "always_sac"),
}


def paired_test(a, b):
    wins = sum(a[s] == 1 and b[s] == 0 for s in a)
    losses = sum(a[s] == 0 and b[s] == 1 for s in a)
    return {"wins": wins, "losses": losses, "net": wins - losses,
            "mcnemar_p": exact_mcnemar_p(wins, losses)}


def holm(pvalues):
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    result, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        result[name] = running
    return result


def build_outcomes(pairs, fixed, specific):
    outcomes = {name: {} for name in METHODS}
    for seed, row in pairs.items():
        bc, sac, x, y = row["bc"], row["sac"], row["cube_x"], row["cube_y"]
        outcomes["bc_only"][seed] = bc
        outcomes["always_sac"][seed] = sac
        for label, config, mode in (
            ("fixed_safety", fixed, "safety"),
            ("fixed_balanced", fixed, "balanced"),
            ("policy_safety", specific, "safety"),
            ("policy_balanced", specific, "balanced"),
        ):
            use = gate_uses_sac(config, mode, x, y)
            outcomes[label][seed] = sac if use else bc
    return outcomes


def method_counts(pairs, outcome, gate, mode=None):
    n = len(pairs)
    success = sum(outcome.values())
    rescues = regressions = activations = 0
    for seed, row in pairs.items():
        use = mode == "always" or (mode is not None and gate_uses_sac(
            gate, mode, row["cube_x"], row["cube_y"]
        ))
        result = outcome[seed]
        rescues += int(row["bc"] == 0 and result == 1)
        regressions += int(row["bc"] == 1 and result == 0)
        activations += int(use)
    return {
        "successes": success, "episodes": n,
        "success_rate": success / n,
        "rescues": rescues, "regressions": regressions,
        "net_improvement": rescues - regressions,
        "sac_activations": activations,
        "sac_activation_rate": activations / n,
        "delta_vs_bc_pp": 100 * (success - sum(r["bc"] for r in pairs.values())) / n,
    }


def quadrant_counts(pairs, outcome, gate, mode=None):
    result = {q: {"episodes": 0, "rescues": 0, "regressions": 0,
                  "sac_activations": 0} for q in QUADRANTS}
    for seed, row in pairs.items():
        x, y = row["cube_x"], row["cube_y"]
        if not outside_demonstrations(x, y, gate["center_x"], gate["center_y"],
                                      gate["inner_range"]):
            continue
        q = quadrant(x, y, gate["center_x"], gate["center_y"])
        use = mode == "always" or (mode is not None and gate_uses_sac(gate, mode, x, y))
        r = result[q]
        r["episodes"] += 1
        r["rescues"] += int(row["bc"] == 0 and outcome[seed] == 1)
        r["regressions"] += int(row["bc"] == 1 and outcome[seed] == 0)
        r["sac_activations"] += int(use)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-gate", required=True)
    parser.add_argument("--input", action="append", required=True,
                        help="TRAIN_SEED=paired_eval.csv,policy_gate.json")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    fixed = json.loads(Path(args.fixed_gate).read_text(encoding="utf-8"))
    if fixed.get("modes") != {"safety": ["EN"], "balanced": ["EN", "WS"]}:
        raise ValueError("Fixed gate must remain Safety=EN, Balanced=EN/WS")

    inputs = {}
    for spec in args.input:
        seed_text, pair = spec.split("=", 1)
        eval_path, gate_path = pair.split(",", 1)
        seed = int(seed_text)
        if seed in inputs:
            raise ValueError(f"Duplicate SAC seed {seed}")
        inputs[seed] = (Path(eval_path), Path(gate_path))
    if tuple(sorted(inputs)) != TRAIN_SEEDS:
        raise ValueError(f"Expected SAC seeds {TRAIN_SEEDS}; got {tuple(sorted(inputs))}")

    results, shared_positions, shared_bc = {}, None, None
    raw_p = {name: {} for name in COMPARISONS}
    quadrant_rows, method_rows = [], []
    for train_seed in TRAIN_SEEDS:
        eval_path, gate_path = inputs[train_seed]
        specific = json.loads(gate_path.read_text(encoding="utf-8"))
        if specific.get("sac_training_seed") != train_seed:
            raise ValueError(f"Gate model seed mismatch in {gate_path}")
        pairs = load_pairs(eval_path)
        if len(pairs) != 2000 or set(pairs) != set(range(190000, 192000)):
            raise ValueError(f"Final evaluation must be exactly seeds 190000-191999: {eval_path}")
        if set(specific.get("calibration_seeds", [])) & set(pairs):
            raise ValueError(f"Calibration/evaluation seed overlap for SAC seed {train_seed}")
        positions = {s: (p["cube_x"], p["cube_y"]) for s, p in pairs.items()}
        bc = {s: p["bc"] for s, p in pairs.items()}
        if shared_positions is None:
            shared_positions, shared_bc = positions, bc
        elif positions != shared_positions or bc != shared_bc:
            raise ValueError("Position or BC outcome mismatch across SAC models")

        fixed_totals, _ = summarize(pairs, fixed)
        policy_totals, _ = summarize(pairs, specific)
        outcomes = build_outcomes(pairs, fixed, specific)
        totals = {
            "bc_only": method_counts(pairs, outcomes["bc_only"], fixed),
            "always_sac": method_counts(pairs, outcomes["always_sac"], fixed, "always"),
            "fixed_safety": method_counts(pairs, outcomes["fixed_safety"], fixed, "safety"),
            "fixed_balanced": method_counts(pairs, outcomes["fixed_balanced"], fixed, "balanced"),
            "policy_safety": method_counts(pairs, outcomes["policy_safety"], specific, "safety"),
            "policy_balanced": method_counts(pairs, outcomes["policy_balanced"], specific, "balanced"),
        }
        for method, metrics in totals.items():
            metrics["delta_vs_always_sac_pp"] = 100 * (
                metrics["successes"] - totals["always_sac"]["successes"]
            ) / len(pairs)
            metrics["delta_vs_matching_fixed_gate_pp"] = None
        totals["policy_safety"]["delta_vs_matching_fixed_gate_pp"] = 100 * (
            totals["policy_safety"]["successes"] - totals["fixed_safety"]["successes"]
        ) / len(pairs)
        totals["policy_balanced"]["delta_vs_matching_fixed_gate_pp"] = 100 * (
            totals["policy_balanced"]["successes"] - totals["fixed_balanced"]["successes"]
        ) / len(pairs)
        for method, source in (("always_sac", "always_sac"),
                               ("fixed_safety", "safety"),
                               ("fixed_balanced", "balanced"),
                               ("policy_safety", "safety"),
                               ("policy_balanced", "balanced")):
            # Assert the reusable analyzer agrees with this report's arithmetic.
            expected = fixed_totals[source] if method.startswith("fixed") else (
                policy_totals[source] if method.startswith("policy") else None
            )
            if expected is not None:
                assert totals[method]["successes"] == expected["successes"]
                assert totals[method]["rescues"] == expected["rescues"]
                assert totals[method]["regressions"] == expected["regressions"]
                assert totals[method]["sac_activations"] == expected["sac_episodes"]

        comparisons = {}
        for name, (method, comparator) in COMPARISONS.items():
            comparison = paired_test(outcomes[method], outcomes[comparator])
            comparisons[name] = comparison
            raw_p[name][train_seed] = comparison["mcnemar_p"]
        results[train_seed] = {
            "evaluation_csv": str(eval_path),
            "policy_gate_config": str(gate_path),
            "selected_quadrants": specific["modes"],
            "methods": totals,
            "comparisons": comparisons,
        }
        for method in METHODS:
            mode = {"fixed_safety": "safety", "fixed_balanced": "balanced",
                    "policy_safety": "safety", "policy_balanced": "balanced"}.get(method)
            gate = specific if method.startswith("policy") else fixed
            qstats = quadrant_counts(pairs, outcomes[method], gate, mode)
            for q in QUADRANTS:
                quadrant_rows.append({"sac_training_seed": train_seed,
                                      "method": method, "quadrant": q,
                                      **qstats[q]})
            m = totals[method]
            method_rows.append({"sac_training_seed": train_seed, "method": method,
                                **m})

    adjusted = {family: holm({str(seed): p for seed, p in pvals.items()})
                for family, pvals in raw_p.items()}
    for seed in TRAIN_SEEDS:
        for family, comparison in COMPARISONS.items():
            results[seed]["comparisons"][family]["holm_p_across_five_models"] = adjusted[family][str(seed)]

    # Per-method across-model variability is descriptive (five independent model seeds).
    cross_seed = {}
    for method in METHODS:
        rates = [results[s]["methods"][method]["success_rate"] for s in TRAIN_SEEDS]
        deltas = [results[s]["methods"][method]["delta_vs_bc_pp"] for s in TRAIN_SEEDS]
        cross_seed[method] = {
            "success_rate_mean": sum(rates) / len(rates),
            "success_rate_sd_sample": math.sqrt(sum((x-sum(rates)/len(rates))**2 for x in rates)/(len(rates)-1)),
            "success_rate_min": min(rates), "success_rate_max": max(rates),
            "delta_vs_bc_pp_mean": sum(deltas) / len(deltas),
            "delta_vs_bc_pp_sd_sample": math.sqrt(sum((x-sum(deltas)/len(deltas))**2 for x in deltas)/(len(deltas)-1)),
            "improved_model_count_vs_bc": sum(x > 0 for x in deltas),
            "worsened_model_count_vs_bc": sum(x < 0 for x in deltas),
        }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_name(prefix.name + "_summary.json")).write_text(json.dumps({
        "evaluation_seeds": [190000, 191999],
        "calibration_seed_range": [180000, 180499],
        "episodes_per_sac_model": 2000,
        "sac_training_seeds": list(TRAIN_SEEDS),
        "shared_positions_and_bc_outcomes_verified": True,
        "fixed_gate_config": args.fixed_gate,
        "per_seed": results,
        "cross_seed_descriptive_summary": cross_seed,
        "holm_adjusted_p_values": adjusted,
        "statistical_note": (
            "Exact paired tests condition on each frozen model and position set. "
            "The five model seeds are the unit for cross-seed generalization; shared "
            "positions do not create 10,000 independent observations. Holm adjustment "
            "is within each pre-specified five-test comparison family."
        ),
    }, indent=2), encoding="utf-8")
    for suffix, rows in (("methods.csv", method_rows), ("quadrants.csv", quadrant_rows)):
        path = prefix.with_name(prefix.name + "_" + suffix)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {prefix}_summary.json, {prefix}_methods.csv, {prefix}_quadrants.csv")


if __name__ == "__main__":
    main()
