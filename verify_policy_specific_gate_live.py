"""Execute fixed and policy-specific gates in MuJoCo on calibration cases."""

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import SAC

from analyze_selective_residual_gate import (
    gate_uses_sac,
    load_pairs,
    outside_demonstrations,
    quadrant,
)
from evaluate_selective_residual_sac import run_episode
from franka_bc_residual_env import BCResidualEnv


def select_cases(pairs, gate):
    cases = {}
    for mode in ("safety", "balanced"):
        # Exercise each SAC-selected quadrant and a BC fallback inside and outside.
        for q in gate["modes"][mode]:
            match = next((s for s, r in sorted(pairs.items())
                          if outside_demonstrations(r["cube_x"], r["cube_y"],
                                                    gate["center_x"], gate["center_y"], gate["inner_range"])
                          and quadrant(r["cube_x"], r["cube_y"], gate["center_x"], gate["center_y"]) == q), None)
            if match is not None:
                cases[(mode, match)] = f"selected_{q}"
        inside = next((s for s, r in sorted(pairs.items())
                       if not outside_demonstrations(r["cube_x"], r["cube_y"],
                                                     gate["center_x"], gate["center_y"], gate["inner_range"])), None)
        if inside is not None:
            cases[(mode, inside)] = "inside_demo_bc_fallback"
        outside_unselected = next((s for s, r in sorted(pairs.items())
                                   if outside_demonstrations(r["cube_x"], r["cube_y"],
                                                             gate["center_x"], gate["center_y"], gate["inner_range"])
                                   and quadrant(r["cube_x"], r["cube_y"], gate["center_x"], gate["center_y"])
                                   not in gate["modes"][mode]), None)
        if outside_unselected is not None:
            cases[(mode, outside_unselected)] = "outer_unselected_bc_fallback"
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sac-training-seed", required=True, type=int)
    parser.add_argument("--sac-model", required=True)
    parser.add_argument("--bc-model", default="bc_models/fair_wide_expert_seed42.pt")
    parser.add_argument("--calibration", required=True,
                        help="Paired MuJoCo CSV used as the live verification reference")
    parser.add_argument("--fixed-gate", required=True)
    parser.add_argument("--specific-gate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pairs = load_pairs(args.calibration)
    fixed = json.loads(Path(args.fixed_gate).read_text(encoding="utf-8"))
    specific = json.loads(Path(args.specific_gate).read_text(encoding="utf-8"))
    if specific.get("sac_training_seed") != args.sac_training_seed:
        raise ValueError("Specific gate SAC seed does not match requested model")
    cases = {}
    for name, gate in (("fixed", fixed), ("policy_specific", specific)):
        for (mode, seed), label in select_cases(pairs, gate).items():
            cases[(name, mode, seed)] = label

    model = SAC.load(args.sac_model, device="cpu")
    env = BCResidualEnv(bc_model=args.bc_model, position_range=0.20,
                       residual_meters=0.005)
    rows = []
    try:
        for (gate_name, mode, seed), case_label in sorted(cases.items()):
            gate = fixed if gate_name == "fixed" else specific
            result = run_episode(env, model, gate, mode, seed)
            reference = pairs[seed]
            if abs(result["cube_x"] - reference["cube_x"]) > 1e-9 or abs(result["cube_y"] - reference["cube_y"]) > 1e-9:
                raise AssertionError(f"Initial position mismatch for seed {seed}")
            expected_use = gate_uses_sac(gate, mode, reference["cube_x"], reference["cube_y"])
            expected_success = reference["sac"] if expected_use else reference["bc"]
            if result["uses_sac"] != int(expected_use) or result["success"] != expected_success:
                raise AssertionError(f"Live rollout disagrees with paired result: {gate_name}/{mode}/{seed}")
            rows.append({"sac_training_seed": args.sac_training_seed,
                         "gate": gate_name, "mode": mode, "case": case_label,
                         **result})
            print(f"{gate_name}/{mode} seed={seed} case={case_label} "
                  f"uses_sac={result['uses_sac']} success={result['success']}")
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} live gate decisions; saved {output}")


if __name__ == "__main__":
    main()
