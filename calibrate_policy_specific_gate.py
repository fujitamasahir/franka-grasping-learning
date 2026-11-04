"""Fit one frozen spatial gate for one SAC checkpoint from paired calibration data."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_selective_residual_gate import fit_gate, load_pairs


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate(calibration_path, sac_training_seed, sac_checkpoint):
    pairs = load_pairs(calibration_path)
    gate = fit_gate(pairs, calibration_path)
    gate.update({
        "calibration_episodes": len(pairs),
        "sac_training_seed": int(sac_training_seed),
        "sac_checkpoint": str(sac_checkpoint),
        "sac_checkpoint_sha256": sha256(sac_checkpoint),
        "selection_method": "policy_specific_calibration",
    })
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--sac-training-seed", required=True, type=int)
    parser.add_argument("--sac-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not Path(args.sac_checkpoint).is_file():
        parser.error(f"SAC checkpoint not found: {args.sac_checkpoint}")
    gate = calibrate(args.calibration, args.sac_training_seed, args.sac_checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"SAC seed {args.sac_training_seed}: {len(gate['calibration_seeds'])} calibration episodes")
    print(f"Selected quadrants: {gate['modes']}")
    print(f"Calibration SHA-256: {gate['calibration_sha256']}")
    print(f"Checkpoint SHA-256: {gate['sac_checkpoint_sha256']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
