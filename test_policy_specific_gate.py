"""Unit tests for per-checkpoint calibration and paired policy comparisons."""

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_policy_specific_gate import holm, paired_test
from calibrate_policy_specific_gate import calibrate


class PolicySpecificGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.model = root / "model.zip"
        self.model.write_bytes(b"frozen checkpoint")
        self.calibration = root / "calibration.csv"
        self._write_calibration()

    def _write_calibration(self):
        positions = {
            "EN": (0.73, 0.18, 5, 0),
            "ES": (0.73, -0.18, 3, 0),
            "WN": (0.37, 0.18, 0, 0),
            "WS": (0.37, -0.18, 4, 1),
        }
        with self.calibration.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "cube_x", "cube_y", "success", "policy"])
            writer.writeheader()
            seed = 180000
            for x_name, (x, y, rescues, regressions) in positions.items():
                del x_name
                for i in range(30):
                    bc, sac = (1, 1)
                    if i < rescues:
                        bc, sac = 0, 1
                    elif i < rescues + regressions:
                        bc, sac = 1, 0
                    for policy, success in (("bc_only", bc), ("bc_plus_sac", sac)):
                        writer.writerow({"seed": seed, "cube_x": x, "cube_y": y,
                                         "success": success, "policy": policy})
                    seed += 1

    def test_policy_specific_config_records_provenance_and_thresholds(self):
        gate = calibrate(self.calibration, 19, self.model)
        self.assertEqual(gate["modes"], {"safety": ["EN"], "balanced": ["EN", "ES", "WS"]})
        self.assertEqual(gate["sac_training_seed"], 19)
        self.assertEqual(gate["calibration_episodes"], 120)
        self.assertEqual(gate["calibration_seeds"][0], 180000)
        self.assertEqual(gate["calibration_seeds"][-1], 180119)
        self.assertEqual(len(gate["calibration_sha256"]), 64)
        self.assertEqual(len(gate["sac_checkpoint_sha256"]), 64)

    def test_empty_eligible_gate_falls_back_to_bc(self):
        path = Path(self.temp.name) / "empty.csv"
        path.write_text(self.calibration.read_text().replace(",5,0", ",0,0"), encoding="utf-8")
        # Build a deliberately low-rescue calibration directly with valid paired rows.
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "cube_x", "cube_y", "success", "policy"])
            writer.writeheader()
            for seed in range(180000, 180120):
                q = (seed - 180000) // 30
                x, y = ((0.73, 0.18), (0.73, -0.18), (0.37, 0.18), (0.37, -0.18))[q]
                for policy in ("bc_only", "bc_plus_sac"):
                    writer.writerow({"seed": seed, "cube_x": x, "cube_y": y,
                                     "success": 1, "policy": policy})
        gate = calibrate(path, 7, self.model)
        self.assertEqual(gate["modes"], {"safety": [], "balanced": []})

    def test_paired_comparison_and_holm(self):
        test = paired_test({1: 1, 2: 1, 3: 0, 4: 1},
                           {1: 0, 2: 1, 3: 1, 4: 1})
        self.assertEqual((test["wins"], test["losses"], test["net"]), (1, 1, 0))
        self.assertEqual(test["mcnemar_p"], 1.0)
        adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertEqual(adjusted, {"a": 0.03, "c": 0.06, "b": 0.06})


if __name__ == "__main__":
    unittest.main()
