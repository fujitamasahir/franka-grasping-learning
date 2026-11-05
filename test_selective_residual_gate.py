"""Pure-Python unit tests; run: python3 -m unittest test_selective_residual_gate.py"""

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_selective_residual_gate import (
    exact_mcnemar_p, fit_gate, gate_uses_sac, load_pairs,
    outside_demonstrations, summarize,
)


class SelectiveGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "calibration.csv"
        self.path.write_text("dummy calibration bytes", encoding="utf-8")

    def make_calibration(self):
        positions = {
            "EN": (0.73, 0.18),
            "ES": (0.73, -0.18),
            "WN": (0.37, 0.18),
            "WS": (0.37, -0.18),
        }
        outcomes = {
            "EN": (6, 0), "ES": (1, 4),
            "WN": (0, 0), "WS": (4, 1),
        }
        rows = {}
        seed = 100
        for q, (x, y) in positions.items():
            rescue, regress = outcomes[q]
            for i in range(30):
                bc, sac = (1, 1)
                if i < rescue:
                    bc, sac = (0, 1)
                elif i < rescue + regress:
                    bc, sac = (1, 0)
                rows[seed] = {
                    "seed": seed, "cube_x": x, "cube_y": y,
                    "bc": bc, "sac": sac,
                }
                seed += 1
        return rows

    def test_fit_gate(self):
        calibration = self.make_calibration()
        gate = fit_gate(calibration, self.path)
        self.assertEqual(gate["modes"]["safety"], ["EN"])
        self.assertEqual(gate["modes"]["balanced"], ["EN", "WS"])
        self.assertFalse(gate_uses_sac(gate, "safety", 0.60, 0.05))
        self.assertTrue(gate_uses_sac(gate, "safety", 0.73, 0.18))
        self.assertFalse(gate_uses_sac(gate, "safety", 0.73, -0.18))
        self.assertTrue(gate_uses_sac(gate, "balanced", 0.37, -0.18))

    def test_paired_counterfactual(self):
        calibration = self.make_calibration()
        gate = fit_gate(calibration, self.path)
        summary, details = summarize(calibration, gate)
        self.assertEqual(len(details), 120)
        self.assertEqual(summary["safety"]["rescues"], 6)
        self.assertEqual(summary["safety"]["regressions"], 0)
        self.assertEqual(summary["balanced"]["rescues"], 10)
        self.assertEqual(summary["balanced"]["regressions"], 1)
        self.assertEqual(summary["safety"]["sac_episodes"], 30)
        self.assertEqual(summary["balanced"]["sac_episodes"], 60)

    def test_square_boundary(self):
        self.assertFalse(outside_demonstrations(0.55 + 0.12, 0.0))
        self.assertTrue(outside_demonstrations(0.55 + 0.121, 0.0))
        self.assertTrue(outside_demonstrations(0.55, -0.121))

    def test_mcnemar(self):
        self.assertAlmostEqual(exact_mcnemar_p(22, 4), 0.00053352, places=6)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)

    def test_reject_duplicate_seed_policy(self):
        path = Path(self.temp.name) / "bad.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=["seed", "cube_x", "cube_y", "success", "policy"]
            )
            writer.writeheader()
            row = {
                "seed": 1, "cube_x": 0.5, "cube_y": 0.1,
                "success": 1, "policy": "bc_only",
            }
            writer.writerow(row)
            writer.writerow(row)
        with self.assertRaises(ValueError):
            load_pairs(path)


if __name__ == "__main__":
    unittest.main()
