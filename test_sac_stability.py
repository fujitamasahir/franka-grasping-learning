import unittest

from analyze_sac_stability import exact_mcnemar, holm_adjust, select_checkpoint


class SacStabilityAnalysisTests(unittest.TestCase):
    def test_exact_mcnemar_has_no_discordant_difference(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(1, 1), 1.0)

    def test_exact_mcnemar_two_sided_binomial(self):
        self.assertEqual(exact_mcnemar(5, 0), 0.0625)
        self.assertAlmostEqual(exact_mcnemar(6, 0), 0.03125)

    def test_holm_adjustment_is_monotone_in_rank(self):
        values = holm_adjust([("a", 0.01), ("b", 0.04), ("c", 0.03)])
        self.assertAlmostEqual(values["a"], 0.03)
        self.assertAlmostEqual(values["c"], 0.06)
        self.assertAlmostEqual(values["b"], 0.06)

    def test_checkpoint_selection_uses_success_then_steps_then_earliest(self):
        rows = [
            {"timesteps": 10, "rate": 0.9, "mean_steps": 80},
            {"timesteps": 20, "rate": 0.95, "mean_steps": 90},
            {"timesteps": 30, "rate": 0.95, "mean_steps": 85},
        ]
        selected = select_checkpoint(rows, "rate", "mean_steps")
        self.assertEqual(selected["timesteps"], 30)

        rows[2]["mean_steps"] = 90
        selected = select_checkpoint(rows, "rate", "mean_steps")
        self.assertEqual(selected["timesteps"], 20)


if __name__ == "__main__":
    unittest.main()
