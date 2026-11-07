"""Check pooled diagnostics without opening real Final outcomes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_h4_adaptive_replanning as analysis


class AdaptiveAnalysisTests(unittest.TestCase):
    def test_noise_diagnostics_include_each_policy_and_early_termination(self):
        cells = {}
        for mi, model in enumerate(analysis.MODELS):
            for controller in analysis.CONTROLLERS:
                for noise in analysis.NOISE:
                    rows = {}
                    for seed in analysis.SEEDS:
                        # A single early termination in each model/noise cell.
                        decided = seed != analysis.SEEDS.start
                        interval = 1 if mi == 2 else 4
                        rows[seed] = {
                            'success': '1', 'bc_inference_calls': '10',
                            'bc_total_inference_ms': '1', 'episode_wall_seconds': '1',
                            'steps': '40', 'decision_made': str(int(decided)),
                            'estimated_sigma_m': str((mi + 1) * .001) if decided else '',
                            'chosen_interval': str(interval) if decided else '', 'region': 'EN',
                        }
                    cells[model, controller, noise] = rows
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'final_integrity_audit.json').write_text('{}')
            with patch.object(analysis, 'OUT', out), patch.object(analysis, 'B', 10), \
                 patch.object(analysis, 'read_all', return_value=(cells, {
                     'calibration_selected_fixed_baseline': 'R4'}, {})), \
                 patch.object(analysis, 'figures'), contextlib.redirect_stdout(io.StringIO()):
                analysis.main()
            summary = json.loads((out / 'final_summary.json').read_text())
        for noise in analysis.NOISE:
            estimate = summary['estimated_sigma_by_true_noise_mm'][str(noise)]
            switch = summary['switching_by_true_noise_mm'][str(noise)]
            conditional = summary['conditional_success_by_chosen_interval'][str(noise)]
            self.assertEqual(estimate['n_estimates'], 2997)
            self.assertEqual(estimate['episodes_without_decision'], 3)
            self.assertAlmostEqual(estimate['mean_m'], .002)
            self.assertEqual(switch['n_decisions'], 2997)
            self.assertEqual((switch['R1'], switch['R4']), (999, 1998))
            self.assertEqual((conditional['1']['n'], conditional['4']['n']), (999, 1998))


if __name__ == '__main__':
    unittest.main()
