"""Synthetic-only preparation tests; never acquire or open Final observations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import validate_h4_kalman_independent as validation
import run_h4_motion_estimator_stage1 as stage
import analyze_h4_motion_estimator_stage1 as analysis
from action_chunking import sha256


class IndependentValidationTests(unittest.TestCase):
    def protocol(self):return json.loads(validation.PROTOCOL.read_text())

    def test_fixed_sources_criteria_and_disjoint_seeds(self):
        p,ph=validation.load_protocol()
        old=json.loads(Path('bc_results/h4_motion_estimator_protocol.json').read_text())
        self.assertEqual(p['pass_criteria'],old['pass_criteria'])
        self.assertFalse(set(p['seeds'])&set(old['seeds']))
        self.assertNotEqual(p['noise_seed'],old['noise_seed'])
        self.assertEqual(len(validation.expected_keys(p)),120)
        self.assertEqual(p['estimators']['kalman']['sigma_grid_m'],[0,.03,.00025])

    def test_reuse_restores_legacy_globals(self):
        old=(stage.OUT,stage.ESTIMATORS,analysis.ESTIMATORS)
        with self.assertRaises(RuntimeError):
            with validation.reuse(Path('/private/tmp/synthetic-only')):
                self.assertEqual(set(stage.ESTIMATORS),{'position','motion_MAD','kalman'})
                raise RuntimeError('deliberate')
        self.assertEqual((stage.OUT,stage.ESTIMATORS,analysis.ESTIMATORS),old)

    def test_three_estimators_and_exact_transition_without_truth_input(self):
        p=self.protocol();xy=np.zeros((64,2));xyz=np.zeros((64,3))
        with validation.reuse(Path('/private/tmp/synthetic-only')):
            a=stage.evaluate_sequence(xy,xyz,'clean_to_noisy',990000,p,{})
            b=stage.evaluate_sequence(xy,xyz,'noisy',990000,p,{})
            changed_truth=xyz.copy();changed_truth[:,0]=np.arange(64)*.01
            c=stage.evaluate_sequence(xy,changed_truth,'clean_to_noisy',990000,p,{})
        np.testing.assert_array_equal(np.asarray(a['observed_xy'])[:40],0)
        np.testing.assert_array_equal(np.asarray(a['observed_xy'])[40:],np.asarray(b['observed_xy'])[40:])
        self.assertEqual({r['estimator'] for r in a['windows']},set(validation.ESTIMATORS))
        self.assertEqual([r['sigma_hat'] for r in a['windows']],[r['sigma_hat'] for r in c['windows']])
        self.assertTrue(any(r['mixed'] for r in a['windows']))
        self.assertTrue(all(r['target_sigma']==.01 for r in a['windows'] if r['step']>=60))

    def test_gates_include_every_criterion_and_missing_metrics_fail(self):
        p=self.protocol();base={'n':100,'MAE_mm':1.,'bias_mm':0.,'classification_accuracy':1.,'wrong_interval_fraction':0.,'wrong_direction_actual_switches_per_decision':0.}
        metrics={e:{phase:{sigma:copy.deepcopy(base) for sigma in ('0','10')} for phase in ('stationary','moving')} for e in validation.ESTIMATORS}
        delays={e:{profile:{'p90_delay_steps':20.,'miss_fraction':0.} for profile in ('clean_to_noisy','noisy_to_clean')} for e in validation.ESTIMATORS}
        rows=[{'estimator':e,'motion':'moving','mixed':False,'error':value} for e,value in (('position',.002),('kalman',.001))]
        result=validation.gates(metrics,delays,rows,'kalman',p['pass_criteria'])
        self.assertTrue(result['all_passed']);self.assertEqual(len(result['checks']),16)
        metrics['kalman']['moving']['10']['classification_accuracy']=.94
        self.assertEqual(validation.gates(metrics,delays,rows,'kalman',p['pass_criteria'])['checks']['moving_classification_10mm'],'FAIL')
        metrics['kalman']['stationary']['0']['MAE_mm']=None
        self.assertEqual(validation.gates(metrics,delays,rows,'kalman',p['pass_criteria'])['checks']['stationary_MAE_0mm'],'FAIL')

    def test_summary_retains_twenty_seed_units_and_mixed_windows(self):
        p=self.protocol();p['seeds']=list(range(990000,990020));xy=np.zeros((64,2));xyz=np.zeros((64,3))
        with validation.reuse(Path('/private/tmp/synthetic-only')):
            templates=[stage.evaluate_sequence(xy,xyz,profile,990000,p,{}) for profile in p['profiles']]
        # Synthetic test fixtures only: clone to exercise seed aggregation,
        # not independently acquired observations and never written as Final.
        seq=[]
        for seed in p['seeds']:
            for template in templates:
                a=copy.deepcopy(template);a['seed']=seed
                for row in a['windows']:row['seed']=seed
                seq.append(a)
        p['bootstrap_replicates']=10
        result=validation.summarize(p,'synthetic',seq)
        self.assertEqual(set(result['per_seed']),set(map(str,p['seeds'])))
        self.assertEqual(set(result['criteria_by_estimator']),set(validation.ESTIMATORS))
        self.assertGreater(result['mixed_windows']['kalman']['clean_to_noisy']['n'],0)
        self.assertFalse(result['kalman_accepted'])  # No moving-window coverage.
        json.dumps(result,allow_nan=False)

    def test_generate_refuses_existing_output_before_collecting(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(stage,'collect') as collect:
            path=Path(tmp);(path/'diagnostic_sequences.jsonl').write_text('evidence')
            with self.assertRaises(FileExistsError):validation.generate(self.protocol(),'x',path)
            collect.assert_not_called()
            self.assertEqual((path/'diagnostic_sequences.jsonl').read_text(),'evidence')

    def test_json_writer_refuses_overwrite_and_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'summary.json';validation.write_json(path,{'passed':False})
            with self.assertRaises(FileExistsError):validation.write_json(path,{})
            with self.assertRaises(ValueError):validation.write_json(Path(tmp)/'bad.json',{'n':float('nan')})
            self.assertFalse((Path(tmp)/'bad.json').exists())


if __name__=='__main__':unittest.main()
