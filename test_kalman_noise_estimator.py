import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from kalman_noise_estimator import kalman_noise
from compare_h4_noise_estimators import compare


class KalmanNoiseTests(unittest.TestCase):
    def test_static_velocity_and_acceleration(self):
        t=np.arange(20)*.1
        for x in (t*0,t*.02,.005*t*t):
            result=kalman_noise(np.column_stack([x,x*0]))
            self.assertTrue(np.isfinite(result['sigma_m']))
            self.assertLessEqual(result['sigma_m'],.001)
            self.assertGreater(result['mean_projected_state_prediction_variance_m2'],0)

    def test_sensor_scale_over_small_gaussian_sample(self):
        rng=np.random.default_rng(7)
        values=[kalman_noise(rng.normal(0,.01,(20,2)))['sigma_m'] for _ in range(24)]
        self.assertLess(abs(np.mean(values)-.01),.003)

    def test_contact_is_not_claimed_identifiable(self):
        y=np.zeros((20,2));y[10:,0]=.03
        result=kalman_noise(y)
        self.assertGreater(result['sigma_m'],0)
        self.assertIn('Contact',result['interpretation'])
        # Different latent decompositions can produce identical observations.
        self.assertEqual(result['sigma_m'],kalman_noise(np.zeros_like(y)+y)['sigma_m'])

    def test_Q_assumption_changes_noise_attribution(self):
        y=np.column_stack([.01*np.sin(np.arange(20)),np.zeros(20)])
        self.assertNotEqual(kalman_noise(y,acceleration_std=0)['sigma_m'],kalman_noise(y,acceleration_std=5)['sigma_m'])

    def test_input_validation(self):
        for y in (np.zeros((19,2)),np.full((20,2),np.nan)):
            with self.assertRaises(ValueError):kalman_noise(y)
        with self.assertRaises(ValueError):kalman_noise(np.zeros((20,2)),dt=0)

    def test_cli_limit_and_reserved_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.jsonl';r={'seed':350000,'observed_xy':np.zeros((24,2)).tolist()}
            path.write_text((json.dumps(r)+'\n')*3)
            s=compare(path,limit=1);self.assertEqual((s['sequences'],s['windows']),(1,2))
            self.assertEqual(set(s['mean_sigma_m']),{'position','MAD','kalman'})
            with self.assertRaises(ValueError):compare(path,1,'final')


if __name__=='__main__':unittest.main()
