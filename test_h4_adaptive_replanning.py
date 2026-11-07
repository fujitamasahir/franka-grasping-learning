import unittest
import numpy as np

from evaluate_h4_adaptive_replanning import estimate_sigma


class AdaptiveEstimatorTests(unittest.TestCase):
    def test_sigma_hat_uses_sample_variance_of_only_noisy_xy_window(self):
        # A symmetric 20-point sequence with unit sample variance on both axes.
        x=np.arange(20,dtype=np.float64); x=(x-x.mean())/x.std(ddof=1)
        obs=np.column_stack([x, -x])
        self.assertAlmostEqual(estimate_sigma(obs),1.0,places=12)

    def test_sigma_hat_detects_independent_xy_scale_without_sigma_input(self):
        rng=np.random.default_rng(14)
        obs=rng.normal(0,.005,size=(20,2))
        self.assertGreater(estimate_sigma(obs),.002)
        self.assertLess(estimate_sigma(obs),.008)

    def test_estimator_rejects_wrong_window_or_nonfinite_observations(self):
        with self.assertRaises(ValueError): estimate_sigma(np.zeros((19,2)))
        bad=np.zeros((20,2));bad[3,1]=np.nan
        with self.assertRaises(ValueError): estimate_sigma(bad)


if __name__=='__main__': unittest.main()
