import inspect
import unittest
import numpy as np
from motion_aware_noise_estimator import position_variance,second_difference_mad


class MotionEstimatorTests(unittest.TestCase):
    def test_only_observations_input(self):
        for fn in (position_variance,second_difference_mad):
            self.assertEqual(tuple(inspect.signature(fn).parameters),('noisy_xy',))
            for bad in (np.zeros((19,2)),np.full((20,2),np.nan)):
                with self.assertRaises(ValueError):fn(bad)

    def test_replay_zero_to_noisy_transition_preserves_float_sigma(self):
        from run_h4_motion_estimator_stage1 import evaluate_sequence
        xy=np.zeros((100,2));xyz=np.zeros((100,3))
        p={'profiles':{'change':[0,.01],'always':[.01,.01]},'noise_seed':20261108}
        change=evaluate_sequence(xy,xyz,'change',350000,p,{})
        always=evaluate_sequence(xy,xyz,'always',350000,p,{})
        a=np.asarray(change['observed_xy']);b=np.asarray(always['observed_xy'])
        np.testing.assert_array_equal(a[:40],0)
        np.testing.assert_array_equal(a[40:],b[40:])
        self.assertGreater(np.abs(a[40:]).max(),.001)
        for e in change['windows']:
            if e['step']>=60:self.assertAlmostEqual(e['target_sigma'],.01)

    def test_affine_motion_cancelled(self):
        x=np.arange(20,dtype=float);xy=np.column_stack([x*.01,x*-.02])
        self.assertLess(second_difference_mad(xy),1e-15)
        self.assertGreater(position_variance(xy),.01)

    def test_translation_invariance_and_scale(self):
        x=np.random.default_rng(5).normal(size=(20,2))
        for fn in (position_variance,second_difference_mad):
            self.assertAlmostEqual(fn(x+12),fn(x),places=12)
            self.assertAlmostEqual(fn(x*3),fn(x)*3,places=12)

    def test_iid_gaussian_scale_in_aggregate(self):
        rng=np.random.default_rng(9);est=[second_difference_mad(rng.normal(0,.01,(20,2))) for _ in range(4000)]
        self.assertLess(abs(np.mean(est)-.01),.001)

    def test_acceleration_not_identifiable_from_observation_only(self):
        x=np.arange(20);xy=np.column_stack([.01*np.sin(x*np.pi/2),.01*np.cos(x*np.pi/2)])
        self.assertGreater(second_difference_mad(xy),.003)
        # Exactly the same samples admit moving/no-noise and static/noisy decompositions.
        np.testing.assert_array_equal(xy,np.zeros_like(xy)+xy)
        self.assertEqual(second_difference_mad(xy),second_difference_mad(np.zeros_like(xy)+xy))


if __name__=='__main__':unittest.main()
