import unittest

import numpy as np

from franka_bc_residual_env import BCResidualEnv


BC = "bc_models/fair_wide_expert_seed42.pt"


class ResidualPenaltyInterventionTests(unittest.TestCase):
    def make_env(self, coefficient=None):
        kwargs = {
            "bc_model": BC,
            "position_range": 0.20,
            "residual_meters": 0.005,
            "reward_mode": "shaped",
        }
        if coefficient is not None:
            kwargs["residual_penalty_coefficient"] = coefficient
        return BCResidualEnv(**kwargs)

    def test_default_coefficient_preserves_original_penalty(self):
        env = self.make_env()
        try:
            self.assertEqual(env.residual_penalty_coefficient, 0.005)
            self.assertEqual(env.residual_penalty_coef, 0.005)
        finally:
            env.close()

    def test_abbreviated_keyword_remains_backward_compatible(self):
        env = BCResidualEnv(bc_model=BC, residual_penalty_coef=0.020)
        try:
            self.assertEqual(env.residual_penalty_coefficient, 0.020)
        finally:
            env.close()

    def test_only_penalty_term_changes_for_same_state_and_action(self):
        arm_a = self.make_env(0.005)
        arm_b = self.make_env(0.020)
        action = np.array([0.4, -0.2, 0.1], dtype=np.float32)
        try:
            obs_a, _ = arm_a.reset(seed=240123)
            obs_b, _ = arm_b.reset(seed=240123)
            np.testing.assert_array_equal(obs_a, obs_b)
            next_a, reward_a, term_a, trunc_a, info_a = arm_a.step(action)
            next_b, reward_b, term_b, trunc_b, info_b = arm_b.step(action)
            expected = (0.020 - 0.005) * float(np.sum(np.clip(action, -1, 1) ** 2))
            self.assertAlmostEqual(reward_a - reward_b, expected, places=7)
            np.testing.assert_array_equal(next_a, next_b)
            self.assertEqual((term_a, trunc_a), (term_b, trunc_b))
            for key in ("is_success", "lift_height", "distance"):
                self.assertEqual(info_a[key], info_b[key])
            self.assertAlmostEqual(info_a["residual_action_penalty"], -0.005 * float(np.sum(action ** 2)))
            self.assertAlmostEqual(info_b["residual_action_penalty"], -0.020 * float(np.sum(action ** 2)))
            self.assertAlmostEqual(
                info_a["task_reward"] + info_a["shaping_reward"] + info_a["residual_action_penalty"],
                reward_a,
                places=7,
            )
        finally:
            arm_a.close()
            arm_b.close()

    def test_diagnostic_state_snapshot_does_not_advance_rng_and_restores_dynamics(self):
        env_a = self.make_env(0.005)
        env_b = self.make_env(0.005)
        first = np.array([0.2, -0.3, 0.1], dtype=np.float32)
        second = np.array([-0.25, 0.1, 0.05], dtype=np.float32)
        try:
            env_a.reset(seed=240321)
            env_b.reset(seed=999)
            env_a.step(first)
            snapshot = env_a.training_state_dict()
            expected_rng = env_a.env.np_random.bit_generator.state
            self.assertEqual(snapshot["base_rng"], expected_rng)
            expected = env_a.step(second)
            env_b.load_training_state_dict(snapshot)
            actual = env_b.step(second)
            np.testing.assert_array_equal(expected[0], actual[0])
            self.assertAlmostEqual(expected[1], actual[1], places=7)
            self.assertEqual(expected[2:4], actual[2:4])
            self.assertEqual(expected[4]["is_success"], actual[4]["is_success"])
        finally:
            env_a.close()
            env_b.close()


if __name__ == "__main__":
    unittest.main()
