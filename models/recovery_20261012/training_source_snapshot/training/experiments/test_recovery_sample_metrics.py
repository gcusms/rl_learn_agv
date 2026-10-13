import unittest
from experiments.recovery_sample_metrics import RecoverySampleMetrics


class RecoverySampleMetricsTests(unittest.TestCase):
    def state(self):
        obs = [0.] * 300
        obs[235] = 1.
        obs[250] = 1.  # Truthful 'none': voluntary idle, not fabricated rejection.
        obs[266] = .04
        return obs

    def test_voluntary_wait_and_episode_boundaries(self):
        metrics = RecoverySampleMetrics()
        mask = [True] * 12
        probabilities = [1/12] * 12
        metrics.record(self.state(), mask, 0, probabilities)
        metrics.end_episode()
        metrics.record(self.state(), mask, 0, probabilities)
        metrics.record(self.state(), mask, 5, probabilities)
        result = metrics.summary()
        self.assertEqual(result['recovery_counts']['decisions'], 3)
        self.assertEqual(result['recovery_counts']['wait'], 2)
        self.assertEqual(result['recovery_counts']['pure_turn'], 1)
        self.assertEqual(result['longest_recovery_wait_run'], 1)
        self.assertEqual(result['previous_reason_index_counts'], {'0': 3})

    def test_goal_and_no_alternatives_are_not_recovery_samples(self):
        metrics = RecoverySampleMetrics()
        obs = self.state()
        obs[250] = 0.; obs[264] = 1.
        metrics.record(obs, [True] * 12, 0, [1/12] * 12)
        obs = self.state(); obs[235] = 0.
        metrics.record(obs, [True] + [False] * 11, 0, [1.] + [0.] * 11)
        self.assertEqual(metrics.summary()['total_decisions'], 2)
        self.assertEqual(metrics.summary()['recovery_counts'].get('decisions', 0), 0)

    def test_illegal_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            RecoverySampleMetrics().record(self.state(), [True] + [False] * 11,
                                           5, [1.] + [0.] * 11)


if __name__ == '__main__':
    unittest.main()
