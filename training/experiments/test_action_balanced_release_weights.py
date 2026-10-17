import unittest

import numpy as np

from experiments.action_balanced_release_weights import MODE, fit_weights
from experiments.release_fit_weights import fit_weights as original_weights


class WeightsTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 1, 1, 1, 4, 11, 1, 6, 6], np.int64)
        self.features = np.arange(40, dtype=np.float32).reshape(10, 4)
        self.waiting = np.array([0, 3, 0, 0, 0, 0, 0, 2, 3, 4.])

    def weights(self, **kwargs):
        return fit_weights(self.features, self.labels, self.waiting, mode=MODE, **kwargs)

    def test_phase_and_stop_budgets_preserved(self):
        weights, report = self.weights()
        self.assertAlmostEqual(weights[self.labels == 0].sum(), .5)
        self.assertAlmostEqual(weights[2:7].sum(), .25)
        self.assertAlmostEqual(weights[7:].sum(), .25)
        self.assertTrue((weights > 0).all())
        self.assertTrue(report['no_action_override'])

    def test_rare_ordinary_turn_has_equal_class_budget(self):
        weights, _ = self.weights()
        for indices in [[2, 3, 4], [5], [6]]:
            self.assertAlmostEqual(weights[indices].sum(), .25 / 3)
        self.assertAlmostEqual(weights[7], .125)
        self.assertAlmostEqual(weights[8:10].sum(), .125)

    def test_duplicate_inputs_share_class_weight(self):
        self.features[3] = self.features[2]
        weights, _ = self.weights()
        self.assertAlmostEqual(weights[2], weights[3])
        self.assertAlmostEqual(weights[2] + weights[3], weights[4])

    def test_stop_examples_are_numerically_unchanged(self):
        self.features[1] = self.features[0]
        weights, _ = self.weights()
        old, _ = original_weights(self.features, self.labels, self.waiting, mode='recovery_phase')
        np.testing.assert_array_equal(weights[:2], old[:2])

    def test_existing_modes_remain_exact(self):
        for mode in ['stop_move', 'recovery_phase']:
            old, report = original_weights(self.features, self.labels, self.waiting, mode=mode)
            actual, actual_report = fit_weights(self.features, self.labels, self.waiting, mode=mode)
            np.testing.assert_array_equal(actual, old)
            self.assertEqual(actual_report, report)

    def test_identical_input_conflicts_still_rejected(self):
        self.features[5] = self.features[2]
        with self.assertRaises(ValueError):
            self.weights()


if __name__ == '__main__':
    unittest.main()
