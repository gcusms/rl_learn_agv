import unittest

import numpy as np
import torch

from experiments.release_fit_weights import fit_weights


class ReleaseFitWeightTests(unittest.TestCase):
    def test_legacy_loss_and_gradients_match_original(self):
        features = np.arange(12, dtype=np.float32).reshape(6, 2)
        labels = np.array([0, 0, 1, 6, 6, 11])
        weights, report = fit_weights(features, labels, np.array([0, 1, 0, 0, 2, 5]))
        logits = torch.arange(72, dtype=torch.float64).reshape(6, 12).sin().requires_grad_()
        losses = torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels), reduction='none')
        old = .5 * losses[:2].mean() + .5 * losses[2:].mean()
        new = (losses * torch.from_numpy(weights)).sum()
        torch.testing.assert_close(new, old)
        torch.testing.assert_close(torch.autograd.grad(new, logits, retain_graph=True)[0],
            torch.autograd.grad(old, logits)[0])
        self.assertEqual(report['mode'], 'stop_move')

    def test_recovery_budget_preserves_stop_and_shares_duplicate_weight(self):
        # Three copies of one ordinary example must not outweigh its peer.
        features = np.array([[0, 0], [1, 1], [2, 2], [2, 2], [2, 2], [3, 3], [4, 4]], np.float32)
        labels = np.array([0, 0, 1, 1, 1, 6, 11])
        wait = np.array([0, 3, 0, 0, 0, 0, 5])
        weights, report = fit_weights(features, labels, wait, mode='recovery_phase')
        self.assertAlmostEqual(weights[:2].sum(), .5)
        self.assertAlmostEqual(weights[2:6].sum(), .25)
        self.assertAlmostEqual(weights[6], .25)
        self.assertAlmostEqual(weights[2:5].sum(), weights[5])
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertTrue((weights > 0).all())
        self.assertEqual(report['phases'][1]['unique_head_inputs'], 2)

    def test_input_reordering_does_not_change_example_weight(self):
        features = np.arange(16, dtype=np.float32).reshape(8, 2)
        labels = np.array([0, 0, 1, 1, 6, 6, 11, 11])
        wait = np.array([0, 5, 0, 0, 0, 2, 3, 5])
        order = np.array([6, 3, 1, 0, 5, 7, 2, 4])
        expected, _ = fit_weights(features, labels, wait, mode='recovery_phase')
        actual, _ = fit_weights(features[order], labels[order], wait[order], mode='recovery_phase')
        np.testing.assert_array_equal(actual, expected[order])

    def test_missing_recovery_or_conflicting_labels_are_rejected(self):
        features = np.arange(8, dtype=np.float32).reshape(4, 2)
        with self.assertRaisesRegex(ValueError, 'collect data first'):
            fit_weights(features, np.array([0, 1, 1, 1]), np.zeros(4), mode='recovery_phase')
        features[1] = features[0]
        with self.assertRaisesRegex(ValueError, 'Conflicting action'):
            fit_weights(features, np.array([0, 1, 6, 11]), np.array([0, 0, 0, 2]), mode='recovery_phase')

    def test_fractional_actions_and_invalid_time_are_rejected(self):
        features = np.array([[0, 0], [1, 1], [2, 2]], np.float32)
        with self.assertRaises(ValueError):
            fit_weights(features, np.array([0., 1., 6.]), np.array([0, 0, 5]))
        with self.assertRaises(ValueError):
            fit_weights(features, np.array([0, 1, 6]), np.array([0, -1, 5]))


if __name__ == '__main__':
    unittest.main()
