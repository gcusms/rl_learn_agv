import unittest

import torch

from experiments.phase_tail_loss import phase_tail_loss


class PhaseTailLossTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([0, 0, 1, 11, 5, 9])
        self.wait = torch.tensor([0., 3., 0., .2, 2., 5.])
        self.weights = torch.tensor([.25, .25, .125, .125, .125, .125], dtype=torch.float64)

    def test_control_is_exact_original_loss(self):
        losses = torch.tensor([.1, .2, 1., 4., .2, .5], dtype=torch.float64)
        result = phase_tail_loss(losses, self.labels, self.wait, self.weights, focus_share=0.)
        torch.testing.assert_close(result, (losses * self.weights).sum(), rtol=0, atol=0)

    def test_phase_gradient_budgets_and_stop_gradients_preserved(self):
        losses = torch.tensor([.1, .2, 1., 7., .2, .5], dtype=torch.float64, requires_grad=True)
        value = phase_tail_loss(losses, self.labels, self.wait, self.weights)
        value.backward()
        torch.testing.assert_close(losses.grad[:2], self.weights[:2], rtol=0, atol=0)
        torch.testing.assert_close(losses.grad[4:], self.weights[4:], rtol=0, atol=0)
        torch.testing.assert_close(losses.grad[2:4].sum(), torch.tensor(.25, dtype=torch.float64))
        self.assertTrue(bool((losses.grad > 0).all()))
        self.assertGreater(float(losses.grad[3]), float(losses.grad[2]))

    def test_constant_loss_retains_same_mean_and_gradient(self):
        losses = torch.full((6,), 3., dtype=torch.float64, requires_grad=True)
        result = phase_tail_loss(losses, self.labels, self.wait, self.weights)
        torch.testing.assert_close(result, torch.tensor(3., dtype=torch.float64))
        result.backward()
        torch.testing.assert_close(losses.grad, self.weights, rtol=0, atol=1e-15)

    def test_unstable_weights_and_nonfinite_losses_rejected(self):
        losses = torch.ones(6, dtype=torch.float64)
        with self.assertRaises(ValueError):
            phase_tail_loss(losses, self.labels, self.wait, self.weights * .9)
        losses[3] = float('nan')
        with self.assertRaises(ValueError):
            phase_tail_loss(losses, self.labels, self.wait, self.weights)


if __name__ == '__main__':
    unittest.main()
