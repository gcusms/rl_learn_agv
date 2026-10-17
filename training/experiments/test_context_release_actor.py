"""Contract checks for sensor-context support and unmodified action permission."""
import unittest

import torch

from experiments.context_release_actor import ContextReleaseActor


class Parent(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x[:, :2])


class SupportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.actor = ContextReleaseActor(Parent()).eval()
        self.x = torch.zeros(1, 625)
        self.x[:, 144:216] = 1.
        self.x[:, 218] = .5
        self.x[:, 235:244] = 1.
        self.x[:, 267:269] = 1.
        self.x[:, 435] = self.x[:, 438] = 1.

    def test_clear_motion_keeps_parent(self):
        self.assertFalse(self.actor.support(self.x).item())

    def test_near_obstacle_keeps_head_available_while_moving(self):
        self.x[:, 144] = .5 / 12.
        self.assertTrue(self.actor.support(self.x).item())
        self.assertEqual(int(self.actor.masks(self.x).sum()), 12)

    def test_restricted_action_context_does_not_unmask_it(self):
        self.x[:, 235] = 0.
        self.assertTrue(self.actor.support(self.x).item())
        self.assertFalse(self.actor.masks(self.x)[0, 1].item())

    def test_valid_forecast_and_previous_wait_remain_contexts(self):
        for index in [419, 433, 434, 451]:
            x = self.x.clone(); x[:, index] = 1.
            self.assertTrue(self.actor.support(x).item(), index)

    def test_invalid_scan_or_forecast_and_near_goal_keep_parent(self):
        self.x[:, 144] = .4 / 12.
        for index in [435, 438]:
            x = self.x.clone(); x[:, index] = 0.
            self.assertFalse(self.actor.support(x).item())
        self.x[:, 218:220] = 0.
        self.assertFalse(self.actor.support(self.x).item())

    def test_stop_only_keeps_parent_and_stop_legal(self):
        self.x[:, 144] = .4 / 12.
        self.x[:, 235:244] = self.x[:, 267:269] = 0.
        self.assertFalse(self.actor.support(self.x).item())
        self.assertEqual(int(self.actor.masks(self.x).sum()), 1)


if __name__ == '__main__':
    unittest.main()
