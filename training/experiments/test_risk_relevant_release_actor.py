"""Recovery triggers reflect contact/wait context, never track validity alone."""
import unittest

import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor
from experiments.risk_relevant_release_actor import RiskRelevantReleaseActor
from experiments.test_guard_reason_release_actor import Parent


class RiskRelevantTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.actor = RiskRelevantReleaseActor(Parent()).eval()
        self.old = GuardReasonReleaseActor(Parent()).eval()
        self.old.load_state_dict(self.actor.state_dict())
        self.x = torch.zeros(1, 625)
        self.x[:, 144:216] = 1.7 / 12.
        self.x[:, 218] = .5
        self.x[:, 235:244] = self.x[:, 267:269] = 1.
        self.x[:, 435] = self.x[:, 438] = 1.

    def test_valid_zero_risk_history_does_not_override_parent(self):
        self.x[:, 465] = 1.  # The observed historical valid frame, all scores zero.
        self.assertTrue(self.old.support(self.x).item())
        self.assertFalse(self.actor.support(self.x).item())
        with torch.no_grad():
            self.actor.head[-1].bias[11] = 100.
            torch.testing.assert_close(self.actor(self.x), self.actor.parent(self.x[:, :439]), rtol=0, atol=0)

    def test_supported_nonzero_historical_risk_keeps_anticipation(self):
        self.x[:, 453] = .4
        self.assertFalse(self.actor.support(self.x).item())
        self.x[:, 465] = 1.
        self.assertTrue(self.actor.support(self.x).item())

    def test_actual_wait_near_geometry_or_restricted_mask_still_supports_recovery(self):
        for index, value in [(433, .1), (434, .1), (144, .5 / 12.), (235, 0.), (419, .1)]:
            x = self.x.clone(); x[:, index] = value
            self.assertTrue(self.actor.support(x).item(), index)

    def test_invalid_and_terminal_keep_parent_even_with_risk(self):
        self.x[:, 453] = .4; self.x[:, 465] = 1.
        for index in [435, 438]:
            x = self.x.clone(); x[:, index] = 0.
            self.assertFalse(self.actor.support(x).item())
        x = self.x.clone(); x[:, 218:220] = 0.
        self.assertFalse(self.actor.support(x).item())

    def test_stop_only_and_permissions_are_unchanged(self):
        self.x[:, 235:244] = self.x[:, 267:269] = 0.
        self.assertFalse(self.actor.support(self.x).item())
        torch.testing.assert_close(self.actor.masks(self.x), self.old.masks(self.x))
        self.assertEqual(self.actor.masks(self.x).sum().item(), 1)


if __name__ == '__main__':
    unittest.main()
