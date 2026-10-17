"""Check migration and permission invariants for explicit guard explanations."""
import unittest

import torch

from experiments.context_release_actor import ContextReleaseActor
from experiments.guard_reason_release_actor import GuardReasonReleaseActor


class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return torch.zeros_like(x[:, :2]) + self.bias


class GuardReasonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(231016)
        self.previous = ContextReleaseActor(Parent()).eval()
        self.actor = GuardReasonReleaseActor(Parent()).eval()
        self.actor.initialize_release_head(self.previous)
        self.x = torch.zeros(6, 625)
        self.x[:, 144:216] = .4 / 12.
        self.x[:, 218] = .5
        self.x[:, 235:244] = 1.
        self.x[:, 267:269] = 1.
        self.x[:, 435] = self.x[:, 438] = 1.
        self.x[:, 303:399] = torch.rand(6, 96)

    def test_migration_preserves_logits_and_decisions(self):
        with torch.no_grad():
            torch.testing.assert_close(self.actor.logits(self.x), self.previous.logits(self.x), rtol=0, atol=0)
            torch.testing.assert_close(self.actor(self.x), self.previous(self.x), rtol=0, atol=0)

    def test_current_reasons_can_be_learned_separately_from_previous_reason(self):
        other = self.x.clone()
        other[:, 303:399] = 0.
        with torch.no_grad():
            old, _ = self.previous.features(self.x)
            unchanged, _ = self.previous.features(other)
            changed, _ = self.actor.features(self.x)
            blank, _ = self.actor.features(other)
        torch.testing.assert_close(old, unchanged, rtol=0, atol=0)
        self.assertFalse(torch.equal(changed, blank))
        self.assertTrue(torch.equal(self.x[:, 399:419], other[:, 399:419]))

    def test_learning_cannot_allow_a_forbidden_action(self):
        self.x[:, 235:244] = self.x[:, 267:269] = 0.
        self.x[:, 243] = 1.  # Only stop and action 9 remain legal.
        with torch.no_grad():
            self.actor.head[-1].weight.zero_()
            self.actor.head[-1].bias.zero_()
            self.actor.head[-1].bias[1] = 1e4  # Forbidden forward action.
            result = self.actor(self.x)
        allowed = self.actor.actions[[0, 9]]
        self.assertTrue(((result[:, None] == allowed[None]).all(2).any(1)).all())
        torch.testing.assert_close(self.actor.masks(self.x), self.previous.masks(self.x))

    def test_invalid_stop_only_and_docking_keep_parent(self):
        cases = []
        for index in [435, 438]:
            x = self.x.clone(); x[:, index] = 0.; cases.append(x)
        x = self.x.clone(); x[:, 235:244] = x[:, 267:269] = 0.; cases.append(x)
        x = self.x.clone(); x[:, 218:220] = 0.; cases.append(x)
        with torch.no_grad():
            self.actor.head[-1].bias[1] = 1e4
            for x in cases:
                self.assertFalse(self.actor.support(x).any())
                torch.testing.assert_close(self.actor(x), self.actor.parent(x[:, :439]), rtol=0, atol=0)

    def test_training_cannot_update_frozen_parent(self):
        self.assertFalse(any(p.requires_grad for p in self.actor.parent.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.actor.head.parameters()))


if __name__ == '__main__':
    unittest.main()
