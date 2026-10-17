"""Meaningful boundaries: unknown returns, frozen behavior and original masks."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.long_risk_actor import LongRiskActor
from experiments.release_actor import ReleaseActor
from neural_navigation.release_observation import augment, scan_quality, to_training


ROOT = Path(__file__).resolve().parents[2]


class ReleaseContextTests(unittest.TestCase):
    def test_unknown_and_no_return_cannot_be_confused(self):
        lidar = SimpleNamespace(beams=72, range_min=.1, range_max=12.)
        values = np.full(72, np.nan)
        values[:7] = [1., np.inf, 12., -np.inf, .05, 13., .1]
        quality = scan_quality(values, lidar).reshape(2, 72)
        np.testing.assert_array_equal(quality[:, :7],
            [[1., 0., 0., 0., 0., 0., 1.], [0., 1., 1., 0., 0., 0., 0.]])
        np.testing.assert_array_equal(quality[:, 7:], 0.)
        observation = np.zeros(481, np.float32)
        x = augment(observation, values, lidar)
        np.testing.assert_array_equal(x[:481], observation)
        self.assertEqual(to_training(x[None]).shape, (1, 622))
        with self.assertRaises(ValueError):
            scan_quality(values[:71], lidar)

    def test_initialization_preserves_all_old_decisions(self):
        model = ROOT / 'models/stop_feedback_20261015/long_risk_bc1000_s231016'
        prior = LongRiskActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
        prior.load_state_dict(torch.load(model / 'checkpoint.pt', map_location='cpu', weights_only=False)['actor'])
        actor = ReleaseActor(torch.jit.load(str(model / 'parent_actor.pt'), map_location='cpu')).eval()
        actor.initialize_risk_head(prior)
        with np.load(ROOT / 'reports/stop_feedback_20261015/release_sources_20261016/demonstrations.npz') as data:
            x = torch.from_numpy(data['observations'].copy())
        with torch.no_grad():
            np.testing.assert_array_equal(actor(x).numpy(), prior(x[:, :481]).numpy())
            features, _ = actor.features(x)
            self.assertEqual(features.shape[1], 312)
        self.assertFalse(any(p.requires_grad for p in actor.parent.parameters()))

    def test_original_mask_and_directional_fallback_remain_authoritative(self):
        class Parent(torch.nn.Module):
            def forward(self, x):
                return torch.ones_like(x[:, :2]) * .123
        actor = ReleaseActor(Parent()).eval()
        x = torch.zeros(4, 625)
        x[:, 218] = .2
        x[:, 435] = 1.
        x[:, 438] = 1.
        x[:, 433] = 1.
        x[:, 235:244] = 1.
        x[:, 267:269] = 1.
        with torch.no_grad():
            for p in actor.head.parameters():
                p.zero_()
            actor.head[-1].bias[4] = 100.
        x[1, 238] = 0.  # Turn action 4 is forbidden by the original mask.
        x[2, 435] = 0.  # Unsupported prediction input: exact frozen fallback.
        x[3, 218] = 0.  # Near-goal directional docking: exact frozen fallback.
        with torch.no_grad():
            command = actor(x).numpy()
        np.testing.assert_array_equal(command[0], actor.actions[4].numpy())
        self.assertFalse(np.array_equal(command[1], actor.actions[4].numpy()))
        np.testing.assert_array_equal(command[2:], np.full((2, 2), .123, np.float32))
        self.assertTrue(bool(actor.masks(x)[:, 0].all()))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
