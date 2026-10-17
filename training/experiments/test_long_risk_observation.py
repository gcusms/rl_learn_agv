"""Causal-time, unknown-data and frozen-parent checks for the new contract."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.long_risk_actor import LongRiskActor
from neural_navigation.long_risk_observation import LongRiskHistory
from neural_navigation.masked_actions import ACTIONS


class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.register_buffer('actions', torch.from_numpy(ACTIONS.copy()))

    def forward(self, x):
        selected = (x[:, 235] > .5).long()
        return self.actions[selected] + self.anchor * 0.


class LongRiskTests(unittest.TestCase):
    def setup_frame(self):
        observation = np.zeros(439, np.float32)
        observation[[435, 438]] = 1.
        observation[235:244] = 1.
        observation[267:269] = 1.
        observation[218] = .2
        core = SimpleNamespace(elapsed=0., odom=np.zeros(3), measured_velocity=np.zeros(2))
        track = dict(time=0., velocity=[-.1, 0.], points=[[1., 0.]], hits=4)
        predictor = SimpleNamespace(timestamp=0., tracks=[track])
        guard = SimpleNamespace(half=np.array([.393924904, .329460022]) / 2,
            predictive_constraint=predictor)
        return observation, core, guard, track

    def test_physical_intervals_and_unknown_samples(self):
        x, core, guard, track = self.setup_frame()
        history = LongRiskHistory()
        for timestamp in (0., .1, .2, .4):
            core.elapsed = guard.predictive_constraint.timestamp = track['time'] = timestamp
            value = history.augment(x, core, guard)
            np.testing.assert_array_equal(value[:439], x)
            valid = value[439:].reshape(3, 14)[:, 12]
            if timestamp < .2:
                np.testing.assert_array_equal(valid, [0, 0, 1])
            if timestamp == .4:
                np.testing.assert_array_equal(valid, [1, 1, 1])
        core.elapsed = guard.predictive_constraint.timestamp = track['time'] = .6
        x[[435, 438]] = 0.
        value = history.augment(x, core, guard)
        np.testing.assert_array_equal(value[-14:], np.zeros(14))
        with self.assertRaises(ValueError):
            history.augment(x, core, guard)
        history.reset()
        value = history.augment(x, core, guard)
        np.testing.assert_array_equal(value[439:], np.zeros(42))

    def test_unsupported_tracks_are_not_valid_clear(self):
        x, core, guard, track = self.setup_frame()
        track['hits'] = 2
        value = LongRiskHistory().augment(x, core, guard)
        np.testing.assert_array_equal(value[-14:], np.zeros(14))

    def test_model_controls_wait_and_turn_but_keeps_parent_outside_support(self):
        actor = LongRiskActor(Parent()).eval()
        x = torch.zeros((1, 481))
        x[:, 235:244] = x[:, 267:269] = 1.
        x[:, 218] = .2
        x[:, [435, 438]] = 1.
        x[:, -2] = 1.
        with torch.no_grad():
            actor.head[-1].bias[5] = 20.
            np.testing.assert_array_equal(actor(x).numpy()[0], ACTIONS[5])
            x[:, 239] = 0.  # Right turn is rejected by the original mask.
            np.testing.assert_array_equal(actor(x).numpy()[0], ACTIONS[1])
            actor.head[-1].bias[0] = 40.
            np.testing.assert_array_equal(actor(x).numpy()[0], ACTIONS[0])
            x[:, 435] = 0.
            np.testing.assert_array_equal(actor(x).numpy()[0], ACTIONS[1])
            x[:, 435] = 1.
            x[:, 439:] = 0.
            x[:, 433] = .2  # Track loss during a real wait stays learnable.
            self.assertTrue(actor.support(x).item())
            x[:, 218:220] = 0.
            np.testing.assert_array_equal(actor(x).numpy()[0], ACTIONS[1])
        self.assertFalse(any(parameter.requires_grad for parameter in actor.parent.parameters()))


if __name__ == '__main__':
    unittest.main()
