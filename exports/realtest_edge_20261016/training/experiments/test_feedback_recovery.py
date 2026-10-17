from types import SimpleNamespace
import unittest

import numpy as np

from experiments.evaluate_feedback_recovery import SensorPrefixActor
from neural_navigation.masked_actions import ACTIONS


class LearnedActor:
    calls = 0

    def predict(self, observation):
        self.calls += 1
        return ACTIONS[1].copy()


class RecoveryPrefixTests(unittest.TestCase):
    def setUp(self):
        self.core = SimpleNamespace(elapsed=0., scan=np.array([1., np.nan]),
                                    odom=np.zeros(3), measured_velocity=np.zeros(2))
        self.prefix = dict(times=np.array([0.]), choices=np.array([8]),
                           scans=self.core.scan[None].copy(), odometry=np.zeros((1, 3)),
                           measured_velocity=np.zeros((1, 2)))
        self.actor = LearnedActor()
        self.wrapper = SensorPrefixActor(self.actor, self.core, self.prefix, .2)
        self.observation = np.ones(439)

    def test_prefix_is_validated_then_control_returns_to_the_model(self):
        original = self.observation.copy()
        np.testing.assert_array_equal(self.wrapper.predict(self.observation), ACTIONS[8])
        self.assertEqual(self.actor.calls, 0)
        self.core.elapsed = .2
        np.testing.assert_array_equal(self.wrapper.predict(self.observation), ACTIONS[1])
        self.assertEqual(self.actor.calls, 1)
        self.assertEqual(self.wrapper.released_frames, 1)
        np.testing.assert_array_equal(original, self.observation)

    def test_different_sensor_history_is_rejected(self):
        self.core.odom[0] = .01
        with self.assertRaises(ValueError): self.wrapper.predict(self.observation)

    def test_rejected_prefix_command_is_not_forced(self):
        self.observation[235:244] = 0.; self.observation[267:269] = 0.
        with self.assertRaises(ValueError): self.wrapper.predict(self.observation)
        self.assertEqual(self.wrapper.cursor, 0)

    def test_skipped_frames_and_early_release_are_rejected(self):
        self.core.elapsed = .2
        with self.assertRaises(ValueError): self.wrapper.predict(self.observation)
        self.core.elapsed = 0.
        self.wrapper.predict(self.observation)
        with self.assertRaises(ValueError): self.wrapper.predict(self.observation)


if __name__ == '__main__': unittest.main()
