from types import SimpleNamespace
import unittest

import numpy as np

from legal_turn_policy import LegalClockwiseManeuver
from neural_navigation.masked_actions import ACTIONS


def observation(choices):
    result = np.zeros(625, np.float32)
    for choice in choices:
        if 1 <= choice <= 9:
            result[234+choice] = 1.
        elif choice >= 10:
            result[267+choice-10] = 1.
    return result


class FrozenWait:
    def predict(self, obs):
        return ACTIONS[0].copy()


class OriginalPermissionTests(unittest.TestCase):
    def setUp(self):
        self.core = SimpleNamespace(odom=np.zeros(3), estimate=np.zeros(3), goal=np.array([0., 2.]), elapsed=0.)

    def test_only_creep_then_pure_turn_as_permission_changes(self):
        policy = LegalClockwiseManeuver(FrozenWait(), self.core, 6.)
        np.testing.assert_array_equal(policy.predict(observation([9])), ACTIONS[9])
        np.testing.assert_array_equal(policy.predict(observation([5, 9])), ACTIONS[5])

    def test_stop_only_is_never_overridden(self):
        policy = LegalClockwiseManeuver(FrozenWait(), self.core, 4.5)
        np.testing.assert_array_equal(policy.predict(observation([])), ACTIONS[0])

    def test_measured_turn_releases_to_frozen_policy(self):
        policy = LegalClockwiseManeuver(FrozenWait(), self.core, 4.5)
        for step in range(1, 6):
            self.core.odom[2] = -float(step)
            action = policy.predict(observation([5, 9]))
        self.assertEqual(policy.phase, 'learned')
        np.testing.assert_array_equal(action, ACTIONS[0])

    def test_yaw_wrap_is_a_small_turn(self):
        self.core.odom[2] = -3.1
        policy = LegalClockwiseManeuver(FrozenWait(), self.core, 6.)
        self.core.odom[2] = 3.1
        policy.predict(observation([5]))
        self.assertAlmostEqual(policy.clockwise_turn, 2*np.pi-6.2)

    def test_goal_alignment_alone_does_not_bypass_blocked_forward(self):
        self.core.goal[:] = [2., 0.]
        policy = LegalClockwiseManeuver(FrozenWait(), self.core, 'path_heading')
        self.core.odom[2] = -.5
        np.testing.assert_array_equal(policy.predict(observation([5])), ACTIONS[5])
        self.assertEqual(policy.phase, 'maneuver')


if __name__ == '__main__':
    unittest.main()
