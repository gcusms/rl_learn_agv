from types import SimpleNamespace
import unittest

import numpy as np

from experiments.probe_observed_turn_release import TurnWhenLegal
from neural_navigation.masked_actions import ACTIONS


class Delegate:
    def predict(self, observation):
        return ACTIONS[1].copy()


def observation(*legal):
    value = np.zeros(625, np.float32)
    for choice in legal:
        if 1 <= choice <= 9:
            value[234 + choice] = 1.
        elif choice in (10, 11):
            value[257 + choice] = 1.
    return value


class ObservedTurnTests(unittest.TestCase):
    def setUp(self):
        self.core = SimpleNamespace(odom=np.zeros(3), elapsed=0.)
        self.policy = TurnWhenLegal(Delegate(), self.core, .3)

    def test_trigger_uses_permission_and_not_episode_time(self):
        for timestamp in (1., 1000000.):
            core = SimpleNamespace(odom=np.zeros(3), elapsed=timestamp)
            actor = TurnWhenLegal(Delegate(), core, .3)
            np.testing.assert_array_equal(actor.predict(observation(1, 5)), ACTIONS[5])

    def test_forbidden_turn_always_stays_stopped(self):
        np.testing.assert_array_equal(self.policy.predict(observation(9)), ACTIONS[0])
        np.testing.assert_array_equal(self.policy.predict(observation(5)), ACTIONS[5])
        np.testing.assert_array_equal(self.policy.predict(observation(9)), ACTIONS[0])

    def test_command_and_elapsed_time_do_not_fake_measured_turn(self):
        np.testing.assert_array_equal(self.policy.predict(observation(1, 5)), ACTIONS[5])
        self.core.elapsed = 100.
        np.testing.assert_array_equal(self.policy.predict(observation(1, 5)), ACTIONS[5])
        self.assertEqual(self.policy.clockwise_turn, 0.)
        self.core.odom[2] = -.31
        np.testing.assert_array_equal(self.policy.predict(observation(1, 5)), ACTIONS[1])
        self.assertEqual(self.policy.phase, 'learned')


if __name__ == '__main__':
    unittest.main()
