from types import SimpleNamespace
import unittest
import numpy as np

from experiments.probe_feedback_alternatives import ProbeActor
from neural_navigation.masked_actions import ACTIONS


class Actor:
    def predict(self, observation): return ACTIONS[1].copy()


class ProbeTests(unittest.TestCase):
    def test_outside_window_policy_is_unchanged(self):
        core = SimpleNamespace(elapsed=0.)
        probe = ProbeActor(Actor(), core, dict(start_s=1., end_s=2., choice=5))
        observation = np.zeros(439)
        np.testing.assert_array_equal(probe.predict(observation), ACTIONS[1])
        self.assertEqual(probe.entries, [])

    def test_illegal_intervention_is_skipped(self):
        core = SimpleNamespace(elapsed=1.)
        probe = ProbeActor(Actor(), core, dict(start_s=1., end_s=2., choice=5))
        observation = np.zeros(439); observation[235] = 1.  # Forward only.
        np.testing.assert_array_equal(probe.predict(observation), ACTIONS[1])
        self.assertFalse(probe.entries[0]['requested_choice_legal'])
        self.assertFalse(probe.entries[0]['replaced_reference'])

    def test_legal_choice_uses_sensor_time_and_keeps_input_unchanged(self):
        core = SimpleNamespace(elapsed=1.)
        probe = ProbeActor(Actor(), core, dict(start_s=1., end_s=2., choice=5))
        observation = np.ones(439); before = observation.copy()
        np.testing.assert_array_equal(probe.predict(observation), ACTIONS[5])
        np.testing.assert_array_equal(observation, before)
        core.elapsed = 2.
        np.testing.assert_array_equal(probe.predict(observation), ACTIONS[1])
        self.assertEqual(len(probe.entries), 1)


if __name__ == '__main__': unittest.main()
