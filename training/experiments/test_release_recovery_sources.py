import unittest

import numpy as np

from experiments.build_release_recovery_sources import training_indices
from neural_navigation.masked_actions import COMMANDS


class ReleaseRecoverySourceTests(unittest.TestCase):
    def source(self):
        obs = np.zeros((3, 625), np.float32)
        obs[-1, 218] = .1 / 12.
        return dict(choices=np.array([1, 1, 1]), commands=np.array([COMMANDS[1], COMMANDS[1], [0., 0.]]),
            observations=obs), dict(stop_reasons={'estimated_goal_reached': 1}), np.array([True, True, False])

    def test_completion_override_is_excluded_without_relabeling_policy_action(self):
        trace, row, support = self.source()
        original_choices = trace['choices'].copy()
        np.testing.assert_array_equal(training_indices(trace, row, support), [0, 1])
        np.testing.assert_array_equal(trace['choices'], original_choices)

    def test_nonterminal_projection_rejects_the_source(self):
        trace, row, support = self.source()
        trace['commands'][1] = 0.
        with self.assertRaises(ValueError):
            training_indices(trace, row, support)

    def test_missing_arrival_evidence_and_trainable_terminal_are_rejected(self):
        trace, row, support = self.source()
        row['stop_reasons'] = {}
        with self.assertRaises(ValueError):
            training_indices(trace, row, support)
        row['stop_reasons'] = {'estimated_goal_reached': 1}
        support[-1] = True
        with self.assertRaises(ValueError):
            training_indices(trace, row, support)

    def test_nonfinite_execution_is_rejected(self):
        trace, row, support = self.source()
        trace['commands'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            training_indices(trace, row, support)


if __name__ == '__main__':
    unittest.main()
