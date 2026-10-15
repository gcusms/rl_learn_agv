import unittest
import numpy as np
from experiments.audit_decision_waits import intervals, measure


class WaitAuditTests(unittest.TestCase):
    def trace(self, choices, available, distance=1.):
        n = len(choices)
        labels = np.zeros((n, 11))
        labels[:, 0] = np.arange(n) * .2
        labels[:, 7] = distance
        masks = np.zeros((n, 12), bool)
        masks[:, 0] = True
        masks[:, 1] = available
        commands = np.c_[np.array(choices) * .2, np.zeros(n)]
        return dict(labels=labels, masks=masks, choices=np.array(choices), commands=commands)

    def test_goal_settling_does_not_become_enroute_waiting(self):
        trace = self.trace([0] * 20, True, distance=.05)
        result = measure(trace, 3.8, .15)
        self.assertAlmostEqual(result['voluntary_wait_s'], 3.8)
        self.assertEqual(result['enroute_voluntary_wait_s'], 0.)
        self.assertFalse(result['long_unfinished_task_waits'])

    def test_position_reached_but_heading_wrong_is_still_pending(self):
        trace = self.trace([0] * 20, True, distance=.05)
        trace['labels'][:, 8] = .17
        result = measure(trace, 3.8, .12, yaw_tolerance=.15)
        self.assertAlmostEqual(result['heading_alignment_wait_s'], 3.8)
        self.assertEqual(result['enroute_voluntary_wait_s'], 0.)
        self.assertAlmostEqual(result['unfinished_task_wait_s'], 3.8)
        self.assertEqual(len(result['long_unfinished_task_waits']), 1)

    def test_release_clock_starts_after_forced_stop(self):
        trace = self.trace([0, 0, 0, 0, 1, 1], [False, False, True, True, True, True])
        result = measure(trace, 1.2, .2, [2., 2., 2., 2., 1.99, 1.97])
        self.assertAlmostEqual(result['only_stop_s'], .4)
        self.assertAlmostEqual(result['enroute_voluntary_wait_s'], .4)
        self.assertAlmostEqual(result['resumptions'][0]['first_motion_delay_s'], .4)
        self.assertAlmostEqual(result['resumptions'][0]['geodesic_progress_m'], .03)

    def test_long_wait_uses_elapsed_time_not_a_frame_count(self):
        trace = self.trace([0] * 5, True)
        trace['labels'][:, 0] = [0., .1, .3, 1.8, 2.1]
        result = measure(trace, 2.2, .2)
        self.assertAlmostEqual(result['longest_unfinished_task_wait_s'], 2.2)
        self.assertEqual(len(result['long_unfinished_task_waits']), 1)

    def test_invalid_time_or_mask_rejects_misleading_metrics(self):
        with self.assertRaises(ValueError):
            intervals([0., .2, .1], .4)
        with self.assertRaises(ValueError):
            intervals([0., .2], .1)
        with self.assertRaises(ValueError):
            measure(self.trace([1], False), .2, .2)


if __name__ == '__main__':
    unittest.main()
