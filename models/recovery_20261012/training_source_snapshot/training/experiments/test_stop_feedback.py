import unittest
import numpy as np
from audit_stop_feedback import audit
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM, REASONS, REASON_START


class FeedbackAuditTest(unittest.TestCase):
    def row(self, time, reason, prior='none', prior_action=(0, 0)):
        obs = np.zeros(OBSERVATION_DIM)
        obs[BASE_DIM] = 1
        obs[BASE_DIM+12:BASE_DIM+14] = prior_action
        obs[BASE_DIM+REASON_START+REASONS.index(prior)] = 1
        return dict(timestamp=time, stop_reason=reason, command=[0, 0],
                    estimate=[0, 0, 0], policy_observation=obs.tolist())

    def test_correct_feedback_and_corruption(self):
        rows = [self.row(0, 'reverse_without_rear_visibility'),
                self.row(.2, None, 'reverse_without_rear_visibility')]
        trace = dict(case=dict(id='test'), samples=rows)
        self.assertEqual(audit(trace)['counts']['feedback_mismatch'], 0)
        rows[1]['policy_observation'][BASE_DIM+12] = .2
        self.assertEqual(audit(trace)['counts']['feedback_mismatch'], 1)

    def test_missing_is_not_a_pass(self):
        row = self.row(0, None)
        row.pop('policy_observation')
        counts = audit(dict(case=dict(id='test'), samples=[row]))['counts']
        self.assertEqual(counts['captured'], 0)
        self.assertEqual(counts['missing'], 1)


if __name__ == '__main__':
    unittest.main()
