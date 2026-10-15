import unittest
import numpy as np
from prototypes.balanced_recovery import BalancedSchedule, SupportedWaitCost


class BalancedRecoveryTests(unittest.TestCase):
    def test_general_half_and_all_training_templates_before_reuse(self):
        first = BalancedSchedule(15, 7)
        rows = [first.next() for _ in range(30)]
        self.assertEqual(rows[1::2], [None]*15)
        self.assertEqual(set(rows[::2]), set(range(15)))
        second = BalancedSchedule(15, 7)
        self.assertEqual(rows, [second.next() for _ in range(30)])

    def test_stopping_for_unknown_motion_or_no_margin_is_not_penalized(self):
        obs = np.zeros(300, np.float32)
        obs[238] = obs[294] = 1
        info = dict(measured_body_gap_m=.2,safety_termination=False,verified_mission_success=False)
        cost = SupportedWaitCost()
        for _ in range(40): cost.step(obs,0,info,.2)
        self.assertGreater(cost.total, 0)
        for field, value in [(294,0),(295,1),(296,1),(297,1),(298,1),(238,0),(286,.8)]:
            altered = obs.copy(); altered[field] = value
            self.assertEqual(cost.step(altered,0,info,.2),0)
        self.assertEqual(cost.step(obs,0,dict(info,measured_body_gap_m=.03),.2),0)
        self.assertEqual(cost.step(obs,0,dict(info,safety_termination=True),.2),0)
        self.assertEqual(cost.step(obs,0,dict(info,verified_mission_success=True),.2),0)
        self.assertEqual(cost.step(obs,4,info,.2),0)

    def test_episode_budget_and_grace_are_bounded(self):
        obs = np.zeros(300,np.float32); obs[238]=obs[294]=1
        info = dict(measured_body_gap_m=.2,safety_termination=False,verified_mission_success=False)
        cost = SupportedWaitCost()
        for _ in range(10): self.assertAlmostEqual(cost.step(obs,0,info,.2),0)
        for _ in range(1000): cost.step(obs,0,info,.2)
        self.assertAlmostEqual(cost.total,2)
        self.assertEqual(cost.step(obs,0,info,.2),0)
        cost.reset(); self.assertEqual(cost.total,0)


if __name__ == '__main__': unittest.main()
