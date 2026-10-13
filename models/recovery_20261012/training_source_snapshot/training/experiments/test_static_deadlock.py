import unittest
import numpy as np
from static_deadlock_curriculum import StaticDeadlockDetector
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from static_deadlock_curriculum import StaticDeadlockCurriculum


class StaticDeadlockTest(unittest.TestCase):
    def setUp(self):
        self.detector = StaticDeadlockDetector()
        self.obs = np.zeros(OBSERVATION_DIM)
        self.pose = np.zeros(3)
        self.health = dict(localization_scan_update_accepted=True,
                           localization_missing_scan_updates=0, localization_map_agreement=1.)

    def step(self, reason='static_braking_footprint', free=False, health=None):
        return self.detector.step(self.obs, self.pose, free, reason,
                                  self.health if health is None else health, .2)

    def test_persistent_origin_deadlock_only(self):
        for _ in range(14):
            self.assertFalse(self.step())
        self.assertTrue(self.step())

    def test_sensor_dynamic_wait_and_feasible_action_exempt(self):
        for reason in ['scan_unavailable', 'stale_or_future_scan', 'localization_uncertain',
                       'laser_braking_footprint', 'reverse_without_rear_visibility']:
            for _ in range(20):
                self.assertFalse(self.step(reason=reason))
        for _ in range(20):
            self.assertFalse(self.step(free=True))
        self.obs[BASE_DIM+3] = 1
        for _ in range(20):
            self.assertFalse(self.step())

    def test_localization_recovery_and_motion_reset_clock(self):
        for _ in range(14):
            self.step()
        self.pose[0] = .04
        self.assertFalse(self.step())
        self.assertAlmostEqual(self.detector.duration, .2)
        for health in [{}, dict(self.health, localization_scan_update_accepted=False),
                       dict(self.health, localization_missing_scan_updates=1),
                       dict(self.health, localization_map_agreement=.7)]:
            self.assertFalse(self.step(health=health))
            self.assertEqual(self.detector.duration, 0.)

    def test_real_transitions_unchanged_outside_deadlock(self):
        import torch
        from navrl.geometry import ROOT
        torch.set_num_threads(1)
        results = []
        for cls in (RecoveryNavigationCurriculum, StaticDeadlockCurriculum):
            env = cls(ROOT/'runs/safety_context_20261011/recovery_probes_v3_16000/candidate',
                      stage='static', footprint_clearance_cost=1.)
            env.scenes = [ROOT/'assets/room_navigation_scripted']
            rows = []
            try:
                env.reset(seed=94022)
                for _ in range(16):
                    obs, reward, done, truncated, info = env.step(np.array([-.05, .3]))
                    self.assertFalse(done or truncated)
                    self.assertFalse(info.get('training_static_deadlock_termination', False))
                    rows.append((obs.copy(), reward, info['commanded_velocity'].copy(),
                                 env.active.truth.copy()))
            finally:
                env.close()
            results.append(rows)
        for original, changed in zip(*results):
            for first, second in zip(original, changed):
                np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    unittest.main()
