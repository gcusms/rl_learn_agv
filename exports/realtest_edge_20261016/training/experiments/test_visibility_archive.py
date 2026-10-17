import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.motion_prediction import ScanMotionPredictor
from visibility_archive import VisibilityArchivePredictor


class VisibilityArchiveTest(unittest.TestCase):
    def setUp(self):
        self.core = SimpleNamespace(estimate=np.zeros(3), scan=np.full(72, np.nan),
            lidar=SimpleNamespace(mount=np.zeros(4), angles=np.linspace(-1.9, 1.9, 72),
                                  angle_min=-1.9, angle_max=1.9, range_min=.1, range_max=12.))
        self.config = dict(horizon_s=2.4, minimum_speed_m_s=.04, uncertainty_m_s=.025)
        self.predictor = VisibilityArchivePredictor(self.config, self.core)

    def cluster(self, x):
        return np.c_[np.full(8, x), np.linspace(-.12, .12, 8)]

    def moving(self):
        for time in [0., .4, .8, 1.2]:
            self.predictor.update(self.cluster(1.+.1*time), time)

    def test_fresh_and_static_behavior_matches_original(self):
        for moving in [False, True]:
            old = ScanMotionPredictor(self.config)
            new = VisibilityArchivePredictor(self.config, self.core)
            for time in [0., .4, .8, 1.2, 1.6]:
                points = self.cluster(1.+(.1*time if moving else 0.))
                np.testing.assert_array_equal(old.update(points, time), new.update(points, time))

    def test_archive_reacquires_without_stale_forecasts(self):
        self.moving()
        identity = self.predictor.tracks[0]['id']
        self.assertEqual(len(self.predictor.update([], 2.4)), 0)
        self.assertEqual(self.predictor.moving_tracks, 0)
        self.assertEqual(len(self.predictor.tracks), 1)
        self.assertGreater(len(self.predictor.update(self.cluster(1.4), 4.)), 0)
        self.assertEqual(self.predictor.tracks[0]['id'], identity)
        self.assertEqual(self.predictor.reassociations, 1)
        self.assertEqual(len(self.predictor.update([], 8.1)), 0)
        self.assertEqual(len(self.predictor.tracks), 0)

    def test_reversal_and_bad_alignment_require_fresh_confirmation(self):
        self.moving()
        self.assertEqual(len(self.predictor.update(self.cluster(.95), 2.4)), 0)
        self.assertEqual(self.predictor.tracks[0]['hits'], 1)
        self.assertEqual(len(self.predictor.update(self.cluster(.91), 2.8, False)), 0)
        self.assertEqual(len(self.predictor.tracks), 0)

    def test_clear_evidence_removes_archive_but_blind_or_missing_scans_do_not(self):
        self.moving()
        self.core.estimate[2] = np.pi
        self.core.scan[:] = 5.
        self.predictor.update([], 2.4)
        self.predictor.update([], 2.8)
        self.assertEqual(len(self.predictor.tracks), 1)
        self.core.estimate[2] = 0.
        self.predictor.update([], 3.2)
        self.assertEqual(len(self.predictor.tracks), 1)
        self.predictor.update([], 3.6)
        self.assertEqual(len(self.predictor.tracks), 0)
        self.assertEqual(self.predictor.clear_removals, 1)

    def test_stopped_reappearance_does_not_inherit_motion(self):
        self.moving()
        self.assertEqual(len(self.predictor.update(self.cluster(1.12), 2.4)), 0)
        self.assertEqual(self.predictor.tracks[0]['hits'], 1)

    def test_ambiguous_groups_do_not_reuse_old_identity(self):
        self.moving()
        groups = np.vstack([np.c_[np.full(8, 1.3), np.linspace(-.26, -.14, 8)],
                            np.c_[np.full(8, 1.3), np.linspace(.14, .26, 8)]])
        self.assertEqual(len(self.predictor.update(groups, 2.4)), 0)
        self.assertEqual(self.predictor.reassociations, 0)
        self.assertEqual(len(self.predictor.tracks), 3)


if __name__ == '__main__':
    unittest.main()
