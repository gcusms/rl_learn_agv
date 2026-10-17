import gc
from types import SimpleNamespace
import unittest
import weakref
import numpy as np
from neural_navigation.motion_prediction import ScanMotionPredictor
from reconfirmed_archive import ReconfirmedArchivePredictor


class SensorState(SimpleNamespace):
    pass


class ReconfirmedArchiveTest(unittest.TestCase):
    def setUp(self):
        self.core = SensorState(estimate=np.zeros(3), scan=np.full(72, np.nan),
            lidar=SimpleNamespace(mount=np.zeros(4), angles=np.linspace(-1.9, 1.9, 72),
                angle_min=-1.9, angle_max=1.9, range_min=.1, range_max=12.))
        self.config = dict(horizon_s=2.4, minimum_speed_m_s=.04, uncertainty_m_s=.025)
        self.p = ReconfirmedArchivePredictor(self.config, self.core)

    def cluster(self, x):
        return np.c_[np.full(8, x), np.linspace(-.12, .12, 8)]

    def moving(self):
        for t in [0., .4, .8, 1.2]:
            self.p.update(self.cluster(1.+.1*t), t)

    def test_reappearance_requires_another_scan_without_losing_identity(self):
        self.moving()
        identity = self.p.tracks[0]['id']
        self.assertEqual(len(self.p.update([], 2.4)), 0)
        self.assertEqual(len(self.p.update(self.cluster(1.4), 4.)), 0)
        self.assertEqual(self.p.tracks[0]['id'], identity)
        self.assertEqual(self.p.tracks[0]['hits'], 2)
        self.assertGreater(len(self.p.update(self.cluster(1.44), 4.4)), 0)
        self.assertEqual(self.p.reconfirmation_resets, 1)

    def test_uninterrupted_outputs_still_match_original(self):
        old = ScanMotionPredictor(self.config)
        for t in [0., .4, .8, 1.2, 1.6]:
            points = self.cluster(1.+.1*t)
            np.testing.assert_array_equal(old.update(points, t), self.p.update(points, t))

    def test_reversal_keeps_stricter_three_hit_confirmation(self):
        self.moving()
        self.assertEqual(len(self.p.update(self.cluster(.95), 2.4)), 0)
        self.assertEqual(self.p.tracks[0]['hits'], 1)
        self.assertEqual(len(self.p.update(self.cluster(.91), 2.8)), 0)
        self.assertGreater(len(self.p.update(self.cluster(.87), 3.2)), 0)

    def test_predictor_does_not_keep_retired_core_alive(self):
        core = SensorState()
        predictor = ReconfirmedArchivePredictor(self.config, core)
        core.predictor = predictor
        reference = weakref.ref(core)
        del core
        gc.collect()
        self.assertIsNone(reference())


if __name__ == '__main__':
    unittest.main()
