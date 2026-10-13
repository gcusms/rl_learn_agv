import unittest
import numpy as np
from neural_navigation.motion_prediction import ScanMotionPredictor


class MotionPredictionTests(unittest.TestCase):
    def cluster(self,x):return np.c_[np.full(8,x),np.linspace(-.12,.12,8)]
    def test_static_returns_do_not_create_a_moving_forecast(self):
        p=ScanMotionPredictor({'horizon_s':2.4})
        for t in np.arange(0,2,.4):self.assertEqual(len(p.update(self.cluster(1.),t)),0)
        self.assertEqual(p.moving_tracks,0)
    def test_measured_motion_predicts_forward_then_expires_after_occlusion(self):
        p=ScanMotionPredictor({'horizon_s':2.4})
        for t in [0.,.4,.8,1.2]:future=p.update(self.cluster(1.+.1*t),t)
        self.assertGreater(future[:,0].max(),1.12)
        np.testing.assert_array_equal(p.update(self.cluster(1.12),1.2),future)
        self.assertGreater(len(p.update(np.empty((0,2)),1.6)),0)
        self.assertEqual(len(p.update(np.empty((0,2)),2.4)),0)
        self.assertEqual(len(p.tracks),0)
    def test_reversal_drops_old_direction_and_poor_alignment_resets_tracks(self):
        p=ScanMotionPredictor({'horizon_s':2.4})
        for t in [0.,.4,.8,1.2]:p.update(self.cluster(1.+.1*t),t)
        self.assertEqual(len(p.update(self.cluster(1.08),1.6)),0)
        for t,x in [(2.,1.04),(2.4,1.)]:future=p.update(self.cluster(x),t)
        self.assertGreater(len(future),0);self.assertLess(future[:,0].min(),1.)
        self.assertEqual(len(p.update(self.cluster(.96),2.8,alignment_supported=False)),0)
        self.assertEqual(len(p.tracks),0)


if __name__=='__main__':unittest.main(verbosity=2)
