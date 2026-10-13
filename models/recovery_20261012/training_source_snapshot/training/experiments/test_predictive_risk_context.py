import copy
from types import SimpleNamespace
import unittest
import numpy as np
from experiments.predictive_risk_context import PredictiveRiskContext, RISK_DIMENSIONS


class SensorOnlyCore(SimpleNamespace):
    @property
    def truth(self): raise AssertionError('Simulator truth must not be accessed')
    @property
    def actors(self): raise AssertionError('Actor truth must not be accessed')


class PredictiveRiskContextTests(unittest.TestCase):
    def core(self, points=None):
        track = dict(time=1., hits=4, velocity=np.array([-.2, 0.]),
            points=np.array([[.6, -.02], [.6, .02]]) if points is None else points)
        return SensorOnlyCore(localizer=SimpleNamespace(scan_update_accepted=True, map_agreement=1., std=np.zeros(3)),
            estimate=np.zeros(3), elapsed=1.,
            laser_fusion=SimpleNamespace(predictor=SimpleNamespace(tracks=[track])))

    def test_sensor_only_features_bounded_and_track_state_unchanged(self):
        context = PredictiveRiskContext([.4, .3])
        core = self.core(); before = copy.deepcopy(core.laser_fusion.predictor.tracks)
        features = context.features(core)
        self.assertEqual(features.shape, (RISK_DIMENSIONS,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(((features >= 0) & (features <= 1)).all())
        self.assertGreater(features[0], 0.)
        self.assertEqual(features[-1], 1.)
        for key in ['points', 'velocity']:
            np.testing.assert_array_equal(before[0][key], core.laser_fusion.predictor.tracks[0][key])
        self.assertFalse(context.contract()['runtime_wired'])

    def test_unknown_and_supported_no_contact_are_distinct(self):
        context = PredictiveRiskContext([.4, .3])
        core = self.core(points=np.array([[20., 0.], [20., .02]]))
        clear = context.features(core)
        self.assertEqual(clear[0], 0.); self.assertEqual(clear[-1], 1.)
        core.elapsed = 2.
        unknown = context.features(core)
        self.assertTrue((unknown == 0).all())
        core.elapsed = 1.; core.localizer.scan_update_accepted = False
        self.assertTrue((context.features(core) == 0).all())
        core.localizer.scan_update_accepted = True; core.localizer.std[0] = .21
        self.assertTrue((context.features(core) == 0).all())

    def test_boundary_contact_has_positive_score(self):
        context = PredictiveRiskContext([.4, .3], dict(horizon_s=.2, uncertainty_m_s=0.))
        # At t=.2 the observed point reaches the 4cm body margin.
        core = self.core(points=np.array([[.28, 0.], [.28, .01]]))
        self.assertGreater(context.features(core)[0], 0.)


if __name__ == '__main__': unittest.main()
