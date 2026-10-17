import unittest
import numpy as np
from experiments.predicted_body_risk import predicted_body_risk


class PredictedBodyRiskTests(unittest.TestCase):
    def track(self, **changes):
        value = dict(time=1., hits=4, velocity=np.array([-.2, 0.]),
            points=np.array([[.6, -.02], [.6, .02]]))
        value.update(changes); return value

    def test_stationary_hazard_and_motion_away_have_different_time_risk(self):
        result = predicted_body_risk([self.track()], [0., 0., 0.], 1.,
            [[0., 0.], [-.2, 0.]], [.4, .3], uncertainty_m_s=0.)
        self.assertAlmostEqual(result[0]['predicted_margin_contact_s'], 1.8)
        self.assertIsNone(result[1]['predicted_margin_contact_s'])
        # Diagnostic reverse is not permission to reverse without rear sensing.

    def test_stale_unconfirmed_and_static_tracks_do_not_certify_clear_space(self):
        for track in [self.track(time=0.), self.track(hits=2), self.track(velocity=[0., 0.])]:
            value = predicted_body_risk([track], [0., 0., 0.], 1., [[0., 0.]], [.4, .3])[0]
            self.assertEqual(value['supported_tracks'], 0)
            self.assertIsNone(value['minimum_predicted_clearance_m'])
            self.assertIsNone(value['predicted_margin_contact_s'])

    def test_world_rotation_preserves_risk(self):
        first = predicted_body_risk([self.track()], [0., 0., 0.], 1., [[0., 0.]], [.4, .3])
        track = self.track(points=np.array([[.02, .6], [-.02, .6]]), velocity=[0., -.2])
        rotated = predicted_body_risk([track], [0., 0., np.pi/2], 1., [[0., 0.]], [.4, .3])
        self.assertEqual(first[0]['predicted_margin_contact_s'], rotated[0]['predicted_margin_contact_s'])
        self.assertAlmostEqual(first[0]['minimum_predicted_clearance_m'], rotated[0]['minimum_predicted_clearance_m'])


if __name__ == '__main__': unittest.main()
