"""Forecast equivalence, including aged contours and contact boundaries."""
import unittest
from unittest.mock import patch
import numpy as np

from neural_navigation.directional_risk import directional_body_risk_reference as reference
from neural_navigation.directional_risk_batch import batched_directional_body_risk as batch
from neural_navigation.masked_actions import COMMANDS


class DirectionalBatchTests(unittest.TestCase):
    def compare(self, tracks, pose, **kwargs):
        args = (tracks, pose, 2., COMMANDS, [.393924904, .329460022])
        expected, actual = reference(*args, **kwargs), batch(*args, **kwargs)
        for before, after in zip(expected, actual):
            self.assertEqual(before['command'], after['command'])
            self.assertEqual(before['supported_tracks'], after['supported_tracks'])
            self.assertEqual(before['predicted_margin_contact_s'], after['predicted_margin_contact_s'])
            if before['minimum_predicted_clearance_m'] is None:
                self.assertIsNone(after['minimum_predicted_clearance_m'])
            else:
                self.assertAlmostEqual(before['minimum_predicted_clearance_m'],
                    after['minimum_predicted_clearance_m'], delta=2e-13)

    def test_random_aged_contours_both_motion_models(self):
        rng = np.random.default_rng(441016)
        for i in range(32):
            tracks = []
            for j in range(i % 4):
                residual = rng.normal(0, .03, (2, 2))
                tracks.append(dict(points=rng.normal(0, 2, (1 + (i * 13 + j) % 70, 2)),
                    velocity=rng.uniform(-.4, .4, 2), time=2. - [.0, .8, .81][j],
                    hits=[2, 3, 5][j], velocity_error_second_moment=residual @ residual.T))
            kwargs = dict(horizon_s=[1.2, 2.4][i % 2], period_s=[.1, .2][i % 2])
            if i % 3:
                kwargs['measured_velocity'] = rng.uniform([0., -.5], [.2, .5])
            self.compare(tracks, rng.uniform([-5, -5, -3.14], [5, 5, 3.14]), **kwargs)

    def test_long_contour_chunking_and_anisotropic_uncertainty(self):
        rng = np.random.default_rng(1016)
        tracks = [dict(points=rng.normal([1.2, .1], [.3, .4], (907, 2)),
            velocity=np.array([-.12, .04]), time=1.8, hits=4,
            velocity_error_second_moment=np.array([[.003, .001], [.001, .0005]]))]
        self.compare(tracks, [.1, -.1, .3], measured_velocity=[.15, -.1])

    def test_no_tracks_preserves_validation_and_unknown_outputs(self):
        self.compare([], [0., 0., 0.], measured_velocity=[0., 0.])
        for function in (reference, batch):
            with self.assertRaises(ValueError):
                function([], np.zeros(3), 2., COMMANDS, [.4, .3], measured_velocity=[np.nan, 0.])
            with self.assertRaises(ValueError):
                function([dict(points=np.zeros((2, 2)), velocity=np.ones(2), time=2., hits=3,
                    velocity_error_second_moment=np.diag([-1., 1.]))],
                    np.zeros(3), 2., COMMANDS, [.4, .3])

    def test_numerical_contact_boundary_retains_original_decision(self):
        for offset in (-1e-10, 0., 1e-10):
            track = dict(points=np.array([[.2 + .04 + 1e-9 + offset, 0.]]),
                velocity=np.array([-.04, 0.]), time=2., hits=3)
            args = ([track], np.zeros(3), 2., np.array([[0., 0.]]), [.4, .3])
            expected = reference(*args, uncertainty_floor_m_s=0.)
            self.assertEqual(batch(*args, uncertainty_floor_m_s=0.), expected)
        track['points'][0, 0] = .2 + .04 + 1e-9
        with patch('neural_navigation.directional_risk.directional_body_risk_reference', wraps=reference) as fallback:
            actual = batch(iter([track]), np.zeros(3), 2., np.array([[0., 0.]]), [.4, .3], uncertainty_floor_m_s=0.)
            self.assertTrue(fallback.called)
            self.assertEqual(actual, reference([track], np.zeros(3), 2., np.array([[0., 0.]]), [.4, .3], uncertainty_floor_m_s=0.))


if __name__ == '__main__':
    unittest.main()
