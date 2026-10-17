import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from neural_navigation.aligned_risk_observation import AlignedRiskHistory, contract


class NativeForecastAlignment(unittest.TestCase):
    def setUp(self):
        self.observation = np.zeros(439, np.float32)
        self.observation[[435, 438]] = 1.
        self.core = SimpleNamespace(elapsed=1., odom=np.zeros(3), measured_velocity=np.zeros(2))
        # A cumulative guard can still retain a one-second-old hypothesis,
        # while the v1 long predictor filters measurements older than0.8s.
        track = dict(points=np.array([[.5, 0.]]), velocity=np.array([-.1, 0.]),
                     time=0., hits=3)
        contacts = np.full(12, np.inf)
        contacts[0] = .7
        self.guard = SimpleNamespace(half=np.array([.2, .16]),
            predictive_constraint=SimpleNamespace(timestamp=1., contacts=contacts,
                                                  supported_tracks=1, tracks=[track]))

    def test_real_old_age_filter_no_longer_hides_guard_stop_contact(self):
        history = AlignedRiskHistory()
        output = history.augment(self.observation, self.core, self.guard)
        self.assertEqual(history.original.last_observation[-2], 0.)
        self.assertAlmostEqual(output[-14], 1.-.7/5.2, places=6)
        self.assertEqual(output[-2], 1.)
        np.testing.assert_array_equal(output[:439], self.observation)

    def test_missing_tracks_do_not_invent_a_forecast(self):
        self.guard.predictive_constraint.tracks = []
        self.guard.predictive_constraint.contacts[:] = np.inf
        self.guard.predictive_constraint.supported_tracks = 0
        output = AlignedRiskHistory().augment(self.observation, self.core, self.guard)
        np.testing.assert_array_equal(output[439:], np.zeros(42))

    def test_reset_clears_both_histories(self):
        history = AlignedRiskHistory()
        history.augment(self.observation, self.core, self.guard)
        history.reset()
        self.core.elapsed = self.guard.predictive_constraint.timestamp = 0.
        self.guard.predictive_constraint.tracks = []
        self.guard.predictive_constraint.contacts[:] = np.inf
        self.guard.predictive_constraint.supported_tracks = 0
        output = history.augment(self.observation, self.core, self.guard)
        np.testing.assert_array_equal(output[439:], np.zeros(42))

    def test_v1_bundle_is_rejected_before_loading_weights(self):
        from neural_navigation.aligned_release_runtime import AlignedReleaseRuntime
        root = Path(__file__).resolve().parents[3]
        bundle = root/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable'
        with self.assertRaisesRegex(ValueError, 'Aligned v2'):
            AlignedReleaseRuntime(bundle)
        original = json.loads((bundle/'long_risk_observation.json').read_text())
        self.assertNotEqual(original, contract())


if __name__ == '__main__':
    unittest.main()
