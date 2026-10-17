import unittest

import numpy as np

from forecast_alignment import contract, fuse_current_frame


class ForecastEvidenceRegression(unittest.TestCase):
    def test_approaching_object_keeps_known_stop_contact(self):
        # The actual failing trace has a supported long frame with zero stop
        # score while the guard already forecasts contact at0.7seconds.
        frame = np.zeros(14, np.float32)
        frame[12:] = [1., .25]
        contacts = np.full(12, np.nan)
        contacts[0] = .7
        original = frame.copy()
        result = fuse_current_frame(frame, contacts, 2, 10.6, 10.6)
        self.assertAlmostEqual(result[0], 1.-.7/5.2, places=6)
        self.assertEqual(result[12], 1.)
        self.assertEqual(result[13], .5)
        np.testing.assert_array_equal(frame, original)

    def test_earlier_long_contact_is_preserved(self):
        frame = np.zeros(14, np.float32)
        frame[0] = 1.-.2/5.2
        frame[12:] = [1., .25]
        contacts = np.full(12, np.inf)
        contacts[0] = .7
        result = fuse_current_frame(frame, contacts, 1, 0., 0.)
        self.assertEqual(result[0], frame[0])

    def test_retained_guard_evidence_is_not_lost_when_old_long_frame_unknown(self):
        frame = np.zeros(14, np.float32)
        contacts = np.full(12, np.inf)
        contacts[5] = .3
        result = fuse_current_frame(frame, contacts, 1, 4., 4.)
        self.assertAlmostEqual(result[5], 1.-.3/5.2, places=6)
        self.assertEqual(result[12], 1.)
        self.assertEqual(result[0], 0.)

    def test_missing_evidence_does_not_create_validity(self):
        result = fuse_current_frame(np.zeros(14), np.full(12, np.nan), 0, 0., 0.)
        np.testing.assert_array_equal(result, np.zeros(14))

    def test_stale_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            fuse_current_frame(np.zeros(14), np.full(12, np.nan), 0, 1., .8)

    def test_invalid_scan_does_not_promote_retained_forecast(self):
        contacts = np.full(12, np.inf)
        contacts[0] = .5
        result = fuse_current_frame(np.zeros(14), contacts, 1, 0., 0., scan_valid=False)
        np.testing.assert_array_equal(result, np.zeros(14))

    def test_invalid_evidence_rejected(self):
        for contact, count in [(-.1, 1), (1.3, 1), (.2, 0), (-np.inf, 1)]:
            contacts = np.full(12, np.inf)
            contacts[0] = contact
            with self.assertRaises(ValueError):
                fuse_current_frame(np.zeros(14), contacts, count, 0., 0.)

    def test_contract_declares_semantic_change_and_separate_training(self):
        value = contract()
        self.assertNotEqual(value['schema'], 'long_action_risk_history_v1')
        self.assertTrue(value['requires_retraining_and_versioned_export'])
        self.assertTrue(value['original_current_guard_masks_and_stop_permission_unchanged'])


if __name__ == '__main__':
    unittest.main()
