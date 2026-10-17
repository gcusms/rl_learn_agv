"""Prevent unsafe or hardware-mismatched teacher labels from becoming training data."""
import unittest

import numpy as np

from curate_v2 import conflicts, eligible


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(hardware='original_memory', safe_success=True, physical_arrival=True,
            false_arrival=False, collisions=0, physical_guards=0, minimum_dynamic_gap_m=.06,
            final_xy_error_m=.1, post_release_seconds=22., original_prefix_sensor_and_physics_exact=True,
            snapshot_two_restores_exact=True)

    def test_complete_safe_original_can_be_considered(self):
        self.assertTrue(eligible(self.row))

    def test_same_path_on_alternative_hardware_requires_separate_data(self):
        self.assertFalse(eligible(dict(self.row, hardware='three_frame_memory')))

    def test_safety_and_arrival_evidence_cannot_be_omitted(self):
        for key in ('false_arrival', 'collisions', 'physical_guards', 'original_prefix_sensor_and_physics_exact'):
            row = self.row.copy()
            del row[key]
            self.assertFalse(eligible(row), key)
        for changed in (dict(minimum_dynamic_gap_m=.03), dict(final_xy_error_m=.3),
                dict(minimum_dynamic_gap_m=float('nan')), dict(post_release_seconds=35.),
                dict(physical_guards=1), dict(safe_success=False)):
            self.assertFalse(eligible(dict(self.row, **changed)))

    def test_conflicting_actions_for_exact_input_are_reported(self):
        x = np.ones((3, 625), np.float32)
        self.assertEqual(conflicts(x, [0, 0, 0]), [])
        found = conflicts(x, [0, 0, 8])
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0]['first_action'], found[0]['second_action']), (0, 8))


if __name__ == '__main__':
    unittest.main()
