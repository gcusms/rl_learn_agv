"""A success flag alone must not admit unsafe or initialized-prefix labels."""
import unittest

from experiments.fit_sensor_retention_pair import admitted_row


class SourceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(safe_success=True, physical_arrival=True,
            protocol_valid=True, learned_model_only=True, no_teacher=True,
            collisions=0, timeout=False, false_arrival=False, physical_guards=0)

    def test_complete_safe_autonomous_source_admitted(self):
        admitted_row(self.row)

    def test_each_unsafe_outcome_overrides_success_flag(self):
        for field in ('collisions', 'timeout', 'false_arrival', 'physical_guards'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                admitted_row(dict(self.row, **{field: True}))

    def test_initialized_recovery_is_not_full_mission_source(self):
        for field in ('recovery_test', 'not_a_full_mission_score'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                admitted_row(dict(self.row, **{field: True}))

    def test_missing_evidence_cannot_default_to_safe(self):
        for field in self.row:
            incomplete = dict(self.row)
            incomplete.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                admitted_row(incomplete)


if __name__ == '__main__':
    unittest.main()
