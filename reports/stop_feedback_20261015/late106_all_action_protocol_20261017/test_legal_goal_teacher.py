import unittest
import numpy as np

from legal_goal_teacher import select_current_legal


def mask(*choices):
    result = np.zeros(12, bool)
    result[[0, *choices]] = True
    return result


class LegalGoalTeacherTests(unittest.TestCase):
    def test_front_target_uses_available_forward_speed(self):
        self.assertEqual(select_current_legal(mask(1, 2, 3, 4, 5, 8, 9), [1., 0.], [0., 0.]), 1)

    def test_pure_turn_direction_follows_target(self):
        self.assertEqual(select_current_legal(mask(4, 5), [0., 1.], [0., 0.]), 4)
        self.assertEqual(select_current_legal(mask(4, 5), [0., -1.], [0., 0.]), 5)

    def test_all_legal_directions_remain_available(self):
        self.assertEqual(select_current_legal(mask(9), [1., .2], [0., 0.]), 9)
        self.assertEqual(select_current_legal(mask(8), [1., .2], [0., 0.]), 8)

    def test_stop_only_is_never_overridden(self):
        self.assertEqual(select_current_legal(mask(), [1., 0.], [0., 0.]), 0)

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            select_current_legal(mask(1), [np.nan, 0.], [0., 0.])


if __name__ == '__main__':
    unittest.main()
