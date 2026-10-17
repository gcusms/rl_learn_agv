import unittest

import numpy as np

from causal_history import CausalHistory


class PhysicalAgeHistory(unittest.TestCase):
    def test_physical_ages_skip_missing_sample(self):
        history = CausalHistory()
        first = np.zeros(14, np.float32)
        first[0] = .8
        history.append(first, 0.)
        frame = np.zeros(14, np.float32)
        frame[0] = .4
        result = history.append(frame, .4)
        self.assertEqual(result[0], first[0])
        self.assertEqual(result[14], 0.)
        self.assertEqual(result[28], frame[0])

    def test_reset_prevents_episode_leakage(self):
        history = CausalHistory()
        history.append(np.ones(14, np.float32), 1.)
        with self.assertRaises(ValueError):
            history.append(np.zeros(14), 0.)
        history.reset()
        np.testing.assert_array_equal(history.append(np.zeros(14), 0.), np.zeros(42))


if __name__ == '__main__':
    unittest.main()
