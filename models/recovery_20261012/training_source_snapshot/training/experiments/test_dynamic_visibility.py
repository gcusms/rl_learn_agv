import unittest
from types import SimpleNamespace
import numpy as np
from audit_dynamic_visibility import score_frame


class DynamicVisibilityTest(unittest.TestCase):
    def setUp(self):
        self.lidar = SimpleNamespace(range_min=.1, range_max=12., mount=np.zeros(4),
                                     angles=np.linspace(-.1, .1, 5), angle_min=-.1, angle_max=.1)
        self.map = SimpleNamespace(distance=None,
                                   lookup=lambda points, field, outside: np.ones(len(points)))
        self.row = dict(scan=[1.]*5, truth=[0, 0, 0], estimate=[0, 0, 0],
                        timestamp=.2, actors=[[1.15, 0.]])
        self.previous = dict(timestamp=0., actors=[[1.13, 0.]])

    def score(self):
        return score_frame(self.row, self.previous, self.lidar, self.map, [.25])[0]

    def test_visible_motion_returns_and_static_subtraction(self):
        result = self.score()
        self.assertTrue(result['moving'])
        self.assertTrue(result['body_intersects_fov'])
        self.assertEqual(result['retained_nonstatic_endpoints'], 5)
        self.map.lookup = lambda points, field, outside: np.zeros(len(points))
        self.assertEqual(self.score()['retained_nonstatic_endpoints'], 0)

    def test_blind_region_and_placement_jump(self):
        self.row['actors'] = [[-1.15, 0.]]
        self.previous['actors'] = [[-1.13, 0.]]
        self.assertFalse(self.score()['body_intersects_fov'])
        self.assertEqual(self.score()['nearby_scan_endpoints'], 0)
        self.previous['actors'] = [[10., 0.]]
        result = self.score()
        self.assertTrue(result['placement_discontinuity'])
        self.assertFalse(result['moving'])


if __name__ == '__main__':
    unittest.main()
