import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from navrl.map import StaticMap
from prototypes.narrow_passage import swept_segment_clear,rotation_clear
from prototypes.entrance_staging import connect_proposal,entrance_connection


class EntranceStagingTests(unittest.TestCase):
    def core(self):
        m=StaticMap('training/runs/narrow_long_50000_v2_20261013/scenes/gap_150/decision_map.yaml')
        c=SimpleNamespace(static_map=m,footprint=np.array([.393924904,.329460022]),margin=.04,
            estimate=np.array([1.40,2.047,.02]),scan=np.ones(220),lidar=object())
        c._segment_clear=lambda a,b:swept_segment_clear(m,a,b,c.footprint,c.margin)
        return c

    def test_offset_entrance_gets_a_swept_and_rotation_checked_approach_corner(self):
        c=self.core();proposal=[c.estimate[:2].copy(),*[np.array([x,2.003333 if x>=1.82 else 2.043]) for x in np.arange(1.50,5.01,.08)],np.array([5.,2.])]
        baseline,index=connect_proposal(c,proposal)
        f=np.array([1.,.24,-.047/.25,-.02/.35,.16,.16])
        with patch('prototypes.narrow_observation.corridor_features',return_value=f):
            path=entrance_connection(c,proposal,baseline,index)
        self.assertIsNotNone(c.entrance_staging_point)
        self.assertLess(c.entrance_staging_point[0],1.75)
        np.testing.assert_allclose(path[-1],[5.,2.])
        yaw=c.estimate[2]
        for a,b in zip(path[:-1],path[1:]):
            heading=np.arctan2((b-a)[1],(b-a)[0])
            self.assertTrue(c._segment_clear(a,b))
            self.assertTrue(rotation_clear(c.static_map,a,yaw,heading,c.footprint,c.margin));yaw=heading

    def test_missing_wall_support_cannot_create_staging_route(self):
        c=self.core();proposal=[c.estimate[:2].copy(),np.array([2.,2.]),np.array([5.,2.])]
        baseline,index=connect_proposal(c,proposal)
        with patch('prototypes.narrow_observation.corridor_features',return_value=np.zeros(6)):
            path=entrance_connection(c,proposal,baseline,index)
        np.testing.assert_array_equal(path,baseline);self.assertIsNone(c.entrance_staging_point)

    def test_scan_blocked_connections_cannot_create_staging_route(self):
        c=self.core();c._segment_clear=lambda a,b:False
        proposal=[c.estimate[:2].copy(),np.array([2.,2.]),np.array([5.,2.])]
        baseline,index=connect_proposal(c,proposal)
        f=np.array([1.,.24,0.,0.,.16,.16])
        with patch('prototypes.narrow_observation.corridor_features',return_value=f):
            path=entrance_connection(c,proposal,baseline,index)
        np.testing.assert_array_equal(path,baseline);self.assertIsNone(c.entrance_staging_point)

if __name__=='__main__':unittest.main()
