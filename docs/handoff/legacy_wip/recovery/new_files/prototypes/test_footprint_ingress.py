import unittest
from types import SimpleNamespace
import numpy as np
from prototypes.footprint_ingress import rectangular_segment_clear
from neural_navigation.static_footprint import static_footprint_clear


class IngressTests(unittest.TestCase):
    def test_swept_body_rejects_wall_unknown_and_map_boundary(self):
        blocked=np.zeros((100,100),bool)
        m=SimpleNamespace(res=.02,origin=np.zeros(3),width=100,height=100,blocked=blocked)
        self.assertTrue(rectangular_segment_clear(m,[1,1],[1.3,1],0))
        # Occupied or unknown cells use the same blocked field.
        m.blocked[50,60]=True
        self.assertFalse(rectangular_segment_clear(m,[1,1],[1.3,1],0))
        m.blocked[:]=False
        self.assertFalse(rectangular_segment_clear(m,[.1,1],[.3,1],0))
        self.assertFalse(rectangular_segment_clear(m,[1,1],[1.6,1],0))

    def test_circle_overlap_does_not_imply_rectangle_collision(self):
        blocked=np.zeros((100,100),bool);blocked[:,70:]=True
        m=SimpleNamespace(res=.02,origin=np.zeros(3),width=100,height=100,blocked=blocked)
        # Center-wall gap .26: smaller than the planning radius but enough
        # for the real body's width at yaw pi/2 plus the 4 cm safety margin.
        pose=np.array([1.14,1,np.pi/2])
        self.assertTrue(static_footprint_clear(m,pose,[.393924904,.329460022],.04))
        self.assertTrue(rectangular_segment_clear(m,pose[:2],[1.04,1],pose[2]))
        self.assertFalse(rectangular_segment_clear(m,pose[:2],[1.25,1],pose[2]))


if __name__=='__main__':unittest.main()
