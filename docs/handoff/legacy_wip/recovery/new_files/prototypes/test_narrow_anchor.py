import unittest
from types import SimpleNamespace
import numpy as np
from prototypes.anchored_narrow_runtime import feasible_anchor
from neural_navigation.static_footprint import static_footprint_clear

class AnchorTests(unittest.TestCase):
    def test_refines_within_cell_preserving_real_body_margin(self):
        blocked=np.zeros((200,300),bool);blocked[:89]=True;blocked[111:]=True
        m=SimpleNamespace(res=.02,origin=np.zeros(3),width=300,height=200,blocked=blocked)
        p=np.array([3.01,2.03]);body=[.393924904,.329460022]
        contains=lambda xy:2.96<=xy[0]<3.04 and 2.0<=xy[1]<2.08
        self.assertFalse(static_footprint_clear(m,np.r_[p,0],body,.04))
        anchor=feasible_anchor(m,p,body,.04,.08,contains)
        self.assertIsNotNone(anchor);self.assertTrue(contains(anchor))
        self.assertTrue(static_footprint_clear(m,np.r_[anchor,0],body,.04))
        np.testing.assert_array_equal(p,[3.01,2.03])
    def test_no_point_when_corridor_is_too_narrow_or_unknown(self):
        blocked=np.zeros((200,300),bool);blocked[:90]=True;blocked[110:]=True
        m=SimpleNamespace(res=.02,origin=np.zeros(3),width=300,height=200,blocked=blocked)
        p=np.array([3.01,2.03]);body=[.393924904,.329460022]
        self.assertIsNone(feasible_anchor(m,p,body,.04,.08,lambda _:True))
        blocked[:]=True
        self.assertIsNone(feasible_anchor(m,p,body,.04,.08,lambda _:True))
    def test_rejects_candidates_outside_their_cell(self):
        m=SimpleNamespace(res=.02,origin=np.zeros(3),width=300,height=200,blocked=np.zeros((200,300),bool))
        self.assertIsNone(feasible_anchor(m,np.array([3.01,2.03]),[.393924904,.329460022],.04,.08,lambda _:False))

if __name__=='__main__':unittest.main()
