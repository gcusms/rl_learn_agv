import unittest
from types import SimpleNamespace
import numpy as np
from prototypes.narrow_passage import swept_segment_clear,rotation_clear

class NarrowGeometryTests(unittest.TestCase):
    def setUp(self):
        blocked=np.zeros((200,300),bool);blocked[:88]=True;blocked[112:]=True
        self.map=SimpleNamespace(res=.02,origin=np.zeros(3),width=300,height=200,blocked=blocked)
        self.body=[.393924904,.329460022]
    def test_straight_fits_but_turn_does_not(self):
        self.assertTrue(swept_segment_clear(self.map,[1,2],[4,2],self.body,.04))
        self.assertFalse(rotation_clear(self.map,[2,2],0,np.pi/2,self.body,.04))
    def test_sweep_rejects_hidden_wall_and_boundary(self):
        self.map.blocked[100,130]=True
        self.assertFalse(swept_segment_clear(self.map,[1,2],[4,2],self.body,.04))
        self.assertFalse(swept_segment_clear(self.map,[.1,2],[1,2],self.body,.04))
    def test_clear_rotation_in_wide_room(self):
        self.map.blocked[:]=False
        self.assertTrue(rotation_clear(self.map,[2,2],0,np.pi,self.body,.04))
        self.assertFalse(rotation_clear(self.map,[.1,2],0,np.pi,self.body,.04))

if __name__=='__main__':unittest.main()
