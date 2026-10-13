"""Check that scheduled column release uses real continuous physics."""
import unittest
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.env import LearnedNavigationEnv
from navrl.scenarios import configure_column_task

class ColumnTests(unittest.TestCase):
    def test_timed_release_moves_column_continuously_without_driving_robot(self):
        torch.set_num_threads(4)
        e=LearnedNavigationEnv(scene_dir=ROOT/'assets/column_navigation_tasks/columns_00',dynamic=True)
        try:
            e.reset(seed=26000,options={'task':0});configure_column_task(e,0)
            w=e.walkers[0];w['block']['trigger_distance_m']=100.;w['block']['hold_seconds']=.20
            point=e._humans()[0].copy()
            _,_,_,_,info=e.step([0,0])
            np.testing.assert_allclose(e._humans()[0],point,atol=.005)
            self.assertEqual(info['column_block_events'][0]['phase'],'holding')
            positions=[e._humans()[0].copy()]
            for _ in range(14):
                _,_,_,_,info=e.step([0,0]);positions.append(e._humans()[0].copy())
                self.assertFalse(info['supervisor_override'])
                np.testing.assert_allclose(info['commanded_velocity'],[0,0])
            steps=np.linalg.norm(np.diff(positions,axis=0),axis=1)
            self.assertGreater(np.linalg.norm(positions[-1]-point),.25)
            self.assertLess(steps.max(),.22)
            self.assertIn(info['column_block_events'][0]['phase'],['releasing','cleared'])
            self.assertGreaterEqual(info['minimum_dynamic_gap_m'],-.001)
        finally:e.close()

if __name__=='__main__':unittest.main(verbosity=2)
