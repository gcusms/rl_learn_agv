"""Two manually controlled columns: safe spawn, persistent hold and physical drag."""
import unittest
import numpy as np
import torch
from neural_navigation.waypoint_interference import build_scene,WaypointInterferenceEnv
from navrl.geometry import ROOT

class ManualColumnsTests(unittest.TestCase):
    def test_only_two_columns_stay_parked_across_goals(self):
        torch.set_num_threads(2);e=WaypointInterferenceEnv(scene_dir=build_scene(),components=ROOT/'runs/remote_room_100k')
        try:
            e.reset(seed=32100,options={'task':0});positions=np.array(e._humans())
            self.assertEqual(len(positions),2);self.assertGreater(positions[0,1],4);self.assertLess(positions[1,1],4);self.assertGreater(positions[1,0],5)
            doors=np.array([[2.5,4],[7.5,4],[5,4]])
            self.assertGreater(np.linalg.norm(positions[:,None]-doors,axis=2).min(),1.5)
            for j in range(4):
                before=e.data.qpos.copy();e.select_goal(j);np.testing.assert_array_equal(before,e.data.qpos)
                e.set_manual_mode(False);self.assertTrue(e.manual_mode)
                for _ in range(10):e.step([0,0])
                np.testing.assert_allclose(e._humans(),positions,atol=.005)
            self.assertEqual(e.hard_guard_activations,0)
        finally:e.close()

    def test_drag_is_continuous_and_stops_before_robot(self):
        torch.set_num_threads(2);e=WaypointInterferenceEnv(scene_dir=build_scene(),components=ROOT/'runs/remote_room_100k')
        try:
            e.reset(seed=32100,options={'task':0});positions=np.array(e._humans());before=e.data.qpos.copy();target=positions[0]+[-.7,0]
            e.drag_column(0,target);np.testing.assert_array_equal(e.data.qpos,before);previous=positions.copy()
            for _ in range(25):
                _,_,_,_,info=e.step([0,0]);current=np.array(e._humans())
                self.assertLess(np.linalg.norm(current-previous,axis=1).max(),.15);previous=current
                self.assertGreaterEqual(info['minimum_dynamic_gap_m'],-.001)
            self.assertLess(np.linalg.norm(e._humans()[0]-target),.1)
            np.testing.assert_allclose(e._humans()[1],positions[1],atol=.005)
            e.drag_column(1,e.truth[:2])
            for _ in range(30):e.step([0,0])
            self.assertGreaterEqual(e.robot_gap(e._humans()[1]),.1);self.assertEqual(e.hard_guard_activations,0)
            self.assertFalse(e._info('running')['supervisor_override'])
        finally:e.close()

if __name__=='__main__':unittest.main(verbosity=2)
