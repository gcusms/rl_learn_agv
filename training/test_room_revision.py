import unittest
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.room_env import RoomNavigationEnv
from neural_navigation.env import LearnedNavigationEnv

class RevisionTests(unittest.TestCase):
    def test_signed_actions_and_truth_free_inputs(self):
        torch.set_num_threads(2);e=RoomNavigationEnv(scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=False)
        try:
            obs,_=e.reset(seed=29000,options={'start':[1.5,1.5,0],'goal':[3.2,2.]})
            self.assertEqual(obs.shape,(235,));before=e.truth.copy()
            obs,_,_,_,info=e.step([-.5,0]);self.assertAlmostEqual(info['commanded_velocity'][0],-.04)
            self.assertLess(e.truth[0],before[0]);self.assertFalse(info['supervisor_override'])
            a=e._observation(e.scan,initialize=True);e.truth=np.array([8,7,2.]);b=e._observation(e.scan,initialize=True);np.testing.assert_array_equal(a,b)
        finally:e.close()
    def test_completed_route_prefix_cannot_be_selected_again(self):
        torch.set_num_threads(2);e=RoomNavigationEnv(scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=False)
        try:
            def forbidden(*args,**kwargs):raise AssertionError('Classical robot planning invoked')
            e.static_map.plan=forbidden
            e.reset(seed=29002,options={'start':[1.5,1.5,0],'goal':[3.2,2.]})
            e.predicted_path=np.array([[1.5,1.5],[1.7,1.5],[1.9,1.5],[2.1,1.5]])
            e.path=e.predicted_path.copy();e.routing_ready=True;e.last_route_time=e.elapsed
            e.estimate=np.array([1.9,1.5,0.]);e._observation(e.scan)
            np.testing.assert_array_equal(e.path,np.array([[1.9,1.5],[2.1,1.5]]))
            e.estimate=np.array([1.65,1.5,0.]);e._observation(e.scan)
            self.assertGreaterEqual(e.path[:,0].min(),1.9)
        finally:e.close()
    def test_neural_grid_centers_match_free_map_samples(self):
        torch.set_num_threads(2);e=RoomNavigationEnv(scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=False)
        try:
            cells=np.argwhere(e.free>0)
            points=np.array([e.xy(tuple(c)) for c in cells])
            self.assertTrue(np.all(e.static_map.lookup(points,e.static_map.free,outside=False)))
            for cell,point in zip(cells,points):self.assertEqual(e.cell(point),tuple(cell))
        finally:e.close()
    def test_partial_fov_keeps_unseen_obstacle_and_clears_visible_free_cell(self):
        torch.set_num_threads(2);e=RoomNavigationEnv(scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=False)
        try:
            e.reset(seed=29003,options={'start':[1.5,1.5,0],'goal':[3.2,2.]})
            unseen=e.cell([.7,1.5]);visible=e.cell([2.5,1.5]);e.expiry[unseen]=6.;e.expiry[visible]=6.
            e._neural_route()
            self.assertGreater(e.expiry[unseen],e.elapsed);self.assertEqual(e.expiry[visible],0.)
            self.assertEqual(e.dynamic_memory_seconds,8.)
        finally:e.close()
    def test_parent_policy_action_semantics_remain_compatible(self):
        torch.set_num_threads(2);e=LearnedNavigationEnv(scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=False)
        try:
            obs,_=e.reset(seed=29001,options={'start':[1.5,1.5,0],'goal':[3.2,2.]});self.assertEqual(obs.shape,(231,))
            _,_,_,_,info=e.step([-.5,0]);np.testing.assert_array_equal(info['commanded_velocity'],[0,0])
        finally:e.close()

if __name__=='__main__':unittest.main(verbosity=2)
