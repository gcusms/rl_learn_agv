import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.models import scan_map_agreement
from neural_navigation.room_env import RoomNavigationEnv
from neural_navigation.curriculum import NavigationCurriculum
from neural_navigation.training_distance import TrainingDistanceGraph


class CurriculumTests(unittest.TestCase):
    def test_u_wall_progress_rewards_moving_away_from_euclidean_goal(self):
        free=np.ones((24,24),bool);free[5:18,7]=False;free[5:18,16]=False;free[17,7:17]=False
        m=SimpleNamespace(free=free,res=.1,origin=np.zeros(3),height=24,width=24,
                          cells=lambda xy:(23-int(np.floor(xy[1]/.1)),int(np.floor(xy[0]/.1))))
        xy=lambda row,col:np.array([(col+.5)*.1,(23-row+.5)*.1])
        goal=xy(20,12);inside=xy(12,12);toward_exit=xy(11,12)
        field=TrainingDistanceGraph(m).field(goal)
        self.assertGreater(np.linalg.norm(toward_exit-goal),np.linalg.norm(inside-goal))
        self.assertLess(field.value(toward_exit),field.value(inside))
        loop=[inside,toward_exit,xy(10,12),inside]
        self.assertAlmostEqual(sum(field.value(a)-field.value(b) for a,b in zip(loop[:-1],loop[1:])),0.)

    def test_route_replacement_does_not_create_progress_reward_and_actions_pass_through(self):
        torch.set_num_threads(1);e=NavigationCurriculum(ROOT/'runs/remote_room_100k')
        try:
            obs,_=e.reset(seed=56001);s=e.active;action=np.array([.3,.2])
            def step(executed):
                s.path=np.array([s.truth[:2],s.truth[:2]+[1,0],s.goal]);s.remaining=100.
                return obs,1500.,False,False,{'true_goal_distance_m':10.}
            with patch.object(s,'step',side_effect=step) as spy:
                _,reward,_,_,info=e.step(action)
            np.testing.assert_array_equal(spy.call_args.args[0],action)
            self.assertEqual(info['fixed_geodesic_progress_m'],0.)
            self.assertLess(reward,0.)
            self.assertFalse(info['training_actions_overridden'])
        finally:e.close()

    def test_alignment_support_detects_a_large_heading_mismatch(self):
        torch.set_num_threads(1)
        e=RoomNavigationEnv(components=ROOT/'runs/remote_room_100k',scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=False)
        try:
            e.reset(seed=41001,options={'start':[1.2,1.2,0.],'goal':[8.6,1.2]})
            good=scan_map_agreement(e.static_map,e.lidar,e.truth,e.scan)
            wrong=scan_map_agreement(e.static_map,e.lidar,e.truth+[0,0,.65],e.scan)
            self.assertGreater(good,.9);self.assertLess(wrong,.35)
            self.assertEqual(scan_map_agreement(e.static_map,e.lidar,e.truth,np.full(220,np.nan)),0.)
        finally:e.close()


if __name__=='__main__':unittest.main(verbosity=2)
