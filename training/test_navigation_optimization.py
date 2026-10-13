import unittest
from unittest.mock import patch
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.room_env import RoomNavigationEnv
from train_room_navigation_v3 import TrainingMission


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env=RoomNavigationEnv(components=ROOT/'runs/remote_room_100k',scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=False)
        self.env.reset(seed=41001,options={'start':[1.2,1.2,0.],'goal':[8.6,1.2]})

    def tearDown(self):
        self.env.close()

    def test_missing_scans_preserve_pose_and_accumulate_uncertainty(self):
        loc=self.env.localizer
        loc.reset([1.2,1.2,0.],np.random.default_rng(1))
        missing=np.full(self.env.lidar.beams,np.inf)
        initial=loc.std.copy()
        with patch.object(loc.network,'forward',side_effect=AssertionError('No evidence must not invoke scorer')):
            np.testing.assert_allclose(loc.update([0.,0.,0.],missing),[1.2,1.2,0.])
            np.testing.assert_allclose(loc.update([.1,0.,0.],missing),[1.3,1.2,0.])
        self.assertTrue(np.all(loc.std>initial))
        self.assertEqual(loc.missing_scan_updates,2)
        self.assertFalse(loc.scan_update_accepted)
        self.assertTrue(np.isfinite(loc.update([.1,0.,0.],self.env.scan)).all())
        self.assertTrue(loc.scan_update_accepted)

    def test_exact_router_cache_matches_uncached_and_invalidates_goal(self):
        env=self.env
        env.router_cache.clear()
        with patch.object(env.router,'forward',wraps=env.router.forward) as forward:
            env._neural_route();first=env.predicted_path.copy();misses=env.router_cache_misses
            env._neural_route()
            self.assertEqual(forward.call_count,1)
            np.testing.assert_array_equal(first,env.predicted_path)
            self.assertEqual(env.router_cache_misses,misses)
            env.goal=np.array([1.2,6.6]);env._neural_route()
            self.assertEqual(forward.call_count,2)

    def test_dynamic_occupancy_is_not_hidden_by_static_cache(self):
        env=self.env
        path=np.array([[1.2,1.2],[1.4,1.2],[1.6,1.2]])
        env.goal=path[-1]
        env.laser_fusion.hard[:]=False
        self.assertTrue(env._route_usable(path))
        env.laser_fusion.hard[env.cell(path[1])]=True
        self.assertFalse(env._route_usable(path))
        env.laser_fusion.hard[:]=False
        self.assertTrue(env._route_usable(path))

    def test_changed_scan_occupancy_invalidates_router_cache(self):
        env=self.env;env.router_cache.clear()
        layer=np.zeros_like(env.free)
        with patch.object(env,'_routing_dynamic_layer',side_effect=lambda:layer.copy()), patch.object(env.router,'forward',wraps=env.router.forward) as forward:
            env._neural_route();env._neural_route()
            self.assertEqual(forward.call_count,1)
            layer[env.cell([2.0,1.2])]=1.
            env._neural_route()
            self.assertEqual(forward.call_count,2)

    def test_penetration_guard_cannot_count_as_success_in_training(self):
        mission=TrainingMission(ROOT/'runs/remote_room_100k',safety_training=True)
        try:
            obs,_=mission.reset(seed=41001)
            def guarded(action):
                mission.active.hard_guard_activations+=1
                return obs,50.,True,False,{'true_goal_distance_m':0.}
            with patch.object(mission.active,'step',side_effect=guarded):
                _,reward,done,truncated,info=mission.step([0.,0.])
            self.assertTrue(done);self.assertFalse(truncated)
            self.assertFalse(info['verified_mission_success'])
            self.assertEqual(mission.index,0);self.assertEqual(mission.verified,[])
            self.assertTrue(info['safety_termination']);self.assertEqual(reward,-25.)
        finally:mission.close()

    def test_time_cost_changes_reward_without_changing_action(self):
        mission=TrainingMission(ROOT/'runs/remote_room_100k',time_cost=.04)
        try:
            obs,_=mission.reset(seed=41001)
            with patch.object(mission.active,'step',return_value=(obs,1.,False,False,{})) as step:
                action=np.array([.5,.2]);_,reward,done,truncated,_=mission.step(action)
            np.testing.assert_array_equal(step.call_args.args[0],action)
            self.assertAlmostEqual(reward,.96)
            self.assertFalse(done);self.assertFalse(truncated)
        finally:mission.close()


if __name__=='__main__':unittest.main(verbosity=2)
