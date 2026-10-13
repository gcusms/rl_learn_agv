import copy
import json
import subprocess
import sys
import unittest
from collections import deque
import numpy as np
import torch
from navrl.geometry import ROOT,LidarProfile
from navrl.map import StaticMap,segment_is_free
from neural_navigation.core import NavigationCore,PortablePolicy
from neural_navigation.room_env import RoomNavigationEnv


class CoreTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_core_import_does_not_load_simulator_or_training_library(self):
        script="import sys;from neural_navigation.core import NavigationCore;assert not any(n in sys.modules for n in ['mujoco','gymnasium','stable_baselines3','rclpy'])"
        subprocess.run([sys.executable,'-c',script],check=True)

    def test_runtime_matches_physics_observations_given_same_sensor_state(self):
        components=ROOT/'runs/remote_room_100k'
        env=RoomNavigationEnv(components=components,scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=False)
        core=NavigationCore(StaticMap(ROOT/'assets/room_navigation_scripted/decision_map.yaml',radius=.3),LidarProfile(),components)
        try:
            env.reset(seed=52001,options={'start':[1.2,1.2,0.],'goal':[8.6,1.2]})
            core.reset([1.2,1.2,0.],env.goal)
            # Snapshot only sensor-derived navigation state, never physics truth.
            names=['estimate','elapsed','odom','measured_velocity','last_action','scan','history','pose_progress',
                   'expiry','router_cache','router_cache_hits','router_cache_misses','last_route_time','last_route_switch',
                   'plan_revision','routing_ready','recent_progress','shortcut_candidate','shortcut_since','replan_reason',
                   'route_retained','path','predicted_path','previous_remaining','remaining','static_probability','current_probability']
            for name in names:setattr(core,name,copy.deepcopy(getattr(env,name)))
            for name in ['pose','std','last_odom','valid_scan_fraction','scan_update_accepted','missing_scan_updates']:
                setattr(core.localizer,name,copy.deepcopy(getattr(env.localizer,name)))
            for name in ['first_seen','last_seen','hard_expiry','hard','layer','cleared','far_pending']:
                setattr(core.laser_fusion,name,copy.deepcopy(getattr(env.laser_fusion,name)))
            core._first_observation=False
            for _ in range(12):
                action=np.array([.6,.15]);obs,_,_,_,_=env.step(action)
                core.record_action(action)
                actual,info=core.update(env.scan,env.odom,env.measured_velocity,env.elapsed)
                np.testing.assert_array_equal(actual,obs)
                np.testing.assert_array_equal(core.path,env.path)
                self.assertFalse(info['runtime_robot_astar'])
            self.assertFalse(hasattr(core,'truth'));self.assertFalse(hasattr(core,'data'))
        finally:env.close()

    def test_supercover_disallows_corner_cutting(self):
        free=np.ones((4,4),bool);free[2,2]=False
        self.assertFalse(segment_is_free([1.5,1.5],[2.5,2.5],np.zeros(3),1.,free))
        free[2,2]=True
        self.assertTrue(segment_is_free([1.5,1.5],[2.5,2.5],np.zeros(3),1.,free))

    def test_shortcuts_shorten_route_without_cutting_dynamic_or_static_cells(self):
        components=ROOT/'runs/remote_room_100k'
        core=NavigationCore(StaticMap(ROOT/'assets/room_navigation_scripted/decision_map.yaml',radius=.3),LidarProfile(),components,route_mode='shortcut')
        core.reset([1.2,1.2,0.],[2.2,2.2])
        path=np.array([[1.2,1.2],[1.2,1.8],[1.2,2.2],[1.8,2.2],[2.2,2.2]])
        result=core._shortcut_route(path)
        length=lambda p:np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
        self.assertLess(length(result),length(path))
        self.assertTrue(core._route_usable(result))
        core.laser_fusion.hard[core.cell([1.7,1.7])]=True
        result=core._shortcut_route(path)
        self.assertTrue(core._route_usable(result))
        self.assertGreater(length(result),np.sqrt(2))

    def test_runtime_rejects_backward_time_and_invalid_goal(self):
        core=NavigationCore(StaticMap(ROOT/'assets/room_navigation_scripted/decision_map.yaml',radius=.3),LidarProfile(),ROOT/'runs/remote_room_100k')
        core.reset([1.2,1.2,0.],[2.2,2.2],timestamp=1.)
        with self.assertRaises(ValueError):core.set_goal([0.,0.])
        with self.assertRaises(ValueError):core.update(np.ones(220),np.zeros(3),np.zeros(2),0.)


if __name__=='__main__':unittest.main(verbosity=2)
