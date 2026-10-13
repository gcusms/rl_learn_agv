import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.guarded_curriculum import GuardedNavigationCurriculum,footprint_scan_penalty
from neural_navigation.runtime import SensorNavigationRuntime


class GuardedCurriculumTests(unittest.TestCase):
    def test_clearance_reward_measures_body_not_sensor_and_ignores_unknown(self):
        from types import SimpleNamespace
        lidar=SimpleNamespace(range_min=.1,range_max=12.,angles=np.array([0.]),mount=np.array([.118,0.,0.,0.]))
        far,_=footprint_scan_penalty(np.array([.5]),lidar,[.394,.330],1.)
        near,gap=footprint_scan_penalty(np.array([.12]),lidar,[.394,.330],1.)
        self.assertEqual(far,0.);self.assertAlmostEqual(gap,.041);self.assertGreater(near,.5)
        self.assertEqual(footprint_scan_penalty(np.array([np.nan]),lidar,[.394,.330],1.),(0.,None))
        self.assertEqual(footprint_scan_penalty(np.array([.12]),lidar,[.394,.330],0.)[0],0.)
    def setUp(self):
        torch.set_num_threads(1)
        self.components=ROOT/'runs/optimization_stage3_20261011/portable_forecast_directional'
        self.env=GuardedNavigationCurriculum(self.components,stage='static')
        self.env.scenes=[ROOT/'assets/room_navigation_scripted']
    def tearDown(self):self.env.close()

    def test_training_uses_identical_deployed_core_and_executed_action_history(self):
        observation,_=self.env.reset(seed=90001);e=self.env.active;source=json.loads(self.env.training_case_json)
        runtime=SensorNavigationRuntime(self.components,e.scene_dir/'decision_map.yaml',source['options']['start'],source['options']['goal'],'shortcut')
        reference,_=runtime.core.update(e.scan,e.observed_odometry,e.measured_velocity,e.elapsed)
        np.testing.assert_array_equal(observation,reference)
        from neural_navigation.runtime import execute_policy_action
        for proposed in [np.array([-.8,.4]),np.array([.6,.2]),np.array([0.,-.4])]:
            result=execute_policy_action(runtime.core,runtime.guard,proposed,e.elapsed)
            observation,reward,done,truncated,info=self.env.step(proposed)
            np.testing.assert_allclose(info['commanded_velocity'],result['command'],atol=0,rtol=0)
            expected,_=runtime.core.update(e.scan,e.observed_odometry,e.measured_velocity,e.elapsed)
            np.testing.assert_array_equal(observation,expected)
            np.testing.assert_array_equal(self.env.core.last_action,info['executed_action'])
            self.assertTrue(np.isfinite(reward));self.assertEqual(info['execution_mode'],'shared_sensor_core_and_guard')

    def test_contact_remains_failure_even_if_legacy_environment_reports_success(self):
        observation,_=self.env.reset(seed=90001);e=self.env.active
        original=e.step
        def collide(action):
            obs,reward,done,truncated,info=original(action);e.collision_events+=1;info['true_goal_distance_m']=0.
            return obs,50.,True,False,info
        with patch.object(e,'step',side_effect=collide):
            _,reward,done,truncated,info=self.env.step(np.array([.2,0.]))
        self.assertTrue(done);self.assertFalse(truncated);self.assertFalse(info['verified_mission_success']);self.assertLessEqual(reward,-100.)


if __name__=='__main__':unittest.main(verbosity=2)
