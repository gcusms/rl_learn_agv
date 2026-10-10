"""Check learned routing/localization provenance and actual navigation inputs."""
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.env import LearnedNavigationEnv
from neural_navigation.models import LearnedLocalizer,NeuralRouter


class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.components=Path(os.environ.get('NAV_COMPONENTS',str(ROOT/'runs/neural_components')))
    def test_robot_reset_and_motion_do_not_call_classical_planning(self):
        env=LearnedNavigationEnv(components=self.components,localizer='learned_attention',dynamic=False)
        def forbidden(*args,**kwargs):raise AssertionError('Robot called classical A*')
        try:
            env.static_map.plan=forbidden
            obs,info=env.reset(seed=4,options={'start':[1,1,0],'goal':[3,1]})
            self.assertEqual(obs.shape,(231,));self.assertIsInstance(env.localizer,LearnedLocalizer)
            obs,_,_,_,info=env.step([.5,0]);self.assertFalse(info['supervisor_override'])
            self.assertEqual(info['decision']['planner_source'],'trained_neural_convolutional_router')
            self.assertGreater(info['decision']['static_reachability_probability'],.5)
        finally:env.close()
    def test_user_initial_pose_is_accepted_independently_of_simulator_spawn(self):
        env=LearnedNavigationEnv(components=self.components,dynamic=False)
        try:
            obs,info=env.reset(seed=7,options={'start':[1,1,0],'goal':[3,1],'initial_pose':[1.10,.94,.04]})
            self.assertEqual(info['provided_initial_pose'],[1.10,.94,.04])
            self.assertEqual(obs.shape,(231,))
            self.assertLess(info['localization_error_m'],.20)
        finally:env.close()
    def test_truth_does_not_enter_neural_observation(self):
        env=LearnedNavigationEnv(components=self.components,dynamic=False)
        try:
            env.reset(seed=7,options={'start':[1,1,0],'goal':[3,1]})
            a=env._observation(env.scan,initialize=True);env.truth=np.array([8,7,2.])
            b=env._observation(env.scan,initialize=True);np.testing.assert_array_equal(a,b)
        finally:env.close()
    def test_learned_route_on_open_map_respects_static_wall_clearance(self):
        env=LearnedNavigationEnv(components=self.components,dynamic=False)
        try:
            env.reset(seed=9,options={'start':[1,1,0],'goal':[8.5,6.8]})
            self.assertLess(np.linalg.norm(env.path[-1]-env.goal),.25)
            for a,b in zip(env.path[:-1],env.path[1:]):self.assertTrue(env.static_map.visible(a,b))
        finally:env.close()
    def test_neural_reachability_distinguishes_disconnected_regions(self):
        d=torch.load(self.components/'router.pt',map_location='cpu',weights_only=True)
        model=NeuralRouter(d['iterations']);model.load_state_dict(d['state_dict']);model.eval()
        x=torch.zeros((2,4,40,50));x[:,0,2:38,2:48]=1.;x[:,2,20,40]=1.
        x[1,0,:,25]=0.
        with torch.no_grad():_,_,r=model(x)
        self.assertGreater(float(r[0,0,20,10]),.5);self.assertLess(float(r[1,0,20,10]),.5)


if __name__=='__main__':unittest.main(verbosity=2)
