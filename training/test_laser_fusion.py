"""Real MuJoCo laser regressions for blocked/cleared passages and distant returns."""
import json
import unittest
import mujoco
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.waypoint_interference import build_scene,WaypointInterferenceEnv

class LaserFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.e=WaypointInterferenceEnv(scene_dir=build_scene(),components=ROOT/'runs/remote_room_100k')
        self.e.reset(seed=32110,options={'start':[3.,2.8,0.],'goal':[8.05,1.7]})
        self.e.lidar.noise=0.;self.e.lidar.dropout=0.
        # Exact pose isolates map/scan fusion from localization error in this test.
        self.e.estimate=self.e.truth.copy()
        self.e.laser_fusion.reset();self.e.expiry[:]=0.;self.e.routing_ready=False
        self.place([[5.,3.5],[3.6,6.8]])
        self.e.static_map.plan=lambda *args,**kwargs: self.fail('Runtime classical robot planner called')
    def tearDown(self):self.e.close()
    def place(self,positions):
        for name,position in zip(self.e.names[1:],positions):
            q,d=self.e.addresses[name];self.e.data.qpos[q]=position;self.e.data.qvel[d]=0.
        mujoco.mj_forward(self.e.model,self.e.data)
    def observe(self,t):
        self.e.elapsed=t;self.e.scan=self.e._scan();self.e._neural_route()
        path=self.e.predicted_path.copy()
        return dict(time_s=t,length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),reason=self.e.replan_reason,hard_cells=int(self.e.laser_fusion.hard.sum()),far_pending=self.e.laser_fusion.far_pending,cleared_cells=self.e.laser_fusion.cleared,path=path.tolist())
    def test_real_laser_cleared_passage_shortens_route(self):
        blocked=[self.observe(t) for t in [.4,.8,1.2,1.6,2.,2.4]]
        self.assertGreater(blocked[-1]['hard_cells'],0)
        self.place([[8.8,6.8],[3.6,6.8]])
        cleared=[self.observe(t) for t in [2.8,3.2,3.6,4.,4.4,4.8]]
        self.assertTrue(any(x['cleared_cells']>0 for x in cleared))
        self.assertTrue(any(x['reason']=='observed_clear_shorter_route' for x in cleared))
        self.assertLess(cleared[-1]['length_m'],blocked[-1]['length_m']-1.)
        self.assertTrue(self.e._route_usable(self.e.predicted_path))
        output=ROOT/'runs/map_laser_fusion_validation';output.mkdir(exist_ok=True)
        (output/'report.json').write_text(json.dumps(dict(note='Physical lidar routing regression with exact pose, not policy success-rate evaluation.',blocked=blocked,cleared=cleared),indent=2))
    def test_distant_single_observation_is_not_hard_blockage(self):
        self.e.truth=np.array([1.,1.,0.]);self.e._sync();self.e.estimate=self.e.truth.copy()
        self.place([[4.,1.2],[8.8,6.8]])
        first=self.observe(.4)
        self.assertGreater(first['far_pending'],0)
        self.assertEqual(first['hard_cells'],0)
        original=self.e.predicted_path.copy()
        second=self.observe(.8)
        self.assertEqual(second['hard_cells'],0)
        np.testing.assert_array_equal(original,self.e.predicted_path)
        self.place([[8.8,6.8],[8.5,6.8]])
        self.observe(1.2)
        self.assertFalse(self.e.laser_fusion.hard.any())
    def test_invalid_returns_do_not_erase_remembered_obstacle(self):
        cell=self.e.cell([3.8,2.8]);self.e.expiry[cell]=6.
        self.e.elapsed=1.;self.e.scan=np.full(len(self.e.lidar.angles),np.inf)
        self.e._routing_dynamic_layer()
        self.assertEqual(self.e.expiry[cell],6.)

if __name__=='__main__':unittest.main(verbosity=2)
