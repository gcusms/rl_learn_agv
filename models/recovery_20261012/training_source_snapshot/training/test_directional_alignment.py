import unittest
from unittest.mock import patch
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.room_env import RoomNavigationEnv
from neural_navigation.models import alignment_features


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env=RoomNavigationEnv(components=ROOT/'runs/remote_room_100k',scene_dir=ROOT/'assets/room_navigation_scripted',dynamic=False)
        self.env.reset(seed=64001,options={'start':[1.2,1.2,0.],'goal':[8.6,1.2]})
    def tearDown(self):self.env.close()

    def test_new_features_preserve_old_evidence_and_missing_scan_shape(self):
        e=self.env
        old,h=alignment_features(e.map,e.lidar,e.estimate,e.scan)
        new,nh=alignment_features(e.map,e.lidar,e.estimate,e.scan,directional=True)
        np.testing.assert_array_equal(new[:,:11],old);np.testing.assert_array_equal(h,nh)
        self.assertEqual(new.shape,(125,15));self.assertTrue(np.isfinite(new).all())
        self.assertTrue(np.all((new[:,11:]>=0)&(new[:,11:]<=1)))
        empty,_=alignment_features(e.map,e.lidar,e.estimate,np.full(e.lidar.beams,np.inf),points=7,directional=True)
        self.assertEqual(empty.shape,(343,15));self.assertTrue(np.isfinite(empty).all())
        np.testing.assert_array_equal(empty[:,11:13],1.);np.testing.assert_array_equal(empty[:,13:],0.)

    def test_escape_origin_does_not_erase_obstacles_or_static_walls(self):
        e=self.env;e.escape_start=True;e.router_cache.clear()
        sr,sc=e.cell(e.estimate[:2]);layer=np.zeros_like(e.free);layer[sr,sc]=1.
        other=(sr,sc+1);layer[other]=1.
        with patch.object(e,'_routing_dynamic_layer',return_value=layer),patch.object(e.router,'forward',wraps=e.router.forward) as run:
            e._neural_route();inputs=run.call_args.args[0].numpy()
        self.assertTrue(e.start_inflation_overlap)
        self.assertEqual(inputs[0,0,sr,sc],e.free[sr,sc]);self.assertEqual(inputs[0,1,sr,sc],0.)
        self.assertEqual(inputs[(0,0)+other],0.);self.assertEqual(inputs[(0,1)+other],1.)
        self.assertEqual(layer[sr,sc],1.)
        e.free[sr,sc]=0.;e.router_cache.clear()
        with patch.object(e,'_routing_dynamic_layer',return_value=layer),patch.object(e.router,'forward',wraps=e.router.forward) as run:
            e._neural_route();inputs=run.call_args.args[0].numpy()
        self.assertFalse(e.start_inflation_overlap);self.assertEqual(inputs[0,0,sr,sc],0.);self.assertEqual(inputs[0,1,sr,sc],1.)


if __name__=='__main__':unittest.main(verbosity=2)
