import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.laser_fusion import LaserMapFusion
from neural_navigation.routing_contract import soft_forecast_contract


def environment(config):
    env=SimpleNamespace(grid_h=20,grid_w=20,grid_res=.2,free=np.ones((20,20)),elapsed=1.,
        estimate=np.array([1.01,1.01,0.]),scan=np.array([.5]),expiry=np.zeros((20,20)),
        routing_contract=config,dynamic_memory_seconds=8.,
        lidar=SimpleNamespace(mount=np.zeros(4),angles=np.array([0.]),range_min=.1,range_max=12.,angle_min=-.2,angle_max=.2),
        localizer=SimpleNamespace(scan_update_accepted=True,map_agreement=1.,std=np.array([.01,.01,.01])))
    env.xy=lambda cell:np.array([cell[1]+.5,cell[0]+.5])*.2
    env.cell=lambda xy:(int(xy[1]/.2),int(xy[0]/.2))
    env.static_map=SimpleNamespace(distance=np.ones((80,80)),lookup=lambda xy,*a,**kw:np.ones(len(xy)),
        cells=lambda xy:(np.floor(xy[:,1]/.05).astype(int),np.floor(xy[:,0]/.05).astype(int)))
    fusion=LaserMapFusion(env)
    fusion.predictor=SimpleNamespace(update=lambda *a:np.array([[2.6,2.6]]))
    return env,fusion


class SoftForecastRouteTests(unittest.TestCase):
    def test_measured_cells_remain_hard_while_future_only_cells_have_soft_cost(self):
        env,fusion=environment(soft_forecast_contract());layer=fusion.update(env)
        self.assertEqual(layer[5,7],1.);self.assertTrue(fusion.routing_hard[5,7])
        self.assertTrue(fusion.hard[13,13]);self.assertFalse(fusion.routing_hard[13,13])
        self.assertAlmostEqual(float(layer[13,13]),.02)
        self.assertEqual(fusion.hard_expiry[13,13],0.)

    def test_legacy_future_occupancy_and_persistent_measurements_are_unchanged(self):
        env,fusion=environment(None);layer=fusion.update(env)
        np.testing.assert_array_equal(fusion.routing_hard,fusion.hard)
        self.assertEqual(layer[13,13],1.)
        env.routing_contract=soft_forecast_contract();env.scan[:]=np.nan;env.elapsed=1.2
        layer=fusion.update(env)
        self.assertTrue(fusion.routing_hard[5,7]);self.assertEqual(layer[5,7],1.)


if __name__=='__main__':unittest.main()
