import unittest
from types import SimpleNamespace
import numpy as np
from navrl.geometry import LidarProfile
from experiments.evaluate_route_recovery import BoundedRouteRecoveryGuard


class RecoveryTests(unittest.TestCase):
    def core(self):
        lidar=LidarProfile()
        return SimpleNamespace(scan=np.full(lidar.beams,2.),lidar=lidar,elapsed=1.,
            estimate=np.zeros(3),odom=np.zeros(3),odometry_sanity_failure=False,
            localizer=SimpleNamespace(std=np.array([.02,.02,.01])),
            static_map=SimpleNamespace(free=np.ones((1,1)),lookup=lambda *a,**kw:True),
            diagnostics=lambda:{'route_complete':False})

    def test_limit_preserves_model_turn_and_real_safety_checks(self):
        core=self.core();guard=BoundedRouteRecoveryGuard(memory_config={'near_range_m':1.5})
        cmd,reason=guard.check(core,[.2,.25],1.,1.)
        np.testing.assert_allclose(cmd,[.05,.25]);self.assertEqual(reason,'learned_route_incomplete')
        self.assertFalse(core.diagnostics()['route_complete'])
        cmd,reason=guard.check(core,[.05,0.],1.,1.)
        np.testing.assert_allclose(cmd,[.05,0.]);self.assertIsNone(reason)
        for age,scan_valid in [(1.,True),(0.,False)]:
            cmd,reason=guard.check(core,[.2,0.],1.,1.+age,scan_valid=scan_valid)
            np.testing.assert_array_equal(cmd,[0.,0.]);self.assertIn(reason,['stale_or_future_scan','scan_unavailable'])
        core.static_map.lookup=lambda *a,**kw:False
        cmd,reason=guard.check(core,[.2,0.],1.,1.)
        np.testing.assert_array_equal(cmd,[0.,0.]);self.assertEqual(reason,'static_braking_footprint')

    def test_reverse_and_nonfinite_inputs_do_not_gain_permission(self):
        core=self.core();guard=BoundedRouteRecoveryGuard(memory_config={'near_range_m':1.5})
        cmd,_=guard.check(core,[-.08,0.],1.,1.)
        np.testing.assert_array_equal(cmd,[0.,0.])
        cmd,reason=guard.check(core,[np.inf,0.],1.,1.)
        np.testing.assert_array_equal(cmd,[0.,0.]);self.assertEqual(reason,'nonfinite_input')


if __name__=='__main__':unittest.main()
