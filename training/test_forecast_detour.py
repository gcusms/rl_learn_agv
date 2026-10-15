"""Forecast-only route changes must not delay measured-obstacle response."""
import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.core import RoomRoutingMixin
from neural_navigation.routing_contract import contract


class Backend:
    def _neural_route(self):
        self.predicted_path=self.candidate.copy();self.plan_revision+=1;self.last_route_time=self.elapsed


class Router(RoomRoutingMixin,Backend):
    def __init__(self):
        self.routing_contract=contract();self.elapsed=0.;self.plan_revision=5;self.routing_ready=True
        self.predicted_path=np.array([[0.,0.],[1.,0.],[2.,0.]])
        self.candidate=np.array([[0.,0.],[-.5,0.],[-.5,1.],[2.,1.],[2.,0.]])
        self.goal=np.array([2.,0.]);self.estimate=np.zeros(3);self.route_mode='raw'
        self.free=np.ones((20,20));self.grid_res=.2
        self.static_map=SimpleNamespace(origin=np.array([-1.,-1.]))
        self.laser_fusion=SimpleNamespace(observed_hard=np.zeros((20,20),bool))
        self.detour_candidate=None;self.detour_since=None;self.static_clear=True
    def _route_usable(self,path):return False  # Forecast cells obstruct the old route.
    def _static_visible(self,a,b):return self.static_clear


class ForecastDetourTests(unittest.TestCase):
    def test_disruptive_forecast_change_is_confirmed_then_accepted(self):
        router=Router();old=router.predicted_path.copy()
        for t in (0.,.4):
            router.elapsed=t;router._neural_route()
            np.testing.assert_array_equal(router.predicted_path,old)
            self.assertEqual(router.plan_revision,5)
            self.assertEqual(router.replan_reason,'confirming_forecast_only_detour')
        router.elapsed=.8;router._neural_route()
        np.testing.assert_array_equal(router.predicted_path,router.candidate)
        self.assertFalse(router.route_retained)

    def test_measured_or_static_obstacle_is_never_delayed(self):
        for kind in ('measured','static'):
            router=Router()
            if kind=='measured':router.laser_fusion.observed_hard[:]=True
            else:router.static_clear=False
            router._neural_route()
            np.testing.assert_array_equal(router.predicted_path,router.candidate)
            self.assertEqual(router.replan_reason,'confirmed_obstacle_or_route_invalid')

    def test_legacy_contract_accepts_original_candidate_immediately(self):
        router=Router();router.routing_contract=None;router._neural_route()
        np.testing.assert_array_equal(router.predicted_path,router.candidate)

    def test_unhashed_route_contract_is_rejected(self):
        import tempfile,json
        from pathlib import Path
        from neural_navigation.goal_pose_runtime import check_files
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'routing_contract.json').write_text(json.dumps(contract()))
            with self.assertRaises(ValueError):check_files(root,{'files':{}})


if __name__=='__main__':unittest.main()
