import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.route_efficiency import route_cost, efficient_learned_route, sensor_route_efficiency_allowed


class RouteEfficiencyTests(unittest.TestCase):
    def test_unknown_or_moving_obstacle_uses_native_routing(self):
        predictor=SimpleNamespace(tracks=[])
        core=SimpleNamespace(efficient_route_selection=True,elapsed=1.,estimate=np.zeros(3),
            localizer=SimpleNamespace(scan_update_accepted=True,map_agreement=1.,std=np.array([.05,.05,.02])),
            laser_fusion=SimpleNamespace(predictor=predictor))
        self.assertFalse(sensor_route_efficiency_allowed(core))
        core.unmatched_returns=0
        self.assertTrue(sensor_route_efficiency_allowed(core))
        core.unmatched_returns=1
        track=dict(time=1.,center=np.array([1.,0.]),velocity=np.zeros(2),hits=2)
        predictor.tracks=[track]
        self.assertFalse(sensor_route_efficiency_allowed(core))
        track['hits']=3
        self.assertTrue(sensor_route_efficiency_allowed(core))
        track['velocity']=np.array([.1,0.])
        self.assertFalse(sensor_route_efficiency_allowed(core))
        track['velocity']=np.zeros(2)
        core.elapsed=2.;track['time']=2.
        self.assertFalse(sensor_route_efficiency_allowed(core))
        core.elapsed=3.5;track['time']=3.5
        self.assertTrue(sensor_route_efficiency_allowed(core))
        core.localizer.scan_update_accepted=False
        self.assertFalse(sensor_route_efficiency_allowed(core))

    def test_occluded_or_far_track_does_not_prove_stationary_local_scene(self):
        track=dict(time=0.,center=np.array([1.,0.]),velocity=np.zeros(2),hits=3)
        core=SimpleNamespace(efficient_route_selection=True,elapsed=1.,estimate=np.zeros(3),
            localizer=SimpleNamespace(scan_update_accepted=True,map_agreement=1.,std=np.array([.05,.05,.02])),
            laser_fusion=SimpleNamespace(predictor=SimpleNamespace(tracks=[track])))
        self.assertFalse(sensor_route_efficiency_allowed(core))
        track['time']=1.;track['center']=np.array([4.,0.])
        self.assertFalse(sensor_route_efficiency_allowed(core))
        track['center']=np.array([1.,0.]);track['velocity'][0]=np.nan
        self.assertFalse(sensor_route_efficiency_allowed(core))

    def test_full_length_beats_initial_goalward_direction(self):
        position=np.array([0.,0.]);goal=np.array([2.,0.])
        long=np.array([[0,0],[1,0],[1,4],[2,4],[2,0]],float)
        retreat=np.array([[0,0],[-.2,0],[-.2,-.3],[2,-.3],[2,0]],float)
        self.assertLess(route_cost(retreat,position,goal,0),route_cost(long,position,goal,0))

    def test_goalward_direction_breaks_length_tie(self):
        position=np.array([0.,0.]);goal=np.array([2.,0.])
        forward=np.array([[0,0],[1,0],[1,1],[2,1],[2,0]],float)
        sideways=np.array([[0,0],[0,1],[2,1],[2,0]],float)
        self.assertLess(route_cost(forward,position,goal,0),route_cost(sideways,position,goal,0))

    def test_never_selects_invalid_shortcut_and_retains_failure_semantics(self):
        original=np.array([[0,0],[0,1],[2,1],[2,0]],float)
        core=SimpleNamespace(estimate=np.array([0.,0.,0.]),goal=np.array([2.,0.]),
            router=SimpleNamespace(directions=[(-1,0),(1,0),(0,-1),(0,1)]),
            cell=lambda p:(0,int(p[0])),xy=lambda cell:np.array([cell[1],cell[0]],float),
            _shortcut_route=lambda p:p,
            _route_usable=lambda p:np.array_equal(p,original))
        q=np.zeros((4,3,3));q[3]=1
        selected=efficient_learned_route(core,q,np.ones((3,3),bool),original)
        np.testing.assert_array_equal(selected,original)
        core._route_usable=lambda p:False
        np.testing.assert_array_equal(efficient_learned_route(core,q,np.zeros((3,3),bool),original),original)

if __name__ == '__main__': unittest.main()
