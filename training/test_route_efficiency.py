import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.route_efficiency import route_cost, efficient_learned_route


class RouteEfficiencyTests(unittest.TestCase):
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
