import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.runtime import EmergencyStop
from prototypes.narrow_observation import fine_masks, FINE_COMMANDS


class FineProbeTests(unittest.TestCase):
    def setUp(self):
        self.core=SimpleNamespace(scan=np.full(30,3.),
            lidar=SimpleNamespace(range_min=.1,range_max=12.,angles=np.linspace(-.5,.5,30),mount=np.zeros(4)),
            localizer=SimpleNamespace(std=np.zeros(3)),odometry_sanity_failure=False,
            diagnostics=lambda:dict(route_complete=True),estimate=np.zeros(3),
            static_map=SimpleNamespace(free=True,lookup=lambda *args,**kwargs:True))
        self.guard=EmergencyStop();self.guard.reverse_started=123.

    def test_same_native_guard_accepts_exact_commands_without_mutating_it(self):
        allowed=fine_masks(self.core,self.guard,1.)
        self.assertTrue(allowed.all());self.assertEqual(self.guard.reverse_started,123.)
        for command in FINE_COMMANDS:
            actual,_=EmergencyStop().check(self.core,command,1.,1.)
            np.testing.assert_array_equal(actual,command)

    def test_missing_stale_or_odometry_jump_blocks_all_fine_actions(self):
        self.assertFalse(fine_masks(self.core,self.guard,1.,scan_valid=False).any())
        self.assertFalse(fine_masks(self.core,self.guard,1.,received_timestamp=2.).any())
        self.core.odometry_sanity_failure=True
        self.assertFalse(fine_masks(self.core,self.guard,1.).any())
        self.assertEqual(self.guard.reverse_started,123.)

if __name__=='__main__':unittest.main()
