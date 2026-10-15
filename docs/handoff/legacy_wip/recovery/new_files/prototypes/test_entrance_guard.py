import unittest
from types import SimpleNamespace
import numpy as np
from navrl.map import StaticMap
from neural_navigation.runtime import EmergencyStop
from neural_navigation.static_footprint import static_footprint_clear
from prototypes.entrance_guard import retains_alignment_space


class EntranceReserveTests(unittest.TestCase):
    def setup_case(self,x=1.76,yaw=.08):
        guard=EmergencyStop(memory_config=dict(static_footprint_schema='oriented_rectangle_v1',static_footprint_margin_m=.04))
        m=StaticMap('training/runs/narrow_long_50000_v2_20261013/scenes/gap_150/decision_map.yaml')
        core=SimpleNamespace(static_map=m,estimate=np.array([x,2.,yaw]),measured_velocity=np.zeros(2),
            corridor_alignment_feature=np.array([1.,.24,0.,-yaw/.35,.16,.16]))
        return guard,core

    def test_currently_clear_but_cornering_space_losing_forward_command_is_rejected(self):
        guard,core=self.setup_case()
        self.assertTrue(static_footprint_clear(core.static_map,core.estimate,guard.half*2,.04))
        self.assertFalse(retains_alignment_space(guard,core,np.array([.05,0.]),0.))

    def test_early_approach_and_aligned_entry_remain_available(self):
        for x,yaw in [(1.5,.08),(1.85,0.),(2.5,0.)]:
            guard,core=self.setup_case(x,yaw)
            self.assertTrue(retains_alignment_space(guard,core,np.array([.05,0.]),0.))

    def test_turns_not_invented_or_hidden_by_forward_reserve(self):
        guard,core=self.setup_case()
        for w in [-.2,-.1,0.,.1,.2]:
            self.assertTrue(retains_alignment_space(guard,core,np.array([0.,w]),0.))

if __name__=='__main__':unittest.main()
