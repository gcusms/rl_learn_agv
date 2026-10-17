"""Safety and regression checks for the opt-in evidence experiment."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from hybrid_memory import HybridFreeEvidenceMemory
from neural_navigation.safety_memory import NearFieldMemory


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.lidar = SimpleNamespace(angles=np.linspace(-.4,.4,5), angle_min=-.4,
            angle_max=.4, range_min=.1, range_max=12., mount=np.zeros(4))
        self.config = dict(resolution_m=.04, near_range_m=1.5, forget_radius_m=2.5,
            max_points=4096, uncertainty_m_s=.002, max_uncertainty_m=.04)
        self.memory = HybridFreeEvidenceMemory(self.config)
        self.core = SimpleNamespace(elapsed=0., odom=np.zeros(3), lidar=self.lidar,
            scan=np.array([np.nan,np.nan,.5,np.nan,np.nan]))
        self.memory.update(self.core)
        self.key = next(iter(self.memory.entries))

    def observe(self, scan, timestamp, pose=None):
        self.core.scan = np.full(5,scan) if np.isscalar(scan) else np.asarray(scan,float)
        self.core.elapsed = timestamp
        if pose is not None:
            self.core.odom = np.asarray(pose,float)
        return self.memory.update(self.core)

    def test_original_obvious_free_ray_clears_immediately(self):
        self.observe(.7,.2)
        self.assertNotIn(self.key,self.memory.entries)

    def test_weaker_evidence_needs_three_distinct_frames(self):
        self.observe(.56,.2)
        self.observe(.56,.4)
        self.assertIn(self.key,self.memory.entries)
        self.observe(.56,.6)
        self.assertNotIn(self.key,self.memory.entries)

    def test_unknown_and_no_return_do_not_clear_or_accumulate(self):
        for scan in [np.nan,np.inf,-np.inf,.05,12.]:
            self.observe(scan,self.core.elapsed+.2)
            self.assertIn(self.key,self.memory.entries)
        self.observe(.56,self.core.elapsed+.2)
        self.assertIn(self.key,self.memory.entries)

    def test_same_timestamp_probes_do_not_accumulate_frames(self):
        for _ in range(5):
            self.observe(.56,.2)
        self.assertIn(self.key,self.memory.entries)
        self.observe(.56,.4)
        self.assertIn(self.key,self.memory.entries)
        self.observe(.56,.6)
        self.assertNotIn(self.key,self.memory.entries)

    def test_unknown_interruption_resets_weaker_evidence(self):
        self.observe(.56,.2)
        self.observe(.56,.4)
        self.observe(np.nan,.6)
        self.observe(.56,.8)
        self.observe(.56,1.)
        self.assertIn(self.key,self.memory.entries)
        self.observe(.56,1.2)
        self.assertNotIn(self.key,self.memory.entries)

    def test_long_gap_does_not_bridge_confirmation(self):
        self.observe(.56,.2)
        self.observe(.56,.4)
        self.observe(.56,1.2)
        self.assertIn(self.key,self.memory.entries)
        self.observe(.56,1.4)
        self.assertIn(self.key,self.memory.entries)
        self.observe(.56,1.6)
        self.assertNotIn(self.key,self.memory.entries)

    def test_reobserved_hit_resets_confirmation_and_age(self):
        self.observe(.56,.2)
        self.observe(.56,.4)
        self.observe([np.nan,np.nan,.5,np.nan,np.nan],.6)
        self.assertEqual(self.memory.entries[self.key][1],.6)
        self.observe(.56,.8)
        self.observe(.56,1.)
        self.assertIn(self.key,self.memory.entries)
        self.observe(.56,1.2)
        self.assertNotIn(self.key,self.memory.entries)

    def test_blind_sector_and_time_do_not_clear(self):
        self.observe(.7,20.,[0.,0.,np.pi])
        self.assertIn(self.key,self.memory.entries)
        self.assertEqual(self.memory.entries[self.key][1],0.)

    def test_odometry_frame_and_uncertainty_are_preserved(self):
        points,uncertainty = self.observe(np.nan,.2,[.1,0.,0.])
        np.testing.assert_allclose(points,[[.4,0.]],rtol=0,atol=1e-12)
        np.testing.assert_allclose(uncertainty,[.0004],rtol=0,atol=1e-12)

    def test_cap_after_combination_matches_original_initial_selection(self):
        config = dict(self.config,max_points=2)
        original,hybrid = NearFieldMemory(config),HybridFreeEvidenceMemory(config)
        self.core.scan = np.linspace(.4,.7,5)
        a = original.update(self.core)
        b = hybrid.update(self.core)
        self.assertEqual(set(original.entries),set(hybrid.entries))
        np.testing.assert_array_equal(a[0],b[0])
        self.assertLessEqual(len(hybrid.entries),2)

    def test_probe_copy_cannot_mutate_live_confirmations(self):
        self.observe(.56,.2)
        shadow = copy.deepcopy(self.memory)
        self.core.elapsed = .4
        shadow.update(self.core)
        self.core.elapsed = .6
        shadow.update(self.core)
        self.assertNotIn(self.key,shadow.entries)
        self.assertIn(self.key,self.memory.entries)
        self.assertEqual(self.memory.free_confirmations[self.key][1],1)

    def test_reset_discards_both_rules_and_confirmation_history(self):
        self.observe(.56,.2)
        self.memory.reset()
        self.assertFalse(self.memory.entries)
        self.assertFalse(self.memory.strong.entries)
        self.assertFalse(self.memory.confirmed.entries)
        self.assertFalse(self.memory.free_confirmations)


if __name__ == '__main__':
    unittest.main()
