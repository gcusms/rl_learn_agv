"""Check existing memory clearing under a proposed multi-frame configuration."""
from types import SimpleNamespace
import unittest

import numpy as np

from neural_navigation.safety_memory import NearFieldMemory


class MemoryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.lidar=SimpleNamespace(beams=220,angles=np.linspace(-1.9,1.9,220),
            angle_min=-1.9,angle_max=1.9,range_min=.1,range_max=12.,mount=np.zeros(4))
        self.config=dict(resolution_m=.04,near_range_m=1.5,forget_radius_m=2.5,
            uncertainty_m_s=.002,max_uncertainty_m=.04,free_evidence_margin_m=.03,
            free_evidence_frames=3,free_evidence_max_gap_s=.4)

    def start(self,heading=0.,config=None):
        memory=NearFieldMemory(self.config if config is None else config)
        scan=np.full(220,np.nan);scan[109:112]=.5
        core=SimpleNamespace(elapsed=0.,odom=np.array([0.,0.,heading]),scan=scan,lidar=self.lidar)
        memory.update(core)
        point=np.array([.5*np.cos(self.lidar.angles[110]),.5*np.sin(self.lidar.angles[110])])
        c,s=np.cos(heading),np.sin(heading);world=point@np.array([[c,s],[-s,c]])
        key=tuple(np.floor(world/.04).astype(int))
        self.assertIn(key,memory.entries)
        return memory,core,key

    def feed(self,memory,core,time,scan):
        core.elapsed=time;core.scan=np.full(220,scan)
        return memory.update(core)

    def test_three_distinct_finite_free_frames_clear_old_endpoint(self):
        memory,core,key=self.start()
        for time in [.2,.4]:
            self.feed(memory,core,time,.56);self.assertIn(key,memory.entries)
        self.feed(memory,core,.6,.56)
        self.assertNotIn(key,memory.entries)

    def test_same_timestamp_probe_calls_do_not_count_as_new_frames(self):
        memory,core,key=self.start()
        for _ in range(12):self.feed(memory,core,.2,.56)
        self.assertIn(key,memory.entries)
        self.assertEqual(memory.free_confirmations[key][1],1)

    def test_unknown_and_no_return_are_not_free_evidence(self):
        for scan in [np.nan,np.inf,-np.inf,12.,0.,13.]:
            with self.subTest(scan=scan):
                memory,core,key=self.start()
                for time in [.2,.4,.6,20.,100.]:self.feed(memory,core,time,scan)
                self.assertIn(key,memory.entries)

    def test_occlusion_keeps_the_old_endpoint(self):
        memory,core,key=self.start()
        for time in [.2,.4,.6,1.]:self.feed(memory,core,time,.45)
        self.assertIn(key,memory.entries)

    def test_unknown_interruption_resets_consecutive_confirmations(self):
        memory,core,key=self.start()
        for time,scan in [(.2,.56),(.4,np.nan),(.6,.56),(.8,.56)]:
            self.feed(memory,core,time,scan)
        self.assertIn(key,memory.entries)
        self.feed(memory,core,1.,.56)
        self.assertNotIn(key,memory.entries)

    def test_long_gap_does_not_accumulate_old_confirmations(self):
        memory,core,key=self.start()
        for time in [.2,1.,1.2]:self.feed(memory,core,time,.56)
        self.assertIn(key,memory.entries)
        self.feed(memory,core,1.4,.56)
        self.assertNotIn(key,memory.entries)

    def test_rear_blind_sector_does_not_clear_with_front_free_rays(self):
        memory,core,key=self.start(heading=np.pi)
        core.odom[2]=0.
        for time in [.2,.4,.6,100.]:self.feed(memory,core,time,.56)
        self.assertIn(key,memory.entries)

    def test_time_alone_retains_point_and_uncertainty(self):
        memory,core,key=self.start()
        _,uncertainty=self.feed(memory,core,100.,np.nan)
        self.assertIn(key,memory.entries)
        self.assertAlmostEqual(float(uncertainty.max()),.04)

    def test_original_10cm_threshold_keeps_the_low_endpoint(self):
        config=dict(self.config,free_evidence_margin_m=.1,free_evidence_frames=1)
        memory,core,key=self.start(config=config)
        for time in [.2,.4,.6,1.]:self.feed(memory,core,time,.56)
        self.assertIn(key,memory.entries)


if __name__=='__main__':
    unittest.main()
