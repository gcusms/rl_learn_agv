import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.stress_env import StressNavigationEnv


class DynamicProfilesTests(unittest.TestCase):
    def world(self,case):
        return SimpleNamespace(case=case,elapsed=1.,intervention=dict(time_s=0.,hold_s=0.,start=[1.,1.],end=[2.2,1.],speed_m_s=.12),
                               walkers=[{'state':'walking'}],_humans=lambda:[np.array([1.3,1.])])
    def velocity(self,world):return StressNavigationEnv._walker_velocity(world,0,.005)
    def test_reversal_changes_actual_world_velocity_without_teleporting(self):
        world=self.world({'reverse_after_event_s':2.})
        np.testing.assert_allclose(self.velocity(world),[.12,0.]);before=world._humans()[0].copy()
        world.elapsed=2.1;np.testing.assert_allclose(self.velocity(world),[-.12,0.])
        np.testing.assert_array_equal(world._humans()[0],before);self.assertAlmostEqual(world.intervention['reversal_time_s'],2.1)
    def test_stop_go_pauses_then_resumes_and_records_both_phases(self):
        world=self.world({'actor_pause_after_event_s':[[2.,4.]]})
        np.testing.assert_allclose(self.velocity(world),[.12,0.]);world.elapsed=2.1
        np.testing.assert_array_equal(self.velocity(world),[0.,0.]);world.elapsed=4.1
        np.testing.assert_allclose(self.velocity(world),[.12,0.]);self.assertIn('pause_completed_time_s',world.intervention)


if __name__=='__main__':unittest.main(verbosity=2)
