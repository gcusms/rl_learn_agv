"""Behavior checks for a physical, one-shot doorway blockage."""
import unittest
import numpy as np
import torch
from navrl.geometry import ROOT
from navrl.scenarios import configure_column_task
from neural_navigation.env import LearnedNavigationEnv

class ScriptedColumnTests(unittest.TestCase):
    def test_far_robot_does_not_trigger_on_elapsed_time(self):
        torch.set_num_threads(2);e=LearnedNavigationEnv(scene_dir=ROOT/'assets/room_navigation_scripted')
        try:
            e.reset(seed=28000,options={'task':0});configure_column_task(e,0);before=np.array(e._humans());e.elapsed=100.
            for i in range(2):np.testing.assert_array_equal(e._walker_velocity(i,.05),[0,0]);self.assertEqual(e.walkers[i]['block']['phase'],'armed')
            np.testing.assert_array_equal(e._humans(),before)
        finally:e.close()

    def test_enter_hold_return_and_remain_parked_without_random_routes(self):
        torch.set_num_threads(2);e=LearnedNavigationEnv(scene_dir=ROOT/'assets/room_navigation_scripted')
        try:
            e.reset(seed=28001,options={'task':0,'start':[2.5,2.1,1.57]});configure_column_task(e,0);start=np.array(e._humans());positions=[start];phases=set()
            def forbid_random_route(*args,**kwargs):raise AssertionError('Scripted columns requested a random destination')
            e._new_route=forbid_random_route
            parked=None
            for _ in range(140):
                _,_,_,_,info=e.step([0,0]);positions.append(np.array(e._humans()));phases.update(b['phase'] for b in info['column_block_events'])
                self.assertFalse(info['supervisor_override']);np.testing.assert_array_equal(info['commanded_velocity'],[0,0])
                self.assertGreaterEqual(info['minimum_dynamic_gap_m'],-.001)
                if all(b['phase']=='cleared' for b in info['column_block_events']):
                    if parked is None:parked=np.array(e._humans())
                    else:np.testing.assert_allclose(e._humans(),parked,atol=.008)
            self.assertTrue({'entering','holding','releasing','cleared'}<=phases)
            np.testing.assert_allclose(e._humans(),start,atol=.08)
            self.assertLess(np.linalg.norm(np.diff(np.array(positions),axis=0),axis=2).max(),.12)
            for b in info['column_block_events']:
                self.assertFalse(b['entry_aborted']);self.assertAlmostEqual(b['release_time_s']-b['entered_time_s'],12.,places=6)
            self.assertEqual(info['penetration_guard_activations'],0)
        finally:e.close()

if __name__=='__main__':unittest.main(verbosity=2)
