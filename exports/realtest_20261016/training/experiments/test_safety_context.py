import copy
import unittest
from types import SimpleNamespace
import numpy as np
from neural_navigation.runtime import EmergencyStop
from navrl.geometry import LidarProfile
from experiments.safety_context import SafetyContext,PROBES,CONTEXT_DIM


class ContextTests(unittest.TestCase):
    def core(self):
        lidar=LidarProfile()
        return SimpleNamespace(scan=np.full(lidar.beams,2.),lidar=lidar,odom=np.zeros(3),elapsed=1.,estimate=np.zeros(3),
                               localizer=SimpleNamespace(std=np.array([.02,.02,.01])),odometry_sanity_failure=False,
                               static_map=SimpleNamespace(free=np.ones((1,1)),lookup=lambda *a,**kw:True),diagnostics=lambda:{'route_complete':True})

    def test_probes_match_real_guard_without_changing_its_memory_or_clocks(self):
        c=self.core();guard=EmergencyStop(memory_config={'near_range_m':1.5})
        guard.memory.entries[(-4,5)]=(np.array([-.15,.22]),.8)
        guard.reverse_started=.2;before=copy.deepcopy(guard)
        ctx=SafetyContext().observation(c,guard,1.)
        self.assertEqual(ctx.shape,(CONTEXT_DIM,))
        for i,cmd in enumerate(PROBES):
            executed,_=copy.deepcopy(before).check(c,cmd,1.,1.)
            self.assertEqual(ctx[i],float(np.allclose(cmd,executed,atol=1e-9,rtol=0)))
        self.assertEqual(guard.reverse_started,.2);self.assertEqual(guard.memory.last_time,None)
        self.assertEqual(guard.memory.entries.keys(),before.memory.entries.keys())
        np.testing.assert_array_equal(guard.memory.entries[(-4,5)][0],before.memory.entries[(-4,5)][0])

    def test_stale_scan_blocks_probe_flags_and_executed_wait_is_observable(self):
        c=self.core();guard=EmergencyStop();context=SafetyContext()
        context.record(dict(timestamp=1.,proposed_action=[-.5,.8],command=[0.,0.],safety_override=True))
        c.elapsed=4.
        out=context.observation(c,guard,4.,received_timestamp=5.)
        np.testing.assert_array_equal(out[:9],np.zeros(9));np.testing.assert_allclose(out[10:12],[-.5,.8]);self.assertAlmostEqual(out[-1],.6,places=6)
        context.record(dict(timestamp=4.,proposed_action=[.5,0.],command=[.1,0.],safety_override=False))
        self.assertEqual(context.observation(c,guard,4.)[-1],0.)
        with self.assertRaises(ValueError):context.observation(c,guard,3.)


if __name__=='__main__':unittest.main()
