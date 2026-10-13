import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from navrl.geometry import ROOT,LidarProfile
from neural_navigation.runtime import EmergencyStop,SensorNavigationRuntime


class ConstantActor(torch.nn.Module):
    def forward(self,x):return x[:,:2]*0+torch.tensor([1.,0.])


class RuntimeTests(unittest.TestCase):
    def fixture(self):
        lidar=LidarProfile()
        return SimpleNamespace(scan=np.full(220,2.),lidar=lidar,localizer=SimpleNamespace(std=np.array([.02,.02,.01])),
                               odometry_sanity_failure=False,estimate=np.zeros(3),
                               static_map=SimpleNamespace(free=np.ones((1,1)),lookup=lambda *a,**k:True),
                               diagnostics=lambda:{'route_complete':True})

    def test_stale_or_unavailable_scan_stops(self):
        core=self.fixture();guard=EmergencyStop()
        command,reason=guard.check(core,[.2,0.],1.,1.5)
        np.testing.assert_array_equal(command,[0,0]);self.assertEqual(reason,'stale_or_future_scan')
        core.scan[:]=np.nan;_,reason=guard.check(core,[.2,0.],1.,1.)
        self.assertEqual(reason,'scan_unavailable')

    def test_braking_footprint_and_blind_reverse_stop(self):
        core=self.fixture();guard=EmergencyStop()
        core.scan[np.argmin(abs(core.lidar.angles))]=.25
        _,reason=guard.check(core,[.2,0.],1.,1.)
        self.assertEqual(reason,'laser_braking_footprint')
        core.scan[:]=2.;_,reason=guard.check(core,[-.08,0.],1.,1.)
        self.assertEqual(reason,'reverse_without_rear_visibility')
        _,reason=guard.check(core,[-.08,0.],1.,1.,rear_clearance_known=True)
        self.assertIsNone(reason)
        _,reason=guard.check(core,[-.08,0.],3.,3.,rear_clearance_known=True)
        self.assertEqual(reason,'reverse_duration_limit')

    def test_blind_reverse_preserves_only_collision_checked_model_rotation(self):
        core=self.fixture();guard=EmergencyStop()
        command,reason=guard.check(core,[-.08,.4],1.,1.)
        np.testing.assert_allclose(command,[0.,.4])
        self.assertEqual(reason,'reverse_without_rear_visibility')
        # No synthesized turn or clearance bypass: a near body return still
        # vetoes rotation as well as translation.
        core.scan[np.argmin(abs(core.lidar.angles))]=.10
        command,reason=guard.check(core,[-.08,.4],1.,1.)
        np.testing.assert_array_equal(command,[0.,0.])
        self.assertEqual(reason,'laser_braking_footprint')

    def test_executed_action_history_records_emergency_stop(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            bundle=Path(directory)
            actor=torch.jit.trace(ConstantActor(),torch.zeros(1,235));actor.save(str(bundle/'actor.pt'))
            for name in ['localizer.pt','router.pt']:shutil.copy2(ROOT/'runs/remote_room_100k'/name,bundle/name)
            (bundle/'hardware.json').write_text(json.dumps({'lidar':LidarProfile().signature,'robot_footprint_m':[.393924904,.329460022]}))
            runtime=SensorNavigationRuntime(bundle,ROOT/'assets/room_navigation_scripted/decision_map.yaml',[1.2,1.2,0.],[8.6,1.2])
            result=runtime.step(np.full(220,2.),np.zeros(3),np.zeros(2),1.,received_timestamp=2.)
            self.assertEqual(result['stop_reason'],'stale_or_future_scan')
            self.assertEqual(result['proposed_action'],[1.,0.]);self.assertEqual(result['command'],[0.,0.])
            np.testing.assert_array_equal(runtime.core.last_action,[0.,0.])


if __name__=='__main__':unittest.main(verbosity=2)
