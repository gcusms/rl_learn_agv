import unittest
from types import SimpleNamespace
import numpy as np
from navrl.geometry import LidarProfile
from neural_navigation.runtime import EmergencyStop
from experiments.evaluate_current_motion_guard import CurrentMotionGuard


class CurrentMotionTests(unittest.TestCase):
    def core(self,velocity=(0.,0.)):
        lidar=LidarProfile()
        return SimpleNamespace(scan=np.full(lidar.beams,2.),lidar=lidar,elapsed=1.,
            estimate=np.zeros(3),odom=np.zeros(3),measured_velocity=np.array(velocity),odometry_sanity_failure=False,
            localizer=SimpleNamespace(std=np.array([.02,.02,.01])),
            static_map=SimpleNamespace(free=np.ones((1,1)),lookup=lambda *a,**kw:True),
            diagnostics=lambda:{'route_complete':True})

    def point(self,core,point):
        offset=np.asarray(point)-core.lidar.mount[:2]
        index=np.argmin(abs(core.lidar.angles-np.arctan2(offset[1],offset[0])))
        core.scan[index]=np.linalg.norm(offset)

    def guard(self,kind):return kind(memory_config={'near_range_m':1.5})

    def test_slow_request_does_not_hide_current_forward_speed(self):
        core=self.core((.2,0.));self.point(core,[.30,0.])
        old,reason=self.guard(EmergencyStop).check(core,[.005,0.],1.,1.)
        np.testing.assert_allclose(old,[.005,0.]);self.assertIsNone(reason)
        new,reason=self.guard(CurrentMotionGuard).check(core,[.005,0.],1.,1.)
        np.testing.assert_array_equal(new,[0.,0.]);self.assertEqual(reason,'laser_braking_footprint')

    def test_zero_yaw_request_does_not_hide_current_rotation(self):
        core=self.core((0.,.5));self.point(core,[.24,-.17])
        _,old_reason=self.guard(EmergencyStop).check(core,[0.,0.],1.,1.)
        self.assertIsNone(old_reason)
        _,reason=self.guard(CurrentMotionGuard).check(core,[0.,0.],1.,1.)
        self.assertEqual(reason,'laser_braking_footprint')

    def test_clear_motion_is_preserved_and_old_rejections_remain(self):
        core=self.core((.1,.2));guard=self.guard(CurrentMotionGuard)
        cmd,reason=guard.check(core,[.05,.1],1.,1.)
        np.testing.assert_array_equal(cmd,[.05,.1]);self.assertIsNone(reason)
        for proposal,now,expected in [([-.08,0.],1.,'reverse_without_rear_visibility'),
                                    ([.1,0.],2.,'stale_or_future_scan')]:
            cmd,reason=guard.check(core,proposal,1.,now)
            np.testing.assert_array_equal(cmd,[0.,0.]);self.assertEqual(reason,expected)
        core.measured_velocity=np.array([np.nan,0.])
        cmd,reason=guard.check(core,[.1,0.],1.,1.)
        np.testing.assert_array_equal(cmd,[0.,0.]);self.assertEqual(reason,'nonfinite_input')


if __name__=='__main__':unittest.main()
