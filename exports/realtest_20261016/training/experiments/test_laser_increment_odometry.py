import unittest
from types import SimpleNamespace as NS
import numpy as np
from neural_navigation.laser_increment_odometry import LaserIncrementOdometry


def room_scan(spec,pose):
    c,s=np.cos(pose[2]),np.sin(pose[2]);mount=spec.mount
    origin=pose[:2]+[c*mount[0]-s*mount[1],s*mount[0]+c*mount[1]]
    angles=pose[2]+mount[3]+spec.angles;directions=np.c_[np.cos(angles),np.sin(angles)]
    ranges=np.full(len(angles),np.inf)
    for axis,coordinate,low,high in [(0,-3.,-4.,4.),(0,6.,-4.,4.),(1,-4.,-3.,6.),(1,4.,-3.,6.)]:
        with np.errstate(divide='ignore',invalid='ignore'):t=(coordinate-origin[axis])/directions[:,axis]
        other=origin[1-axis]+t*directions[:,1-axis]
        ok=(t>0)&(other>=low)&(other<=high);ranges=np.minimum(ranges,np.where(ok,t,np.inf))
    return ranges


class IncrementTests(unittest.TestCase):
    def setUp(self):
        self.spec=NS(range_min=.1,range_max=10.,angles=np.linspace(-1.89,1.938,220),mount=np.array([.118389,0,.11668,0]))
    def test_recovers_small_translation_and_turn_from_measured_scans(self):
        matcher=LaserIncrementOdometry(self.spec);pose=np.array([1.,.5,0.]);truth_delta=np.array([.035,.003,.065])
        previous=room_scan(self.spec,pose);current=room_scan(self.spec,pose+truth_delta)
        candidate,reason=matcher.match(current,previous,np.array([.041,.005,.07]),.2)
        self.assertEqual(reason,'accepted_scan_increment')
        np.testing.assert_allclose(candidate,truth_delta,atol=.002)
    def test_missing_frame_keeps_continuous_raw_odometry_without_fake_match(self):
        matcher=LaserIncrementOdometry(self.spec);matcher.update(room_scan(self.spec,np.array([1.,.5,0.])),np.zeros(3),0.)
        value=matcher.update(np.full(220,np.nan),np.array([.04,0,0]),.2)
        np.testing.assert_allclose(value,[.04,0,0]);self.assertEqual(matcher.last_reason,'insufficient_returns')
    def test_parallel_wall_does_not_invent_along_wall_displacement(self):
        previous=np.where(np.sin(self.spec.angles)>.1,2/np.maximum(np.sin(self.spec.angles),.1),np.inf)
        current=np.where(np.sin(self.spec.angles)>.1,1.98/np.maximum(np.sin(self.spec.angles),.1),np.inf)
        candidate,reason=LaserIncrementOdometry(self.spec).match(current,previous,np.array([.06,.03,0.]),.2)
        self.assertEqual(reason,'accepted_scan_increment')
        self.assertAlmostEqual(candidate[0],.06,places=6);self.assertAlmostEqual(candidate[1],.02,places=3)

    def test_changed_foreground_cannot_move_stationary_odometry_prior(self):
        matcher=LaserIncrementOdometry(self.spec);scan=room_scan(self.spec,np.array([1.,.5,0.]))
        matcher.update(scan,np.zeros(3),0.)
        for step in range(1,20):
            altered=scan.copy();altered[70:130]=.8+.01*step
            pose=matcher.update(altered,np.zeros(3),step*.2)
        np.testing.assert_array_equal(pose,np.zeros(3));self.assertEqual(matcher.last_reason,'stationary_odometry_prior')


if __name__=='__main__':unittest.main()
