import unittest
import numpy as np
from neural_navigation.sensors import LidarSpec,OdometryBuffer
from navrl.geometry import LidarProfile


class SensorTests(unittest.TestCase):
    def test_odometry_interpolation_wraps_yaw_and_rejects_large_gap(self):
        buffer=OdometryBuffer()
        buffer.append(1.,[0,0,np.pi-.1],[.1,.1]);buffer.append(1.2,[.02,0,-np.pi+.1],[.1,.1])
        pose,velocity=buffer.sample(1.1,max_gap=.21)
        self.assertAlmostEqual(pose[0],.01);self.assertAlmostEqual(abs(pose[2]),np.pi)
        self.assertIsNone(buffer.sample(1.1,max_gap=.1))
        self.assertIsNone(buffer.sample(.9));self.assertIsNone(buffer.sample(1.3))
        self.assertIsNotNone(buffer.sample(1.22))

    def test_scan_resampling_handles_circular_sector_and_invalid_returns(self):
        spec=LidarSpec(LidarProfile().signature)
        ranges=np.full(1440,2.);ranges[:10]=0.
        scan,valid=spec.resample(ranges,-np.pi,2*np.pi/1440,range_min=.1,range_max=12.)
        self.assertTrue(valid);self.assertEqual(scan.shape,(220,))
        self.assertTrue(np.all(scan[np.isfinite(scan)]==2.))
        _,valid=spec.resample(np.ones(20),0.,.01)
        self.assertFalse(valid)

    def test_odometry_duplicate_time_replaces_value_and_backwards_time_fails(self):
        buffer=OdometryBuffer();buffer.append(1.,[0,0,0],[0,0]);buffer.append(1.,[.1,0,0],[0,0])
        self.assertEqual(len(buffer.rows),1);self.assertAlmostEqual(buffer.sample(1.)[0][0],.1)
        with self.assertRaises(ValueError):buffer.append(.9,[0,0,0],[0,0])


if __name__=='__main__':unittest.main(verbosity=2)
