"""Measured motion, causal timing and navigation-stop invariants."""
import json
from types import SimpleNamespace
import unittest
import numpy as np
from neural_navigation.measured_odometry import MeasuredOdometry,MeasuredNavigationAdapter


class MeasuredOdometryTests(unittest.TestCase):
    def sample(self, odom, t, wheel=(.2,0.,0.), gyro=0., **kwargs):
        return odom.update(t,wheel,t,gyro,t,**kwargs)

    def test_measured_arc_and_lateral_motion(self):
        odom=MeasuredOdometry()
        for i in range(101):self.sample(odom,i*.01,(.2,.05,.5),.5)
        truth=[(.2*np.sin(.5)+.05*(np.cos(.5)-1))/.5,
               (.2*(1-np.cos(.5))+.05*np.sin(.5))/.5,.5]
        np.testing.assert_allclose(odom.pose,truth,atol=3e-7)
        self.assertTrue(odom.state(1.)['valid'])

    def test_command_is_not_an_input_or_motion_source(self):
        odom=MeasuredOdometry()
        for i in range(101):self.sample(odom,i*.01,(0.,0.,0.),0.)
        np.testing.assert_array_equal(odom.pose,np.zeros(3))

    def test_duplicate_cannot_integrate_and_reverse_time_rejected(self):
        odom=MeasuredOdometry();self.sample(odom,0.);self.sample(odom,.01)
        before=odom.pose.copy();self.sample(odom,.01)
        np.testing.assert_array_equal(odom.pose,before)
        with self.assertRaises(ValueError):self.sample(odom,.005)

    def test_stale_future_invalid_and_count_jump_hold(self):
        examples=[dict(wheel_timestamp=-.2),dict(imu_timestamp=-.2),dict(imu_timestamp=.02),
                  dict(wheel_valid=False),dict(gyro_z=np.nan),dict(wheel_velocity=[2.,0.,0.])]
        for patch in examples:
            with self.subTest(patch=patch):
                odom=MeasuredOdometry();self.sample(odom,0.)
                args=dict(timestamp=.01,wheel_velocity=[.2,0.,0.],wheel_timestamp=.01,gyro_z=0.,imu_timestamp=.01)
                args.update(patch);state=odom.update(**args)
                self.assertFalse(state['valid']);self.assertTrue(state['continuity_lost'])
                np.testing.assert_array_equal(odom.pose,np.zeros(3))

    def test_gap_requires_explicit_reset_no_implicit_command_fallback(self):
        odom=MeasuredOdometry();self.sample(odom,0.);self.sample(odom,.01)
        before=odom.pose.copy();state=self.sample(odom,.3)
        np.testing.assert_array_equal(odom.pose,before)
        self.assertEqual(state['reason'],'motion_interval_gap')
        self.assertFalse(self.sample(odom,.31)['valid'])
        odom.reset(before);self.assertTrue(self.sample(odom,.32)['valid'])

    def test_stationary_bias_requires_duration_and_motion_breaks_window(self):
        odom=MeasuredOdometry()
        for i in range(190):self.sample(odom,i*.01,(0.,0.,0.),.005)
        self.assertEqual(odom.bias_updates,0)
        self.sample(odom,1.90,(0.,0.,.1),.1)
        for i in range(191,350):self.sample(odom,i*.01,(0.,0.,0.),.005)
        self.assertEqual(odom.bias_updates,0)
        for i in range(350,600):self.sample(odom,i*.01,(0.,0.,0.),.005)
        self.assertGreater(odom.bias_updates,0)
        self.assertGreater(odom.gyro_bias,0.)
        self.assertLess(odom.gyro_bias,.005)

    def test_output_cannot_extrapolate_past_freshness_or_use_future(self):
        odom=MeasuredOdometry();self.sample(odom,1.)
        self.assertTrue(odom.state(1.01)['valid'])
        self.assertFalse(odom.state(1.1)['valid']);self.assertFalse(odom.state(.99)['valid'])

    def test_adapter_does_not_invoke_policy_with_missing_motion(self):
        class Runtime:
            def __init__(self):
                self.called=0;self.actions=[]
                self.core=SimpleNamespace(record_action=self.actions.append)
                self.context=SimpleNamespace(last_executed=np.ones(2),last_reason=None)
            def step(self,*args,**kwargs):
                self.called+=1;return dict(command=[.2,0.],velocity=np.asarray(args[2]).tolist())
        runtime=Runtime();adapter=MeasuredNavigationAdapter(runtime)
        stopped=adapter.step(np.zeros(220),0.);json.dumps(stopped)
        self.assertEqual(runtime.called,0);self.assertEqual(stopped['command'],[0.,0.])
        self.sample(adapter.odometry,0.,(.1,.02,.2),.2)
        result=adapter.step(np.zeros(220),0.)
        self.assertEqual(runtime.called,1);self.assertEqual(result['velocity'],[.1,.2])
        json.dumps(result)


if __name__=='__main__':unittest.main()
