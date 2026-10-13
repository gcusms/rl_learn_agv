import unittest
from types import SimpleNamespace
import numpy as np
from navrl.geometry import integrate
from neural_navigation.motion_response import force_response_trajectory
from neural_navigation.runtime import EmergencyStop
from neural_navigation.motion_risk import predicted_body_risk


class MotionResponseTests(unittest.TestCase):
    def test_optional_risk_forecast_detects_coasting_and_leaves_default_unchanged(self):
        tracks=[dict(time=0.,hits=3,velocity=np.array([0.,.04]),points=np.array([[.27,0.]]))]
        args=(tracks,np.zeros(3),0.,np.array([[0.,0.]]),[.4,.2])
        old=predicted_body_risk(*args,horizon_s=.4,period_s=.2)
        finite=predicted_body_risk(*args,horizon_s=.4,period_s=.2,measured_velocity=[.2,0.])
        self.assertIsNone(old[0]['predicted_margin_contact_s'])
        self.assertIsNotNone(finite[0]['predicted_margin_contact_s'])
        self.assertEqual(old,predicted_body_risk(*args,horizon_s=.4,period_s=.2))

    def test_constant_motion_matches_geometry(self):
        command=np.array([.2,.5])
        poses=force_response_trajectory(command,command,1.)
        expected=np.array([integrate(np.zeros(3),*command,t) for t in np.linspace(0,1,9)])
        np.testing.assert_allclose(poses,expected,atol=1e-7)

    def test_stop_has_finite_coasting_distance(self):
        poses=force_response_trajectory([.2,0],[0,0],1.)
        # Saturated deceleration for .35s, followed by exponential tail.
        expected=.2*.35-.5*.5*.35**2+.025/20*(1-np.exp(-20*.65))
        self.assertAlmostEqual(poses[-1,0],expected,delta=1e-6)
        self.assertGreater(poses[-1,0],.03)
        np.testing.assert_allclose(poses[:,1:],0,atol=1e-12)

    def test_turn_reversal_keeps_initial_rotation(self):
        poses=force_response_trajectory([.1,.5],[.1,-.5],1.4)
        self.assertGreater(poses[1,2],0)
        self.assertLess(poses[-1,2],0)

    def test_guard_detects_coasting_into_laser_point(self):
        core=SimpleNamespace(estimate=np.array([1.,1.,.7]),measured_velocity=np.array([.2,0.]))
        legacy=EmergencyStop(footprint=[.4,.2])
        response=EmergencyStop(footprint=[.4,.2],memory_config={'motion_response_schema':'force_response_v1'})
        legacy._static_pose_clear=lambda *args:True
        response._static_pose_clear=lambda *args:True
        points=np.array([[.27,0.]])
        self.assertIsNone(legacy._trajectory_check(core,np.zeros(2),points,np.array([.04]),0.))
        self.assertEqual(response._trajectory_check(core,np.zeros(2),points,np.array([.04]),0.),'laser_braking_footprint')
        local,world,_=list(response._trajectory(core,np.zeros(2),0.))[-1]
        np.testing.assert_allclose(world[:2],core.estimate[:2]+local[0]*np.array([np.cos(.7),np.sin(.7)]))

    def test_nonfinite_velocity_and_invalid_parameters_rejected(self):
        guard=EmergencyStop(memory_config={'motion_response_schema':'force_response_v1'})
        core=SimpleNamespace(measured_velocity=np.array([np.nan,0]))
        cmd,reason=guard.check(core,[.1,0],0,0)
        self.assertEqual(reason,'nonfinite_input')
        np.testing.assert_array_equal(cmd,0)
        for acceleration in ([0,1],[.5,-1]):
            with self.assertRaises(ValueError):
                force_response_trajectory([0,0],[.1,0],1,acceleration=acceleration)


if __name__=='__main__':unittest.main()
