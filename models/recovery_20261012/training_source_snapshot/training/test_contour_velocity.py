import copy
import unittest
import numpy as np
from neural_navigation.contour_velocity import ContourAwareVelocity


def track(time,count=30,extent=.4,hits=5,velocity=(.1,0)):
    return dict(id=1,time=time,hits=hits,center=np.zeros(2),velocity=np.array(velocity),
        points=np.c_[np.linspace(-extent/2,extent/2,count),np.zeros(count)])


class ContourVelocityTests(unittest.TestCase):
    def test_degradation_holds_previous_velocity_without_mutating_input(self):
        quality=ContourAwareVelocity();quality.update([track(0)],0)
        t=track(.4,7,.18,6,(.14,.11));before=copy.deepcopy(t)
        out=quality.update([t],.4)[0]
        self.assertEqual(out['contour_velocity_source'],'held');self.assertTrue(out['contour_degraded'])
        np.testing.assert_array_equal(out['velocity'],[.1,0]);np.testing.assert_array_equal(t['velocity'],before['velocity'])
        np.testing.assert_array_equal(t['points'],before['points'])

    def test_duplicate_scan_does_not_recover_and_hold_expires(self):
        q=ContourAwareVelocity();q.update([track(0)],0);bad=track(.4,7,.18,6)
        q.update([bad],.4)
        a=q.update([bad],.5)[0];self.assertEqual(a['contour_recovery_measurements'],0)
        b=q.update([bad],.7)[0];self.assertEqual(b['contour_velocity_source'],'unknown')
        self.assertFalse(b['contour_velocity_supported'])

    def test_two_fresh_stable_measurements_recover(self):
        q=ContourAwareVelocity();q.update([track(0)],0);q.update([track(.4,7,.18,6)],.4)
        a=q.update([track(.8,8,.19,7)],.8)[0];self.assertTrue(a['contour_degraded'])
        b=q.update([track(1.2,9,.2,8)],1.2)[0];self.assertFalse(b['contour_degraded'])
        self.assertEqual(b['contour_velocity_source'],'observed')

    def test_reconfirmation_does_not_inherit_old_direction(self):
        q=ContourAwareVelocity();q.update([track(0)],0)
        out=q.update([track(.4,hits=1,velocity=(-.1,0))],.4)[0]
        self.assertEqual(out['contour_velocity_source'],'unknown')
        np.testing.assert_array_equal(out['velocity'],[-.1,0])

    def test_geometry_is_rotation_invariant_and_invalid_inputs_fail(self):
        t=track(0);a=ContourAwareVelocity.geometry(t['points']);angle=.72
        c,s=np.cos(angle),np.sin(angle)
        b=ContourAwareVelocity.geometry(t['points']@np.array([[c,-s],[s,c]]))
        self.assertAlmostEqual(a[0],b[0]);self.assertEqual(a[1],b[1])
        q=ContourAwareVelocity();q.update([t],0)
        with self.assertRaises(ValueError):q.update([t],-.1)


if __name__=='__main__':unittest.main()
