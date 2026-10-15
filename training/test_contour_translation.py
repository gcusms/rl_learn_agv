import unittest
import numpy as np
from neural_navigation.contour_translation import contour_translation,RegisteredContourMotion
from neural_navigation.contour_history import CumulativeContourPrediction


def arc(lo=-2.5,hi=-.7,shift=(0.,0.)):
    theta=np.linspace(lo,hi,40)
    return .25*np.c_[np.cos(theta),np.sin(theta)]+shift


def track(t,points):
    return dict(id=1,time=t,center=np.median(points,axis=0),points=points,
        velocity=np.array([.2,.1]),hits=1)


class ContourRegistrationTests(unittest.TestCase):
    def test_partial_visibility_is_not_translation(self):
        a=arc();b=arc(-2.15,-.7)
        self.assertGreater(np.linalg.norm(np.median(b,axis=0)-np.median(a,axis=0)),.03)
        fit=contour_translation(a,b)
        self.assertIsNotNone(fit)
        self.assertLess(np.linalg.norm(fit[0]),.004)

    def test_translating_partial_contour_retains_motion_through_centroid_changes(self):
        motion=RegisteredContourMotion();history=CumulativeContourPrediction(registered_motion=True)
        velocity=np.array([.03,-.09])
        for i,t in enumerate(np.arange(0.,1.01,.2)):
            points=arc(-2.5+.2*i,-.7,velocity*t)
            rows=history.update(motion.update([track(float(t),points)],float(t)),float(t))
        self.assertEqual(rows[0]['source'],'observed')
        np.testing.assert_allclose(rows[0]['velocity'],velocity,atol=.012)
        self.assertGreater(rows[0]['contour_position_uncertainty_m'],0.)
        # Native tracking may retain the same measurement, but time alone
        # cannot renew its support, stamp or bounded velocity hold.
        stale=history.update(motion.update([track(1.,points)],1.4),1.4)[0]
        self.assertTrue(stale['retained_without_measurement'])
        self.assertEqual(stale['source'],'held')
        stale=history.update(motion.update([track(1.,points)],1.8),1.8)[0]
        self.assertEqual(stale['source'],'unknown');self.assertAlmostEqual(stale['age'],.8)

    def test_single_straight_edge_does_not_certify_two_dimensional_motion(self):
        a=np.c_[np.linspace(-.2,.2,20),np.zeros(20)]
        self.assertIsNone(contour_translation(a,a+[.02,.01]))

    def test_reversal_and_stop_are_measured_without_centroid_velocity_fallback(self):
        motion=RegisteredContourMotion()
        for i in range(14):
            t=i*.2
            offset=np.array([.04,0.])*min(t,1.)-np.array([.04,0.])*max(0.,min(t-1.,.8))
            row=motion.update([track(t,arc(shift=offset))],t)[0]
            if i==8:self.assertLess(row['velocity'][0],-.025)
        self.assertTrue(row['motion_registration_supported'])
        self.assertLess(np.linalg.norm(row['velocity']),.01)


if __name__=='__main__':unittest.main()
