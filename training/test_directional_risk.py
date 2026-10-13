import unittest
import numpy as np
from neural_navigation.directional_risk import directional_body_risk
from neural_navigation.motion_risk import predicted_body_risk


def track(second):
    return dict(time=0.,hits=3,velocity=np.array([.1,0.]),points=np.array([[.6,0.]]),
        velocity_error_second_moment=np.asarray(second,float))


class DirectionalRiskTests(unittest.TestCase):
    def test_isotropic_finite_response_equivalence(self):
        t=track(np.eye(2)*.01)
        args=([t],[.1,.2,.7],0.,[[.2,-.5],[0,0]],[.4,.3])
        old=predicted_body_risk(*args,period_s=.2,uncertainty_m_s=.2,measured_velocity=[.2,.4])
        new=directional_body_risk(*args,measured_velocity=[.2,.4])
        for a,b in zip(old,new):
            self.assertEqual(a['predicted_margin_contact_s'],b['predicted_margin_contact_s'])
            self.assertAlmostEqual(a['minimum_predicted_clearance_m'],b['minimum_predicted_clearance_m'])

    def test_invalid_measured_velocity_rejected(self):
        with self.assertRaises(ValueError):
            directional_body_risk([],np.zeros(3),0.,[[0,0]],[.4,.3],measured_velocity=[np.nan,0])

    def test_isotropic_equivalence(self):
        t=track(np.eye(2)*.01)
        old=predicted_body_risk([t],[0,0,0],0.,[[0,0]],[.4,.3],period_s=.2,uncertainty_m_s=.2)
        new=directional_body_risk([t],[0,0,0],0.,[[0,0]],[.4,.3])
        self.assertEqual(old[0]['predicted_margin_contact_s'],new[0]['predicted_margin_contact_s'])
        self.assertAlmostEqual(old[0]['minimum_predicted_clearance_m'],new[0]['minimum_predicted_clearance_m'])

    def test_tangential_residual_does_not_expand_normal_identically(self):
        normal=directional_body_risk([track(np.diag([.04,0]))],[0,0,0],0.,[[0,0]],[.4,.3])[0]
        tangent=directional_body_risk([track(np.diag([0,.04]))],[0,0,0],0.,[[0,0]],[.4,.3])[0]
        self.assertIsNotNone(normal['predicted_margin_contact_s'])
        self.assertIsNone(tangent['predicted_margin_contact_s'])

    def test_rotation_preserves_forecast(self):
        t=track(np.diag([.04,.002]));angle=.73
        c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s],[s,c]])
        transformed=dict(t,points=t['points']@rot.T,velocity=t['velocity']@rot.T,
            velocity_error_second_moment=rot@t['velocity_error_second_moment']@rot.T)
        a=directional_body_risk([t],[0,0,0],0.,[[.1,-.2]],[.4,.3])[0]
        b=directional_body_risk([transformed],[0,0,angle],0.,[[.1,-.2]],[.4,.3])[0]
        self.assertEqual(a['predicted_margin_contact_s'],b['predicted_margin_contact_s'])
        self.assertAlmostEqual(a['minimum_predicted_clearance_m'],b['minimum_predicted_clearance_m'])

    def test_invalid_matrix_rejected_and_no_support_is_unknown(self):
        with self.assertRaises(ValueError):
            directional_body_risk([track(np.diag([-1,0]))],[0,0,0],0.,[[0,0]],[.4,.3])
        t=track(np.eye(2));t['hits']=2
        r=directional_body_risk([t],[0,0,0],0.,[[0,0]],[.4,.3])[0]
        self.assertEqual(r['supported_tracks'],0)
        self.assertIsNone(r['minimum_predicted_clearance_m'])

    def test_projection_never_exceeds_corresponding_isotropic_envelope(self):
        rng=np.random.default_rng(81)
        for _ in range(24):
            matrix=rng.normal(size=(2,2))*.05;second=matrix@matrix.T
            t=track(second);t['points']=rng.uniform(-1,1,(6,2))
            command=rng.uniform([0,-.5],[.2,.5],(1,2))
            spread=max(.025,2*np.sqrt(np.linalg.eigvalsh(second)[-1]))
            old=predicted_body_risk([t],[0,0,.4],0.,command,[.4,.3],period_s=.2,uncertainty_m_s=spread)[0]
            new=directional_body_risk([t],[0,0,.4],0.,command,[.4,.3])[0]
            self.assertGreaterEqual(new['minimum_predicted_clearance_m']+1e-12,old['minimum_predicted_clearance_m'])
            if old['predicted_margin_contact_s'] is None:
                self.assertIsNone(new['predicted_margin_contact_s'])


if __name__=='__main__':unittest.main()
