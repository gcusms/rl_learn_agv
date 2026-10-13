import copy
import unittest
import numpy as np
from neural_navigation.track_uncertainty import TrackInnovationEnvelope
from neural_navigation.motion_risk import predicted_body_risk


class TrackUncertaintyTests(unittest.TestCase):
    def track(self,time,center,velocity,hits=3):
        return dict(id=0,time=time,center=np.asarray(center,float),velocity=np.asarray(velocity,float),
            hits=hits,points=np.asarray([center],float))

    def test_constant_observed_motion_keeps_floor_and_does_not_mutate_tracks(self):
        estimator=TrackInnovationEnvelope()
        for time in (0.,.2,.4,.6):
            tracks=[self.track(time,[1+.1*time,0],[.1,0])];before=copy.deepcopy(tracks)
            result=estimator.update(tracks,time)
            self.assertEqual(result[0]['velocity_uncertainty_m_s'],.025)
            self.assertNotIn('velocity_uncertainty_m_s',tracks[0])
            np.testing.assert_array_equal(tracks[0]['center'],before[0]['center'])

    def test_smoothed_velocity_lag_increases_envelope(self):
        estimator=TrackInnovationEnvelope();estimator.update([self.track(0,[0,0],[0,0],1)],0)
        result=estimator.update([self.track(.4,[.08,0],[.09,0],2)],.4)
        self.assertAlmostEqual(result[0]['velocity_uncertainty_m_s'],.22)
        repeated=estimator.update([self.track(.4,[.08,0],[.09,0],2)],.6)
        self.assertEqual(repeated[0]['velocity_uncertainty_samples'],1)
        self.assertEqual(repeated[0]['velocity_uncertainty_m_s'],result[0]['velocity_uncertainty_m_s'])

    def test_reconfirmation_and_occlusion_expiry_clear_old_residuals(self):
        estimator=TrackInnovationEnvelope();estimator.update([self.track(0,[0,0],[0,0],1)],0)
        estimator.update([self.track(.4,[.08,0],[.09,0],4)],.4)
        result=estimator.update([self.track(.6,[.1,0],[-.1,0],1)],.6)
        self.assertEqual(result[0]['velocity_uncertainty_samples'],0)
        self.assertEqual(estimator.update([],1.5),[])
        self.assertEqual(estimator.states,{})

    def test_uncertainty_changes_only_explicit_shadow_risk_call(self):
        track=self.track(0,[.4,0],[-.05,0]);track['velocity_uncertainty_m_s']=.15
        args=([track],np.zeros(3),0,np.zeros((1,2)),[.4,.2])
        old=predicted_body_risk(*args,horizon_s=1.)
        shadow=predicted_body_risk(*args,horizon_s=1.,adaptive_track_uncertainty=True)
        self.assertIsNone(old[0]['predicted_margin_contact_s'])
        self.assertIsNotNone(shadow[0]['predicted_margin_contact_s'])
        self.assertEqual(old,predicted_body_risk(*args,horizon_s=1.))


if __name__=='__main__':unittest.main()
