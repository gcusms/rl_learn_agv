"""Experimental append-only prediction inputs shared by training/inference.

The 282-field prefix remains unchanged. Flags distinguish absent moving
tracks from unsupported pose/velocity. These features never permit a command.
"""
import numpy as np
from neural_navigation.contour_velocity import ContourAwareVelocity
from neural_navigation.track_uncertainty import TrackInnovationEnvelope
from neural_navigation.directional_risk import directional_body_risk
from neural_navigation.predictive_risk import RISK_COMMANDS

PREFIX_DIM=282
QUALITY_DIM=18
TOTAL_DIM=PREFIX_DIM+QUALITY_DIM


def contract():
    return dict(schema='quality_prediction_append_v1',prefix_dimensions=PREFIX_DIM,
        dimensions=TOTAL_DIM,commands_m_s_rad_s=RISK_COMMANDS.tolist(),
        fields=['12_directional_response_risk_scores','pose_supported','observed_moving_track',
            'held_moving_track','unknown_velocity_track','degraded_contour_track','maximum_track_age_over_0.8'],
        horizon_s=2.4,period_s=.2,margin_m=.04,velocity_error_floor_m_s=.025,
        residual_scale=2.,response_gain=20.,response_acceleration=[.5,1.],
        contour_hold_s=.6,contour_recovery_measurements=2,
        thresholds_calibrated=False,absence_of_support_means='unknown',
        simulator_truth_used=False,commands_issued=False)


class QualityPredictionContext:
    def __init__(self,footprint):
        self.footprint=np.asarray(footprint,float).copy()
        # Validate declared geometry without requiring a simulator.
        directional_body_risk([],np.zeros(3),0.,RISK_COMMANDS,self.footprint)
        self.reset()

    def reset(self):
        self.quality=ContourAwareVelocity();self.innovations=TrackInnovationEnvelope()

    def features(self,core):
        loc=core.localizer;std=np.asarray(loc.std,float)
        pose=np.asarray(core.estimate,float);velocity=np.asarray(core.measured_velocity,float)
        supported=(std.shape==(3,) and pose.shape==(3,) and velocity.shape==(2,)
            and np.isfinite(np.r_[std,pose,velocity,loc.map_agreement,core.elapsed]).all()
            and bool(loc.scan_update_accepted) and loc.map_agreement>=.8
            and max(std[:2])<=.2 and std[2]<=.25
            and not getattr(core,'odometry_sanity_failure',False))
        if not supported:
            self.reset()
            return np.zeros(QUALITY_DIM,np.float32)  # pose_supported=0 explicitly.
        predictor=getattr(getattr(core,'laser_fusion',None),'predictor',None)
        tracks=predictor.tracks if predictor is not None else []
        processed=self.quality.update(tracks,float(core.elapsed))
        annotated=self.innovations.update(processed,float(core.elapsed))
        reliable=[];observed=False;held=False;unknown=False;degraded=False;age=0.
        for t in annotated:
            source=t['contour_velocity_source'];speed=np.linalg.norm(t['velocity'])
            unknown|=source=='unknown';degraded|=t['contour_degraded']
            age=max(age,float(core.elapsed)-float(t['time']))
            observed|=source=='observed' and speed>=.04
            held|=source=='held' and speed>=.04
            if source=='unknown':continue
            residuals=np.asarray(self.innovations.states[t['id']]['residuals']).reshape(-1,2)
            t['velocity_error_second_moment']=(residuals.T@residuals/len(residuals) if len(residuals) else np.zeros((2,2)))
            reliable.append(t)
        risk=directional_body_risk(reliable,pose,float(core.elapsed),RISK_COMMANDS,
            self.footprint,measured_velocity=velocity)
        scores=[0. if r['predicted_margin_contact_s'] is None else
            1.-r['predicted_margin_contact_s']/2.6 for r in risk]
        return np.asarray(scores+[1.,float(observed),float(held),float(unknown),float(degraded),
            float(np.clip(age/.8,0,1))],np.float32)

    def augment(self,base,core):
        base=np.asarray(base,np.float32)
        if base.shape!=(PREFIX_DIM,) or not np.isfinite(base).all():
            raise ValueError('Finite unchanged V4 prefix required')
        return np.r_[base,self.features(core)].astype(np.float32)
