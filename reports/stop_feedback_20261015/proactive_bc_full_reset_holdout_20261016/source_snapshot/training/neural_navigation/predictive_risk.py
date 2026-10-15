"""Actor features for supported scan-track body risk.

Enabled only by a versioned observation contract. No command selection, guard relaxation or
simulator truth access. Risk scores describe approximate forecasts only.
"""
import numpy as np
from neural_navigation.safety_context import RECOVERY_PROBES
from neural_navigation.motion_risk import predicted_body_risk


RISK_COMMANDS = np.vstack([np.zeros(2), RECOVERY_PROBES, [.2, .5], [.2, -.5]])
RISK_DIMENSIONS = len(RISK_COMMANDS)+1


class PredictiveRiskContext:
    def __init__(self, footprint, config=None):
        config = dict(config or {})
        self.footprint = np.asarray(footprint, float).copy()
        self.horizon = float(config.get('horizon_s', 2.4))
        self.period = .2
        self.uncertainty = float(config.get('uncertainty_m_s', .025))
        # Validate fixed geometry/options even if the first frame has no tracks.
        predicted_body_risk([], np.zeros(3), 0., RISK_COMMANDS, self.footprint,
            self.horizon, self.period, uncertainty_m_s=self.uncertainty)

    def contract(self):
        return dict(schema='predicted_body_risk_features_v1', dimensions=RISK_DIMENSIONS,
            commands_m_s_rad_s=RISK_COMMANDS.tolist(), horizon_s=self.horizon,
            period_s=self.period, margin_m=.04, uncertainty_m_s=self.uncertainty,
            score='zero_if_no_predicted_contact_else_one_minus_time_over_horizon_plus_period',
            final_feature='at_least_one_supported_moving_track',
            absence_of_support_means='unknown', instantaneous_command_following=True,
            simulator_truth_used=False, commands_issued=False)

    def features(self, core):
        # Reject unsupported localization before reading/predicting tracks.
        localizer = core.localizer
        std = np.asarray(localizer.std, float)
        supported_pose = (bool(localizer.scan_update_accepted)
            and np.isfinite(localizer.map_agreement) and localizer.map_agreement >= .8
            and std.shape == (3,) and np.isfinite(std).all()
            and max(std[:2]) <= .2 and std[2] <= .25)
        if not supported_pose:
            return np.zeros(RISK_DIMENSIONS, dtype=np.float32)
        fusion = getattr(core, 'laser_fusion', None)
        predictor = getattr(fusion, 'predictor', None)
        tracks = predictor.tracks if predictor is not None else []
        risk = predicted_body_risk(tracks, core.estimate, core.elapsed, RISK_COMMANDS,
            self.footprint, self.horizon, self.period, uncertainty_m_s=self.uncertainty)
        scores = [0. if row['predicted_margin_contact_s'] is None else
            1.-row['predicted_margin_contact_s']/(self.horizon+self.period) for row in risk]
        return np.asarray(scores+[float(risk[0]['supported_tracks'] > 0)], dtype=np.float32)


class BoundedPoseRiskContext(PredictiveRiskContext):
    """Retain warnings briefly across support flicker, never authorize motion.

    A previous supported estimate is checked against measured odometry. This
    is a bounded approximation, not calibrated localization confidence.
    """
    def __init__(self,footprint):
        super().__init__(footprint)
        self.anchor=None

    def contract(self):
        value=super().contract()
        value.update(schema='predicted_body_risk_bounded_pose_v1',
            pose_support_hold_s=.6,held_scan_map_agreement_floor=.7,
            odometry_innovation_limit_m=.1,odometry_innovation_limit_rad=.1,
            held_extra_uncertainty_m_s=.05,
            final_feature='at_least_one_moving_track_with_current_or_bounded_historical_pose_support',
            localization_confidence_calibrated=False)
        return value

    def features(self,core):
        from navrl.geometry import wrap
        now=float(core.elapsed);loc=core.localizer
        pose=np.asarray(core.estimate,float);odom=np.asarray(core.odom,float)
        std=np.asarray(loc.std,float)
        finite=(pose.shape==(3,) and odom.shape==(3,) and std.shape==(3,)
            and np.isfinite(np.r_[pose,odom,std,now]).all())
        valid=finite and max(std[:2])<=.2 and std[2]<=.25 and not getattr(core,'odometry_sanity_failure',False)
        agreement=float(loc.map_agreement)
        if not valid or not np.isfinite(agreement):
            self.anchor=None
            return np.zeros(RISK_DIMENSIONS,np.float32)
        current=bool(loc.scan_update_accepted) and agreement>=.8
        held_age=0.
        if current:
            self.anchor=(now,pose.copy(),odom.copy())
        else:
            if self.anchor is None:return np.zeros(RISK_DIMENSIONS,np.float32)
            stamp,old_pose,old_odom=self.anchor;held_age=now-stamp
            # Compare the current estimate against SE(2) odometry propagation.
            yaw=old_pose[2]-old_odom[2];c,s=np.cos(yaw),np.sin(yaw)
            propagated=old_pose[:2]+(odom[:2]-old_odom[:2])@np.array([[c,s],[-s,c]])
            propagated_yaw=old_pose[2]+wrap(odom[2]-old_odom[2])
            supported=(0<=held_age<=.6+1e-9
                and (not loc.scan_update_accepted or agreement>=.7)
                and np.linalg.norm(pose[:2]-propagated)<=.1
                and abs(wrap(pose[2]-propagated_yaw))<=.1)
            if not supported:
                self.anchor=None
                return np.zeros(RISK_DIMENSIONS,np.float32)
        predictor=getattr(getattr(core,'laser_fusion',None),'predictor',None)
        tracks=predictor.tracks if predictor is not None else []
        risk=predicted_body_risk(tracks,pose,now,RISK_COMMANDS,self.footprint,
            self.horizon,self.period,margin_m=.04+.05*held_age,
            uncertainty_m_s=self.uncertainty)
        scores=[0. if row['predicted_margin_contact_s'] is None else
            1.-row['predicted_margin_contact_s']/(self.horizon+self.period) for row in risk]
        return np.asarray(scores+[float(risk[0]['supported_tracks']>0)],np.float32)
