"""Opt-in, sensor-only short-horizon vetoes, shared by probes and execution.

Tracks live in the continuous odometry frame, so a map correction cannot create
object motion. This is a bounded engineering forecast, not a safety certificate.
It never chooses a velocity and never forbids the explicit stop action.
"""
import numpy as np
from neural_navigation.motion_prediction import ScanMotionPredictor
from neural_navigation.contour_velocity import ContourAwareVelocity
from neural_navigation.track_uncertainty import TrackInnovationEnvelope
from neural_navigation.directional_risk import directional_body_risk


def contract():
    return dict(schema='odometry_frame_predictive_mask_v1',horizon_s=1.2,period_s=.1,
        reaction_s=.2,minimum_check_s=.6,margin_m=.04,velocity_floor_m_s=.025,
        minimum_speed_m_s=.04,track_max_age_s=.8,minimum_track_hits=3,
        response_gain=20.,response_acceleration=[.5,1.],
        contour_hold_s=.6,contour_recovery_measurements=2,
        wait_always_allowed=True,simulator_truth_used=False,calibrated=False)


def cumulative_contract():
    return dict(contract(),schema='odometry_frame_predictive_mask_v2',
        cumulative_contour_reference=True,contour_count_ratio=.8,contour_diameter_ratio=.75,
        retained_hypothesis_s=1.2,unknown_velocity_reachable_envelope=True,
        retained_measurement_age_preserved=True)


class PredictiveConstraint:
    def __init__(self,config,footprint,braking_acceleration):
        if config not in (contract(),cumulative_contract()):raise ValueError('Predictive mask contract mismatch')
        self.config=config;self.footprint=np.asarray(footprint,float).copy()
        self.braking=float(braking_acceleration)
        if self.braking<=0 or not np.isfinite(self.braking):raise ValueError('Positive braking acceleration required')
        from neural_navigation.masked_actions import COMMANDS
        self.commands=COMMANDS.copy();self.reset()

    def reset(self):
        self.tracker=ScanMotionPredictor({'horizon_s':1.2})
        self.quality=ContourAwareVelocity();self.innovations=TrackInnovationEnvelope()
        from neural_navigation.contour_history import CumulativeContourPrediction
        self.contours=CumulativeContourPrediction()
        self.timestamp=None;self.contacts=np.full(len(self.commands),np.inf)
        self.supported_tracks=0;self.tracks=[];self.update_ms=0.

    def prepare(self,core):
        import time
        now=float(core.elapsed)
        if self.timestamp==now:return
        if self.timestamp is not None and now<self.timestamp:self.reset()
        started=time.perf_counter();self.timestamp=now;self.contacts[:]=np.inf
        self.supported_tracks=0;self.tracks=[]
        odom=np.asarray(core.odom,float);scan=np.asarray(core.scan,float)
        velocity=np.asarray(core.measured_velocity,float)
        if (odom.shape!=(3,) or velocity.shape!=(2,) or
            not np.isfinite(np.r_[odom,velocity,now]).all() or getattr(core,'odometry_sanity_failure',False)):
            self.reset();self.timestamp=now;return
        valid=np.isfinite(scan)&(scan>=core.lidar.range_min)&(scan<core.lidar.range_max)
        angle=core.lidar.angles[valid]+core.lidar.mount[3]
        body=scan[valid,None]*np.c_[np.cos(angle),np.sin(angle)]+core.lidar.mount[:2]
        c,s=np.cos(odom[2]),np.sin(odom[2]);points=body@np.array([[c,s],[-s,c]])+odom[:2]
        self.tracker.update(points,now)
        if self.config==cumulative_contract():
            self._cumulative_risk(core,now,odom,velocity)
            self.update_ms=(time.perf_counter()-started)*1000;return
        annotated=self.innovations.update(self.quality.update(self.tracker.tracks,now),now)
        reliable=[]
        for track in annotated:
            if track['contour_velocity_source']=='unknown':continue
            residuals=np.asarray(self.innovations.states[track['id']]['residuals']).reshape(-1,2)
            track['velocity_error_second_moment']=(residuals.T@residuals/len(residuals) if len(residuals) else np.zeros((2,2)))
            reliable.append(track)
        self.tracks=reliable
        reliable=[track for track in reliable if track['hits']>=self.config['minimum_track_hits']
            and np.linalg.norm(track['velocity'])>=self.config['minimum_speed_m_s']]
        if not reliable:
            self.update_ms=(time.perf_counter()-started)*1000;return
        risk=directional_body_risk(reliable,odom,now,self.commands,self.footprint,
            horizon_s=self.config['horizon_s'],period_s=self.config['period_s'],
            margin_m=self.config['margin_m'],uncertainty_floor_m_s=self.config['velocity_floor_m_s'],
            measured_velocity=velocity,response_gain=self.config['response_gain'],
            response_acceleration=self.config['response_acceleration'])
        self.supported_tracks=risk[0]['supported_tracks']
        self.contacts=np.array([np.inf if row['predicted_margin_contact_s'] is None else row['predicted_margin_contact_s'] for row in risk])
        self.update_ms=(time.perf_counter()-started)*1000

    def _cumulative_risk(self,core,now,odom,velocity):
        tracks=self.contours.update(self.tracker.tracks,now)
        self.innovations.update([t for t in tracks if now-t['time']<=.8],now)
        self.tracks=tracks
        for track in tracks:
            if track['hits']<3 or np.linalg.norm(track['velocity'])<.04:continue
            age=float(track['age']);unknown=track['source']=='unknown'
            spread=track['reachable_speed_hypothesis_m_s'] if unknown else self.config['velocity_floor_m_s']
            residuals=np.asarray(self.innovations.states.get(track['id'],{}).get('residuals',[])).reshape(-1,2)
            second=residuals.T@residuals/len(residuals) if len(residuals) else np.zeros((2,2))
            # Propagate an aged hypothesis explicitly; the uncertainty includes
            # the entire original measurement age, never a fictitious fresh ray.
            current=dict(track,points=track['points']+track['velocity']*age,time=now,
                velocity_error_second_moment=second)
            risk=directional_body_risk([current],odom,now,self.commands,self.footprint,
                horizon_s=self.config['horizon_s'],period_s=self.config['period_s'],
                margin_m=self.config['margin_m']+track['contour_position_uncertainty_m']+spread*age,
                uncertainty_floor_m_s=spread,measured_velocity=velocity,
                response_gain=self.config['response_gain'],response_acceleration=self.config['response_acceleration'])
            self.supported_tracks+=risk[0]['supported_tracks']
            self.contacts=np.minimum(self.contacts,[np.inf if r['predicted_margin_contact_s'] is None else r['predicted_margin_contact_s'] for r in risk])

    def rejection(self,core,command,age=0.):
        self.prepare(core);command=np.asarray(command,float)
        if np.allclose(command,0.,rtol=0,atol=1e-9):return None
        difference=np.max(abs(self.commands-command),axis=1);index=int(difference.argmin())
        if difference[index]>2e-8:return 'predictive_command_contract'
        velocity=np.asarray(core.measured_velocity,float)
        horizon=max(self.config['minimum_check_s'],self.config['reaction_s']+max(age,0.)+
            max(abs(command[0]),abs(velocity[0]))/self.braking,
            self.config['reaction_s']+max(age,0.)+abs(velocity[1]-command[1])/self.config['response_acceleration'][1])
        return 'predicted_dynamic_contact' if self.contacts[index]<=min(horizon,self.config['horizon_s'])+1e-9 else None

    def diagnostics(self,core):
        self.prepare(core)
        return dict(schema=self.config['schema'],supported_tracks=self.supported_tracks,
            predicted_contact_seconds=[None if not np.isfinite(t) else float(t) for t in self.contacts],
            rejected_nonzero_choices=sum(self.rejection(core,command) is not None for command in self.commands[1:]),
            compute_ms=self.update_ms,calibrated=False)
