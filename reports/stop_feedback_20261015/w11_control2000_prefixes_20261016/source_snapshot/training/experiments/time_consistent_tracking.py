"""Experimental time-based confirmation/filtering for sensor-only tracks."""
import numpy as np
import json
from pathlib import Path
from neural_navigation.motion_prediction import ScanMotionPredictor
from experiments.isolated_frame_prediction import IsolatedFramePredictionRuntime
from experiments.fresh_perception_runtime import FreshPerceptionCore
from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime


def sensor_core_contract():
    return dict(fusion_every_sensor_frame=True,confirmation_window_s=.8,
                velocity_filter_reference_period_s=.4,velocity_filter_reference_retention=.55)


class TimeConsistentPredictor(ScanMotionPredictor):
    def update(self,points,timestamp,alignment_supported=True):
        if self.last_time is not None and timestamp<=self.last_time:
            return self.predicted_points
        previous={t['id']:dict(time=t['time'],center=t['center'].copy(),
            velocity=t['velocity'].copy(),hits=t['hits'],
            confirmation_started=t.get('confirmation_started',t['time'])) for t in self.tracks}
        super().update(points,timestamp,alignment_supported)
        for track in self.tracks:
            old=previous.get(track['id'])
            if old is None:
                track['confirmation_started']=track['time']
            elif track['time']>old['time']:
                if track['hits']!=old['hits']+1:
                    track['confirmation_started']=track['time']
                else:
                    dt=track['time']-old['time']
                    measured=(track['center']-old['center'])/dt
                    retention=.55**(dt/.4)
                    track['velocity']=retention*old['velocity']+(1-retention)*measured
                    track['confirmation_started']=old['confirmation_started']
            # Three legacy samples at .4 s span .8 s. Fresh measurements
            # cannot shorten that physical confirmation window.
            if track['time']-track['confirmation_started']<.8-1e-9:
                track['hits']=min(track['hits'],2)
        # Rebuild forecasts after adjusting tracks; never expose stale output.
        predicted=[];self.moving_tracks=0
        for track in self.tracks:
            if track['hits']<3 or np.linalg.norm(track['velocity'])<float(self.config.get('minimum_speed_m_s',.04)):
                continue
            self.moving_tracks+=1;age=timestamp-track['time']
            for future in np.arange(.4,float(self.config.get('horizon_s',2.4))+.001,.4):
                displaced=track['points']+track['velocity']*(age+future)
                uncertainty=float(self.config.get('uncertainty_m_s',.025))*(age+future)
                for offset in ([0,0],[uncertainty,0],[-uncertainty,0],[0,uncertainty],[0,-uncertainty]):
                    predicted.extend(displaced+offset)
        self.predicted_points=np.asarray(predicted,float).reshape(-1,2)
        return self.predicted_points


class TimeConsistentRiskRuntime(IsolatedFramePredictionRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.prediction_view.laser_fusion.predictor=TimeConsistentPredictor(self.core.motion_prediction_config)


class TimeConsistentCore(FreshPerceptionCore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.laser_fusion.predictor=TimeConsistentPredictor(self.motion_prediction_config)


class TimeConsistentFreshRuntime(QualitySensorNavigationRuntime):
    experiment_variant='fresh_fusion_time_consistent_confirmation_filter'
    sensor_core_variants=('native','time-consistent')
    def __init__(self,bundle,map_path,initial_pose,goal,route_mode='raw'):
        manifest=json.loads((Path(bundle)/'manifest.json').read_text())
        if manifest.get('sensor_core')=='time-consistent' and manifest.get('sensor_core_parameters')!=sensor_core_contract():
            raise ValueError('Time-consistent sensor core contract mismatch')
        super().__init__(bundle,map_path,initial_pose,goal,route_mode)
        core=TimeConsistentCore(self.core.static_map,self.core.lidar,bundle,route_mode)
        core.reset(initial_pose,goal)
        self.core=core;self.policy.core=core

    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        result['perception_update_count']=self.core.perception_update_count
        result['perception_update_time']=self.core.last_perception_time
        return result
