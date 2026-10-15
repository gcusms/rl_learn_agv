"""Fresh sensor tracker for actor risk inputs, leaving native routing intact."""
from types import SimpleNamespace
import numpy as np
from neural_navigation.motion_prediction import ScanMotionPredictor
from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime,QualityPolicyInput


class PredictionCoreView:
    def __init__(self,core):
        self.core=core
        config=core.motion_prediction_config
        if not config:raise ValueError('Configured sensor motion prediction required')
        self.laser_fusion=SimpleNamespace(predictor=ScanMotionPredictor(config))
        self.last_alignment_supported=False
        self.update_count=0

    def __getattr__(self,name):return getattr(self.core,name)

    def update_tracker(self):
        core=self.core;predictor=self.laser_fusion.predictor
        if predictor.last_time is not None and core.elapsed<=predictor.last_time:return
        scan=np.asarray(core.scan);lidar=core.lidar;pose=core.estimate;mount=lidar.mount
        valid=np.isfinite(scan)&(scan>=lidar.range_min)&(scan<lidar.range_max)
        c,s=np.cos(pose[2]),np.sin(pose[2])
        origin=pose[:2]+[c*mount[0]-s*mount[1],s*mount[0]+c*mount[1]]
        angles=lidar.angles[valid]+pose[2]+mount[3]
        ends=origin+scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]
        ends=ends[core.static_map.lookup(ends,core.static_map.distance,outside=0.)>.22]
        if core.localizer.scan_update_accepted:
            self.last_alignment_supported=core.localizer.map_agreement>=.5
        supported=(self.last_alignment_supported and max(core.localizer.std[:2])<.2
                   and core.localizer.std[2]<.25 and not core.odometry_sanity_failure)
        predictor.update(ends,float(core.elapsed),supported)
        self.update_count+=1


class FramePredictionInput(QualityPolicyInput):
    def predict(self,prefix):
        self.core.update_tracker()
        return super().predict(prefix)


class IsolatedFramePredictionRuntime(QualitySensorNavigationRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        old=self.policy
        self.prediction_view=PredictionCoreView(self.core)
        self.policy=FramePredictionInput(old.policy,self.prediction_view,old.prediction)
