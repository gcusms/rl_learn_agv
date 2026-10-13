"""Temporal known-map/partial-laser fusion, with free-space evidence and near risk."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from navrl.geometry import wrap
from neural_navigation.motion_prediction import ScanMotionPredictor

class LaserMapFusion:
    def __init__(self,env):
        self.points=np.array([env.xy((r,c)) for r in range(env.grid_h) for c in range(env.grid_w)]).reshape(env.grid_h,env.grid_w,2)
        self.shape=env.free.shape;self.reset()
        config=getattr(env,'motion_prediction_config',None)
        self.predictor=ScanMotionPredictor(config) if config else None
    def reset(self):
        self.first_seen=np.full(self.shape,np.nan);self.last_seen=np.full(self.shape,-100.);self.hard_expiry=np.zeros(self.shape)
        self.hard=np.zeros(self.shape,bool);self.layer=np.zeros(self.shape,np.float32);self.cleared=0;self.far_pending=0
        self.forecast_cells=0
        self.last_alignment_supported=False
        if getattr(self,'predictor',None):self.predictor.reset()
    def update(self,env):
        now=env.elapsed;pose=env.estimate;yaw=pose[2];mount=env.lidar.mount
        self.hard_expiry=np.maximum(self.hard_expiry,env.expiry)
        origin=pose[:2]+[np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1],np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1]]
        scan=np.asarray(env.scan);valid=np.isfinite(scan)&(scan>=env.lidar.range_min)&(scan<env.lidar.range_max)
        angles=env.lidar.angles[valid]+yaw+mount[3]
        ends=origin+scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]
        keep=env.static_map.lookup(ends,env.static_map.distance,outside=0.)>.22;ends=ends[keep]
        hit=np.zeros(self.shape,bool);close=np.zeros(self.shape,bool);urgent=np.zeros(self.shape,bool)
        for point in ends:
            r,c=env.cell(point)
            if not (0<=r<env.grid_h and 0<=c<env.grid_w):continue
            distance=np.linalg.norm(point-pose[:2]);hit[r,c]=True
            if distance<2.5:close[r,c]=True
            if distance<.8:urgent[r,c]=True
        # Clear remembered occupancy only with actual, finite free-space returns.
        delta=self.points-origin;distance=np.linalg.norm(delta,axis=-1)
        bearing=wrap(np.arctan2(delta[...,1],delta[...,0])-yaw-mount[3])
        idx=np.clip(np.searchsorted(env.lidar.angles,bearing),0,len(scan)-1)
        ranges=np.where(valid,scan,0.)
        observed=np.minimum.reduce([ranges[np.clip(idx+k,0,len(scan)-1)] for k in [-1,0,1]])
        clear=(bearing>=env.lidar.angle_min)&(bearing<=env.lidar.angle_max)&(observed>distance+.22)&~hit
        self.cleared=int(np.count_nonzero(clear&(self.hard_expiry>now)))
        self.first_seen[clear]=np.nan;self.last_seen[clear]=-100.;self.hard_expiry[clear]=0.
        fresh=hit&((now-self.last_seen>.65)|np.isnan(self.first_seen));self.first_seen[fresh]=now
        self.last_seen[hit]=now;age=now-self.first_seen
        confirmed=hit&(urgent|(close&(age>=.35))|(age>=2.0))
        self.hard_expiry[confirmed]=now+env.dynamic_memory_seconds
        # The legacy expiry field remains available for diagnostics and old tests.
        env.expiry[:]=self.hard_expiry
        occupied=self.hard_expiry>now
        self.hard=distance_transform_edt(~occupied)*env.grid_res<=.32 if occupied.any() else np.zeros(self.shape,bool)
        if self.predictor is not None:
            if getattr(env.localizer,'scan_update_accepted',True):
                self.last_alignment_supported=getattr(env.localizer,'map_agreement',1.)>=.5
            # A brief missing scan is occlusion evidence, not proof of a bad
            # map pose. Retain bounded prediction from the last supported scan.
            supported=self.last_alignment_supported and max(env.localizer.std[:2])<.2 and env.localizer.std[2]<.25
            predicted=self.predictor.update(ends,now,supported)
            future=np.zeros(self.shape,bool)
            if len(predicted):
                r,c=env.static_map.cells(predicted);r=r//4;c=c//4
                inside=(r>=0)&(r<env.grid_h)&(c>=0)&(c<env.grid_w)
                future[r[inside],c[inside]]=True
                future=distance_transform_edt(~future)*env.grid_res<=.32
            self.forecast_cells=int(np.count_nonzero(future&~self.hard))
            # Forecast is recomputed, never written into the eight-second
            # occupancy memory or cleared using today's free rays.
            self.hard|=future
        recent=(now-self.last_seen<.8)&~occupied
        soft=distance_transform_edt(~recent)*env.grid_res<=.32 if recent.any() else np.zeros(self.shape,bool)
        self.layer=np.maximum(self.hard.astype(np.float32),soft.astype(np.float32)*.02)
        self.far_pending=int(np.count_nonzero(hit&~close&~confirmed));env.unmatched_returns=len(ends)
        return self.layer
