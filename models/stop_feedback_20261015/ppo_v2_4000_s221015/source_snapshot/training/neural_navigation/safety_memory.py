"""Near-field measured endpoints retained across the rear blind sector.

Memory is in the supplied continuous odometry frame, not corrected map pose.
Only actual finite free-ray evidence or distance from the robot clears entries;
time alone does not prove that a parked obstacle disappeared. This is an
experimental safety aid, not a calibrated occupancy/velocity estimator.
"""
import numpy as np


class NearFieldMemory:
    def __init__(self,config):
        self.config=config
        margin=float(config.get('free_evidence_margin_m',.1))
        frames=config.get('free_evidence_frames',1)
        gap=float(config.get('free_evidence_max_gap_s',.4))
        if not np.isfinite([margin,gap]).all() or margin<=0 or gap<=0 or not isinstance(frames,int) or frames<1:
            raise ValueError('Positive free-evidence margin, frame count and time gap required')
        self.reset()
    def reset(self):self.entries={};self.last_time=None;self.free_confirmations={}
    def update(self,core):
        now=core.elapsed;pose=core.odom;spec=core.lidar;scan=np.asarray(core.scan)
        valid=np.isfinite(scan)&(scan>=spec.range_min)&(scan<spec.range_max)
        c,s=np.cos(pose[2]),np.sin(pose[2]);rotation=np.array([[c,-s],[s,c]])
        old_keys=list(self.entries)
        if old_keys:
            world=np.array([self.entries[k][0] for k in old_keys]);body=(world-pose[:2])@rotation
            mount=spec.mount;mc,ms=np.cos(mount[3]),np.sin(mount[3]);lidar=(body-mount[:2])@np.array([[mc,-ms],[ms,mc]])
            distance=np.linalg.norm(lidar,axis=1);bearing=np.arctan2(lidar[:,1],lidar[:,0])
            index=np.clip(np.searchsorted(spec.angles,bearing),0,len(scan)-1)
            ranges=np.where(valid,scan,0.)
            observed=np.minimum.reduce([ranges[np.clip(index+k,0,len(scan)-1)] for k in [-1,0,1]])
            supported=(bearing>=spec.angle_min)&(bearing<=spec.angle_max)
            # Optional multi-frame clearing avoids retaining a low noise
            # outlier forever. Unknown/occluded rays never count as free.
            clearance=float(self.config.get('free_evidence_margin_m',.1))
            required=int(self.config.get('free_evidence_frames',1))
            max_gap=float(self.config.get('free_evidence_max_gap_s',.4))
            evidence=supported&(observed>distance+clearance)
            cleared=np.zeros(len(old_keys),bool)
            if required==1:cleared=evidence
            else:
                for i,key in enumerate(old_keys):
                    if not evidence[i]:self.free_confirmations.pop(key,None);continue
                    previous,count=self.free_confirmations.get(key,(now,0))
                    if now-previous>max_gap+1e-8:count=0
                    if count==0 or now>previous+1e-8:count+=1
                    self.free_confirmations[key]=(now,count)
                    cleared[i]=count>=required
            distant=np.linalg.norm(body,axis=1)>float(self.config.get('forget_radius_m',2.5))
            for i in np.flatnonzero(cleared|distant):
                key=old_keys[int(i)];self.entries.pop(key,None);self.free_confirmations.pop(key,None)
        angle=spec.angles[valid]+spec.mount[3]
        body=scan[valid,None]*np.c_[np.cos(angle),np.sin(angle)]+spec.mount[:2]
        body=body[np.linalg.norm(body,axis=1)<=float(self.config.get('near_range_m',1.5))]
        world=body@rotation.T+pose[:2]
        resolution=float(self.config.get('resolution_m',.04))
        for point in world:
            key=tuple(np.floor(point/resolution).astype(int))
            self.entries[key]=(point.copy(),now);self.free_confirmations.pop(key,None)
        keys=list(self.entries)
        if not keys:self.last_time=now;return np.empty((0,2)),np.empty(0)
        points=np.array([self.entries[k][0] for k in keys]);times=np.array([self.entries[k][1] for k in keys])
        body=(points-pose[:2])@rotation
        maximum=int(self.config.get('max_points',4096))
        if len(keys)>maximum:
            selected=np.argsort(np.linalg.norm(body,axis=1))[:maximum];self.entries={keys[int(i)]:self.entries[keys[int(i)]] for i in selected}
            self.free_confirmations={k:v for k,v in self.free_confirmations.items() if k in self.entries}
            body=body[selected];times=times[selected]
        self.last_time=now
        uncertainty=np.minimum(float(self.config.get('max_uncertainty_m',.04)),np.maximum(now-times,0)*float(self.config.get('uncertainty_m_s',.002)))
        return body,uncertainty
