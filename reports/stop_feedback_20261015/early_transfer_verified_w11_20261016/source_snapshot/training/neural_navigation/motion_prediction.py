"""Short-horizon scan-cluster prediction, using measured map-frame endpoints.

No simulator actors, velocities, goals or future trajectories are accepted.
This optional engineering predictor augments neural routing inputs; it never
generates or rewrites the PPO velocity action. Constant velocity is approximate.
"""
import numpy as np


class ScanMotionPredictor:
    def __init__(self,config):
        self.config=config;self.reset()
    def reset(self):
        self.tracks=[];self.next_id=0;self.last_time=None
        self.predicted_points=np.empty((0,2));self.moving_tracks=0
    def update(self,points,timestamp,alignment_supported=True):
        if self.last_time is not None and timestamp<=self.last_time:return self.predicted_points
        self.predicted_points=np.empty((0,2));self.moving_tracks=0
        self.last_time=timestamp
        if not alignment_supported:
            self.tracks=[];return self.predicted_points
        points=np.asarray(points,float).reshape(-1,2);points=points[np.isfinite(points).all(axis=1)];groups=[]
        self.tracks=[t for t in self.tracks if timestamp-t['time']<=.8]
        if len(points):
            # Ordered laser returns split at spatial gaps. Map-matched static
            # returns have already been removed by the caller.
            cuts=np.flatnonzero(np.linalg.norm(np.diff(points,axis=0),axis=1)>.18)+1
            for group in np.split(points,cuts):
                if len(group)<4:continue
                center=np.median(group,axis=0)
                if np.max(np.linalg.norm(group-center,axis=1))>.7:continue
                groups.append((center,group))
        pairs=[]
        for ti,track in enumerate(self.tracks):
            dt=timestamp-track['time'];expected=track['center']+track['velocity']*dt
            for gi,(center,_) in enumerate(groups):
                distance=float(np.linalg.norm(center-expected))
                if distance<=.22+.5*dt:pairs.append((distance,ti,gi))
        used_tracks=set();used_groups=set()
        for _,ti,gi in sorted(pairs):
            if ti in used_tracks or gi in used_groups:continue
            used_tracks.add(ti);used_groups.add(gi);track=self.tracks[ti];center,group=groups[gi]
            dt=timestamp-track['time'];velocity=(center-track['center'])/dt
            if np.linalg.norm(velocity)>.5:
                # Association or pose discontinuity: do not forecast that jump.
                track['velocity']=np.zeros(2);track['hits']=0
            elif np.dot(velocity,track['velocity'])<0 and np.linalg.norm(velocity)>.04:
                # A reversal invalidates the previous confident direction;
                # require fresh confirmations rather than forecasting that way.
                track['velocity']=velocity;track['hits']=0
            else:track['velocity']=.55*track['velocity']+.45*velocity
            track.update(center=center,points=group,time=timestamp,hits=track['hits']+1)
        for gi,(center,group) in enumerate(groups):
            if gi in used_groups:continue
            self.tracks.append(dict(id=self.next_id,center=center,points=group,velocity=np.zeros(2),time=timestamp,hits=1))
            self.next_id+=1
        self.tracks=[t for t in self.tracks if timestamp-t['time']<=.8]
        predicted=[]
        for track in self.tracks:
            speed=float(np.linalg.norm(track['velocity']))
            if track['hits']<3 or speed<float(self.config.get('minimum_speed_m_s',.04)):continue
            self.moving_tracks+=1;age=timestamp-track['time']
            horizon=float(self.config.get('horizon_s',2.4))
            for future in np.arange(.4,horizon+.001,.4):
                displaced=track['points']+track['velocity']*(age+future)
                uncertainty=float(self.config.get('uncertainty_m_s',.025))*(age+future)
                for offset in [[0,0],[uncertainty,0],[-uncertainty,0],[0,uncertainty],[0,-uncertainty]]:
                    predicted.extend(displaced+offset)
        if predicted:self.predicted_points=np.asarray(predicted,float)
        return self.predicted_points
