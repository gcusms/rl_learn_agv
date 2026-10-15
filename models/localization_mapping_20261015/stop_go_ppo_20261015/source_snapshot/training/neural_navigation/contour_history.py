"""Bounded contour history for gradual occlusion, never a steering controller."""
import copy
import numpy as np
from neural_navigation.contour_velocity import ContourAwareVelocity


class CumulativeContourPrediction:
    def __init__(self,registered_motion=False):
        self.registered_motion=registered_motion;self.states={};self.last_time=None

    def update(self,tracks,timestamp):
        if not np.isfinite(timestamp) or (self.last_time is not None and timestamp<self.last_time):raise ValueError('Monotonic finite time required')
        self.last_time=timestamp;present=set();output=[]
        for raw in tracks:
            track={k:copy.deepcopy(raw[k]) for k in ('id','time','center','velocity','hits','points')}
            if self.registered_motion:
                track['motion_registration_supported']=bool(raw.get('motion_registration_supported',False))
                track['motion_velocity_second_moment']=np.asarray(raw.get('motion_velocity_second_moment',np.zeros((2,2))),float).copy()
            key=track['id'];stamp=float(track['time']);points=np.asarray(track['points'],float)
            velocity=np.asarray(track['velocity'],float);center=np.asarray(track['center'],float)
            if points.ndim!=2 or points.shape[1]!=2 or velocity.shape!=(2,) or center.shape!=(2,) or not np.isfinite(np.r_[points.ravel(),velocity,center,stamp]).all() or stamp>timestamp+1e-9:
                raise ValueError('Finite native measurements required')
            present.add(key);diameter,count=ContourAwareVelocity.geometry(points);state=self.states.get(key)
            if state is not None and stamp<state['time']:raise ValueError('Track clock moved backwards')
            if state is None or (not self.registered_motion and track['hits']<state['hits']):
                state=dict(time=stamp,hits=track['hits'],peak_count=count,peak_diameter=diameter,
                    degraded=False,recovery=0,anchor=None,peak_speed=0.,last=track)
            elif stamp>state['time']:
                loss=count<.8*state['peak_count'] and diameter<.75*state['peak_diameter']
                if loss:state['degraded']=True;state['recovery']=0
                elif state['degraded']:
                    restored=count>=.8*state['peak_count'] and diameter>=.85*state['peak_diameter']
                    state['recovery']=state['recovery']+1 if restored else 0
                    if state['recovery']>=2:state['degraded']=False
                state.update(time=stamp,hits=track['hits'],last=track)
            if not state['degraded']:
                state['peak_count']=max(count,state['peak_count']);state['peak_diameter']=max(diameter,state['peak_diameter'])
                if not self.registered_motion and track['hits']>=3 and count>=4:state['anchor']=(stamp,velocity.copy())
            if self.registered_motion and track['motion_registration_supported'] and track['hits']>=3:
                state['anchor']=(stamp,velocity.copy())
            state['peak_speed']=max(state['peak_speed'],float(np.linalg.norm(velocity)))
            state['last']=track;self.states[key]=state
        for key,state in list(self.states.items()):
            age=timestamp-state['time']
            if age>1.2+1e-9:del self.states[key];continue
            track=copy.deepcopy(state['last']);anchor=state['anchor']
            source='observed' if key in present and not state['degraded'] and track['hits']>=3 and len(track['points'])>=4 else 'unknown'
            if self.registered_motion:
                source='observed' if age<1e-8 and track.get('motion_registration_supported') and track['hits']>=3 else 'unknown'
            velocity=np.asarray(track['velocity'],float)
            if source!='observed' and anchor is not None:
                velocity=anchor[1].copy()
                if (key in present or self.registered_motion) and timestamp-anchor[0]<=.6+1e-9:source='held'
            diameter,_=ContourAwareVelocity.geometry(np.asarray(track['points']))
            output.append(dict(track,velocity=velocity,source=source,degraded=state['degraded'],
                contour_position_uncertainty_m=max(0.,state['peak_diameter']-diameter)/2,
                reachable_speed_hypothesis_m_s=max(.04,state['peak_speed'])+.025,
                age=age,retained_without_measurement=(age>1e-8 if self.registered_motion else key not in present)))
        return output

