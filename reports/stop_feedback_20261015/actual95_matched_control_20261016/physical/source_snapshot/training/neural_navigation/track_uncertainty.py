"""Observed centroid innovations for a diagnostic velocity-error envelope.

This is an empirical RMS envelope, not a Kalman filter or calibrated confidence
interval. Partial contours may bias centroid motion; large residuals should
increase uncertainty rather than imply precise object centers.
"""
from collections import deque
import numpy as np


class TrackInnovationEnvelope:
    def __init__(self,floor_m_s=.025,scale=2.,window=8):
        if not np.isfinite([floor_m_s,scale]).all() or floor_m_s<0 or scale<=0 or window<2:
            raise ValueError('Finite nonnegative floor, positive scale and window >=2 required')
        self.floor=float(floor_m_s);self.scale=float(scale);self.window=int(window)
        self.states={};self.last_time=None

    def update(self,tracks,timestamp):
        if not np.isfinite(timestamp):raise ValueError('Finite timestamp required')
        if self.last_time is not None and timestamp<self.last_time:
            raise ValueError('Monotonic diagnostic timestamps required')
        self.last_time=timestamp;output=[];present=set()
        for track in tracks:
            key=track['id'];stamp=float(track['time']);center=np.asarray(track['center'],float)
            velocity=np.asarray(track['velocity'],float)
            if center.shape!=(2,) or velocity.shape!=(2,) or not np.isfinite(np.r_[center,velocity,stamp]).all() or stamp>timestamp+1e-9:
                raise ValueError('Finite measured track state required')
            if timestamp-stamp>.8+1e-9:continue
            present.add(key);state=self.states.get(key)
            if state is None or track['hits']<state['hits']:
                state=dict(time=stamp,center=center.copy(),hits=track['hits'],residuals=deque(maxlen=self.window))
            elif stamp>state['time']:
                dt=stamp-state['time']
                # Compare the fresh observed displacement with the reported
                # smoothed velocity; the second moment also retains bias.
                residual=(center-state['center'])/dt-velocity
                state['residuals'].append(residual.copy())
                state.update(time=stamp,center=center.copy(),hits=track['hits'])
            elif stamp<state['time']:
                raise ValueError('Track measurement time moved backwards')
            residuals=np.asarray(state['residuals']).reshape(-1,2)
            covariance=residuals.T@residuals/len(residuals) if len(residuals) else np.zeros((2,2))
            empirical=self.scale*np.sqrt(max(0.,float(np.linalg.eigvalsh(covariance)[-1])))
            annotated=dict(track,velocity_uncertainty_m_s=max(self.floor,float(empirical)),
                velocity_uncertainty_samples=len(residuals))
            output.append(annotated);self.states[key]=state
        self.states={key:state for key,state in self.states.items() if key in present}
        return output
