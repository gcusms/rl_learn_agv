"""Sensor-only contour quality and bounded velocity hold; experimental.

Not imported by the active runtime. A held velocity is a hypothesis, not a
confidence certificate; unknown quality must remain explicit to consumers.
"""
import numpy as np


class ContourAwareVelocity:
    def __init__(self,hold_s=.6,recovery_measurements=2):
        if not np.isfinite(hold_s) or hold_s<=0 or recovery_measurements<2:
            raise ValueError('Positive finite hold and at least two recovery measurements required')
        self.hold=float(hold_s);self.recovery=int(recovery_measurements)
        self.states={};self.last_time=None

    @staticmethod
    def geometry(points):
        if not len(points):return 0.,0
        # Diameter is invariant to map rotation; x/y spans are not.
        diameter=float(np.sqrt(np.max(np.sum((points[:,None]-points[None,:])**2,axis=2))))
        return diameter,len(points)

    def update(self,tracks,timestamp):
        if not np.isfinite(timestamp) or (self.last_time is not None and timestamp<self.last_time):
            raise ValueError('Finite monotonic timestamps required')
        self.last_time=timestamp;output=[];present=set()
        for track in tracks:
            key=track['id'];stamp=float(track['time']);points=np.asarray(track['points'],float)
            velocity=np.asarray(track['velocity'],float);center=np.asarray(track['center'],float)
            if points.ndim!=2 or points.shape[1]!=2 or velocity.shape!=(2,) or center.shape!=(2,) or not np.isfinite(np.r_[points.ravel(),velocity,center,stamp]).all() or stamp>timestamp+1e-9:
                raise ValueError('Finite native track measurements required')
            if timestamp-stamp>.8+1e-9:continue
            present.add(key);diameter,count=self.geometry(points);state=self.states.get(key)
            if state is not None and stamp<state['time']:raise ValueError('Track time moved backwards')
            new=state is None or track['hits']<state['hits']
            if new:
                state=dict(time=stamp,hits=track['hits'],diameter=diameter,count=count,
                    unstable=False,recovery=0,loss=False,anchor=None)
                if track['hits']>=3 and count>=4:state['anchor']=(stamp,velocity.copy())
            elif stamp>state['time']:
                loss=(count<.5*state['count'] and diameter<.75*state['diameter'])
                if loss:state.update(unstable=True,recovery=0)
                elif state['unstable']:
                    state['recovery']+=1
                    if state['recovery']>=self.recovery:state['unstable']=False
                if not state['unstable'] and track['hits']>=3 and count>=4:
                    state['anchor']=(stamp,velocity.copy())
                state.update(time=stamp,hits=track['hits'],diameter=diameter,count=count,loss=loss)
            anchor=state['anchor'];supported=track['hits']>=3 and count>=4 and not state['unstable']
            source='observed' if supported else 'unknown';selected=velocity.copy()
            if not supported and state['unstable'] and anchor is not None:
                age=timestamp-anchor[0]
                if 0<=age<=self.hold+1e-9:selected=anchor[1].copy();source='held'
            output.append(dict(track,velocity=selected,
                contour_velocity_source=source,contour_velocity_supported=source!='unknown',
                contour_degraded=bool(state['unstable']),contour_loss_event=bool(state['loss']),
                contour_recovery_measurements=state['recovery'],
                contour_velocity_confidence_calibrated=False))
            self.states[key]=state
        self.states={k:s for k,s in self.states.items() if k in present}
        return output
