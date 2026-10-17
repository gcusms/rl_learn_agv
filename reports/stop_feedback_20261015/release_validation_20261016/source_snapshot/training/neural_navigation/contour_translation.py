"""Estimate translation from overlapping measured contours, not their centroid.

This bounded point-to-line registration has no object shape, map or simulator
inputs. A poorly constrained fit remains unsupported; it is not zero motion.
"""
import copy
import numpy as np


def contour_translation(reference, current):
    a=np.asarray(reference,float);b=np.asarray(current,float)
    if min(len(a),len(b))<6:return None
    # Bound quadratic nearest-neighbour work, retaining the scan ordering.
    a=a[np.linspace(0,len(a)-1,min(len(a),64)).astype(int)]
    b=b[np.linspace(0,len(b)-1,min(len(b),64)).astype(int)]
    j=np.arange(len(b));tangent=b[np.minimum(j+2,len(b)-1)]-b[np.maximum(j-2,0)]
    length=np.linalg.norm(tangent,axis=1)
    normals=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(length[:,None],1e-9)
    displacement=np.zeros(2)
    for _ in range(12):
        moved=a+displacement
        distances=np.sum((moved[:,None]-b[None,:])**2,axis=2)
        nearest=distances.argmin(axis=1);n=normals[nearest]
        residual=np.sum(n*(b[nearest]-moved),axis=1)
        weight=np.minimum(1.,.012/np.maximum(abs(residual),1e-9))
        weight*=(distances[np.arange(len(a)),nearest]<.12**2)&(length[nearest]>.005)
        # A missing continuation of a surface is not a correspondence with
        # its new endpoint. Those artificial edge matches bias translation.
        weight*=(nearest>=2)&(nearest<len(b)-2)
        information=(n*weight[:,None]).T@n;eigen=np.linalg.eigvalsh(information)
        if eigen[0]<.6 or eigen[0]/max(eigen[1],1e-9)<.04 or np.count_nonzero(weight) < .6*len(a):return None
        delta=np.linalg.solve(information,n.T@(weight*residual))
        displacement+=delta
        if np.linalg.norm(displacement)>.30:return None
        if np.linalg.norm(delta)<1e-4:break
    error=float(np.sqrt(np.sum(weight*residual**2)/max(weight.sum(),1.)))
    if error>.02:return None
    return displacement,error,np.linalg.inv(information)*max(error**2,.005**2)


class RegisteredContourMotion:
    def __init__(self):self.states={};self.last_time=None

    def update(self,tracks,timestamp):
        if not np.isfinite(timestamp) or self.last_time is not None and timestamp<self.last_time:
            raise ValueError('Finite monotonic measurement time required')
        self.last_time=timestamp;output=[]
        for raw in tracks:
            track=copy.deepcopy(raw);key=track['id'];stamp=float(track['time'])
            points=np.asarray(track['points'],float)
            if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(np.r_[points.ravel(),stamp]).all() or stamp>timestamp+1e-9:
                raise ValueError('Finite measured contours required')
            state=self.states.get(key)
            if state is not None and stamp<state['time']:raise ValueError('Measurement clock moved backwards')
            if state is None:
                state=dict(time=stamp,history=[],velocity=np.zeros(2),hits=1,supported=False,error=None,second=np.zeros((2,2)))
            if stamp>state['time'] or not state['history']:
                fits=[]
                for previous,reference in state['history']:
                    dt=stamp-previous
                    if not .15<=dt<=.61:continue
                    fit=contour_translation(reference,points)
                    if fit is not None:
                        velocity=fit[0]/dt
                        if np.linalg.norm(velocity)<=.5:fits.append((velocity,fit[1],dt,fit[2]/dt**2))
                state['supported']=bool(fits)
                if fits:
                    # Longer baselines reduce derivative noise. A reversal is
                    # kept in the measured estimates, not reset to zero speed.
                    weights=np.array([f[2] for f in fits]);velocities=np.array([f[0] for f in fits])
                    state['velocity']=np.average(velocities,axis=0,weights=weights)
                    state['error']=max(f[1] for f in fits);state['hits']+=1
                    delta=velocities-state['velocity']
                    # Fit disagreement and registration information use the
                    # same measured surface model as the velocity estimate.
                    state['second']=np.average(np.array([f[3] for f in fits])+delta[:,:,None]*delta[:,None,:],axis=0,weights=weights)
                state['history']=[(t,p) for t,p in state['history'] if stamp-t<.61]
                state['history'].append((stamp,points.copy()));state['time']=stamp
            fresh=abs(timestamp-stamp)<1e-8
            track.update(velocity=state['velocity'].copy(),hits=state['hits'],
                motion_registration_supported=bool(fresh and state['supported']),
                motion_registration_error_m=state['error'],motion_velocity_second_moment=state['second'].copy())
            output.append(track);self.states[key]=state
        self.states={key:state for key,state in self.states.items() if timestamp-state['time']<=1.2+1e-9}
        return output
