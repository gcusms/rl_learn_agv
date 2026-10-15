"""Diagnostic forecast using direction-dependent observed velocity residuals.

No runtime integration, command selection, truth labels, or confidence claim.
Point contours still omit unobserved object shape and association ambiguity.
"""
import numpy as np
from navrl.geometry import integrate


def directional_body_risk(tracks,pose,timestamp,commands,footprint,
                          horizon_s=2.4,period_s=.2,margin_m=.04,
                          uncertainty_floor_m_s=.025,residual_scale=2.,
                          measured_velocity=None,response_gain=20.,response_acceleration=(.5,1.)):
    pose=np.asarray(pose,float);commands=np.asarray(commands,float)
    half=np.asarray(footprint,float)/2
    if pose.shape!=(3,) or commands.ndim!=2 or commands.shape[1]!=2 or half.shape!=(2,):
        raise ValueError('Pose, commands and footprint dimensions required')
    if not np.isfinite(np.r_[pose,commands.ravel(),half,timestamp,horizon_s,period_s,margin_m,uncertainty_floor_m_s,residual_scale]).all() or min(half)<=0 or min(horizon_s,period_s,residual_scale)<=0 or min(margin_m,uncertainty_floor_m_s)<0:
        raise ValueError('Finite positive geometry and valid uncertainty required')
    supported=[]
    for track in tracks:
        points=np.asarray(track['points'],float);velocity=np.asarray(track['velocity'],float)
        age=timestamp-float(track['time'])
        second=np.asarray(track.get('velocity_error_second_moment',np.zeros((2,2))),float)
        if velocity.shape!=(2,) or points.ndim!=2 or points.shape[1]!=2 or second.shape!=(2,2) or not np.isfinite(np.r_[points.ravel(),velocity,second.ravel(),age]).all():
            raise ValueError('Finite track points and residual second moment required')
        if not np.allclose(second,second.T,atol=1e-12,rtol=0) or np.linalg.eigvalsh(second)[0]<-1e-12:
            raise ValueError('Symmetric positive semidefinite second moment required')
        if 0<=age<=.8 and track['hits']>=3 and np.linalg.norm(velocity)>=.04 and len(points):
            supported.append((points,velocity,age,second))
    times=np.arange(0.,horizon_s+1e-9,period_s)
    if times[-1]<horizon_s-1e-9:times=np.r_[times,horizon_s]
    trajectories=None
    if measured_velocity is not None:
        from neural_navigation.motion_response import force_response_trajectory
        times=np.linspace(0.,horizon_s,int(np.ceil(horizon_s/period_s))+1)
        trajectories=[force_response_trajectory(measured_velocity,command,horizon_s,
            gain=response_gain,acceleration=response_acceleration,samples=len(times))
            for command in commands]
    result=[]
    for command_index,command in enumerate(commands):
        first=None;minimum=None
        for sample_index,future in enumerate(times):
            if trajectories is None:body=integrate(pose,*command,float(future))
            else:
                local=trajectories[command_index][sample_index]
                c,s=np.cos(pose[2]),np.sin(pose[2])
                body=np.r_[pose[:2]+local[:2]@np.array([[c,s],[-s,c]]),pose[2]+local[2]]
            c,s=np.cos(body[2]),np.sin(body[2]);rotation=np.array([[c,-s],[s,c]])
            for points,velocity,age,second in supported:
                relative=(points+velocity*(age+future)-body[:2])@rotation
                delta=relative-np.clip(relative,-half,half)
                gap=np.linalg.norm(delta,axis=1)
                # Robot-frame normals become map-frame normals for the
                # map-frame residual matrix. Interior points already contact.
                normals=np.divide(delta,gap[:,None],out=np.zeros_like(delta),where=gap[:,None]>1e-12)@rotation.T
                projected=np.maximum(0.,np.einsum('ni,ij,nj->n',normals,second,normals))
                spread=np.maximum(uncertainty_floor_m_s,residual_scale*np.sqrt(projected))
                clearance=float(np.min(gap-margin_m-spread*(age+future)))
                minimum=clearance if minimum is None else min(minimum,clearance)
                if clearance<=1e-9 and first is None:first=float(future)
        result.append(dict(command=command.tolist(),supported_tracks=len(supported),
            predicted_margin_contact_s=first,minimum_predicted_clearance_m=minimum))
    return result
