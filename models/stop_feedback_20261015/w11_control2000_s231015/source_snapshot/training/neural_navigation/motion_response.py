"""Finite-response trajectory for a declared, uncalibrated force-response model."""
import numpy as np


def force_response_trajectory(velocity,command,horizon,gain=20.,acceleration=(.5,1.),samples=9):
    velocity=np.asarray(velocity,float);command=np.asarray(command,float)
    acceleration=np.asarray(acceleration,float)
    if velocity.shape!=(2,) or command.shape!=(2,) or acceleration.shape!=(2,):
        raise ValueError('Two-dimensional velocity, command and acceleration required')
    if not np.isfinite(np.r_[velocity,command,acceleration,horizon,gain]).all() or min(acceleration)<=0 or gain<=0 or horizon<=0 or samples<2:
        raise ValueError('Finite positive response parameters required')
    count=max(samples-1,int(np.ceil(horizon/.005/(samples-1)))*(samples-1))
    dt=horizon/count;edges=np.linspace(0.,horizon,count+1);mid=(edges[:-1]+edges[1:])/2
    def state_and_integral(index,times):
        error=velocity[index]-command[index];magnitude=abs(error);a=acceleration[index]
        saturated=max(0.,(magnitude-a/gain)/a);first=np.minimum(times,saturated)
        amplitude=magnitude-a*saturated;tail=np.maximum(times-saturated,0.)
        residual=np.where(times<=saturated,magnitude-a*times,amplitude*np.exp(-gain*tail))
        integral=magnitude*first-.5*a*first**2+amplitude/gain*(1-np.exp(-gain*tail))
        return command[index]+np.sign(error)*residual,command[index]*times+np.sign(error)*integral
    speed,_=state_and_integral(0,mid);_,yaw=state_and_integral(1,mid)
    increment=dt*speed[:,None]*np.c_[np.cos(yaw),np.sin(yaw)]
    xy=np.vstack([np.zeros(2),np.cumsum(increment,axis=0)])
    _,heading=state_and_integral(1,edges)
    poses=np.c_[xy,heading]
    return poses[::count//(samples-1)]
