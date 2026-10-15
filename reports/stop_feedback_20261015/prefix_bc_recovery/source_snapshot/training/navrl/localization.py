"""Lightweight likelihood-field particle localization; not the Nav2 AMCL binary."""
import numpy as np
from .geometry import wrap


class ParticleLocalizer:
    def __init__(self, static_map, lidar, particles=384):
        self.map=static_map; self.lidar=lidar; self.count=particles
        self.last_odom=None; self.pose=None; self.std=np.ones(3)
        self.rng=np.random.default_rng()

    def reset(self, nominal, rng):
        self.rng=rng
        self.particles=rng.normal(nominal,[.12,.12,.08],size=(self.count,3))
        self.particles[:,2]=wrap(self.particles[:,2]); self.last_odom=None
        self.weights=np.full(self.count,1/self.count)
        self.pose=np.array(nominal,float); self.std=np.array([.12,.12,.08])

    def update(self, odom, scan):
        if self.last_odom is not None:
            dx,dy=odom[:2]-self.last_odom[:2]
            yaw=self.last_odom[2]
            local=np.array([np.cos(yaw)*dx+np.sin(yaw)*dy,-np.sin(yaw)*dx+np.cos(yaw)*dy])
            angle=wrap(odom[2]-yaw); p=self.particles
            p[:,0]+=np.cos(p[:,2])*local[0]-np.sin(p[:,2])*local[1]
            p[:,1]+=np.sin(p[:,2])*local[0]+np.cos(p[:,2])*local[1]
            p[:,2]=wrap(p[:,2]+angle)
            noise=.002+.06*np.linalg.norm(local)+.01*abs(angle)
            p+=self.rng.normal(0,[noise,noise,.002+.05*abs(angle)],p.shape)
        self.last_odom=np.array(odom)
        indices=np.arange(0,len(scan),max(1,len(scan)//60))
        indices=indices[np.isfinite(scan[indices])&(scan[indices]>=self.lidar.range_min)&(scan[indices]<self.lidar.range_max-.02)]
        if len(indices)>=8:
            p=self.particles; yaw=p[:,2]
            mount=self.lidar.mount
            sx=p[:,0]+np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1]
            sy=p[:,1]+np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1]
            angles=yaw[:,None]+mount[3]+self.lidar.angles[indices]
            endpoints=np.stack([sx[:,None]+scan[indices]*np.cos(angles),sy[:,None]+scan[indices]*np.sin(angles)],axis=-1)
            distance=self.map.lookup(endpoints,self.map.distance,outside=2.)
            # Reject beams inconsistent with most hypotheses (often moving obstacles).
            accepted=(distance<.3).mean(axis=0)>.3
            if accepted.sum()<len(indices)*.25: accepted=np.ones(len(indices),bool)
            score=np.log(.9*np.exp(-.5*(distance[:,accepted]/.12)**2)+.1/self.lidar.range_max).sum(axis=1)
            score+=np.log(np.maximum(self.weights,1e-12))
            score[~self.map.lookup(p[:,:2],~self.map.blocked,False)]=-1e6
            score-=score.max(); self.weights=np.exp(score); self.weights/=self.weights.sum()
        self.pose=np.r_[np.average(self.particles[:,:2],axis=0,weights=self.weights),
            np.arctan2(np.dot(self.weights,np.sin(self.particles[:,2])),np.dot(self.weights,np.cos(self.particles[:,2])))]
        deviations=self.particles-self.pose; deviations[:,2]=wrap(deviations[:,2])
        self.std=np.sqrt(np.average(deviations**2,axis=0,weights=self.weights))
        if 1/np.sum(self.weights**2)<self.count*.65:
            positions=(self.rng.random()+np.arange(self.count))/self.count
            indices=np.searchsorted(np.cumsum(self.weights),positions).clip(0,self.count-1)
            self.particles=self.particles[indices].copy()
            self.weights.fill(1/self.count)
        return self.pose.copy()

    def close(self): pass
