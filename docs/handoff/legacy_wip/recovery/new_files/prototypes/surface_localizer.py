"""Opt-in robust map-surface localization; no simulator or route truth input.

The learned coarse pose average can drift inside wide occupancy plateaus. This
adapter instead aligns actual occupied/free boundaries with bounded, robust
point-to-plane updates. Unobservable directions retain odometry, and rejected
scans increase uncertainty. This is geometric localization, not an RL policy.
"""
import numpy as np
from scipy.spatial import cKDTree
from navrl.geometry import wrap
from neural_navigation.models import LearnedLocalizer,scan_map_agreement


def map_surfaces(static_map):
    occupied=static_map.occupied;known_free=~static_map.blocked
    centers=[];normals=[]
    for dr,dc,normal in [(0,1,(1.,0.)),(0,-1,(-1.,0.)),(1,0,(0.,-1.)),(-1,0,(0.,1.))]:
        rr,cc=np.nonzero(occupied);nr,nc=rr+dr,cc+dc
        inside=(nr>=0)&(nr<occupied.shape[0])&(nc>=0)&(nc<occupied.shape[1])
        rr,cc,nr,nc=rr[inside],cc[inside],nr[inside],nc[inside]
        good=known_free[nr,nc];rr,cc=rr[good],cc[good]
        xy=static_map.origin[:2]+np.c_[(cc+.5)*static_map.res,(static_map.height-rr-.5)*static_map.res]
        # A conservatively occupied boundary cell only locates the physical
        # wall within that cell. Use its midpoint, not the free-facing edge
        # (which systematically pushes estimates away from unevenly visible
        # walls). Navigation still uses the complete occupied cell unchanged.
        centers.append(xy)
        normals.append(np.tile(normal,(len(xy),1)))
    return np.vstack(centers),np.vstack(normals)


class SurfaceLocalizer(LearnedLocalizer):
    def __init__(self,static_map,lidar,checkpoint):
        super().__init__(static_map,lidar,checkpoint)
        self.surfaces,self.normals=map_surfaces(static_map)
        self.tree=cKDTree(self.surfaces) if len(self.surfaces) else None

    def reset(self,nominal,rng):
        super().reset(nominal,rng)
        self.surface_residual_m=None;self.observable_rank=0;self.surface_support=0.
        self.correction=np.zeros(3);self.rejection_reason='initial'

    def correspondences(self,pose,points):
        c,s=np.cos(pose[2]),np.sin(pose[2]);rot=np.array([[c,-s],[s,c]])
        rotated=points@rot.T;world=rotated+pose[:2]
        distance,index=self.tree.query(world)
        normal=self.normals[index];surface=self.surfaces[index]
        origin=pose[:2]+rot@self.lidar.mount[:2]
        facing=np.sum((origin-surface)*normal,axis=1)>0
        valid=(distance<.10)&facing
        residual=np.sum((world-surface)*normal,axis=1)
        jacobian=np.c_[normal,np.sum(np.c_[-rotated[:,1],rotated[:,0]]*normal,axis=1)]
        return residual,jacobian,valid

    def update(self,odom,scan):
        odom=np.asarray(odom,float);scan=np.asarray(scan,float)
        if self.last_odom is not None:
            delta=odom[:2]-self.last_odom[:2];a=self.pose[2]-self.last_odom[2]
            c,s=np.cos(a),np.sin(a);self.pose[:2]+=np.array([[c,-s],[s,c]])@delta
            self.pose[2]=wrap(self.pose[2]+wrap(odom[2]-self.last_odom[2]))
        self.last_odom=odom.copy();prior=self.pose.copy()
        valid=np.isfinite(scan)&(scan>=self.lidar.range_min)&(scan<self.lidar.range_max)
        self.valid_scan_fraction=float(valid.mean());self.scan_update_accepted=False
        self.observable_rank=0;self.surface_support=0.;self.surface_residual_m=None
        self.correction=np.zeros(3);self.rejection_reason='insufficient_returns'
        self.std=np.sqrt(self.std**2+np.array([.003,.003,.002])**2)
        if valid.sum()>=20 and self.tree is not None:
            angles=self.lidar.angles[valid]+self.lidar.mount[3]
            points=scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]+self.lidar.mount[:2]
            proposed=prior.copy();eigenvalues=np.zeros(3);eigenvectors=np.eye(3)
            for _ in range(6):
                residual,jacobian,matched=self.correspondences(proposed,points)
                if matched.sum()<20:break
                r=residual[matched];j=jacobian[matched]
                weight=np.minimum(1.,.015/np.maximum(np.abs(r),1e-9))
                # A dense nearby wall must not outweigh its opposite merely
                # because of angular sampling. Balance opposite surface-normal
                # groups only when both have independent multi-ray support.
                for axis in (0,1):
                    positive=j[:,axis]>.5;negative=j[:,axis]<-.5
                    if positive.sum()>=8 and negative.sum()>=8:
                        a,b=weight[positive].sum(),weight[negative].sum()
                        weight[positive]*=(a+b)/(2*a);weight[negative]*=(a+b)/(2*b)
                h=j.T@(weight[:,None]*j);g=j.T@(weight*r)
                eigenvalues,eigenvectors=np.linalg.eigh(h)
                observable=eigenvalues>max(2.,float(eigenvalues[-1])*.01)
                step=-eigenvectors@(np.where(observable,1/np.maximum(eigenvalues,1e-9),0.)*(eigenvectors.T@g))
                step[:2]*=min(1.,.025/max(np.linalg.norm(step[:2]),1e-9));step[2]=np.clip(step[2],-.025,.025)
                proposed+=step;proposed[2]=wrap(proposed[2])
                if np.linalg.norm(step)<1e-5:break
            residual,jacobian,matched=self.correspondences(proposed,points)
            self.surface_support=float(matched.mean())
            self.observable_rank=int(np.sum(eigenvalues>max(2.,float(eigenvalues[-1])*.01)))
            if matched.any():self.surface_residual_m=float(np.median(np.abs(residual[matched])))
            correction=proposed-prior;correction[2]=wrap(correction[2])
            accepted=(matched.sum()>=20 and self.surface_support>=.6 and self.observable_rank>=2
                      and self.surface_residual_m<=.025 and np.linalg.norm(correction[:2])<=.08
                      and abs(correction[2])<=.08)
            self.rejection_reason='surface_support_or_correction_limit'
            if accepted:
                self.pose=proposed;self.correction=correction;self.scan_update_accepted=True
                self.rejection_reason='none';self.missing_scan_updates=0
                observable=eigenvalues>max(2.,float(eigenvalues[-1])*.01)
                # Only observed directions may gain confidence. Preserve the
                # prior variance in the corridor's unsupported longitudinal axis.
                variance=np.diag(eigenvectors.T@np.diag(self.std**2)@eigenvectors)
                variance=np.where(observable,np.maximum(.015**2,.02**2/np.maximum(eigenvalues,1e-9)),variance)
                self.std=np.sqrt(np.diag(eigenvectors@np.diag(variance)@eigenvectors.T))
        if not self.scan_update_accepted:self.missing_scan_updates+=1
        self.map_agreement=scan_map_agreement(self.map,self.lidar,self.pose,scan) if valid.any() else 0.
        self.posterior_entropy=1.-self.observable_rank/3 if self.scan_update_accepted else 1.
        return self.pose.copy()
