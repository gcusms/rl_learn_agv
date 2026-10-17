"""Experimental point-to-line scan increments on a continuous odometry prior.

Uses measured adjacent scans only. Weak geometry axes and absent scans retain
raw odometry. This small independent implementation is inspired by the metric
in Censi (ICRA 2008), not a reimplementation/copy of its closed-form optimizer.
"""
import numpy as np
from scipy.spatial import cKDTree
from navrl.geometry import wrap


def body_delta(current,previous):
    d=current[:2]-previous[:2];c,s=np.cos(previous[2]),np.sin(previous[2])
    return np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1],wrap(current[2]-previous[2])])


class LaserIncrementOdometry:
    def __init__(self,lidar):
        self.lidar=lidar;self.previous=None;self.raw_previous=None;self.timestamp=None
        self.pose=np.zeros(3);self.accepted=0;self.fallback=0;self.last_reason='initial'
    def points(self,scan):
        scan=np.asarray(scan,float);valid=np.isfinite(scan)&(scan>=self.lidar.range_min)&(scan<self.lidar.range_max)
        a=self.lidar.angles+self.lidar.mount[3]
        points=scan[:,None]*np.c_[np.cos(a),np.sin(a)]+self.lidar.mount[:2]
        return points,valid
    def match(self,scan,previous,prior,dt):
        current,valid=self.points(scan);reference,rv=self.points(previous)
        indices=np.flatnonzero(rv);good=np.zeros(len(rv),bool)
        normals=np.zeros_like(reference)
        if len(indices)<20 or valid.sum()<20:return prior,'insufficient_returns'
        # Tangents need nearby measured neighbors, never bridge occlusion edges.
        mid=indices[1:-1];left=indices[:-2];right=indices[2:]
        tangent=reference[right]-reference[left];length=np.linalg.norm(tangent,axis=1)
        keep=(right-left<=5)&(length>.01)&(length<.7)
        good[mid[keep]]=True
        normals[mid[keep]]=np.c_[-tangent[keep,1],tangent[keep,0]]/length[keep,None]
        ref=reference[good];n=normals[good]
        if len(ref)<20:return prior,'insufficient_contours'
        tree=cKDTree(ref);source=current[valid];candidate=prior.copy();initial=None;projection=None;last_cost=None
        for iteration in range(5):
            a=candidate[2];c,s=np.cos(a),np.sin(a)
            rotated=source@np.array([[c,s],[-s,c]])
            transformed=rotated+candidate[:2]
            distances,ids=tree.query(transformed,k=1,distance_upper_bound=.25)
            ok=np.isfinite(distances)
            if ok.sum()<20:return prior,'insufficient_correspondences'
            normals_match=n[ids[ok]];residual=np.sum(normals_match*(transformed[ok]-ref[ids[ok]]),axis=1)
            j=np.c_[normals_match,-normals_match[:,0]*rotated[ok,1]+normals_match[:,1]*rotated[ok,0]]
            # Trim high residuals; moving foreground and contour edges must
            # not dominate the static-motion consensus.
            selected=np.argsort(abs(residual))[:max(20,int(.75*len(residual)))]
            residual=residual[selected];j=j[selected]
            scale=np.array([1.,1.,2.]);js=j/scale
            weight=np.minimum(1.,.025/np.maximum(abs(residual),1e-9))
            h=js.T@(weight[:,None]*js)/len(js);rhs=js.T@(weight*residual)/len(js)
            values,vectors=np.linalg.eigh(h);observed=values>max(.01,.04*float(values.max()))
            inverse=np.where(observed,1/np.maximum(values,1e-9),0.)
            projection=(vectors*observed)@vectors.T
            step=-(vectors*inverse)@vectors.T@rhs/scale
            cost=float(np.mean(np.minimum(abs(residual),.025)))
            if initial is None:initial=cost
            last_cost=cost;candidate+=step;candidate[2]=wrap(candidate[2])
            if np.linalg.norm(step)<1e-5:break
        correction=candidate-prior;correction[2]=wrap(correction[2])
        if (not np.isfinite(candidate).all() or np.linalg.norm(correction[:2])>.025 or abs(correction[2])>.04
            or np.linalg.norm(candidate[:2])>.35*dt+.015 or abs(candidate[2])>.75*dt+.03):
            return prior,'innovation_bounds'
        if last_cost is None or last_cost>initial+1e-5 or last_cost>.02:return prior,'matching_cost'
        if projection is None or np.trace(projection)<.5:return prior,'unobservable'
        return candidate,'accepted_scan_increment'
    def update(self,scan,raw,timestamp):
        raw=np.asarray(raw,float);dt=0. if self.timestamp is None else timestamp-self.timestamp
        if not np.isfinite(np.r_[raw,timestamp]).all() or dt<0:raise ValueError('Finite monotonic odometry required')
        delta=np.zeros(3) if self.raw_previous is None else body_delta(raw,self.raw_previous)
        reason='initial_or_scan_gap'
        if np.linalg.norm(delta[:2])<.0015 and abs(delta[2])<.002:
            reason='stationary_odometry_prior'
        elif self.previous is not None and .01<dt<=.4:delta,reason=self.match(scan,self.previous,delta,dt)
        self.raw_previous=raw.copy();self.previous=np.asarray(scan,float).copy();self.timestamp=float(timestamp)
        c,s=np.cos(self.pose[2]),np.sin(self.pose[2]);self.pose[:2]+=[c*delta[0]-s*delta[1],s*delta[0]+c*delta[1]]
        self.pose[2]=wrap(self.pose[2]+delta[2]);self.last_reason=reason
        if reason=='accepted_scan_increment':self.accepted+=1
        else:self.fallback+=1
        return self.pose.copy()
