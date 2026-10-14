"""Opt-in robust map corrections on a continuous odometry prior.

Unmapped returns are excluded from localization ONLY. Callers retain the
original scan for avoidance. Limits are provisional, not calibrated confidence.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from navrl.geometry import wrap
from neural_navigation.models import LearnedLocalizer, alignment_features


def contract():
    return dict(schema='odom_anchored_static_support_v1',calibrated=False,
                endpoint_gate_m=.15,minimum_hits=12,minimum_support_fraction=.35,
                maximum_translation_rate_m_s=.04,maximum_yaw_rate_rad_s=.06,
                maximum_proposal_translation_m=.12,maximum_proposal_yaw_rad=.12,
                weak_axis_ratio=.12,minimum_cost_improvement_m=.001,
                verified_prior_cost_m=.02,verified_prior_support=.8,
                process_std_per_sqrt_s=[.01,.01,.008])


class OdometryAnchoredLocalizer(LearnedLocalizer):
    def __init__(self,static_map,lidar,checkpoint,config):
        if config!=contract():raise ValueError('Unsupported localization contract')
        super().__init__(static_map,lidar,checkpoint);self.config=config
        self.timestamp=None;self.correction_reason='reset';self.static_support_fraction=0.

    def reset(self,nominal,rng):
        super().reset(nominal,rng)
        self.timestamp=None;self.correction_reason='reset';self.static_support_fraction=0.

    def _endpoints(self,pose,scan,indices):
        yaw=pose[2];m=self.lidar.mount
        origin=pose[:2]+[np.cos(yaw)*m[0]-np.sin(yaw)*m[1],np.sin(yaw)*m[0]+np.cos(yaw)*m[1]]
        a=yaw+m[3]+self.lidar.angles[indices]
        return origin+scan[indices,None]*np.c_[np.cos(a),np.sin(a)]

    def _observable_projection(self,ends,prior):
        if not hasattr(self.map,'_alignment_normals'):
            signed=self.map.distance-distance_transform_edt(self.map.occupied)*self.map.res
            gy,gx=np.gradient(signed,self.map.res);norm=np.maximum(np.hypot(gx,gy),1e-8)
            self.map._alignment_normals=(gx/norm,-gy/norm)
        nx=self.map.lookup(ends,self.map._alignment_normals[0],outside=0.)
        ny=self.map.lookup(ends,self.map._alignment_normals[1],outside=0.)
        lever=ends-prior[:2];j=np.c_[nx,ny,-nx*lever[:,1]+ny*lever[:,0]]
        values,vectors=np.linalg.eigh(j.T@j/max(len(j),1))
        observed=values>max(.02,float(values.max())*self.config['weak_axis_ratio'])
        return (vectors*observed)@vectors.T

    def _accept_measurement(self,projection,dt,reason):
        self.scan_update_accepted=True;self.correction_reason=reason
        diag=np.clip(np.diag(projection),0,1)
        self.std=np.maximum(self.std*np.exp(-.25*dt*diag),[.04,.04,.03])

    def update(self,odom,scan,timestamp=None):
        odom=np.asarray(odom,float);scan=np.asarray(scan,float)
        if timestamp is None:raise ValueError('Timestamp required by time-scaled localization')
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp<self.timestamp):
            raise ValueError('Monotonic finite localization timestamp required')
        dt=0. if self.timestamp is None else timestamp-self.timestamp
        self.timestamp=float(timestamp)
        movement=0.;turn=0.
        if self.last_odom is not None:
            delta=odom[:2]-self.last_odom[:2];yaw=self.last_odom[2]
            local=np.array([np.cos(yaw)*delta[0]+np.sin(yaw)*delta[1],-np.sin(yaw)*delta[0]+np.cos(yaw)*delta[1]])
            c,s=np.cos(self.pose[2]),np.sin(self.pose[2])
            self.pose[:2]+=[c*local[0]-s*local[1],s*local[0]+c*local[1]]
            turn=wrap(odom[2]-self.last_odom[2]);self.pose[2]=wrap(self.pose[2]+turn)
            movement=float(np.linalg.norm(delta))
        self.last_odom=odom.copy();prior=self.pose.copy()
        self.std=np.sqrt(self.std**2+np.asarray(self.config['process_std_per_sqrt_s'])**2*dt
                         +np.array([.02*movement,.02*movement,.02*abs(turn)])**2)
        self.scan_update_accepted=False;self.correction_reason='insufficient_static_support'
        all_indices=np.linspace(0,len(scan)-1,60).astype(int)
        valid=np.isfinite(scan[all_indices])&(scan[all_indices]>=self.lidar.range_min)&(scan[all_indices]<self.lidar.range_max)
        self.valid_scan_fraction=float(valid.mean());indices=all_indices[valid]
        distances=self.map.lookup(self._endpoints(prior,scan,indices),self.map.distance,outside=1.)
        matched=distances<=self.config['endpoint_gate_m'];support=indices[matched]
        self.static_support_fraction=float(matched.mean()) if len(indices) else 0.
        self.map_agreement=self.static_support_fraction
        self.posterior_entropy=1.
        if len(support)<self.config['minimum_hits'] or self.static_support_fraction<self.config['minimum_support_fraction']:
            self.missing_scan_updates+=1;return prior
        self.missing_scan_updates=0
        ends=self._endpoints(prior,scan,support)
        projection=self._observable_projection(ends,prior)
        before=float(np.minimum(distances[matched],.2).mean())
        if (before<=self.config['verified_prior_cost_m'] and
            self.static_support_fraction>=self.config['verified_prior_support']):
            # A valid zero-innovation measurement supports prediction too.
            # "No correction needed" must not mean "pose unsupported" to actor.
            self._accept_measurement(projection,dt,'validated_static_prior');return prior
        # Use the same beam indexing as the trained scorer, keeping only static
        # support at the propagated pose. Do not search for a wall for new objects.
        filtered=np.full_like(scan,np.nan);filtered[support]=scan[support]
        x,hypotheses=alignment_features(self.map,self.lidar,prior,filtered,
            points=self.hypothesis_points,directional=self.feature_dim==15)
        with torch.no_grad():weights=torch.softmax(self.network(torch.from_numpy(x)),dim=0).numpy()
        proposed=np.r_[weights@hypotheses[:,:2],np.arctan2(weights@np.sin(hypotheses[:,2]),weights@np.cos(hypotheses[:,2]))]
        delta=proposed-prior;delta[2]=wrap(delta[2])
        self.posterior_entropy=float(-np.sum(weights*np.log(np.maximum(weights,1e-12)))/np.log(len(weights)))
        if np.linalg.norm(delta[:2])>self.config['maximum_proposal_translation_m'] or abs(delta[2])>self.config['maximum_proposal_yaw_rad']:
            self.correction_reason='innovation_rejected';return prior
        # The normal Jacobian identifies unconstrained displacement along walls.
        # Yaw is represented as an equivalent 1m arc for this observability gate.
        delta=projection@delta
        # No per-frame correction allowance: stationary repeated measurements
        # cannot accumulate a 10Hz fixed spatial jump.
        xy_limit=self.config['maximum_translation_rate_m_s']*dt
        yaw_limit=self.config['maximum_yaw_rate_rad_s']*dt
        delta*=min(1.,xy_limit/max(np.linalg.norm(delta[:2]),1e-12),yaw_limit/max(abs(delta[2]),1e-12))
        candidate=prior+delta;candidate[2]=wrap(candidate[2])
        after=float(np.minimum(self.map.lookup(self._endpoints(candidate,scan,support),self.map.distance,outside=1.),.2).mean())
        if after>before-self.config['minimum_cost_improvement_m']:
            self.correction_reason='no_matching_improvement';return prior
        self.pose=candidate
        # Never replace uncertainty by the network's uncalibrated sharp softmax.
        # Only supported axes shrink; unsupported axes continue growing.
        self._accept_measurement(projection,dt,'accepted_static_correction')
        return self.pose.copy()
