"""Experimental, local SE(2) tracking against static geometry.

No particles, pose lattice, Torch, robot truth, or runtime path planner. This is
a geometric localization frontend, not a newly trained neural localizer. It
requires a supplied initial map pose; it cannot solve global kidnapping or
create an absolute reference in an unobservable environment.
"""
import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt, map_coordinates
from navrl.geometry import wrap


def contract():
    return dict(schema='fast_static_tracking_v1', calibrated=False, beams=60,
        iterations=4, endpoint_gate_m=.18, angular_gate_rad=.02,
        maximum_endpoint_gate_m=.35, minimum_hits=12, minimum_support_fraction=.35,
        huber_m=.05, damping=.02, weak_axis_ratio=.04, minimum_eigenvalue=.02,
        yaw_length_scale_m=3., maximum_correction_m=.12, maximum_correction_rad=.15,
        correction_rate_m_s=.30, correction_rate_rad_s=.50,
        process_std_per_sqrt_s=[.012,.012,.012], measurement_std_m=.06,
        covariance_floor=[.04,.04,.03], correlation_seconds=.5,
        supplied_initial_pose_required=True, global_relocalization=False)


def native_contract():
    return dict(contract(),schema='fast_static_tracking_native_v1',kernel='static_refine_v1')


def sparse_contract(native=False):
    """Use sparse geometric evidence without treating it as a full scan.

    A zero eigenvalue still contributes no correction/information. Fewer hits
    reduce the covariance update; this is not a license to trust blank scans.
    V1 contracts retain their original math for archived replay.
    """
    value=dict(contract(),schema='fast_static_tracking_v2',minimum_hits=3,
        weak_axis_ratio=.01,minimum_eigenvalue=.01,information_reference_hits=60)
    if native:value.update(schema='fast_static_tracking_native_v2',kernel='static_refine_v2')
    return value


def temporal_contract(native=False):
    value=dict(sparse_contract(native),schema='fast_static_tracking_temporal_v1',
        support_history_offsets_s=[-.4,-.2,0.],history_tolerance_s=.05,
        history_current_ray_tolerance_m=.10,history_requires_current_static_hit=True,
        history_information_average=True)
    if native:value['schema']='fast_static_tracking_temporal_native_v1'
    return value


class FastStaticLocalizer:
    uses_timestamps=True

    def __init__(self, static_map, lidar, config=None, kernel_path=None, uncertainty=None):
        self.config=contract() if config is None else config
        if self.config not in (contract(),native_contract(),sparse_contract(),sparse_contract(True),temporal_contract(),temporal_contract(True)):raise ValueError('Unsupported fast localization contract')
        self.map=static_map;self.lidar=lidar
        self.weak_geometry_uncertainty=None
        if uncertainty is not None:
            from neural_navigation.weak_geometry_uncertainty import WeakGeometryUncertainty
            self.weak_geometry_uncertainty=WeakGeometryUncertainty(uncertainty)
        self._native_refine=None
        if 'kernel' in self.config:
            import ctypes
            if kernel_path is None:raise ValueError('Pinned native localization kernel required')
            self._library=ctypes.CDLL(str(kernel_path))
            self._native_refine=getattr(self._library,self.config['kernel'])
            array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
            self._native_refine.argtypes=[ctypes.c_int,ctypes.c_int,*[ctypes.c_double]*3,*[array]*4,
                ctypes.c_int,array,ctypes.c_double,ctypes.c_double,*[array]*3]
            self._native_refine.restype=ctypes.c_int
        self.indices=np.linspace(0,lidar.beams-1,self.config['beams']).astype(int)
        angles=lidar.angles[self.indices]+lidar.mount[3]
        self.directions=np.c_[np.cos(angles),np.sin(angles)]
        self.reset(np.zeros(3),None)

    def reset(self, nominal, rng):
        nominal=np.asarray(nominal,float)
        if nominal.shape!=(3,) or not np.isfinite(nominal).all():raise ValueError('Finite initial pose required')
        self.pose=nominal.copy();self.covariance=np.diag(np.array([.12,.12,.08])**2)
        self.std=np.sqrt(np.diag(self.covariance));self.last_odom=None;self.timestamp=None
        self.valid_scan_fraction=0.;self.static_support_fraction=0.;self.map_agreement=0.
        self.scan_update_accepted=False;self.missing_scan_updates=0;self.posterior_entropy=1.
        self.recovery_attempts=0;self.recovery_accepted=0;self.correction_reason='reset'
        self.observable_eigenvalues=np.zeros(3)
        self.observable_rank=0
        self.valid_static_hits=0
        self.support_history=deque(maxlen=128);self.information_frames=1
        self.observable_projection=np.zeros((3,3));self._odometry_transition=np.eye(3)
        if self.weak_geometry_uncertainty is not None:self.weak_geometry_uncertainty.reset()

    def close(self):pass

    def _field(self, points, gradient=False):
        if not hasattr(self.map,'_fast_static_fields'):
            signed=self.map.distance-distance_transform_edt(self.map.occupied)*self.map.res
            gy,gx=np.gradient(signed,self.map.res)
            self.map._fast_static_fields=(signed,gx,-gy)
        col=(points[:,0]-self.map.origin[0])/self.map.res-.5
        row=self.map.height-.5-(points[:,1]-self.map.origin[1])/self.map.res
        fields=self.map._fast_static_fields if gradient else self.map._fast_static_fields[:1]
        return [map_coordinates(f,[row,col],order=1,mode='constant',cval=1. if i==0 else 0.)
                for i,f in enumerate(fields)]

    @staticmethod
    def _points(pose, body_points):
        c,s=np.cos(pose[2]),np.sin(pose[2])
        return body_points@np.array([[c,s],[-s,c]])+pose[:2]

    def _linearize(self, pose, body_points):
        points=self._points(pose,body_points);r,gx,gy=self._field(points,True)
        lever=points-pose[:2]
        jac=np.c_[gx,gy,-gx*lever[:,1]+gy*lever[:,0]]
        weight=np.minimum(1.,self.config['huber_m']/np.maximum(abs(r),1e-12))
        return r,jac,weight

    def _basis(self, jac, weight):
        metric=np.array([1.,1.,self.config['yaw_length_scale_m']])
        scaled=jac/metric
        h=scaled.T@(weight[:,None]*scaled)/len(jac)
        eig,vec=np.linalg.eigh(h);self.observable_eigenvalues=eig
        strong=eig>max(self.config['minimum_eigenvalue'],eig[-1]*self.config['weak_axis_ratio'])
        self.observable_rank=int(np.count_nonzero(strong))
        self.observable_projection=(vec*strong)@vec.T
        return self.observable_projection,metric

    def _finish_update(self,dt,movement,turn):
        self.std=np.sqrt(np.diag(self.covariance))
        if self.weak_geometry_uncertainty is not None:
            floor=self.weak_geometry_uncertainty.update(self.observable_projection,self._odometry_transition,
                dt,movement,turn,self.valid_static_hits)
            self.std=np.maximum(self.std,floor)
        return self.pose.copy()

    def _cost(self, pose, points):
        r=abs(self._field(self._points(pose,points))[0]);h=self.config['huber_m']
        return float(np.mean(np.where(r<=h,.5*r*r,h*(r-.5*h))))

    def _uncertainty(self, jac, weight, dt):
        # Correlated scans are not independent particle measurements. This is
        # a provisional information model, explicitly not calibrated accuracy.
        projection,metric=self._basis(jac,weight)
        observed=(jac/metric)@projection*metric
        denominator=self.config.get('information_reference_hits',len(jac))*self.information_frames
        information=observed.T@(weight[:,None]*observed)/denominator
        strength=min(1.,dt/self.config['correlation_seconds'])
        self.covariance=np.linalg.inv(np.linalg.inv(self.covariance)+
            information*strength/self.config['measurement_std_m']**2)
        floors=np.array(self.config['covariance_floor'])**2
        self.covariance+=np.diag(np.maximum(0.,floors-np.diag(self.covariance)))

    def _temporal_support(self,points,odom,scan,timestamp,prior):
        """Deskew short endpoint history with measured relative odometry.

        Historical returns need a current, finite, range-consistent ray AND
        agreement with known static geometry. A blank/occluded scan never
        resurrects old evidence. Only the two fixed physical-time offsets are
        used, and their information is averaged rather than counted as three
        independent full scans. Nothing modifies the caller's avoidance scan.
        """
        self.support_history.append((float(timestamp),odom.copy(),points.copy()))
        while self.support_history and timestamp-self.support_history[0][0]>.5:
            self.support_history.popleft()
        if not len(points):return points
        batches=[points];used=set()
        c,s=np.cos(odom[2]),np.sin(odom[2]);rotation=np.array([[c,-s],[s,c]])
        for offset in self.config['support_history_offsets_s'][:-1]:
            target=timestamp+offset
            choices=[row for row in self.support_history if abs(row[0]-target)<=self.config['history_tolerance_s']+1e-8]
            if not choices:continue
            then,old_odom,old_points=min(choices,key=lambda row:abs(row[0]-target))
            if then in used:continue
            used.add(then)
            cc,ss=np.cos(old_odom[2]),np.sin(old_odom[2]);old_rotation=np.array([[cc,-ss],[ss,cc]])
            body=(old_points@old_rotation.T+old_odom[:2]-odom[:2])@rotation
            # Compare in the sensor frame, including the nonzero laser mount.
            mount=self.lidar.mount;mc,ms=np.cos(mount[3]),np.sin(mount[3])
            sensor=(body-mount[:2])@np.array([[mc,-ms],[ms,mc]])
            distance=np.linalg.norm(sensor,axis=1);bearing=np.arctan2(sensor[:,1],sensor[:,0])
            index=np.rint((bearing-self.lidar.angle_min)/(self.lidar.angle_max-self.lidar.angle_min)*(self.lidar.beams-1)).astype(int)
            inside=(index>=0)&(index<self.lidar.beams)&(bearing>=self.lidar.angle_min)&(bearing<=self.lidar.angle_max)
            index=np.clip(index,0,self.lidar.beams-1)
            current=scan[index];valid=inside&np.isfinite(current)&(current>=self.lidar.range_min)&(current<self.lidar.range_max)
            valid&=abs(current-distance)<=self.config['history_current_ray_tolerance_m']
            endpoints=self._points(prior,body);residual=abs(self._field(endpoints)[0])
            gate=np.minimum(self.config['maximum_endpoint_gate_m'],self.config['endpoint_gate_m']+self.config['angular_gate_rad']*distance)
            valid&=(residual<=gate)&self.map.lookup(endpoints,~self.map.unknown,outside=False)
            if valid.any():batches.append(body[valid])
        self.information_frames=len(batches)
        return np.concatenate(batches)

    def _select_static_hits(self, prior, scan, body, valid, keep):
        """Opt-in frontends may exclude evidence; archived contracts are identity."""
        return keep

    def _correction_permitted(self, prior, candidate, scan):
        return True

    def _reject_correction(self, dt, movement, turn):
        # The odometry prediction/covariance already exist. A rejected laser
        # proposal must not contribute either a pose jump or information gain.
        self.scan_update_accepted=False
        self.missing_scan_updates+=1
        self.observable_rank=0;self.observable_eigenvalues=np.zeros(3)
        self.observable_projection=np.zeros((3,3))
        self.support_history.clear()
        return self._finish_update(dt,movement,turn)

    def update(self, odom, scan, timestamp=None):
        odom=np.asarray(odom,float);scan=np.asarray(scan,float)
        if odom.shape!=(3,) or not np.isfinite(odom).all():raise ValueError('Finite odometry required')
        if scan.shape!=(self.lidar.beams,):raise ValueError('Scan contract mismatch')
        if timestamp is None or not np.isfinite(timestamp):raise ValueError('Finite timestamp required')
        if self.timestamp is not None and timestamp<=self.timestamp:
            if timestamp==self.timestamp and np.array_equal(odom,self.last_odom):
                self.scan_update_accepted=False;self.correction_reason='duplicate_timestamp';return self.pose.copy()
            raise ValueError('Strictly increasing localization timestamps required')
        dt=0. if self.timestamp is None else timestamp-self.timestamp
        movement=0.;turn=0.;self._odometry_transition=np.eye(3)
        if self.last_odom is not None:
            d=odom[:2]-self.last_odom[:2];c,s=np.cos(self.last_odom[2]),np.sin(self.last_odom[2])
            local=np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1]])
            c,s=np.cos(self.pose[2]),np.sin(self.pose[2]);delta=np.array([c*local[0]-s*local[1],s*local[0]+c*local[1]])
            transition=np.eye(3);transition[:2,2]=[-delta[1],delta[0]]
            self._odometry_transition=transition
            self.covariance=transition@self.covariance@transition.T
            self.pose[:2]+=delta;turn=float(wrap(odom[2]-self.last_odom[2]));self.pose[2]=wrap(self.pose[2]+turn)
            movement=float(np.linalg.norm(local))
        self.covariance+=np.diag(np.array(self.config['process_std_per_sqrt_s'])**2*dt+
            np.array([.08*movement,.08*movement,.08*abs(turn)])**2)
        self.last_odom=odom.copy();self.timestamp=float(timestamp);prior=self.pose.copy()
        self.scan_update_accepted=False;self.correction_reason='insufficient_static_support'
        self.observable_rank=0;self.observable_eigenvalues=np.zeros(3);self.valid_static_hits=0;self.information_frames=1
        self.observable_projection=np.zeros((3,3))
        selected=scan[self.indices]
        valid=np.isfinite(selected)&(selected>=self.lidar.range_min)&(selected<self.lidar.range_max)
        self.valid_scan_fraction=float(valid.mean())
        body=self.directions[valid]*selected[valid,None]+self.lidar.mount[:2]
        # Unmapped/unknown returns remain untouched in the caller's avoidance
        # scan. Reject them only from the static localization measurement.
        endpoints=self._points(prior,body)
        residual=abs(self._field(endpoints)[0])
        gate=np.minimum(self.config['maximum_endpoint_gate_m'],
            self.config['endpoint_gate_m']+self.config['angular_gate_rad']*selected[valid])
        known=self.map.lookup(endpoints,~self.map.unknown,outside=False)
        keep=(residual<=gate)&known
        keep=self._select_static_hits(prior,scan,body,valid,keep)
        self.static_support_fraction=float(keep.mean()) if len(keep) else 0.
        self.map_agreement=self.static_support_fraction;self.posterior_entropy=1.
        points=body[keep]
        self.valid_static_hits=len(points)
        if 'support_history_offsets_s' in self.config:
            points=self._temporal_support(points,odom,scan,timestamp,prior)
        if len(points)<self.config['minimum_hits'] or self.static_support_fraction<self.config['minimum_support_fraction']:
            self.missing_scan_updates+=1
            return self._finish_update(dt,movement,turn)
        cost=self._cost(prior,points);candidate=prior.copy()
        # No arbitrary minimum translation in weak directions, and no lateral
        # command restriction on measured mecanum odometry/slip.
        xy_cap=self.config['maximum_correction_m'] if dt==0 else min(self.config['maximum_correction_m'],self.config['correction_rate_m_s']*dt)
        yaw_cap=self.config['maximum_correction_rad'] if dt==0 else min(self.config['maximum_correction_rad'],self.config['correction_rate_rad_s']*dt)
        if self._native_refine is not None:
            jac=np.empty((len(points),3));weight=np.empty(len(points));candidate=np.empty(3)
            result=self._native_refine(self.map.height,self.map.width,self.map.res,*self.map.origin[:2],
                *self.map._fast_static_fields,np.ascontiguousarray(points),len(points),prior,xy_cap,yaw_cap,candidate,jac,weight)
            if result!=0 or not np.isfinite(candidate).all():raise RuntimeError('Native localization kernel failed')
            if not self._correction_permitted(prior,candidate,scan):
                return self._reject_correction(dt,movement,turn)
            self.missing_scan_updates=0
            self.pose=candidate;self._uncertainty(jac,weight,dt)
            self.std=np.sqrt(np.diag(self.covariance));self.scan_update_accepted=True
            self.correction_reason='accepted_static_correction' if np.linalg.norm(candidate-prior)>1e-6 else 'validated_static_prior'
            return self._finish_update(dt,movement,turn)
        for _ in range(self.config['iterations']):
            r,jac,weight=self._linearize(candidate,points);projection,metric=self._basis(jac,weight)
            scaled=jac/metric;h=scaled.T@(weight[:,None]*scaled)/len(jac)
            gradient=scaled.T@(weight*r)/len(jac)
            delta=-(projection@np.linalg.solve(h+np.eye(3)*self.config['damping'],gradient))/metric
            trial=candidate+delta;trial[2]=wrap(trial[2])
            innovation=trial-prior;innovation[2]=wrap(innovation[2])
            # Initialization has one bounded correction; subsequent corrections
            # scale with physical elapsed time, not the number of scan messages.
            innovation*=min(1.,xy_cap/max(np.linalg.norm(innovation[:2]),1e-12),yaw_cap/max(abs(innovation[2]),1e-12))
            trial=prior+innovation;trial[2]=wrap(trial[2]);after=self._cost(trial,points)
            if after>=cost-1e-12:break
            candidate=trial;cost=after
            if np.linalg.norm(delta*metric)<1e-4:break
        if not self._correction_permitted(prior,candidate,scan):
            return self._reject_correction(dt,movement,turn)
        self.missing_scan_updates=0
        self.pose=candidate;r,jac,weight=self._linearize(candidate,points)
        self._uncertainty(jac,weight,dt)
        self.std=np.sqrt(np.diag(self.covariance));self.scan_update_accepted=True
        self.correction_reason='accepted_static_correction' if np.linalg.norm(candidate-prior)>1e-6 else 'validated_static_prior'
        # No particle/neural posterior exists here. Preserve v5's conservative
        # entropy sentinel instead of silently redefining this actor feature.
        self.posterior_entropy=1.
        return self._finish_update(dt,movement,turn)
