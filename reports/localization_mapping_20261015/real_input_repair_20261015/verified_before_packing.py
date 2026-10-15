"""Opt-in ray-consistent tracking; raw avoidance scans and maps stay intact.

Endpoint proximity alone can match a return behind a nearer mapped wall.
Exclude that ray from tracking, then veto corrections that create additional
free-space contradictions. This is local tracking, not global recovery/SLAM.
"""
import numpy as np
from neural_navigation.fast_static_localizer import FastStaticLocalizer, temporal_contract


def contract(native=False):
    return dict(schema='ray_consistent_static_tracking_v1', experimental=True,
        tracking=temporal_contract(native), sensor_max_range_m=10.,
        ray_endpoint_allowance_m=.18, maximum_new_contradicted_rays=2,
        maximum_new_contradiction_fraction=.05, thresholds_calibrated=False,
        raw_obstacle_scan_unchanged=True, map_unchanged=True,
        supplied_initial_pose_required=True, global_relocalization=False)


class VerifiedStaticLocalizer(FastStaticLocalizer):
    def __init__(self, static_map, lidar, config=None, kernel_path=None, uncertainty=None):
        selected = contract() if config is None else config
        if selected not in (contract(), contract(True)):
            raise ValueError('Unsupported ray-consistent localization contract')
        self.verification_contract = selected
        super().__init__(static_map, lidar, selected['tracking'], kernel_path, uncertainty)

    def reset(self, nominal, rng):
        super().reset(nominal, rng)
        self.ray_contradicted_hits=0
        self.ray_excluded_static_hits=0
        self.new_contradicted_hits=0
        self._excluded_beams=np.zeros(self.lidar.beams, bool)

    def ray_contradictions(self, pose, scan, indices=None):
        indices=self.indices if indices is None else np.asarray(indices,int)
        values=np.asarray(scan, float)[indices]
        maximum=min(self.lidar.range_max,self.verification_contract['sensor_max_range_m'])
        finite=np.isfinite(values)&(values>=self.lidar.range_min)&(values<maximum)
        out=np.zeros(len(values),bool)
        if not finite.any():return out,finite
        angles=self.lidar.angles[indices[finite]]+self.lidar.mount[3]+pose[2]
        c,s=np.cos(pose[2]),np.sin(pose[2])
        origin=pose[:2]+self.lidar.mount[:2]@np.array([[c,s],[-s,c]])
        distance=np.arange(0.,float(values[finite].max()),self.map.res/2)
        active=distance[None,:]<np.maximum(0.,values[finite,None]-self.verification_contract['ray_endpoint_allowance_m'])
        points=origin+distance[None,:,None]*np.c_[np.cos(angles),np.sin(angles)][:,None,:]
        occupied=self.map.lookup(points.reshape(-1,2),self.map.occupied,outside=False).reshape(active.shape)
        out[finite]=np.any(occupied&active,axis=1)
        return out,finite

    def _select_static_hits(self, prior, scan, body, valid, keep):
        all_contradicted,all_finite=self.ray_contradictions(prior,scan,np.arange(self.lidar.beams))
        contradicted,finite=all_contradicted[self.indices],all_finite[self.indices]
        self._prior_contradictions=contradicted
        self.ray_contradicted_hits=int(contradicted.sum())
        self.new_contradicted_hits=0
        excluded=contradicted|~finite
        self._excluded_beams[:]=all_contradicted|~all_finite
        self.ray_excluded_static_hits=int(np.count_nonzero(keep&excluded[valid]))
        return keep&~excluded[valid]

    def _temporal_support(self, points, odom, scan, timestamp, prior):
        # Do not resurrect an excluded current ray through endpoint history.
        local=np.asarray(scan,float).copy()
        local[self._excluded_beams]=np.nan
        return super()._temporal_support(points,odom,local,timestamp,prior)

    def _correction_permitted(self, prior, candidate, scan):
        contradicted,finite=self.ray_contradictions(candidate,scan)
        self.new_contradicted_hits=int(np.count_nonzero(contradicted&~self._prior_contradictions))
        cfg=self.verification_contract
        limit=max(cfg['maximum_new_contradicted_rays'],int(np.floor(finite.sum()*cfg['maximum_new_contradiction_fraction'])))
        if self.new_contradicted_hits>limit:
            self.correction_reason='ray_inconsistent_correction'
            return False
        return True
