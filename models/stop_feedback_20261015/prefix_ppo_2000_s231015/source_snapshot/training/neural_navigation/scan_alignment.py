"""Optional rolling LaserScan adapter, independent of ROS and simulator state.

The stamp denotes the FIRST beam; time_increment is the interval between beams,
not scan_time. Motion compensation relies on the supplied odometry, which may
be inaccurate on this robot. Calibration and closed-loop hardware validation
are required before enabling it in production.
"""
from dataclasses import dataclass
import numpy as np
from navrl.geometry import wrap


def deskew_endpoints(ranges,angles,poses,reference,mount):
    """Measured finite endpoints in the reference LiDAR frame; no truth input."""
    ranges=np.asarray(ranges,float);angles=np.asarray(angles,float);poses=np.asarray(poses,float)
    reference=np.asarray(reference,float);mount=np.asarray(mount,float)
    if ranges.ndim!=1 or angles.shape!=ranges.shape or poses.shape!=(len(ranges),3) or reference.shape!=(3,) or mount.shape!=(4,):
        raise ValueError('Invalid deskew geometry')
    if not np.isfinite(ranges).all() or not np.isfinite(angles).all() or not np.isfinite(poses).all() or not np.isfinite(reference).all() or not np.isfinite(mount).all():
        raise ValueError('Deskew endpoints require finite inputs')
    yaw=poses[:,2];c=np.cos(yaw);s=np.sin(yaw)
    origins=poses[:,:2]+np.c_[c*mount[0]-s*mount[1],s*mount[0]+c*mount[1]]
    world_angle=yaw+mount[3]+angles
    world=origins+ranges[:,None]*np.c_[np.cos(world_angle),np.sin(world_angle)]
    c,s=np.cos(reference[2]),np.sin(reference[2]);body=(world-reference[:2])@np.array([[c,-s],[s,c]])
    c,s=np.cos(mount[3]),np.sin(mount[3]);return (body-mount[:2])@np.array([[c,-s],[s,c]])


@dataclass
class PreparedScan:
    scan:np.ndarray
    timestamp:float
    pose:np.ndarray
    velocity:np.ndarray
    valid:bool
    compensated:bool
    coverage:float
    reason:str


class NativeScanAdapter:
    def __init__(self,spec,odometry,max_scan_duration=.2):
        self.spec=spec;self.odometry=odometry;self.max_scan_duration=max_scan_duration

    def prepare(self,ranges,angle_min,angle_increment,stamp,time_increment,range_min,range_max):
        ranges=np.asarray(ranges,float);spec=self.spec
        def rejected(reason,timestamp=stamp):
            return PreparedScan(np.full(spec.beams,np.nan),float(timestamp),np.zeros(3),np.zeros(2),False,False,0.,reason)
        if ranges.ndim!=1 or len(ranges)<2 or not np.isfinite([angle_min,angle_increment,stamp,time_increment,range_min,range_max]).all() or angle_increment==0 or time_increment<0 or not 0<=range_min<range_max:
            return rejected('invalid_scan_metadata')
        duration=(len(ranges)-1)*time_increment;reference_time=stamp+duration
        if duration>self.max_scan_duration:return rejected('scan_duration_exceeded',reference_time)
        stamps=stamp+np.arange(len(ranges))*time_increment
        aligned=self.odometry.sample_many(stamps)
        if aligned is None:return rejected('acquisition_odometry_unavailable',reference_time)
        poses,speeds=aligned;reference=poses[-1]
        movement=np.max(np.linalg.norm(poses[:,:2]-reference[:2],axis=1));rotation=np.max(np.abs(wrap(poses[:,2]-reference[2])))
        if movement>.1 or rotation>.2:return rejected('scan_motion_exceeded',reference_time)
        angles=angle_min+np.arange(len(ranges))*angle_increment
        if movement<1e-10 and rotation<1e-10:
            # Preserve the established resampling contract when no deskew is
            # needed; avoid introducing a different wall bias at rest.
            output,valid=spec.resample(ranges,angle_min,angle_increment,range_min,range_max)
            difference=np.abs(wrap(angles[:,None]-spec.angles[None,:]))
            coverage=float((difference.min(axis=0)<=max(abs(angle_increment)*1.5,.025)).mean())
            return PreparedScan(output,float(reference_time),reference.copy(),speeds[-1].copy(),valid,False,coverage,'ok' if valid else 'angular_coverage_insufficient')
        finite=np.isfinite(ranges)&(ranges>=max(range_min,spec.range_min))&(ranges<=min(range_max,spec.range_max))
        projected_range=np.full(len(ranges),np.nan)
        # No-return rays supply angular coverage, never finite clearing evidence.
        bearing=wrap(angles+poses[:,2]-reference[2]);projected_range[np.isposinf(ranges)]=np.inf
        if finite.any():
            points=deskew_endpoints(ranges[finite],angles[finite],poses[finite],reference,spec.mount)
            projected_range[finite]=np.linalg.norm(points,axis=1);bearing[finite]=np.arctan2(points[:,1],points[:,0])
        differences=np.abs(wrap(bearing[:,None]-spec.angles[None,:]))
        # Use native angular resolution, not the coarser model sector width:
        # wide min-pooling itself shifts wall ranges and biases localization.
        tolerance=max(abs(angle_increment)*.75,1e-8)
        contributes=differences<=tolerance;supported=contributes.any(axis=0)
        # Multiple reprojected hits in a sector use the nearest return. Unknown
        # sectors remain NaN; never interpolate free space across an occlusion.
        values=np.where(contributes&~np.isnan(projected_range[:,None]),projected_range[:,None],np.inf)
        output=values.min(axis=0);evidence=(contributes&~np.isnan(projected_range[:,None])).any(axis=0)
        output[~evidence]=np.nan;output[output<spec.range_min]=np.nan;output[output>spec.range_max]=np.inf
        coverage=float(supported.mean());valid=coverage>=.9
        return PreparedScan(output,float(reference_time),reference.copy(),speeds[-1].copy(),valid,duration>0,coverage,'ok' if valid else 'angular_coverage_insufficient')
