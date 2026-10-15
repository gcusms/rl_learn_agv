"""Portable sensor contracts and acquisition-time odometry alignment."""
from collections import deque
import numpy as np
from navrl.geometry import integrate,wrap


class LidarSpec:
    def __init__(self,data):
        self.beams=int(data['beams']);self.angle_min=float(data['angle_min']);self.angle_max=float(data['angle_max'])
        self.range_min=float(data['range_min']);self.range_max=float(data['range_max'])
        self.mount=np.asarray(data['mount'],float)
        if self.beams<2 or self.mount.shape!=(4,) or not np.isfinite(self.mount).all():raise ValueError('Invalid LiDAR geometry')
        if not np.isfinite([self.angle_min,self.angle_max,self.range_min,self.range_max]).all() or not self.angle_min<self.angle_max or not 0<self.range_min<self.range_max:
            raise ValueError('Invalid LiDAR limits')
        self.angles=np.linspace(self.angle_min,self.angle_max,self.beams)

    def resample(self,ranges,angle_min,angle_increment,range_min=0.,range_max=np.inf):
        ranges=np.asarray(ranges,float)
        if ranges.ndim!=1 or len(ranges)<2 or not np.isfinite([angle_min,angle_increment]).all() or angle_increment==0:
            raise ValueError('Invalid scan angular geometry')
        angles=wrap(angle_min+np.arange(len(ranges))*angle_increment)
        difference=np.abs(wrap(angles[:,None]-self.angles[None,:]))
        nearest=np.argmin(difference,axis=0);sample=ranges[nearest].copy()
        supported=difference[nearest,np.arange(self.beams)]<=max(abs(angle_increment)*1.5,.025)
        sample[~supported|(sample<max(range_min,self.range_min))]=np.nan
        sample[sample>min(range_max,self.range_max)]=np.inf
        return sample,bool(supported.mean()>=.9)


class OdometryBuffer:
    def __init__(self,size=100):self.rows=deque(maxlen=size)
    def append(self,timestamp,pose,velocity):
        pose=np.asarray(pose,float);velocity=np.asarray(velocity,float)
        if pose.shape!=(3,) or velocity.shape!=(2,) or not np.isfinite(pose).all() or not np.isfinite(velocity).all() or not np.isfinite(timestamp):
            raise ValueError('Invalid odometry')
        if self.rows and timestamp<self.rows[-1][0]:raise ValueError('Odometry timestamps must be monotonic')
        row=(float(timestamp),pose.copy(),velocity.copy())
        if self.rows and timestamp==self.rows[-1][0]:self.rows[-1]=row
        else:self.rows.append(row)
    def sample(self,timestamp,max_gap=.2,max_extrapolation=.05):
        if not self.rows or not np.isfinite(timestamp):return None
        rows=list(self.rows);stamps=np.array([r[0] for r in rows]);index=int(np.searchsorted(stamps,timestamp))
        if index<len(rows) and abs(rows[index][0]-timestamp)<1e-9:return rows[index][1].copy(),rows[index][2].copy()
        if index==0:return None
        if index==len(rows):
            dt=timestamp-rows[-1][0]
            if dt>max_extrapolation:return None
            return integrate(rows[-1][1],*rows[-1][2],dt),rows[-1][2].copy()
        left,right=rows[index-1],rows[index];gap=right[0]-left[0]
        if gap>max_gap:return None
        t=(timestamp-left[0])/gap;pose=left[1]*(1-t)+right[1]*t
        pose[2]=wrap(left[1][2]+t*wrap(right[1][2]-left[1][2]))
        return pose,left[2]*(1-t)+right[2]*t

    def sample_many(self,timestamps,max_gap=.2,max_extrapolation=.05):
        """Vectorized acquisition poses; reject the whole scan if any pose is missing."""
        query=np.asarray(timestamps,float)
        if query.ndim!=1 or not len(query) or not np.isfinite(query).all() or not self.rows:return None
        rows=list(self.rows);stamps=np.array([r[0] for r in rows]);poses=np.array([r[1] for r in rows]);velocities=np.array([r[2] for r in rows])
        index=np.searchsorted(stamps,query);clipped=np.minimum(index,len(rows)-1)
        exact=np.abs(stamps[clipped]-query)<1e-9
        if np.any((index==0)&~exact):return None
        after=(index==len(rows))&~exact
        if np.any(query[after]-stamps[-1]>max_extrapolation+1e-9):return None
        left=np.clip(index-1,0,len(rows)-1);right=np.clip(index,0,len(rows)-1)
        middle=~exact&~after;gap=stamps[right]-stamps[left]
        if np.any(gap[middle]>max_gap+1e-9):return None
        fraction=np.zeros(len(query));fraction[middle]=(query[middle]-stamps[left[middle]])/gap[middle]
        result=poses[left]*(1-fraction[:,None])+poses[right]*fraction[:,None]
        result[:,2]=wrap(poses[left,2]+fraction*wrap(poses[right,2]-poses[left,2]))
        speed=velocities[left]*(1-fraction[:,None])+velocities[right]*fraction[:,None]
        result[exact]=poses[clipped[exact]];speed[exact]=velocities[clipped[exact]]
        if after.any():
            dt=query[after]-stamps[-1];v,w=velocities[-1];yaw=poses[-1,2]
            if abs(w)<1e-8:
                dx=v*dt*np.cos(yaw);dy=v*dt*np.sin(yaw)
            else:
                dx=v/w*(np.sin(yaw+w*dt)-np.sin(yaw));dy=v/w*(-np.cos(yaw+w*dt)+np.cos(yaw))
            result[after]=poses[-1]+np.c_[dx,dy,w*dt];result[after,2]=wrap(result[after,2]);speed[after]=velocities[-1]
        return result,speed
