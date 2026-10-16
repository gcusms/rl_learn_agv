"""Observable navigation decisions, using map/laser estimates, never simulation truth."""
import copy
from collections import deque
import numpy as np
from scipy.ndimage import distance_transform_edt
from .map import continuous_remaining


class NavigationDecision:
    def __init__(self,static_map):
        self.map=static_map;self.expiry=np.zeros(static_map.free.shape)
        self.revision=0;self.last_plan=-100.;self.positions=deque(maxlen=31)
        self.state='following_path';self.reason='known_map_route';self.blocked=False
        self.static_reachable=True;self.local_points=0

    def update(self,env):
        pose=env.estimate;now=env.elapsed;self.positions.append((now,pose[:2].copy()))
        valid=np.isfinite(env.scan)&(env.scan>=env.lidar.range_min)&(env.scan<env.lidar.range_max)
        m=env.lidar.mount; yaw=pose[2]
        origin=pose[:2]+np.array([np.cos(yaw)*m[0]-np.sin(yaw)*m[1],np.sin(yaw)*m[0]+np.cos(yaw)*m[1]])
        angles=env.lidar.angles[valid]+yaw+m[3]
        points=origin+env.scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]
        points=points[self.map.lookup(points,self.map.distance,outside=0.)>.22]
        self.local_points=len(points)
        r,c=self.map.cells(points);inside=(r>=0)&(c>=0)&(r<self.map.height)&(c<self.map.width)
        self.expiry[r[inside],c[inside]]=now+1.0
        dynamic=(self.expiry>now)
        inflated=distance_transform_edt(~dynamic)*self.map.res<.36 if dynamic.any() else dynamic
        local=copy.copy(self.map);local.free=self.map.free&~inflated
        # Permit the localized start cell and its immediate footprint to escape inflation.
        rr,cc=self.map.cells(pose[:2]);rad=2
        local.free[max(0,rr-rad):rr+rad+1,max(0,cc-rad):cc+rad+1]=self.map.free[max(0,rr-rad):rr+rad+1,max(0,cc-rad):cc+rad+1]
        nearest=int(np.argmin(np.linalg.norm(env.path-pose[:2],axis=1)))
        ahead=env.path[nearest:nearest+22]
        invalid=np.any(local.lookup(ahead,local.free,outside=False)==False)
        stalled=len(self.positions)==31 and np.linalg.norm(self.positions[-1][1]-self.positions[0][1])<.10
        if now-self.last_plan>=1. and (invalid or self.blocked or stalled):
            old=continuous_remaining(env.path,pose[:2]); self.last_plan=now
            try:
                path=local.plan(pose[:2],env.goal)
                changed=len(path)!=len(env.path) or not np.allclose(path,env.path,atol=.05)
                env.path=path
                if hasattr(env,'previous_remaining'):env.previous_remaining+=continuous_remaining(path,pose[:2])-old
                self.blocked=False
                if changed:self.revision+=1
                self.state='replanning';self.reason='laser_obstacle_on_route' if invalid else 'lack_of_progress'
            except ValueError:
                self.blocked=True;self.state='waiting_for_clearance';self.reason='temporary_laser_costmap_blocks_route'
        elif self.blocked:self.state='waiting_for_clearance';self.reason='temporary_laser_costmap_blocks_route'
        else:self.state='following_path';self.reason='known_map_route'
        if np.max(env.localizer.std[:2])>.35:
            self.state='localization_uncertain';self.reason='position_standard_deviation_exceeds_0.35_m'
        self.local_map=local
        return self.evidence(env)

    def evidence(self,env):
        return dict(state=self.state,reason=self.reason,static_map_reachable=self.static_reachable,
                    temporary_route_blocked=self.blocked,plan_revision=self.revision,
                    unmatched_laser_returns=self.local_points,localization_std=env.localizer.std.tolist(),
                    estimated_goal_distance_m=float(np.linalg.norm(env.goal-env.estimate[:2])),
                    remaining_path_m=continuous_remaining(env.path,env.estimate[:2]),
                    planner_source='A_star_with_laser_costmap',action_source='velocity_controller')
