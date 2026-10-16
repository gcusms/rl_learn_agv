"""Optional observed-map navigation driven by external scan/odom SLAM.

Neither map nor localization input may contain simulator truth. Actor, action
choices and guard are the existing portable model. Unknown cells stay blocked.
Frontier subgoals are task inputs, not a substitute velocity controller.
"""
import numpy as np
import copy
from scipy.ndimage import binary_dilation,distance_transform_edt
from scipy.ndimage import label
from navrl.geometry import wrap
from neural_navigation.goal_pose_runtime import DockingRuntime
from neural_navigation.models import scan_map_agreement


def clear_body_unknown(static_map,pose,footprint=(.393924904,.329460022),previous_points=None):
    """The occupied robot body is known; never erase an occupied map cell.

    SLAM may leave a near-sensor blind hole below the laser. Only the actual
    rectangle at the estimated pose is cleared, with no safety-margin expansion.
    Raw SLAM occupancy remains intact; this is an explicit planning-map layer.
    """
    r,c=np.indices(static_map.free.shape)
    x=static_map.origin[0]+(c+.5)*static_map.res-pose[0]
    y=static_map.origin[1]+(static_map.height-r-.5)*static_map.res-pose[1]
    a=pose[2];lx=np.cos(a)*x+np.sin(a)*y;ly=-np.sin(a)*x+np.cos(a)*y
    body=(abs(lx)<=footprint[0]/2)&(abs(ly)<=footprint[1]/2)
    evidence=body.copy()
    if previous_points is not None and len(previous_points):
        rows,cols=static_map.cells(previous_points)
        inside=(rows>=0)&(rows<static_map.height)&(cols>=0)&(cols<static_map.width)
        evidence[rows[inside],cols[inside]]=True
    cleared=evidence&static_map.unknown&~static_map.occupied
    if not cleared.any():return static_map
    m=copy.copy(static_map);m.unknown=static_map.unknown.copy();m.unknown[cleared]=False
    m.blocked=m.occupied|m.unknown;m.clearance=distance_transform_edt(~m.blocked)*m.res
    m.free=(m.clearance>.30+m.res*.71)&~m.blocked
    m.body_unknown_cleared_cells=int(cleared.sum())
    return m


def clear_observed_ray_unknown(static_map,pose,scan,lidar,maximum_distance=1.2):
    """Clear unknown cells crossed by finite measured rays, stopping at walls.

    This planning layer never erases occupancy, never treats invalid returns as
    free, and never changes the raw SLAM map or policy scan. It compensates for
    free evidence withheld with transient returns in the static mapper.
    """
    scan=np.asarray(scan,float);valid=np.isfinite(scan)&(scan>=lidar.range_min)&(scan<lidar.range_max)
    mnt=lidar.mount;a=pose[2];c,s=np.cos(a),np.sin(a)
    origin=pose[:2]+[c*mnt[0]-s*mnt[1],s*mnt[0]+c*mnt[1]]
    cleared=np.zeros_like(static_map.unknown)
    for index in np.flatnonzero(valid):
        length=min(float(scan[index])-.08,maximum_distance)
        if length<=0:continue
        angle=a+mnt[3]+lidar.angles[index]
        points=origin+np.arange(0.,length,static_map.res*.5)[:,None]*[np.cos(angle),np.sin(angle)]
        rows,cols=static_map.cells(points)
        inside=(rows>=0)&(rows<static_map.height)&(cols>=0)&(cols<static_map.width)
        invalid=np.flatnonzero(~inside)
        stop=int(invalid[0]) if len(invalid) else len(rows)
        occupied=np.flatnonzero(static_map.occupied[rows[:stop],cols[:stop]])
        if len(occupied):stop=min(stop,int(occupied[0]))
        cleared[rows[:stop],cols[:stop]]=True
    cleared&=static_map.unknown&~static_map.occupied
    if not cleared.any():return static_map
    m=copy.copy(static_map);m.unknown=static_map.unknown.copy();m.unknown[cleared]=False
    m.blocked=m.occupied|m.unknown;m.clearance=distance_transform_edt(~m.blocked)*m.res
    m.free=(m.clearance>.30+m.res*.71)&~m.blocked
    return m


class SlamPoseLocalizer:
    uses_timestamps=True
    def __init__(self,template):
        self.__dict__.update(template.__dict__)
        self.measurement=None;self.correction_reason='waiting_slam';self.static_support_fraction=0.
    def supply(self,pose,timestamp,std):
        pose=np.asarray(pose,float);std=np.asarray(std,float)
        if pose.shape!=(3,) or std.shape!=(3,) or not np.isfinite(np.r_[pose,std,timestamp]).all() or min(std)<=0:
            raise ValueError('Finite external SLAM pose, time and uncertainty required')
        self.measurement=(pose.copy(),float(timestamp),std.copy())
    def update(self,odom,scan,timestamp):
        if self.last_odom is not None:
            delta=odom[:2]-self.last_odom[:2];a=self.last_odom[2];c,s=np.cos(a),np.sin(a)
            local=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])
            c,s=np.cos(self.pose[2]),np.sin(self.pose[2]);self.pose[:2]+=[c*local[0]-s*local[1],s*local[0]+c*local[1]]
            self.pose[2]=wrap(self.pose[2]+wrap(odom[2]-self.last_odom[2]))
        self.last_odom=np.asarray(odom).copy();self.scan_update_accepted=False
        self.large_correction=False
        if self.measurement is not None and -.05<=timestamp-self.measurement[1]<=1.0:
            self.large_correction=(np.linalg.norm(self.measurement[0][:2]-self.pose[:2])>.25
                or abs(wrap(self.measurement[0][2]-self.pose[2]))>.25)
            self.pose=self.measurement[0].copy();self.std=self.measurement[2].copy()
            self.scan_update_accepted=True;self.missing_scan_updates=0;self.correction_reason='external_slam_pose'
        else:
            self.std=np.sqrt(self.std**2+np.array([.01,.01,.008])**2*.2)
            self.missing_scan_updates+=1;self.correction_reason='slam_stale_odometry_only'
        self.valid_scan_fraction=float(np.isfinite(scan).mean())
        self.map_agreement=scan_map_agreement(self.map,self.lidar,self.pose,scan)
        self.static_support_fraction=self.map_agreement;self.posterior_entropy=1.
        return self.pose.copy()


def select_observed_subgoal(static_map,pose,goal,routing_grid=False):
    """Goal-directed candidate within observed free space; no collision truth."""
    # Only candidates in the robot's observed component are eligible. This
    # is connectivity filtering, not a velocity or shortest-path controller.
    components,_=label(static_map.free)
    r,c=static_map.cells(pose[:2])
    if not (0<=r<static_map.height and 0<=c<static_map.width):return None,False
    component=components[r,c]
    if not component:return None,False
    connected=components==component
    if routing_grid:
        # The neural router receives exactly [2::4, 2::4]. Fine-grid
        # connectivity alone can select a goal across a coarse-grid break.
        # Restrict candidates to the actual router's component; do not open
        # unknown cells or generate a replacement graph-search route.
        coarse,_=label(static_map.free[2::4,2::4])
        coarse_component=coarse[r//4,c//4]
        if not coarse_component:return None,False
        connected&=np.repeat(np.repeat(coarse==coarse_component,4,axis=0),4,axis=1)
    if static_map.lookup(goal[:2],connected,outside=False):return np.asarray(goal,float),True
    cells=np.argwhere(connected[2::4,2::4])
    if not len(cells):return None,False
    fine=cells*4+2
    points=np.array([static_map.xy(cell) for cell in fine])
    distance=np.linalg.norm(points-pose[:2],axis=1);keep=distance>.4
    if not keep.any():return None,False
    near_unknown=binary_dilation(static_map.unknown,iterations=8)[fine[:,0],fine[:,1]]
    cost=np.linalg.norm(points-np.asarray(goal[:2]),axis=1)+.2*distance-.1*near_unknown
    cost[~keep]=np.inf
    return points[int(cost.argmin())],False


class OnlineMappingRuntime:
    def __init__(self,bundle,initial_map,pose,goal,bootstrap_map_path,ray_free_layer=False,coarse_subgoals=False,known_free_dynamics=False):
        # A saved snapshot is only the initial observed map, never a simulator map.
        self.requested_goal=np.asarray(goal,float);self.last_map_key=None
        self.ray_free_layer=bool(ray_free_layer)
        self.coarse_subgoals=bool(coarse_subgoals)
        initial_map=clear_body_unknown(initial_map,pose)
        self.body_evidence=[]
        self._remember_body(initial_map,pose)
        subgoal,final=self._select_subgoal(initial_map,pose)
        if subgoal is None:raise ValueError('No observed navigation subgoal available')
        self.runtime=DockingRuntime(bundle,initial_map,pose,subgoal,'shortcut')
        self.core=self.runtime.core;self.guard=self.runtime.guard
        self.core.routing_dynamic_known_free_only=bool(known_free_dynamics)
        self.core.replace_map(initial_map)
        self.core.localizer=SlamPoseLocalizer(self.core.localizer)
        self.final_subgoal=final;self.subgoal_updates=0;self.map_updates=1

    def _select_subgoal(self,static_map,pose):
        if getattr(self,'coarse_subgoals',False):
            return select_observed_subgoal(static_map,pose,self.requested_goal,routing_grid=True)
        return select_observed_subgoal(static_map,pose,self.requested_goal)

    def _remember_body(self,static_map,pose):
        # Keep cell centers inside the actual body, never its inflated margin.
        extent=.26;r0,c0=static_map.cells(pose[:2]+[-extent,extent]);r1,c1=static_map.cells(pose[:2]+[extent,-extent])
        cells=np.array([(r,c) for r in range(max(0,r0),min(static_map.height,r1+1))
            for c in range(max(0,c0),min(static_map.width,c1+1))])
        if not len(cells):return
        points=np.array([static_map.xy(cell) for cell in cells]);delta=points-pose[:2]
        a=pose[2];local=delta@np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        inside=(abs(local[:,0])<=.393924904/2)&(abs(local[:,1])<=.329460022/2)
        self.body_evidence.append(points[inside]);self.body_evidence=self.body_evidence[-200:]

    def _clear_world_history(self):
        self.core.expiry.fill(0);self.core.laser_fusion.reset()
        self.body_evidence=[];self.last_map_key=None
        if self.guard.memory:self.guard.memory.reset()
        self.runtime.policy.prediction.reset();self.guard.reverse_started=None

    def step(self,scan,odom,velocity,timestamp,observed_map,slam_pose,std,pose_timestamp=None):
        self.core.localizer.supply(slam_pose,timestamp if pose_timestamp is None else pose_timestamp,std)
        if observed_map is not None:
            if self.ray_free_layer:
                observed_map=clear_observed_ray_unknown(observed_map,slam_pose,scan,self.core.lidar)
            previous=np.concatenate(self.body_evidence) if self.body_evidence else None
            observed_map=clear_body_unknown(observed_map,slam_pose,self.guard.half*2,previous)
            self._remember_body(observed_map,slam_pose)
            key=(observed_map.res,observed_map.origin.tobytes(),observed_map.blocked.shape,
                observed_map.occupied.tobytes(),observed_map.unknown.tobytes())
            if key!=self.last_map_key:
                self.core.replace_map(observed_map,preserve_valid_route=True);self.last_map_key=key;self.map_updates+=1
        subgoal,final=self._select_subgoal(self.core.static_map,slam_pose)
        if subgoal is None:
            # A mapping pause still executes STOP. Keep localization, scan
            # history and the next actor's actual-action history advancing.
            self.core.update(scan,odom,velocity,timestamp)
            if getattr(self.core.localizer,'large_correction',False):self._clear_world_history()
            self.core.record_action([0.,0.])
            result=dict(command=[0.,0.],stop_reason='waiting_observed_free_space',task_goal_achieved=False,
                proposed_action=[0.,0.],model_command=[0.,0.],timestamp=float(timestamp),
                safety_override=True,actor_evaluated=False)
            if self.runtime.context:self.runtime.context.record(result)
            return result
        required_yaw=len(subgoal)==3
        if (final!=self.final_subgoal or np.linalg.norm(subgoal[:2]-self.core.goal)>.15
            or required_yaw!=(self.core.goal_yaw is not None)):
            self.core.set_goal(subgoal);self.subgoal_updates+=1
        self.final_subgoal=final
        result=self.runtime.step(scan,odom,velocity,timestamp)
        if self.core.localizer.large_correction:
            # Loop correction changes the world frame of old scan memory.
            # Clear that memory and reobserve before releasing motion.
            self._clear_world_history();self.core.record_action([0.,0.])
            result['command']=[0.,0.];result['stop_reason']='slam_correction_reobserve';result['safety_override']=True
            if self.runtime.context:self.runtime.context.record(result)
        result['task_goal_achieved']=bool(final and result['stop_reason']=='estimated_goal_reached')
        result['mapping_subgoal_final']=final;result['mapping_subgoal_updates']=self.subgoal_updates
        result['mapping_map_updates']=self.map_updates
        if not final and result['stop_reason']=='estimated_goal_reached':result['stop_reason']='observed_subgoal_reached'
        return result
