"""Shared learned navigation inference, independent of MuJoCo, Gym and ROS.

The physics environments and sensor-stream runtime use the same routing and
observation methods. Static map geometry is immutable between resets.
"""
from collections import deque
from pathlib import Path
import json
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from navrl.geometry import wrap
from navrl.map import continuous_remaining,segment_is_free
from neural_navigation.models import LearnedLocalizer,NeuralRouter
from neural_navigation.laser_fusion import LaserMapFusion

def corridor_clearance(scan,angles,mount_x=.118389):
    ranges=np.nan_to_num(np.asarray(scan),nan=12.,posinf=12.,neginf=.1)
    x=mount_x+ranges*np.cos(angles);y=ranges*np.sin(angles)
    selected=(x>.05)&(np.abs(y)<.23)
    return float(np.min(x[selected])) if np.any(selected) else 12.

def extend_observation(base,progress=1.,clearance=None):
    base=np.asarray(base,np.float32);target=base[216:218]*2
    heading=np.arctan2(target[1],target[0]) if np.linalg.norm(target)>.001 else 0.
    if clearance is None:
        angles=np.linspace(-1.8915926535897931,1.938407346410207,72)
        clearance=corridor_clearance(base[144:216]*12,angles)
    return np.r_[base,np.sin(heading),np.cos(heading),np.clip(progress,0,1),np.clip(clearance/2,0,1)].astype(np.float32)


class LearnedRoutingMixin:
    def cell(self,xy):
        r,c=self.static_map.cells(xy);return int(r)//4,int(c)//4

    def xy(self,cell):
        r,c=cell;return self.static_map.origin[:2]+np.array([(c+.5)*self.grid_res,(self.grid_h-r-.5)*self.grid_res])

    def _routing_dynamic_layer(self):
        now=self.elapsed;pose=self.estimate;yaw=pose[2];mount=self.lidar.mount
        valid=np.isfinite(self.scan)&(self.scan>=self.lidar.range_min)&(self.scan<self.lidar.range_max)
        origin=pose[:2]+[np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1],np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1]]
        angles=self.lidar.angles[valid]+yaw+mount[3]
        points=origin+self.scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]
        points=points[self.static_map.lookup(points,self.static_map.distance,outside=0.)>.22]
        for point in points:
            r,c=self.cell(point)
            if 0<=r<self.grid_h and 0<=c<self.grid_w:self.expiry[r,c]=now+getattr(self,'dynamic_memory_seconds',1.)
        self.unmatched_returns=len(points)
        return binary_dilation(self.expiry>now,iterations=2).astype(np.float32)

    def _neural_route(self):
        now=self.elapsed;pose=self.estimate;dyn=self._routing_dynamic_layer()
        gm=np.zeros_like(self.free);gr,gc=self.cell(self.goal)
        if 0<=gr<self.grid_h and 0<=gc<self.grid_w:gm[gr,gc]=1.
        pm=np.zeros_like(self.free);sr,sc=self.cell(pose[:2])
        if 0<=sr<self.grid_h and 0<=sc<self.grid_w:pm[sr,sc]=1.
        current=np.stack([self.free*(1-dyn),dyn,gm,pm]);static=np.stack([self.free,np.zeros_like(dyn),gm,pm])
        self.start_inflation_overlap=False
        if getattr(self,'escape_start',False) and 0<=sr<self.grid_h and 0<=sc<self.grid_w and self.free[sr,sc]>0 and dyn[sr,sc]>0:
            # Inflation can cover the physically occupied robot origin while
            # adjacent escape cells remain free. Permit neural propagation from
            # that origin only; retain every obstacle and all static barriers.
            current[0,sr,sc]=self.free[sr,sc];current[1,sr,sc]=0.
            self.start_inflation_overlap=True
        # Exact-input cache: no quantization, no stale dynamic layer, no change
        # to learned propagation depth or checkpoint outputs.
        inputs=np.stack([current,static]);key=inputs.tobytes()
        cached=self.router_cache.get(key)
        if cached is None:
            with torch.no_grad():value,q,reach=self.router(torch.from_numpy(inputs))
            q=q[0].numpy();reach=reach.numpy();self.router_cache_misses+=1
            if len(self.router_cache)>=4:self.router_cache.pop(next(iter(self.router_cache)))
            self.router_cache[key]=(q,reach)
        else:
            q,reach=cached;self.router_cache_hits+=1
        self.current_probability=float(reach[0,0,sr,sc]) if 0<=sr<self.grid_h and 0<=sc<self.grid_w else 0.
        self.static_probability=float(reach[1,0,sr,sc]) if 0<=sr<self.grid_h and 0<=sc<self.grid_w else 0.
        path=[pose[:2].copy()];r,c=sr,sc;visited=set()
        for _ in range(160):
            if not (0<=r<self.grid_h and 0<=c<self.grid_w) or (r,c) in visited:break
            visited.add((r,c))
            if (r,c)==(gr,gc):path.append(self.goal.copy());break
            direction=int(np.argmax(q[:,r,c]));dr,dc=self.router.directions[direction];r+=dr;c+=dc
            if not (0<=r<self.grid_h and 0<=c<self.grid_w):break
            path.append(self.xy((r,c)))
        new=np.asarray(path)
        if self.routing_ready and (len(new)!=len(self.predicted_path) or not np.allclose(new,self.predicted_path,atol=.08)):self.plan_revision+=1
        self.predicted_path=new;self.routing_ready=True;self.last_route_time=now

    def _observation(self,scan,initialize=False):
        # Parent reset passes a scan before it sets the public scan attribute.
        self.scan=scan
        old=continuous_remaining(self.path,self.estimate[:2]) if self.routing_ready else None
        if initialize or self.elapsed-self.last_route_time>=.4:self._neural_route()
        self.path=self.predicted_path.copy()
        self.remaining=continuous_remaining(self.path,self.estimate[:2]) if len(self.path)>1 else float(np.linalg.norm(self.goal-self.estimate[:2]))
        if old is not None and hasattr(self,'previous_remaining'):self.previous_remaining+=self.remaining-old
        clean=np.nan_to_num(scan,nan=12.,posinf=12.,neginf=.1)
        binned=np.array([p.min() for p in np.array_split(clean,72)])/self.lidar.range_max
        if getattr(self,'timed_scan_history',None) is not None:
            self.history.clear();self.history.extend(self.timed_scan_history.update(self.elapsed,binned))
        elif initialize:
            self.history.clear()
            for _ in range(3):self.history.append(binned.copy())
        else:self.history.append(binned)
        nearest=int(np.argmin(np.linalg.norm(self.path-self.estimate[:2],axis=1)))
        waypoint=self.path[min(nearest+2,len(self.path)-1)]
        yaw=self.estimate[2];rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
        local_waypoint=rotation@(waypoint-self.estimate[:2]);goal=rotation@(self.goal-self.estimate[:2])
        deviation=float(np.min(np.linalg.norm(self.path-self.estimate[:2],axis=1)))
        features=np.r_[local_waypoint/2,goal/12,self.measured_velocity/[.2,.5],self.last_action,self.localizer.std/[.5,.5,.5],deviation/2,self.static_probability,self.current_probability,min(self.remaining/20,1.)]
        return np.clip(np.r_[np.concatenate(self.history),features],-1,1).astype(np.float32)


class RoomRoutingMixin:
    def _static_visible(self,a,b):
        # Static map is immutable for this environment. Dynamic occupancy is
        # checked separately on every route test and is never cached here.
        key=(np.asarray(a,dtype=np.float64).tobytes(),np.asarray(b,dtype=np.float64).tobytes())
        if key not in self.static_visibility_cache:
            if len(self.static_visibility_cache)>=4096:self.static_visibility_cache.clear()
            self.static_visibility_cache[key]=bool(self.static_map.visible(a,b))
        return self.static_visibility_cache[key]

    def _velocity_command(self,action):
        return np.array([action[0]*(.2 if action[0]>=0 else .08),action[1]*.5])

    def xy(self,cell):
        # Match the actual free-map sample [2::4,2::4], including pixel centers.
        # The inherited coarse-center formula differs by half a source pixel.
        return super().xy(cell)+np.array([self.static_map.res/2,-self.static_map.res/2])

    def _routing_dynamic_layer(self):
        # Known-map subtraction + real laser pose; no simulator obstacle positions.
        return self.laser_fusion.update(self)

    def _route_usable(self,path):
        if path is None or len(path)<2 or np.linalg.norm(path[-1]-self.goal)>.12:return False
        if getattr(self,'route_mode','raw')=='shortcut':
            return all(self._segment_clear(a,b) for a,b in zip(path[:-1],path[1:]))
        samples=[]
        for a,b in zip(path[:-1],path[1:]):
            if not self._static_visible(a,b):return False
            samples.extend(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.08)+1)))
        if not samples:return False
        points=np.array(samples);r,c=self.static_map.cells(points);r=r//4;c=c//4
        if np.any((r<0)|(r>=self.grid_h)|(c<0)|(c>=self.grid_w)):return False
        return not np.any(getattr(self.laser_fusion,'routing_hard',self.laser_fusion.hard)[r,c])

    def _segment_clear(self,a,b):
        if not self._static_visible(a,b):return False
        # Supercover also checks both cells when a segment crosses a corner.
        # No sampled gaps, and no cached dynamic occupancy.
        free=(self.free>0)&~getattr(self.laser_fusion,'routing_hard',self.laser_fusion.hard)
        return segment_is_free(a,b,self.static_map.origin,self.grid_res,free)

    def _observed_route_clear(self,path):
        free=(self.free>0)&~self.laser_fusion.observed_hard
        return all(self._static_visible(a,b) and segment_is_free(a,b,self.static_map.origin,self.grid_res,free)
            for a,b in zip(path[:-1],path[1:]))

    def _confirm_forecast_detour(self,old,tail,candidate):
        config=getattr(self,'routing_contract',None)
        if config is None or config.get('schema')!='confirm_forecast_detour_v1':return False
        if not self._observed_route_clear(tail):
            self.detour_candidate=None;self.detour_since=None;return False
        if len(candidate)<2 or np.linalg.norm(candidate[-1]-self.goal)>.12:
            self.detour_candidate=None;self.detour_since=None;return False
        old_length=continuous_remaining(tail,self.estimate[:2]);new_length=continuous_remaining(candidate,self.estimate[:2])
        delta=candidate[min(2,len(candidate)-1)]-self.estimate[:2]
        direction=np.array([np.cos(self.estimate[2]),np.sin(self.estimate[2])])
        behind=float(delta@direction)/max(np.linalg.norm(delta),1e-9)<config['behind_cosine']
        disruptive=behind or new_length-old_length>max(config['excess_length_m'],config['excess_length_ratio']*old_length)
        if not disruptive:
            self.detour_candidate=None;self.detour_since=None;return False
        stable=False
        if self.detour_candidate is not None:
            distances=np.linalg.norm(candidate[:,None]-self.detour_candidate[None,:],axis=2)
            stable=max(distances.min(axis=0).max(),distances.min(axis=1).max())<config['candidate_geometry_tolerance_m']
        if not stable:self.detour_candidate=candidate.copy();self.detour_since=self.elapsed
        if self.elapsed-self.detour_since>=config['confirmation_s']-1e-8:
            self.detour_candidate=None;self.detour_since=None;return False
        return True

    def _shortcut_route(self,path):
        if len(path)<3 or np.linalg.norm(path[-1]-self.goal)>.12:return path
        sparse=[path[0]];index=0
        while index<len(path)-1:
            end=len(path)-1
            while end>index+1 and not self._segment_clear(path[index],path[end]):end-=1
            if not self._segment_clear(path[index],path[end]):return path
            sparse.append(path[end]);index=end
        # Keep the original policy's approximately 0.2m waypoint spacing.
        # This is geometric shortening of a learned route, not graph search.
        dense=[]
        for a,b in zip(sparse[:-1],sparse[1:]):
            count=max(1,int(np.ceil(np.linalg.norm(b-a)/self.grid_res)))
            dense.extend(np.linspace(a,b,count+1)[:-1])
        dense.append(sparse[-1])
        return np.asarray(dense)

    def _neural_route(self):
        old=self.predicted_path.copy() if self.routing_ready else None;revision=self.plan_revision
        super()._neural_route();candidate=self.predicted_path.copy();self.route_retained=False
        if getattr(self,'route_mode','raw')=='shortcut':
            candidate=self._shortcut_route(candidate);self.predicted_path=candidate.copy()
        self.replan_reason='new_goal_or_initial_route'
        if old is None or len(old)<2 or np.linalg.norm(old[-1]-self.goal)>.12:return
        nearest=int(np.argmin(np.linalg.norm(old-self.estimate[:2],axis=1)))
        tail=np.vstack([self.estimate[:2],old[nearest:]])
        deviation=np.linalg.norm(old[nearest]-self.estimate[:2])
        if deviation>.35 or not self._route_usable(tail):
            if deviation<=.35 and self._confirm_forecast_detour(old,tail,candidate):
                self.predicted_path=old;self.plan_revision=revision;self.route_retained=True
                self.replan_reason='confirming_forecast_only_detour';return
            self.shortcut_candidate=None;self.shortcut_since=None;self.replan_reason='confirmed_obstacle_or_route_invalid';self.last_route_switch=self.elapsed
            return
        self.detour_candidate=None;self.detour_since=None
        old_length=continuous_remaining(tail,self.estimate[:2]);new_length=continuous_remaining(candidate,self.estimate[:2])
        improvement=old_length-new_length
        shorter=self._route_usable(candidate) and improvement>max(.5,.08*old_length)
        if shorter:
            # Confirm a better route for .8s; compare geometry, not moving start cells.
            stable=False
            if self.shortcut_candidate is not None:
                distances=np.linalg.norm(candidate[:,None]-self.shortcut_candidate[None,:],axis=2)
                stable=max(distances.min(axis=0).max(),distances.min(axis=1).max())<.4
            if not stable:
                self.shortcut_candidate=candidate.copy();self.shortcut_since=self.elapsed
            if self.elapsed-self.shortcut_since>=.8 and self.elapsed-self.last_route_switch>=1.:
                self.replan_reason='observed_clear_shorter_route';self.last_route_switch=self.elapsed;self.shortcut_candidate=None;self.shortcut_since=None
                return
            self.replan_reason='confirming_shorter_route'
        else:
            self.shortcut_candidate=None;self.shortcut_since=None;self.replan_reason='retain_valid_route_no_material_improvement'
        self.predicted_path=old;self.plan_revision=revision;self.route_retained=True

    def _observation(self,scan,initialize=False):
        base=super()._observation(scan,initialize=initialize)
        # A route can pass near itself. Discard completed prefix vertices so
        # nearest-point tracking cannot jump back to an already traversed leg.
        nearest=int(np.argmin(np.linalg.norm(self.path-self.estimate[:2],axis=1)))
        if nearest>0:
            self.predicted_path=self.path[nearest:].copy();self.path=self.predicted_path.copy()
        if not self.pose_progress or self.elapsed>self.pose_progress[-1][0]+1e-8:
            self.pose_progress.append((self.elapsed,self.estimate[:2].copy()))
        distance=np.linalg.norm(self.estimate[:2]-self.pose_progress[0][1])
        self.recent_progress=float(np.clip(distance/.15,0,1)) if len(self.pose_progress)>5 else 1.
        nearest=int(np.argmin(np.linalg.norm(self.path-self.estimate[:2],axis=1)))
        target_index=min(nearest+2,len(self.path)-1)
        while target_index>nearest and not self._static_visible(self.estimate[:2],self.path[target_index]):
            target_index-=1
        delta=self.path[target_index]-self.estimate[:2];yaw=self.estimate[2]
        base[216:218]=np.clip(np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@delta/2,-1,1)
        clearance=corridor_clearance(scan,self.lidar.angles,self.lidar.mount[0])
        self.last_policy_observation=extend_observation(base,self.recent_progress,clearance)
        return self.last_policy_observation


class NavigationCore(RoomRoutingMixin,LearnedRoutingMixin):
    """Known-map navigation state driven only by scan, odometry and velocity.

    No simulator state, graph search, AMCL, ROS process or physics environment
    is needed. The caller supplies a coarse initial pose and sensor timestamps.
    """
    def __init__(self,static_map,lidar,components,route_mode='raw'):
        if route_mode not in ('raw','shortcut'):raise ValueError('Unknown route mode')
        self.route_mode=route_mode
        self.timed_scan_history=None
        goal_contract_path=Path(components)/'goal_pose_observation.json'
        if goal_contract_path.exists():
            from neural_navigation.goal_pose import contract
            from neural_navigation.timed_scan_history import TimedScanHistory
            if json.loads(goal_contract_path.read_text())!=contract():raise ValueError('Goal-pose contract mismatch')
            self.timed_scan_history=TimedScanHistory(contract()['control_period_s'])
        self.static_map=static_map;self.lidar=lidar;components=Path(components)
        self.routing_contract=None
        if (components/'routing_contract.json').exists():
            from neural_navigation.routing_contract import contract as routing_contract,soft_forecast_contract
            self.routing_contract=json.loads((components/'routing_contract.json').read_text())
            if self.routing_contract not in (routing_contract(),soft_forecast_contract()):raise ValueError('Routing acceptance contract mismatch')
        localization_contract=components/'localization_contract.json'
        if localization_contract.exists():
            localization_config=json.loads(localization_contract.read_text())
            if localization_config.get('schema') in ('fast_static_tracking_v1','fast_static_tracking_native_v1','fast_static_tracking_v2','fast_static_tracking_native_v2','fast_static_tracking_temporal_v1','fast_static_tracking_temporal_native_v1'):
                from neural_navigation.fast_static_localizer import FastStaticLocalizer
                self.localizer=FastStaticLocalizer(static_map,lidar,localization_config,
                    kernel_path=str((components/'localization_kernel.so').resolve()),
                    uncertainty=json.loads((components/'uncertainty_contract.json').read_text()) if (components/'uncertainty_contract.json').exists() else None)
            else:
                from neural_navigation.odom_anchored_localizer import OdometryAnchoredLocalizer
                self.localizer=OdometryAnchoredLocalizer(static_map,lidar,components/'localizer.pt',localization_config)
        else:self.localizer=LearnedLocalizer(static_map,lidar,components/'localizer.pt')
        checkpoint=torch.load(components/'router.pt',map_location='cpu',weights_only=True)
        self.motion_prediction_config=checkpoint.get('motion_prediction')
        self.escape_start=bool(checkpoint.get('escape_start',False))
        self.router=NeuralRouter(checkpoint['iterations'])
        self.router.load_state_dict(checkpoint['state_dict']);self.router.eval()
        h,w=static_map.free.shape
        if h%4 or w%4:raise ValueError('Current routing grid requires map dimensions divisible by four')
        self.free=static_map.free[2:h:4,2:w:4].astype(np.float32)
        self.grid_res=static_map.res*4;self.grid_h,self.grid_w=self.free.shape
        self.expiry=np.zeros_like(self.free);self.router_cache={}
        self.history=deque(maxlen=3);self.pose_progress=deque(maxlen=16)
        self.dynamic_memory_seconds=8.;self.static_visibility_cache={}
        self.laser_fusion=LaserMapFusion(self)
        self.goal=None;self._initialized=False

    def reset(self,initial_pose,goal,timestamp=0.):
        pose=np.asarray(initial_pose,dtype=float)
        if pose.shape!=(3,) or not np.isfinite(pose).all() or not np.isfinite(timestamp):
            raise ValueError('Finite initial x, y, yaw and timestamp required')
        self.localizer.reset(pose,None);self.estimate=pose.copy()
        self.elapsed=float(timestamp);self._last_timestamp=float(timestamp)
        self.odom=np.zeros(3);self.measured_velocity=np.zeros(2);self.last_action=np.zeros(2)
        self.scan=np.full(self.lidar.beams,np.nan);self.history.clear();self.pose_progress.clear()
        if self.timed_scan_history is not None:self.timed_scan_history.reset()
        self.expiry.fill(0);self.laser_fusion.reset();self.router_cache.clear();self.static_visibility_cache.clear()
        self.router_cache_hits=0;self.router_cache_misses=0
        self.last_route_time=-100.;self.last_route_switch=-100.;self.plan_revision=0;self.routing_ready=False
        self.recent_progress=1.;self.shortcut_candidate=None;self.shortcut_since=None
        self.replan_reason='initial';self.route_retained=False;self.path=np.array([pose[:2]])
        self.previous_remaining=0.;self._first_observation=True;self._initialized=True
        self.odometry_sanity_failure=False
        self.set_goal(goal)

    def set_goal(self,goal):
        goal=np.asarray(goal,dtype=float)
        if goal.shape not in ((2,),(3,)) or not np.isfinite(goal).all() or not self.static_map.lookup(goal[:2],self.static_map.free,outside=False):
            raise ValueError('Goal must be finite and inside inflated known free space')
        self.goal_yaw=float(wrap(goal[2])) if len(goal)==3 else None
        self._goal_settled_since=None
        self.goal=goal[:2].copy();self.last_route_time=-100.;self.routing_ready=False
        self.detour_candidate=None;self.detour_since=None

    def replace_map(self,static_map,preserve_valid_route=False):
        """Install an observed map revision without resetting odometry or actor.

        Dynamic memory is remapped in world coordinates; map-dependent caches
        are invalidated. The caller owns map/odom frame consistency.
        """
        h,w=static_map.free.shape
        previous_ready=getattr(self,'routing_ready',False);previous_route_time=getattr(self,'last_route_time',-100.)
        if h%4 or w%4:raise ValueError('Routing map must be padded to multiples of four')
        old=self.laser_fusion;memory=old.points[old.hard_expiry>self.elapsed]
        expiry=old.hard_expiry[old.hard_expiry>self.elapsed]
        self.static_map=static_map;self.localizer.map=static_map
        self.free=static_map.free[2:h:4,2:w:4].astype(np.float32)
        self.grid_res=static_map.res*4;self.grid_h,self.grid_w=self.free.shape
        self.expiry=np.zeros_like(self.free);self.router_cache.clear();self.static_visibility_cache.clear()
        self.laser_fusion=LaserMapFusion(self)
        if len(memory):
            r,c=static_map.cells(memory);r=r//4;c=c//4
            inside=(r>=0)&(r<self.grid_h)&(c>=0)&(c<self.grid_w)
            np.maximum.at(self.laser_fusion.hard_expiry,(r[inside],c[inside]),expiry[inside])
            self.expiry[:]=self.laser_fusion.hard_expiry
            from scipy.ndimage import distance_transform_edt
            self.laser_fusion.hard=distance_transform_edt(~(self.expiry>self.elapsed))*self.grid_res<=.32
        if old.predictor is not None:self.laser_fusion.predictor=old.predictor
        self.last_route_time=-100.;self.routing_ready=False;self.shortcut_candidate=None;self.shortcut_since=None
        self.detour_candidate=None;self.detour_since=None
        if preserve_valid_route and previous_ready and self._route_usable(self.predicted_path):
            # Keep the existing route in world coordinates until the regular
            # 0.4s replanning tick. New geometry and fresh laser guards apply
            # immediately. Map growth alone must not erase route hysteresis.
            self.routing_ready=True;self.last_route_time=previous_route_time

    def record_action(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all():raise ValueError('Finite two-dimensional action required')
        self.last_action=np.clip(action,-1.,1.)

    def update(self,scan,odom,velocity,timestamp):
        if not self._initialized:raise RuntimeError('Reset with an initial pose and goal before updating')
        scan=np.asarray(scan,dtype=float);odom=np.asarray(odom,dtype=float);velocity=np.asarray(velocity,dtype=float)
        if scan.shape!=(self.lidar.beams,) or odom.shape!=(3,) or velocity.shape!=(2,):
            raise ValueError('Expected a configured scan, planar odometry and signed velocity')
        if not np.isfinite(odom).all() or not np.isfinite(velocity).all() or not np.isfinite(timestamp):
            raise ValueError('Nonfinite sensor state')
        if timestamp<self._last_timestamp:raise ValueError('Sensor timestamps must be monotonic')
        dt=max(float(timestamp)-self._last_timestamp,.2);self.odometry_sanity_failure=False
        if self.localizer.last_odom is not None:
            delta=odom-self.localizer.last_odom;delta[2]=wrap(delta[2])
            if np.linalg.norm(delta[:2])>.2*dt*1.5+.1 or abs(delta[2])>.5*dt*1.5+.1:
                # An odometry reset cannot teleport the map pose. Resynchronize
                # the incremental origin, continue laser alignment and flag stop.
                self.odometry_sanity_failure=True;self.localizer.last_odom=odom.copy()
        self.elapsed=float(timestamp);self._last_timestamp=self.elapsed
        self.scan=scan.copy();self.odom=odom.copy();self.measured_velocity=velocity.copy()
        from neural_navigation.odom_anchored_localizer import OdometryAnchoredLocalizer
        if isinstance(self.localizer,OdometryAnchoredLocalizer) or getattr(self.localizer,'uses_timestamps',False):
            self.estimate=self.localizer.update(self.odom,self.scan,timestamp=self.elapsed)
        else:self.estimate=self.localizer.update(self.odom,self.scan)
        from neural_navigation.goal_pose import update_goal_settling
        update_goal_settling(self)
        observation=self._observation(self.scan,initialize=self._first_observation)
        self._first_observation=False;self.previous_remaining=self.remaining
        return observation.copy(),self.diagnostics()

    def diagnostics(self):
        from neural_navigation.goal_pose import goal_status
        return dict(**goal_status(self),estimated_pose=self.estimate.tolist(),localization_std=self.localizer.std.tolist(),
                    localization_correction_reason=getattr(self.localizer,'correction_reason','legacy_unchecked'),
                    localization_observable_rank=getattr(self.localizer,'observable_rank',None),
                    localization_covariance_calibrated=False,
                    localization_weak_odometry_budget=(self.localizer.weak_geometry_uncertainty.floor.tolist()
                        if getattr(self.localizer,'weak_geometry_uncertainty',None) is not None else None),
                    localization_static_support_fraction=getattr(self.localizer,'static_support_fraction',None),
                    localization_scan_update_accepted=self.localizer.scan_update_accepted,
                    localization_valid_scan_fraction=self.localizer.valid_scan_fraction,
                    localization_missing_scan_updates=self.localizer.missing_scan_updates,
                    localization_map_agreement=self.localizer.map_agreement,
                    localization_posterior_entropy=self.localizer.posterior_entropy,
                    localization_recovery_attempts=self.localizer.recovery_attempts,
                    localization_recovery_accepted=self.localizer.recovery_accepted,
                    scan_motion_prediction_enabled=self.laser_fusion.predictor is not None,
                    scan_moving_tracks=self.laser_fusion.predictor.moving_tracks if self.laser_fusion.predictor else 0,
                    scan_forecast_cells=self.laser_fusion.forecast_cells,
                    start_inflation_overlap=getattr(self,'start_inflation_overlap',False),
                    odometry_sanity_failure=self.odometry_sanity_failure,
                    route_complete=bool(len(self.path)>1 and np.linalg.norm(self.path[-1]-self.goal)<.12),
                    route_length_m=float(continuous_remaining(self.path,self.estimate[:2])),
                    estimated_goal_distance_m=float(np.linalg.norm(self.estimate[:2]-self.goal)),
                    static_reachability_probability=self.static_probability,current_reachability_probability=self.current_probability,
                    route_switch_reason=self.replan_reason,router_cache_hits=self.router_cache_hits,
                    router_cache_misses=self.router_cache_misses,runtime_robot_astar=False,runtime_robot_amcl=False)

    def velocity_command(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all():raise ValueError('Finite two-dimensional action required')
        return self._velocity_command(np.clip(action,-1.,1.))


class PortablePolicy:
    """Exported deterministic actor, without Stable-Baselines3 at runtime."""
    def __init__(self,path,observation_dimensions=235):
        self.observation_dimensions=observation_dimensions
        self.actor=torch.jit.load(str(path),map_location='cpu');self.actor.eval()
    def predict(self,observation):
        observation=np.asarray(observation,dtype=np.float32)
        if observation.shape!=(self.observation_dimensions,) or not np.isfinite(observation).all():
            raise ValueError(f'{self.observation_dimensions} finite observation features required')
        with torch.no_grad():action=self.actor(torch.from_numpy(observation[None])).numpy()[0]
        if action.shape!=(2,) or not np.isfinite(action).all():raise RuntimeError('Invalid actor output')
        return np.clip(action,-1.,1.)

