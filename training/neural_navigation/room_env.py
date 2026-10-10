"""Room-navigation revision: learned control with explicit route heading and progress inputs."""
from collections import deque
import numpy as np
from gymnasium import spaces
from navrl.map import continuous_remaining
from neural_navigation.laser_fusion import LaserMapFusion
from navrl.geometry import wrap
from neural_navigation.env import LearnedNavigationEnv


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


class RoomNavigationEnv(LearnedNavigationEnv):
    """No motion overrides: all signed speed/turn commands still come from PPO."""
    def __init__(self,**kwargs):
        self.pose_progress=deque(maxlen=16);self.recent_progress=1.;self.dynamic_memory_seconds=8.
        super().__init__(**kwargs)
        self.observation_space=spaces.Box(-1.,1.,shape=(235,),dtype=np.float32)
        self.scene['action_limits']['vx_m_s']=[-.08,.2]
        self.laser_fusion=LaserMapFusion(self);self.shortcut_candidate=None;self.shortcut_since=None;self.route_choice='initial'
        self.replan_reason='initial';self.last_route_switch=-100.

    def reset(self,**kwargs):
        self.pose_progress.clear();self.recent_progress=1.
        if hasattr(self,'laser_fusion'):self.laser_fusion.reset()
        self.shortcut_candidate=None;self.shortcut_since=None;self.last_route_switch=-100.
        return super().reset(**kwargs)

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
        samples=[]
        for a,b in zip(path[:-1],path[1:]):
            if not self.static_map.visible(a,b):return False
            samples.extend(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.08)+1)))
        if not samples:return False
        points=np.array(samples);r,c=self.static_map.cells(points);r=r//4;c=c//4
        if np.any((r<0)|(r>=self.grid_h)|(c<0)|(c>=self.grid_w)):return False
        return not np.any(self.laser_fusion.hard[r,c])

    def _neural_route(self):
        old=self.predicted_path.copy() if self.routing_ready else None;revision=self.plan_revision
        super()._neural_route();candidate=self.predicted_path.copy();self.route_retained=False
        self.replan_reason='new_goal_or_initial_route'
        if old is None or len(old)<2 or np.linalg.norm(old[-1]-self.goal)>.12:return
        nearest=int(np.argmin(np.linalg.norm(old-self.estimate[:2],axis=1)))
        tail=np.vstack([self.estimate[:2],old[nearest:]])
        if np.linalg.norm(old[nearest]-self.estimate[:2])>.35 or not self._route_usable(tail):
            self.shortcut_candidate=None;self.shortcut_since=None;self.replan_reason='confirmed_obstacle_or_route_invalid';self.last_route_switch=self.elapsed
            return
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
        while target_index>nearest and not self.static_map.visible(self.estimate[:2],self.path[target_index]):
            target_index-=1
        delta=self.path[target_index]-self.estimate[:2];yaw=self.estimate[2]
        base[216:218]=np.clip(np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@delta/2,-1,1)
        clearance=corridor_clearance(scan,self.lidar.angles,self.lidar.mount[0])
        self.last_policy_observation=extend_observation(base,self.recent_progress,clearance)
        return self.last_policy_observation

    def step(self,action):
        old_cos=float(self.last_policy_observation[232]);old_action=self.last_action.copy()
        obs,reward,done,truncated,info=super().step(action)
        reward+=.4*float(obs[232]-old_cos)
        if obs[232]>.65 and obs[234]>.30 and self.current_probability>.5 and abs(info['actual_velocity'][0])<.01:
            reward-=.03*(1-self.recent_progress)
        if old_action[1]*action[1]<-.16:reward-=.02
        if action[0]<0:reward-=.005*abs(action[0])
        return obs,float(reward),done,truncated,info

    def _info(self,status):
        info=super()._info(status)
        info.update(policy_revision='room_v3',observation_dimensions=235,recent_progress=self.recent_progress,
                    commanded_velocity=self._velocity_command(self.last_action),reverse_speed_limit_m_s=.08,neural_route_retained=getattr(self,'route_retained',False),planning_fusion='known_map_plus_temporal_partial_laser',route_switch_reason=getattr(self,'replan_reason','initial'),far_unconfirmed_cells=self.laser_fusion.far_pending if hasattr(self,'laser_fusion') else 0,cleared_dynamic_cells=self.laser_fusion.cleared if hasattr(self,'laser_fusion') else 0)
        return info
