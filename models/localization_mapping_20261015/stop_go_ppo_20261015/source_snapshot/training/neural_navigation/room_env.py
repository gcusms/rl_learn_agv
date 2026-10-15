"""Room-navigation revision: learned control with explicit route heading and progress inputs."""
from collections import deque
import numpy as np
from gymnasium import spaces
from navrl.map import continuous_remaining
from neural_navigation.laser_fusion import LaserMapFusion
from navrl.geometry import wrap
from neural_navigation.env import LearnedNavigationEnv
from neural_navigation.core import RoomRoutingMixin,corridor_clearance,extend_observation






class RoomNavigationEnv(RoomRoutingMixin,LearnedNavigationEnv):
    """No motion overrides: all signed speed/turn commands still come from PPO."""
    def __init__(self,route_mode='raw',**kwargs):
        if route_mode not in ('raw','shortcut'):raise ValueError('Unknown route mode')
        self.route_mode=route_mode
        self.pose_progress=deque(maxlen=16);self.recent_progress=1.;self.dynamic_memory_seconds=8.
        super().__init__(**kwargs)
        self.observation_space=spaces.Box(-1.,1.,shape=(235,),dtype=np.float32)
        self.scene['action_limits']['vx_m_s']=[-.08,.2]
        self.laser_fusion=LaserMapFusion(self);self.shortcut_candidate=None;self.shortcut_since=None;self.route_choice='initial'
        self.replan_reason='initial';self.last_route_switch=-100.
        self.static_visibility_cache={}

    def reset(self,**kwargs):
        self.pose_progress.clear();self.recent_progress=1.
        if hasattr(self,'laser_fusion'):self.laser_fusion.reset()
        self.shortcut_candidate=None;self.shortcut_since=None;self.last_route_switch=-100.
        if hasattr(self,'static_visibility_cache'):self.static_visibility_cache.clear()
        return super().reset(**kwargs)








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
        info['route_mode']=self.route_mode
        fusion=getattr(self,'laser_fusion',None);predictor=getattr(fusion,'predictor',None)
        info.update(scan_motion_prediction_enabled=predictor is not None,
                    scan_moving_tracks=predictor.moving_tracks if predictor else 0,
                    scan_forecast_cells=fusion.forecast_cells if fusion else 0)
        return info
