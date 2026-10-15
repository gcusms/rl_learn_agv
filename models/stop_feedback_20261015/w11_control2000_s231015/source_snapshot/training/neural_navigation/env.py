"""Learned robot localization/routing/control; no runtime robot AMCL or graph planner."""
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from gymnasium import spaces
from navrl.geometry import ROOT
from navrl.physical_env import PhysicalNavigationEnv
from navrl.multimap import MultiMapNavigationEnv
from navrl.map import continuous_remaining
from neural_navigation.models import LearnedLocalizer,NeuralRouter
from neural_navigation.core import LearnedRoutingMixin


class ResetMapView:
    """Legacy reset API needs a provisional path; this supplies no planned route.
    Every observation replaces the straight placeholder with neural routing outputs.
    """
    def __init__(self,m):self.base=m
    def __getattr__(self,key):return getattr(self.base,key)
    def plan(self,start,goal):return np.asarray([start,goal],float)


class LearnedNavigationEnv(LearnedRoutingMixin,PhysicalNavigationEnv):
    def __init__(self,components=None,**kwargs):
        if components is None:
            frozen=ROOT/'runs/full_model_navigation_v2'
            components=frozen if (frozen/'config.json').exists() else ROOT/'runs/neural_components_v2' if (ROOT/'runs/neural_components_v2/config.json').exists() else ROOT/'runs/neural_components'
        components=Path(components)
        kwargs.pop('localizer',None)
        super().__init__(localizer='learned_attention',localizer_factory=lambda m,l:LearnedLocalizer(m,l,components/'localizer.pt'),**kwargs)
        self.static_map=self.map;self.map=ResetMapView(self.static_map)
        d=torch.load(components/'router.pt',map_location='cpu',weights_only=True)
        self.motion_prediction_config=d.get('motion_prediction')
        self.escape_start=bool(d.get('escape_start',False))
        self.router=NeuralRouter(d['iterations']);self.router.load_state_dict(d['state_dict']);self.router.eval()
        m=self.static_map;h,w=m.free.shape
        self.free=m.free[2:h//4*4:4,2:w//4*4:4].astype(np.float32)
        self.grid_res=m.res*4;self.grid_h,self.grid_w=self.free.shape
        self.expiry=np.zeros_like(self.free);self.last_route_time=-100.;self.plan_revision=0;self.routing_ready=False
        self.observation_space=spaces.Box(-1.,1.,shape=(231,),dtype=np.float32)
        self.router_cache={};self.router_cache_hits=0;self.router_cache_misses=0

    def reset(self,**kwargs):
        self.expiry.fill(0);self.last_route_time=-100.;self.plan_revision=0;self.routing_ready=False
        self.router_cache.clear();self.router_cache_hits=0;self.router_cache_misses=0
        obs,info=super().reset(**kwargs)
        options=kwargs.get('options') or {}
        prior=options.get('initial_pose')
        if prior is None and 'initial_prior_offset' in options:
            task=self.scene['tasks'][options.get('task',0)]
            prior=np.asarray(options.get('start',task['start']),float)+np.asarray(options['initial_prior_offset'],float)
        if prior is not None:
            prior=np.asarray(prior,float)
            if prior.shape!=(3,) or not np.isfinite(prior).all():raise ValueError('Initial pose must contain finite x, y, yaw')
            self.localizer.reset(prior,self.np_random)
            self.estimate=self.localizer.update(self.odom,self.scan)
            self.routing_ready=False
            obs=self._observation(self.scan,initialize=True);self.previous_remaining=self.remaining
            info=self._info('running');info['provided_initial_pose']=prior.tolist()
        return obs,info

    def set_navigation_goal(self, goal):
        """Switch mission target while preserving physics and learned localization."""
        goal=np.asarray(goal,float)
        if goal.shape!=(2,) or not np.isfinite(goal).all():
            raise ValueError('Goal must contain finite x and y')
        if not self.static_map.lookup(goal,self.static_map.free,outside=False):
            raise ValueError('Goal must lie in navigable known-map space')
        self.goal=goal.copy();self.last_route_time=-100.;self.routing_ready=False
        obs=self._observation(self.scan,initialize=False);self.previous_remaining=self.remaining
        return obs






    def _info(self,status):
        info=super()._info(status)
        if getattr(self,'routing_ready',False):
            state='model_goal_reached' if status=='success' else 'timeout' if status=='timeout' else 'model_predicts_blockage' if self.current_probability<.5 else 'model_following_route'
            info['decision']=dict(state=state,reason='learned_map_scan_pose_and_neural_route_values',plan_revision=self.plan_revision,
                static_reachability_probability=self.static_probability,current_reachability_probability=self.current_probability,
                localization_std=self.localizer.std.tolist(),unmatched_laser_returns=self.unmatched_returns,
                planner_source='trained_neural_convolutional_router',action_source='PPO',no_robot_amcl_or_astar=True)
        info['supervisor_override']=False;info['map_id']=self.scene.get('map_id','original')
        info['proposed_policy_action']=self.last_action.tolist()
        if hasattr(self,'router_cache_hits'):
            info['router_cache_hits']=self.router_cache_hits;info['router_cache_misses']=self.router_cache_misses
        info['localization_scan_update_accepted']=getattr(self.localizer,'scan_update_accepted',None)
        info['localization_valid_scan_fraction']=getattr(self.localizer,'valid_scan_fraction',None)
        info['localization_missing_scan_updates']=getattr(self.localizer,'missing_scan_updates',None)
        return info


class LearnedMultiMapEnv(MultiMapNavigationEnv):
    def __init__(self,components=None,**kwargs):
        self.components=components
        super().__init__(**kwargs)
    def _get(self,index):
        if index not in self.pool:self.pool[index]=LearnedNavigationEnv(components=self.components,scene_dir=self.entries[index]['directory'],**self.kwargs)
        return self.pool[index]
