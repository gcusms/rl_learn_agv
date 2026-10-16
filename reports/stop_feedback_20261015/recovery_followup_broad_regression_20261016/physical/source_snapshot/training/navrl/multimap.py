"""Native physics, randomized navigation tasks, disjoint scene splits and replanning."""
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from .geometry import ROOT
from .physical_env import PhysicalNavigationEnv
from .decision import NavigationDecision


class DecisionNavigationEnv(PhysicalNavigationEnv):
    def reset(self,**kwargs):
        options=dict(kwargs.get('options') or {});task=self.scene['tasks'][options.get('task',0)]
        start=np.array(options.get('start',task['start']),float);goal=np.array(options.get('goal',task['goal']),float)
        reachable=True
        try:self.map.plan(start[:2],goal)
        except ValueError:reachable=False;options['goal']=start[:2].tolist()
        options.setdefault('task',0);kwargs['options']=options
        self.decision_ready=False;self.supervisor_override=False
        self.routing_map=self.map
        obs,info=super().reset(**kwargs)
        self.goal=goal
        self.decision=NavigationDecision(self.map);self.decision.static_reachable=reachable;self.decision_ready=True
        if not reachable:self.path=np.array([self.estimate[:2]]);self.decision.state='static_unreachable';self.decision.reason='no_collision_free_route_in_known_map'
        obs=self._observation(self.scan,initialize=True);self.previous_remaining=self.remaining
        return obs,self._info('running')

    def _observation(self,scan,initialize=False):
        if getattr(self,'decision_ready',False) and self.decision.static_reachable:
            self.decision.update(self);self.routing_map=self.decision.local_map
        return super()._observation(scan,initialize=initialize)

    def step(self,action):
        proposed=np.array(action,copy=True)
        if not self.decision.static_reachable:
            self.supervisor_override=True
            self.last_action=np.zeros(2);info=self._info('unreachable');info['proposed_policy_action']=proposed.tolist()
            return self._observation(self.scan),0.,True,False,info
        # Supervisory waits are explicitly labelled and never counted as learned stops.
        self.supervisor_override=self.decision.state in ['waiting_for_clearance','localization_uncertain']
        if self.supervisor_override:action=np.zeros(2)
        result=super().step(action);obs,reward,terminated,truncated,info=result
        info['proposed_policy_action']=proposed.tolist()
        if terminated:self.decision.state='goal_reached';self.decision.reason='localized_goal_tolerance'
        elif truncated:self.decision.state='timeout';self.decision.reason='episode_time_budget_exhausted'
        info['decision']=self.decision.evidence(self)
        return obs,reward,terminated,truncated,info

    def _info(self,status):
        info=super()._info(status)
        if getattr(self,'decision_ready',False):info['decision']=self.decision.evidence(self)
        info['supervisor_override']=getattr(self,'supervisor_override',False)
        info['map_id']=self.scene.get('map_id','original')
        return info


class MultiMapNavigationEnv(gym.Env):
    metadata=PhysicalNavigationEnv.metadata
    def __init__(self,dataset=None,split='train',**kwargs):
        self.dataset=Path(dataset or ROOT/'assets/navigation_dataset').resolve();self.split=split;self.kwargs=kwargs
        manifest=json.loads((self.dataset/'manifest.json').read_text());self.entries=manifest['splits'][split]
        self.pool={};self.active=self._get(0)
        self.action_space=self.active.action_space;self.observation_space=self.active.observation_space
        self.task_history=[]

    def _get(self,index):
        if index not in self.pool:
            # Only one live ROS AMCL backend when changing maps.
            if self.kwargs.get('localizer')=='ros-amcl':
                for env in self.pool.values():env.close()
                self.pool.clear()
            self.pool[index]=DecisionNavigationEnv(scene_dir=self.entries[index]['directory'],**self.kwargs)
        return self.pool[index]

    def __getattr__(self,name):
        if name=='active':raise AttributeError(name)
        return getattr(self.active,name)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed);options=dict(options or {})
        index=int(options.pop('map_index',self.np_random.integers(len(self.entries))));self.active=self._get(index)
        if 'start' not in options:
            cells=np.argwhere(self.active.map.clearance>.60)
            for attempt in range(500):
                a,b=[self.active.map.xy(cells[int(self.np_random.integers(len(cells)))]) for _ in range(2)]
                if np.linalg.norm(a-b)<2.0:continue
                try:path=self.active.map.plan(a,b)
                except ValueError:continue
                if np.linalg.norm(np.diff(path,axis=0),axis=1).sum()>9.:continue
                options.update(start=[*a,float(self.np_random.uniform(-np.pi,np.pi))],goal=b.tolist());break
            else:raise RuntimeError('Could not sample a reachable random navigation task')
        options.pop('task',None)
        self.task_history.append(dict(map_id=self.entries[index]['id'],start=options['start'],goal=options['goal'],seed=seed))
        return self.active.reset(seed=seed,options=options)

    def step(self,action):return self.active.step(action)
    def render(self):return self.active.render()
    def close(self):
        for env in self.pool.values():env.close()
        self.pool.clear()
