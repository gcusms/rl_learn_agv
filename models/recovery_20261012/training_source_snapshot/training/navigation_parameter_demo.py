"""Explicitly unequal configuration demonstrations, not training-quality benchmarks."""
import argparse
from pathlib import Path
import numpy as np
from navigation_comparison import ComparisonEnv,evaluate
from navrl.geometry import ROOT

PRESETS={
 'ppo_10k':dict(checkpoint_steps=10000,forward_limit=.11,reverse_limit=.06,yaw_limit=.35,route_shortcut=False,forward_gain=1.,turn_alpha=1.,label='保守限速 / 原始神经路线'),
 'ppo_50k':dict(checkpoint_steps=50000,forward_limit=.15,reverse_limit=.07,yaw_limit=.45,route_shortcut=False,forward_gain=1.,turn_alpha=1.,label='中等限速 / 原始神经路线'),
 'ppo_100k':dict(checkpoint_steps=100000,forward_limit=.2,reverse_limit=.08,yaw_limit=.5,route_shortcut=False,forward_gain=1.1,turn_alpha=.9,lookahead_points=4,label='完整限速 / 原始神经路线 / 转向滤波'),
 'nav2_teb':dict(checkpoint_steps=None,forward_limit=.14,reverse_limit=.08,yaw_limit=.42,route_shortcut=False,forward_gain=1.,turn_alpha=1.,native_lookahead_m=2.5,label='普通导航 / 保守限速配置'),
}

class ParameterDemoEnv(ComparisonEnv):
    preset='ppo_100k'
    def __init__(self,**kwargs):
        self.demo_configuration=PRESETS[self.preset].copy();self.demo_turn=0.
        super().__init__(**kwargs)
        c=self.demo_configuration;self.scene['action_limits']['vx_m_s']=[-c['reverse_limit'],c['forward_limit']]
    def reset(self,**kwargs):
        self.demo_turn=0.;return super().reset(**kwargs)
    def _velocity_command(self,action):
        c=self.demo_configuration
        return np.array([action[0]*(c['forward_limit'] if action[0]>=0 else c['reverse_limit']),action[1]*c['yaw_limit']])
    def step(self,action):
        action=np.asarray(action,float).copy();c=self.demo_configuration
        if action[0]>0:action[0]=min(1.,action[0]*c['forward_gain'])
        self.demo_turn=c['turn_alpha']*action[1]+(1-c['turn_alpha'])*self.demo_turn;action[1]=self.demo_turn
        return super().step(action)
    def _observation(self,scan,initialize=False):
        obs=super()._observation(scan,initialize=initialize)
        lookahead=self.demo_configuration.get('lookahead_points',2)
        if lookahead==2:return obs
        nearest=int(np.argmin(np.linalg.norm(self.path-self.estimate[:2],axis=1)));target=min(nearest+lookahead,len(self.path)-1)
        while target>nearest and not self.static_map.visible(self.estimate[:2],self.path[target]):target-=1
        delta=self.path[target]-self.estimate[:2];yaw=self.estimate[2]
        local=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@delta
        obs[216:218]=np.clip(local/2,-1,1);heading=np.arctan2(local[1],local[0])
        obs[231:233]=[np.sin(heading),np.cos(heading)];self.last_policy_observation=obs
        return obs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preset',choices=PRESETS,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    ParameterDemoEnv.preset=a.preset;c=PRESETS[a.preset].copy();c['comparison_type']='unequal_parameter_demonstration';c['policy_weights_unchanged']=True
    native=a.preset=='nav2_teb';policy=None if native else ROOT/f"runs/remote_room_100k/policy_{c['checkpoint_steps']}.zip"
    evaluate(policy,a.output or ROOT/f'runs/navigation_parameter_demo/{a.preset}',max_leg_steps=1800,native_teb=native,env_class=ParameterDemoEnv,configuration=c)
