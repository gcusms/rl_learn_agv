"""Training-only cost for advancing while a trusted path point is behind.

Uses the pre-action observation and actual executed velocity. Never changes
an action, mask, sensor input or runtime command; stopped/turning actions have
zero added cost and do not receive a stationary reward bonus.
"""
import gymnasium as gym
import numpy as np


def alignment_context(observation):
    obs=np.asarray(observation)
    if obs.shape!=(300,) or not np.isfinite(obs).all():raise ValueError('Finite300-field observation required')
    return bool(obs[294]>.5 and obs[229]>=.5 and np.linalg.norm(obs[216:218])>.025 and obs[232]<-.2)


def forward_alignment_cost(observation,command,coefficient):
    if not np.isfinite(coefficient) or coefficient<0:raise ValueError('Nonnegative finite coefficient required')
    command=np.asarray(command)
    if command.shape!=(2,) or not np.isfinite(command).all():raise ValueError('Finite executed velocity required')
    if not alignment_context(observation):return 0.
    return float(coefficient*np.clip(command[0]/.2,0,1)*np.clip(-observation[232],0,1))


class AlignmentReward(gym.Wrapper):
    def __init__(self,env,coefficient=3.):
        super().__init__(env)
        if not np.isfinite(coefficient) or coefficient<0:raise ValueError('Nonnegative finite coefficient required')
        self.coefficient=float(coefficient);self.previous_observation=None
        self.reward_scale=float(env.reward_scale)

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);self.previous_observation=obs.copy();return obs,info

    def step(self,action):
        if self.previous_observation is None:raise ValueError('Reset before stepping')
        previous=self.previous_observation
        obs,reward,done,truncated,info=self.env.step(action)
        cost=forward_alignment_cost(previous,info['commanded_velocity'],self.coefficient)
        eligible=alignment_context(previous)
        self.previous_observation=obs.copy()
        info=dict(info,native_base_reward=info['native_reward'],native_reward=info['native_reward']-cost,
            alignment_context_eligible=eligible,alignment_forward_cost=cost,
            alignment_forward_motion=bool(eligible and info['commanded_velocity'][0]>0))
        return obs,float(reward-cost*self.reward_scale),done,truncated,info
