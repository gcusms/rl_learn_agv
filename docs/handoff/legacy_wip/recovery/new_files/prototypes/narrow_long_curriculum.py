"""Training-only short-to-full straight-lane curriculum with replay."""
import numpy as np
from prototypes.narrow_training_env import NarrowTrainingEnv


def sample_task(rng,steps):
    # Keep full entrance exposure from the first batch; never train on fixed
    # evaluation starts/seeds. Short exit tasks provide reachable rewards.
    progress=min(1.,steps/20000.)
    choice=rng.random()
    if choice<.2:
        return dict(stage='open_room',start=[float(rng.uniform(.7,1.1)),float(rng.uniform(1.7,2.3)),float(rng.uniform(-.15,.15))],goal=[1.65,2.])
    if choice<.2+.45*(1-progress):
        return dict(stage='short_exit',start=[float(rng.uniform(3.55,3.85)),float(rng.uniform(1.995,2.005)),float(rng.uniform(-.025,.025))],goal=[4.55,2.])
    return dict(stage='full_passage',start=[float(rng.uniform(1.15,1.65)),float(rng.uniform(1.98,2.02)),float(rng.uniform(-.08,.08))],goal=[5.,2.])

class LongNarrowCurriculum(NarrowTrainingEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.training_steps=0
    def reset(self,*,seed=None,options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        self.task=sample_task(self.rng,self.training_steps)
        merged=dict(start=self.task['start'],goal=self.task['goal']);merged.update(options or {})
        observation,info=super().reset(options=merged)
        if not options and self.task['stage']=='short_exit':
            # Some noisy inside-lane resets already violate scan guard
            # clearance. Rehearse a supported state rather than training only
            # the unavoidable zero action. Explicit eval resets stay intact.
            for _ in range(3):
                if self.action_masks()[1:].any():break
                observation,info=super().reset(options=merged)
            if not self.action_masks()[1:].any():
                self.task=dict(stage='exit_room_fallback',start=[4.3,2.,0.],goal=[4.9,2.])
                observation,info=super().reset(options=dict(start=self.task['start'],goal=self.task['goal']))
        self.previous_alignment=self.alignment_potential(observation)
        return observation,dict(info,training_stage=self.task['stage'])
    @staticmethod
    def alignment_potential(observation):
        # Local direction only; no simulator trajectory or obstacle truth.
        target=observation[216:218]
        if np.linalg.norm(target)<.025:return 0.
        return -float(abs(np.arctan2(target[1],target[0])))/np.pi
    def step(self,action):
        observation,reward,done,truncated,info=super().step(action)
        new=self.alignment_potential(observation)
        # Potential-based difference uses the PPO discount and zero terminal
        # potential. It does not prescribe a turn or penalize required waiting.
        alignment_shaping=.3*((0. if done or truncated else .99*new)-self.previous_alignment)
        self.previous_alignment=new;self.training_steps+=1
        info.update(training_stage=self.task['stage'],alignment_shaping=alignment_shaping)
        return observation,float(reward+alignment_shaping*self.reward_scale),done,truncated,info
