"""Alternate recorded training difficulties and general tasks, never force actions."""
from experiments.masked_navigation import FocusedMaskedCurriculum
from experiments.fixed_training_replay import reset_training_episode
from experiments.train_masked_navigation import Progress
import json
import numpy as np
import gymnasium as gym


class ScheduledRecoveryCurriculum(FocusedMaskedCurriculum):
    def __init__(self,*args,recorded_cases=(),**kwargs):
        if not recorded_cases:raise ValueError('Recorded training cases required')
        self.recovery_reset_index=0;self.current_recovery_episode=False
        kwargs['replay_probability']=0.
        super().__init__(*args,recorded_cases=recorded_cases,**kwargs)

    def reset(self,seed=None,options=None):
        if seed is not None:
            self.recovery_reset_index=0;self.replay_rng=np.random.default_rng(seed)
            gym.Env.reset(self,seed=seed)
        focused=not options and self.recovery_reset_index%2==0
        if focused:
            case=self.recorded_cases[(self.recovery_reset_index//2)%len(self.recorded_cases)]
            # Replaying an exact simulator seed must not rewind the curriculum
            # generator and repeat the same general task after every replay.
            curriculum_rng=self.np_random
            obs,info=reset_training_episode(self,case)
            self.np_random=curriculum_rng
        else:obs,info=super().reset(seed=seed,options=options)
        self.recovery_reset_index+=1;self.current_recovery_episode=focused
        return obs,dict(info,focused_training_replay=focused)

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        return obs,reward,done,truncated,dict(info,focused_training_replay=self.current_recovery_episode)


class RecoveryCoverageProgress(Progress):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.coverage={label:dict(frames=0,episodes=0,arrivals=0,collisions=0,timeouts=0,
            voluntary_wait_frames=0) for label in ('recorded_training_recovery','general_training')}

    def _on_step(self):
        for info in self.locals['infos']:
            label='recorded_training_recovery' if info['focused_training_replay'] else 'general_training'
            counts=self.coverage[label];counts['frames']+=1
            counts['voluntary_wait_frames']+=int(info['masked_chose_wait_with_nonzero_available'])
            if 'episode' in info:
                counts['episodes']+=1;counts['arrivals']+=int(info['verified_mission_success'])
                counts['collisions']+=int(info['collision_events'])
                counts['timeouts']+=int(info.get('TimeLimit.truncated',False))
        result=super()._on_step()
        if self.num_timesteps%250==0 or self.num_timesteps>=self.steps:
            (self.output/'recovery_coverage.json').write_text(json.dumps(self.coverage,indent=2))
        return result
