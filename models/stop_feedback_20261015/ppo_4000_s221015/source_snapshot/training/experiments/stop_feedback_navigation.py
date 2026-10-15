"""New observation wrapper; original physics, rewards, masks and actions."""
import gymnasium as gym
import numpy as np
from neural_navigation.decision_feedback import DecisionFeedback, TRAIN_DIM
from neural_navigation.masked_actions import masks_from_observation
from experiments.quality_masked_navigation import QualityMaskedWrapper


class FeedbackInterface(gym.Env):
    """Model migration/export spec; deliberately cannot simulate transitions."""
    reward_scale=1/60
    observation_space=gym.spaces.Box(-1,1,(TRAIN_DIM,),np.float32)
    action_space=gym.spaces.Discrete(12)
    def reset(self,*,seed=None,options=None):raise RuntimeError('Attach the physical feedback curriculum')
    def step(self,action):raise RuntimeError('Attach the physical feedback curriculum')


class StopFeedbackWrapper(QualityMaskedWrapper):
    def __init__(self, env, reward_scale=1.):
        super().__init__(env, reward_scale)
        env.capture_decision_feedback = True
        self.feedback = DecisionFeedback()
        self.observation_space = gym.spaces.Box(-1, 1, (TRAIN_DIM,), np.float32)

    def _augment(self, prefix):
        observation = self.feedback.augment(prefix, self.env.core, self.env.guard, self.env.context)
        if not np.array_equal(self.feedback.frame_mask, masks_from_observation(prefix[:282])):
            raise ValueError('Current guard explanation disagrees with sampled action masks')
        self.previous_augmented = observation.copy()
        return observation

    def reset(self, **kwargs):
        self.feedback.reset()
        observation, info = super().reset(**kwargs)
        return self._augment(observation), info

    def step(self, action):
        timestamp = float(self.env.core.elapsed)
        alternatives = int(self.feedback.frame_mask[1:].sum())
        observation, reward, terminated, truncated, info = super().step(action)
        self.feedback.record(dict(timestamp=timestamp, command=info['commanded_velocity'],
                                  stop_reason=info['execution_guard_reason']))
        info.update(decision_feedback_previous_reason=self.feedback.previous_reason,
                    decision_feedback_current_alternatives=alternatives,
                    decision_feedback_trainable_choice=bool(alternatives),
                    decision_feedback_reward_changed=False)
        return self._augment(observation), reward, terminated, truncated, info
