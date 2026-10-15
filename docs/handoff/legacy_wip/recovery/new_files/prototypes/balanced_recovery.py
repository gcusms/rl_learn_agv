"""Training-only balanced recovery exposure and bounded voluntary-wait cost.

Uses historical training templates, not evaluation cases. Neither component
changes policy observations, action masks or executed commands.
"""
import gymnasium as gym
import numpy as np
from experiments.masked_navigation import MaskedNavigationCurriculum


class BalancedSchedule:
    def __init__(self, count, seed=0):
        if count < 1:
            raise ValueError('Nonempty training template pool required')
        self.count = count
        self.reset(seed)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.episode = 0
        self.order = []

    def next(self):
        focused = self.episode % 2 == 0
        self.episode += 1
        if not focused:
            return None
        if not self.order:
            self.order = self.rng.permutation(self.count).tolist()
        return self.order.pop()


class BalancedRecoveryCurriculum(MaskedNavigationCurriculum):
    def __init__(self, *args, training_templates, schedule_seed=0, **kwargs):
        self.training_templates = list(training_templates)
        self.schedule = BalancedSchedule(len(self.training_templates), schedule_seed)
        kwargs['failure_templates'] = ()
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.schedule.reset(seed)
        index = self.schedule.next() if not options else None
        options = dict(options or {})
        if index is not None:
            options['training_template'] = self.training_templates[index]
        episode_seed = int(self.schedule.rng.integers(1, 2**31-1))
        obs, info = super().reset(seed=episode_seed, options=options)
        self.targeted_episode = index is not None
        self.targeted_template = index
        return obs, dict(info, focused_training_replay=self.targeted_episode,
                         balanced_template_index=index)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, dict(info,
            focused_training_replay=self.targeted_episode,
            balanced_template_index=self.targeted_template)


class SupportedWaitCost:
    def __init__(self, maximum=.02, episode_budget=2., grace_s=2.):
        if not np.isfinite([maximum, episode_budget, grace_s]).all() or min(maximum, episode_budget, grace_s) < 0:
            raise ValueError('Finite nonnegative wait cost parameters required')
        self.maximum, self.budget, self.grace = maximum, episode_budget, grace_s
        self.reset()

    def reset(self):
        self.duration = 0.
        self.total = 0.

    def step(self, observation, action, info, dt):
        obs = np.asarray(observation)
        if obs.shape != (300,) or not np.isfinite(obs).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Finite 300D observation and positive period required')
        # Mask acceptance never certifies future safety. A modest bounded
        # objective only applies with supported pose, a permitted pure turn,
        # no observed/held/unknown/degraded motion track and measured margin.
        safe_turn = ((obs[238] > .5 and obs[286] < .2)
                     or (obs[239] > .5 and obs[287] < .2))
        eligible = (action == 0 and obs[294] > .5 and safe_turn
                    and (obs[295:299] < .5).all()
                    and info['measured_body_gap_m'] > .08
                    and not info['safety_termination']
                    and not info['verified_mission_success'])
        if not eligible:
            self.duration = 0.
            return 0.
        self.duration += dt
        fraction = float(np.clip((self.duration-self.grace)/2., 0., 1.))
        cost = min(self.maximum*fraction, max(0., self.budget-self.total))
        self.total += cost
        return cost


class WaitReward(gym.Wrapper):
    def __init__(self, env, maximum=.02):
        super().__init__(env)
        self.reward_scale = env.reward_scale
        self.cost = SupportedWaitCost(maximum=maximum)
        self.previous = None

    def reset(self, **kwargs):
        self.cost.reset()
        obs, info = self.env.reset(**kwargs)
        self.previous = obs.copy()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        cost = self.cost.step(self.previous, action, info, self.env.unwrapped.active.dt)
        self.previous = obs.copy()
        info = dict(info, native_reward_before_wait_cost=info['native_reward'],
                    native_reward=info['native_reward']-cost,
                    supported_wait_cost=cost, wait_cost_episode_total=self.cost.total)
        return obs, float(reward-cost*self.reward_scale), done, truncated, info

    def action_masks(self):
        return self.env.action_masks()
