"""Retain general tasks while rehearsing randomized historical training failures."""
import gymnasium as gym
import gc
from neural_navigation.guarded_curriculum import GuardedNavigationCurriculum


class ContextNavigationCurriculum(GuardedNavigationCurriculum):
    def __init__(self, *args, failure_templates=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_templates = list(failure_templates)

    def reset(self, seed=None, options=None):
        # Release heavyweight MuJoCo models at episode boundaries instead of
        # waiting for Python's infrequent generational cyclic collection.
        if self.active is not None:
            self.active.close()
            self.active=None
            if hasattr(self,'core'):self.core=None
            gc.collect()
        gym.Env.reset(self, seed=seed)
        options = dict(options or {})
        if self.failure_templates and 'training_template' not in options and self.np_random.random() < .4:
            options['training_template'] = self.failure_templates[int(self.np_random.integers(len(self.failure_templates)))]
        return super().reset(seed=None, options=options)
