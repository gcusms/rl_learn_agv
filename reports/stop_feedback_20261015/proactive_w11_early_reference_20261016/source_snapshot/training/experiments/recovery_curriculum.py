"""Training-only cost for repeating rejected translation, never an action rule.

Voluntary waiting, sensor-loss stopping and model-selected valid turns retain
their original rewards. Probe acceptance describes current guard geometry,
not future dynamic safety, and never grants permission to issue a command.
"""
import numpy as np
from neural_navigation.context_curriculum import ContextNavigationCurriculum
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM

REJECTED_TRANSLATION = frozenset((
    'reverse_without_rear_visibility', 'reverse_duration_limit',
    'static_braking_footprint', 'laser_braking_footprint',
    'unsafe_translation_blocked', 'learned_route_incomplete'))


class RejectedTranslationCost:
    def __init__(self, grace_s=1., maximum=.2):
        if not np.isfinite([grace_s, maximum]).all() or grace_s < 0 or maximum < 0:
            raise ValueError('Finite nonnegative recovery cost parameters required')
        self.grace_s, self.maximum = float(grace_s), float(maximum)
        self.reset()

    def reset(self):
        self.duration = 0.
        self.previous = None

    def step(self, before, info, dt):
        before = np.asarray(before)
        if before.shape != (OBSERVATION_DIM,) or not np.isfinite(before).all():
            raise ValueError('Current finite safety-context observation required')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite control period required')
        proposed = np.asarray(info['proposed_action'])
        command = np.asarray(info['commanded_velocity'])
        if proposed.shape != (2,) or command.shape != (2,) or not np.isfinite(np.r_[proposed, command]).all():
            raise ValueError('Finite proposed and executed actions required')
        reason = info['execution_guard_reason']
        # A fully accepted turn counts as an alternative; do not force a
        # particular turn or penalize a voluntary zero command.
        alternatives = bool(np.any(before[BASE_DIM:BASE_DIM+9] > .5))
        rejected = (bool(info['training_actions_overridden']) and alternatives and
                    reason in REJECTED_TRANSLATION and abs(proposed[0]) > .01 and
                    abs(command[0]) < .01 and
                    abs(info['fixed_geodesic_progress_m']) < .01)
        if not rejected:
            self.reset()
            return 0.
        signature = (reason, int(np.sign(proposed[0])))
        self.duration = self.duration + dt if signature == self.previous else dt
        self.previous = signature
        # Bounded, gradual floor prevents tiny rejected requests from being
        # almost free under the original squared projection cost.
        return self.maximum * float(np.clip((self.duration-self.grace_s)/2., 0., 1.))


class RecoveryNavigationCurriculum(ContextNavigationCurriculum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.use_safety_context:
            raise ValueError('Recovery curriculum requires observable guard feedback')
        self.rejection_cost = RejectedTranslationCost()

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.rejection_cost.reset()
        self.previous_observation = observation.copy()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        cost = self.rejection_cost.step(self.previous_observation[:OBSERVATION_DIM], info, self.active.dt)
        self.previous_observation = observation.copy()
        info['repeated_rejection_penalty'] = cost
        info['repeated_rejection_seconds'] = self.rejection_cost.duration
        return observation, reward-cost, terminated, truncated, info
