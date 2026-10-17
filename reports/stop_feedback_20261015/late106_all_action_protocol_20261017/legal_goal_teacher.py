"""Offline feasibility teacher over every currently legal original action.

It uses only sensor-derived subgoal, measured velocity and the current mask.
Choosing a legal command is not a future safety certificate. Physical trials
must still stop on contact, guard intervention or insufficient body clearance.
"""
import numpy as np

from navrl.geometry import wrap
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation


def select_current_legal(mask, target, velocity):
    mask = np.asarray(mask, bool)
    target, velocity = np.asarray(target, float), np.asarray(velocity, float)
    if mask.shape != (12,) or not mask[0] or target.shape != (2,) or velocity.shape != (2,):
        raise ValueError('Original twelve-action mask and finite planar sensor inputs required')
    if not np.isfinite(np.r_[target, velocity]).all():
        raise ValueError('Finite target and measured velocity required')
    choices = np.flatnonzero(mask[1:])+1
    if not len(choices):
        return 0
    scores = []
    for choice in choices:
        x = y = yaw = 0.
        measured = velocity.copy()
        for _ in range(12):
            measured += np.clip((COMMANDS[choice]-measured)*20.*.05, -np.array([.5, 1.])*.05,
                                np.array([.5, 1.])*.05)
            x += measured[0]*np.cos(yaw)*.05
            y += measured[0]*np.sin(yaw)*.05
            yaw += measured[1]*.05
        remaining = target-np.array([x, y])
        progress = (np.linalg.norm(target)-np.linalg.norm(remaining))/.6
        error = abs(wrap(np.arctan2(remaining[1], remaining[0])-yaw))
        scores.append(progress-.06*error)
    return int(choices[np.argmax(scores)])


class LegalGoalTeacher:
    def __init__(self, delegate, core, target_kind):
        if target_kind not in ('waypoint', 'goal'):
            raise ValueError('One of the predeclared sensor target modes required')
        self.delegate, self.core, self.target_kind = delegate, core, target_kind
        self.records = []

    def predict(self, observation):
        observation = np.asarray(observation, np.float32)
        mask = masks_from_observation(observation[:282])
        nominal = self.delegate.predict(observation)
        baseline = int(abs(ACTIONS-nominal).max(1).argmin())
        if not mask[baseline]:
            raise ValueError('Frozen model violated its original mask')
        goal = observation[218:220]*12.
        target = observation[216:218]*2. if self.target_kind == 'waypoint' else goal
        if np.linalg.norm(target) < .05:
            target = goal
        terminal = np.linalg.norm(goal) <= .3
        choice = baseline if terminal else select_current_legal(mask, target, observation[220:222]*[.2, .5])
        assert mask[choice]
        self.records.append(dict(sensor_time_s=float(self.core.elapsed), choice=choice,
            frozen_model_choice=baseline, changed_from_model=choice != baseline,
            legal_choices=np.flatnonzero(mask).tolist(), target_kind=self.target_kind,
            target_body_m=target.tolist(), terminal_parent=bool(terminal)))
        return ACTIONS[choice].copy()
