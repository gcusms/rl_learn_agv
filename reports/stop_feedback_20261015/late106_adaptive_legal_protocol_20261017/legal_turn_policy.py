"""Explicit offline teacher; it never overrides the original current mask."""
import numpy as np

from navrl.geometry import wrap
from neural_navigation.masked_actions import ACTIONS, masks_from_observation


class LegalClockwiseManeuver:
    def __init__(self, delegate, core, target):
        if target not in (4.5, 6., 'path_heading'):
            raise ValueError('One of the three predeclared teacher targets required')
        self.delegate, self.core, self.target = delegate, core, target
        self.previous_yaw = float(core.odom[2])
        self.clockwise_turn = 0.
        self.phase = 'maneuver'
        self.records = []

    def predict(self, observation):
        yaw = float(self.core.odom[2])
        self.clockwise_turn -= float(wrap(yaw-self.previous_yaw))
        self.previous_yaw = yaw
        mask = masks_from_observation(np.asarray(observation)[:282])
        nominal = self.delegate.predict(observation)
        baseline = int(np.abs(ACTIONS-nominal).max(1).argmin())
        assert mask[baseline], 'Frozen delegate must respect the same current mask'
        direction = np.asarray(self.core.goal[:2])-np.asarray(self.core.estimate[:2])
        heading_error = float(wrap(np.arctan2(direction[1], direction[0])-self.core.estimate[2]))
        finished = (self.clockwise_turn >= self.target if isinstance(self.target, float)
                    else self.clockwise_turn >= .4 and abs(heading_error) < .22 and mask[1:4].any())
        if finished:
            self.phase = 'learned'
        if self.phase == 'learned':
            choice = baseline
        elif self.clockwise_turn > 2*np.pi+.3:
            choice = 0
        else:
            choice = 5 if mask[5] else 9 if mask[9] else 0
        assert mask[choice]
        self.records.append(dict(sensor_time_s=float(self.core.elapsed), phase=self.phase,
            measured_clockwise_turn_rad=self.clockwise_turn, measured_goal_heading_error_rad=heading_error,
            choice=choice, frozen_model_choice=baseline, changed_from_model=choice != baseline,
            legal_choices=np.flatnonzero(mask).tolist()))
        return ACTIONS[choice].copy()
