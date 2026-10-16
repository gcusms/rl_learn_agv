"""Training-only terminal cost for a persistent static guard-origin deadlock.

No action is selected or rewritten here. Sensor outages, dynamic-clearance
waiting and uncertain/recovering localization must not trigger termination.
"""
import numpy as np
from experiments.recovery_curriculum import RecoveryNavigationCurriculum
from neural_navigation.safety_context import BASE_DIM, OBSERVATION_DIM


class StaticDeadlockDetector:
    def __init__(self, patience_s=3., maximum_pose_change_m=.03):
        if not np.isfinite([patience_s, maximum_pose_change_m]).all() or min(patience_s, maximum_pose_change_m) <= 0:
            raise ValueError('Positive finite static-deadlock bounds required')
        self.patience_s = float(patience_s)
        self.maximum_pose_change_m = float(maximum_pose_change_m)
        self.reset()

    def reset(self):
        self.duration = 0.
        self.anchor = None

    def step(self, observation, pose, origin_free, reason, diagnostics, dt):
        observation = np.asarray(observation, float)
        pose = np.asarray(pose, float)
        if observation.shape != (OBSERVATION_DIM,) or pose.shape != (3,) or not np.isfinite(np.r_[observation, pose, dt]).all() or dt <= 0:
            raise ValueError('Finite context, pose and positive period required')
        healthy = (diagnostics.get('localization_scan_update_accepted') is True
                   and diagnostics.get('localization_missing_scan_updates') == 0
                   and diagnostics.get('localization_map_agreement', 0.) >= .8)
        eligible = (reason == 'static_braking_footprint' and not origin_free
                    and not np.any(observation[BASE_DIM:BASE_DIM+9] > .5) and healthy)
        if not eligible:
            self.reset()
            return False
        if self.anchor is None or np.linalg.norm(pose[:2]-self.anchor) > self.maximum_pose_change_m:
            self.anchor = pose[:2].copy()
            self.duration = 0.
        self.duration += float(dt)
        return self.duration + 1e-9 >= self.patience_s


class StaticDeadlockCurriculum(RecoveryNavigationCurriculum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_deadlock = StaticDeadlockDetector()

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.static_deadlock.reset()
        return observation, info

    def step(self, action):
        before = self.previous_observation.copy()
        pose = self.core.estimate.copy()
        origin_free = bool(self.core.static_map.lookup(pose[:2], self.core.static_map.free, outside=False))
        observation, reward, terminated, truncated, info = super().step(action)
        deadlock = False
        if not terminated and not truncated:
            deadlock = self.static_deadlock.step(before, pose, origin_free,
                info['execution_guard_reason'], info['sensor_core_diagnostics'], self.active.dt)
        else:
            self.static_deadlock.reset()
        info['training_static_deadlock_termination'] = bool(deadlock)
        info['static_deadlock_seconds'] = self.static_deadlock.duration
        if deadlock:
            reward -= 25.
            terminated = True
            info['verified_mission_success'] = False
            info['mission_targets_reached'] = 0
        return observation, reward, terminated, truncated, info
