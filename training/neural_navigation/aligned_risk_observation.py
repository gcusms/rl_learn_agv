"""Opt-in v2 history preserving same-frame short-guard contact evidence.

This changes policy feature semantics, never commands or their admissibility.
Old v1 bundles retain their original runtime. A v2 actor must be trained and
exported with the corresponding release contract before closed-loop use.
"""
from collections import deque

import numpy as np

from neural_navigation.long_risk_observation import (
    LongRiskHistory, contract as original_contract,
)


SCHEMA = 'long_action_risk_history_v2_guard_lower_bound'


def contract():
    return dict(original_contract(), schema=SCHEMA,
        contact_score='earliest existing long forecast or same-frame guard TTC, normalized as1-contact_s/5.2',
        short_guard_contact_is_risk_lower_bound=True, short_guard_horizon_s=1.2,
        short_and_long_supported_track_counts_combined=True,
        supported_prediction_valid='some sensor-supported forecast exists; zero risk does not certify clearance',
        requires_retraining_after_v1=True, changes_action_permission=False,
        additional_neural_inference=False)


def fuse_current_frame(frame, contacts, supported_tracks, timestamp, guard_timestamp,
                       input_valid=True, scan_valid=True):
    frame = np.asarray(frame, np.float32)
    contacts = np.asarray(contacts, float)
    if frame.shape != (14,) or contacts.shape != (12,):
        raise ValueError('One14D history frame and12 guard TTCs required')
    if not np.isfinite(frame).all() or np.any((frame < 0) | (frame > 1)):
        raise ValueError('Normalized finite history features required')
    if not np.isfinite([timestamp, guard_timestamp]).all() or abs(timestamp-guard_timestamp) > 1e-8:
        raise ValueError('Guard forecasts must refer to the current sensor timestamp')
    if not isinstance(supported_tracks, (int, np.integer)) or supported_tracks < 0:
        raise ValueError('Nonnegative integer supported-track count required')
    finite = np.isfinite(contacts)
    if np.any(np.isneginf(contacts)) or np.any(contacts[finite] < 0) or np.any(contacts[finite] > 1.2+1e-8):
        raise ValueError('Guard TTC is missing or lies within its1.2second horizon')
    if not input_valid or not scan_valid:
        if frame.any():
            raise ValueError('An invalid input must not contain a valid long prediction')
        return frame.copy()
    if finite.any() and not supported_tracks:
        raise ValueError('Contact evidence requires at least one supported track')
    result = frame.copy()
    if finite.any():
        lower_bound = np.zeros(12, np.float32)
        lower_bound[finite] = np.clip(1.-contacts[finite]/5.2, 0., 1.)
        result[:12] = np.maximum(result[:12], lower_bound)
        result[12] = 1.
        result[13] = max(result[13], min(supported_tracks/4., 1.))
    return result


class AlignedRiskHistory:
    def __init__(self):
        self.original = LongRiskHistory()
        self.reset()

    def reset(self):
        self.original.reset()
        self.frames = deque()
        self.last_observation = None

    def augment(self, observation, core, guard):
        old = self.original.augment(observation, core, guard)
        timestamp = float(core.elapsed)
        predictor = guard.predictive_constraint
        valid = bool(old[435] > .5 and old[438] > .5)
        frame = fuse_current_frame(old[-14:], predictor.contacts, predictor.supported_tracks,
            timestamp, predictor.timestamp, input_valid=valid, scan_valid=valid)
        self.frames.append((timestamp, frame))
        while self.frames and self.frames[0][0] < timestamp-.451:
            self.frames.popleft()
        history = []
        for age in [.4, .2, 0.]:
            target = timestamp-age
            eligible = [value for value in self.frames if value[0] <= target+1e-8]
            sample = eligible[-1] if eligible else None
            history.append(np.zeros(14, np.float32) if sample is None or target-sample[0] > .05+1e-8
                           else sample[1].copy())
        self.last_observation = np.r_[old[:439], np.concatenate(history)].astype(np.float32)
        return self.last_observation.copy()
