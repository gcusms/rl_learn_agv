"""Causal, explicitly versioned per-action risk history; no action selection."""
from collections import deque

import numpy as np

from neural_navigation.decision_feedback import contract as feedback_contract, WAIT_SCHEMA
from neural_navigation.long_motion_risk import batch_predicted_body_risk
from neural_navigation.masked_actions import COMMANDS


SCHEMA = 'long_action_risk_history_v1'
RUNTIME_DIM = 481
TRAIN_DIM = 478
HISTORY_DIM = 42
FRAME_DIM = 14
HISTORY_AGES = (.4, .2, 0.)


def contract():
    return dict(schema=SCHEMA, dimensions=RUNTIME_DIM, training_dimensions=TRAIN_DIM,
        parent_contract=feedback_contract(WAIT_SCHEMA), unchanged_runtime_prefix_dimensions=439,
        unchanged_training_prefix_dimensions=436, append_dimensions=HISTORY_DIM,
        history_ages_s=list(HISTORY_AGES), maximum_historical_sample_age_s=.05,
        fields_per_frame=['12_forecast_contact_scores', 'supported_prediction_valid',
            'supported_tracks_over_4'], horizon_s=5., forecast_period_s=.2,
        contact_score='0 if no predicted contact, else 1-contact_s/5.2; consult validity',
        unsupported_or_missing_history='zero_values_and_zero_validity_not_certified_clear',
        margin_m=.04, velocity_uncertainty_floor_m_s=.025,
        adaptive_track_uncertainty=True, finite_measured_motion_response=True,
        response_gain=20., response_acceleration=[.5, 1.], action_commands=COMMANDS.tolist(),
        frame='guard tracks and robot pose in continuous odometry coordinates',
        simulator_truth_used=False, future_sensor_frames_used=False,
        absolute_episode_time_input=False, commands_issued=False,
        actions_masks_body_guard_and_blind_memory_unchanged=True)


class LongRiskHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = deque()
        self.last_observation = None

    def augment(self, observation, core, guard):
        observation = np.asarray(observation, np.float32)
        if observation.shape != (439,) or not np.isfinite(observation).all():
            raise ValueError('Unchanged finite 439-dimensional feedback prefix required')
        timestamp = float(core.elapsed)
        if not np.isfinite(timestamp) or (self.frames and timestamp <= self.frames[-1][0]):
            raise ValueError('Strictly increasing sensor time or explicit reset required')
        predictor = guard.predictive_constraint
        if predictor.timestamp != timestamp:
            raise ValueError('Forecast and policy must use the same sensor frame')
        frame = np.zeros(FRAME_DIM, np.float32)
        if observation[435] > .5 and observation[438] > .5:
            forecasts = batch_predicted_body_risk(predictor.tracks, np.asarray(core.odom),
                timestamp, COMMANDS, guard.half * 2, horizon_s=5., period_s=.2,
                margin_m=.04, uncertainty_m_s=.025,
                measured_velocity=np.asarray(core.measured_velocity),
                adaptive_track_uncertainty=True)
            supported = forecasts[0]['supported_tracks']
            if supported:
                for index, forecast in enumerate(forecasts):
                    contact = forecast['predicted_margin_contact_s']
                    if contact is not None:
                        frame[index] = np.clip(1. - contact / 5.2, 0., 1.)
                frame[12] = 1.
                frame[13] = min(supported / 4., 1.)
        self.frames.append((timestamp, frame))
        while self.frames and self.frames[0][0] < timestamp - .451:
            self.frames.popleft()
        history = []
        for age in HISTORY_AGES:
            target = timestamp - age
            eligible = [item for item in self.frames if item[0] <= target + 1e-8]
            sample = eligible[-1] if eligible else None
            if sample is None or target - sample[0] > .05 + 1e-8:
                history.append(np.zeros(FRAME_DIM, np.float32))
            else:
                history.append(sample[1].copy())
        self.last_observation = np.r_[observation, np.concatenate(history)].astype(np.float32)
        assert self.last_observation.shape == (RUNTIME_DIM,)
        return self.last_observation.copy()


def to_training(observations):
    values = np.asarray(observations, np.float32)
    if values.ndim != 2 or values.shape[1] != RUNTIME_DIM or not np.isfinite(values).all():
        raise ValueError('Finite runtime observation matrix required')
    return np.concatenate([values[:, :300], values[:, 303:]], axis=1)
