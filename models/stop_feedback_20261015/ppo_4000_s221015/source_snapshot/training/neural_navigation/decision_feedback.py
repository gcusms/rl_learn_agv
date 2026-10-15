"""Causal action explanations and the guard's own forecast for a new policy.

The old 300/303 input semantics remain intact. No command is chosen here;
masking/stop permissions and dynamics are unchanged. An accepted probe means
currently admissible, not guaranteed future safety or a useful maneuver.
"""
import numpy as np
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.safety_context import REASONS


GROUPS = ('accepted', 'perception_or_odometry', 'localization_or_route',
          'static_geometry', 'laser_geometry', 'dynamic_prediction', 'projection', 'other')
FEEDBACK_REASONS = REASONS + ('predicted_dynamic_contact', 'predictive_command_contract',
                             'policy_wait', 'only_stop_available')
GROUP_DIM = len(COMMANDS) * len(GROUPS)
REASON_SLICE = slice(GROUP_DIM, GROUP_DIM + len(FEEDBACK_REASONS))
RISK_SLICE = slice(REASON_SLICE.stop, REASON_SLICE.stop + len(COMMANDS))
STAT_SLICE = slice(RISK_SLICE.stop, RISK_SLICE.stop + 8)
DIM = STAT_SLICE.stop
TRAIN_DIM = 300 + DIM
RUNTIME_DIM = 303 + DIM


def contract():
    return dict(schema='stop_decision_feedback_v1', prefix_dimensions=300,
        training_dimensions=TRAIN_DIM, heading_prefix_dimensions=303, dimensions=RUNTIME_DIM,
        append_dimensions=DIM, action_commands=COMMANDS.tolist(),
        fields=['12_current_guard_rejection_groups_one_hot', 'previous_executed_stop_reason_one_hot',
                '12_guard_forecast_contact_scores', 'nonzero_choices_over_11', 'scan_age_over_guard_limit',
                'previous_idle_seconds_over_5', 'previous_idle_with_choices_seconds_over_5',
                'prediction_input_valid', 'moving_tracks_over_4', 'oldest_track_age_over_0.8', 'scan_valid'],
        rejection_groups=list(GROUPS), previous_reasons=list(FEEDBACK_REASONS),
        forecast_source='same_odometry_frame_predictive_constraint_as_action_mask',
        contact_score='0 if no forecast contact, otherwise 1-contact_s/(horizon_s+period_s)',
        unknown_forecast_is_not_clear=True, legacy_prefix_unchanged=True,
        stop_always_selectable=True, simulator_truth_used=False, commands_issued=False)


def reason_group(reason):
    if reason is None or reason in ('none', 'estimated_goal_reached'):
        return 0
    if reason in ('nonfinite_input', 'stale_or_future_scan', 'scan_unavailable', 'odometry_jump'):
        return 1
    if reason in ('localization_uncertain', 'learned_route_incomplete'):
        return 2
    if reason == 'static_braking_footprint':
        return 3
    if reason in ('laser_braking_footprint', 'escape_not_increasing_clearance'):
        return 4
    if reason in ('predicted_dynamic_contact', 'predictive_command_contract'):
        return 5
    if reason in ('reverse_without_rear_visibility', 'reverse_duration_limit',
                  'unsafe_turn_blocked', 'unsafe_translation_blocked'):
        return 6
    return 7


class DecisionFeedback:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_reason = 'none'
        self.idle_since = None
        self.idle_with_choices_since = None
        self.recorded_timestamp = None
        self.frame_timestamp = None
        self.frame_mask = None
        self.last_features = None

    def features(self, core, guard, context):
        frame = context.guard_probe_frame
        now = float(core.elapsed)
        if frame is None or frame['timestamp'] != now:
            raise ValueError('Decision feedback requires probes for this exact sensor frame')
        if self.recorded_timestamp is not None and now < self.recorded_timestamp:
            raise ValueError('Reset decision feedback before restarting sensor time')
        if not np.array_equal(frame['commands'], COMMANDS):
            raise ValueError('Guard probe action order changed')
        accepted = np.all(np.isclose(frame['executed'], COMMANDS, atol=1e-9, rtol=0), axis=1)
        # Stopping remains a legal choice even if perception itself is unavailable.
        self.frame_mask = accepted.copy(); self.frame_mask[0] = True
        self.frame_timestamp = now
        groups = np.zeros((len(COMMANDS), len(GROUPS)), dtype=np.float32)
        for index, (ok, reason) in enumerate(zip(accepted, frame['reasons'])):
            group = reason_group(reason)
            groups[index, 0 if ok else (group or len(GROUPS)-1)] = 1.
        previous = np.zeros(len(FEEDBACK_REASONS), dtype=np.float32)
        previous[FEEDBACK_REASONS.index(self.previous_reason)] = 1.
        risk = np.zeros(len(COMMANDS), dtype=np.float32)
        valid_prediction = False; moving = 0; oldest = 0.
        predictor = getattr(guard, 'predictive_constraint', None)
        raw_age = frame['received_timestamp'] - now
        if predictor is not None:
            if predictor.timestamp != now:
                raise ValueError('Guard forecast and policy feedback are out of sync')
            finite = np.isfinite(predictor.contacts)
            denominator = predictor.config['horizon_s'] + predictor.config['period_s']
            risk[finite] = np.clip(1. - predictor.contacts[finite] / denominator, 0., 1.)
            odom = np.asarray(core.odom)
            scan = np.asarray(core.scan)
            has_scan = np.any(np.isfinite(scan) & (scan >= core.lidar.range_min)
                              & (scan < core.lidar.range_max))
            valid_prediction = bool(odom.shape == (3,) and np.isfinite(odom).all()
                and not getattr(core, 'odometry_sanity_failure', False) and frame['scan_valid']
                and has_scan and -.05 <= raw_age <= guard.max_scan_age)
            moving = predictor.supported_tracks
            for track in predictor.tracks:
                oldest = max(oldest, now - float(track['time']))
        held = 0. if self.idle_since is None else max(0., now - self.idle_since)
        voluntary = 0. if self.idle_with_choices_since is None else max(0., now - self.idle_with_choices_since)
        age = max(0., frame['received_timestamp'] - now)
        stats = np.array([accepted[1:].sum() / 11., min(age / guard.max_scan_age, 1.),
            min(held / 5., 1.), min(voluntary / 5., 1.), float(valid_prediction),
            min(moving / 4., 1.), min(oldest / .8, 1.), float(frame['scan_valid'])], np.float32)
        self.last_features = np.r_[groups.ravel(), previous, risk, stats].astype(np.float32)
        if self.last_features.shape != (DIM,) or not np.isfinite(self.last_features).all():
            raise ValueError('Invalid decision feedback')
        return self.last_features.copy()

    def record(self, result):
        now = float(result['timestamp'])
        if self.frame_timestamp != now or self.frame_mask is None:
            raise ValueError('Record execution against its original probe frame')
        if self.recorded_timestamp is not None and now <= self.recorded_timestamp:
            raise ValueError('An executed action cannot be recorded twice')
        command = np.asarray(result['command'], float)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError('Finite executed command required')
        idle = bool(np.max(np.abs(command)) < 1e-8)
        reason = result.get('stop_reason') or 'none'
        if reason == 'none' and idle:
            reason = 'policy_wait' if self.frame_mask[1:].any() else 'only_stop_available'
        self.previous_reason = reason if reason in FEEDBACK_REASONS else 'other'
        if idle and reason != 'estimated_goal_reached':
            if self.idle_since is None:
                self.idle_since = now
        else:
            self.idle_since = None
        if idle and reason == 'policy_wait':
            if self.idle_with_choices_since is None:
                self.idle_with_choices_since = now
        else:
            self.idle_with_choices_since = None
        self.recorded_timestamp = now

    def augment(self, prefix, core, guard, context):
        prefix = np.asarray(prefix, np.float32)
        if prefix.shape not in ((300,), (303,)) or not np.isfinite(prefix).all():
            raise ValueError('Unchanged quality or directional prefix required')
        return np.r_[prefix, self.features(core, guard, context)].astype(np.float32)
