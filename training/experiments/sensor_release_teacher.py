"""Offline sensor-only earlier-release teacher; never a deployed controller."""
import numpy as np

from neural_navigation.masked_actions import COMMANDS, masks_from_observation
from neural_navigation.motion_response import force_response_trajectory


def contract():
    return dict(schema='offline_sensor_checked_wait_release_v1', training_only=True,
        observation_dimensions=625, minimum_prior_legal_wait_s=2.,
        forecast_history='current supported frame plus at least one supported historical frame',
        candidate_contact_scores_must_all_be_zero=True,
        unsupported_forecasts_are_not_clear=True,
        minimum_forward_finite_hit_fraction=.25, maximum_forward_unknown_fraction=.15,
        forward_sector_definition='sensor angles plus mount yaw within 35 degrees of forward',
        waypoint_normalization_m=2., goal_normalization_m=12.,
        rollout_s=1., finite_response_gain=20., finite_response_acceleration=[.5,1.],
        only_override_nominal_stop_inside_trainable_support=True,
        armed_release_can_bridge_temporary_unknown_without_issuing_motion=True,
        no_absolute_episode_time_or_simulator_truth=True,
        original_masks_actions_body_and_guard_unchanged=True,
        positive_labels_require_separate_complete_safe_physical_episode=True)


class SensorReleaseTeacher:
    """Choose a measured-progress alternative to an admitted voluntary wait.

    ``armed`` is an offline demonstration state. Unknown input still returns
    the unchanged model proposal; it never authorizes a motion command.
    """
    def __init__(self, forward_sectors):
        self.forward_sectors = np.asarray(forward_sectors, bool)
        if self.forward_sectors.shape != (72,) or not self.forward_sectors.any():
            raise ValueError('At least one declared forward LiDAR sector required')
        self.armed = False

    def select(self, observation, nominal, trainable):
        x = np.asarray(observation, np.float32)
        if x.shape != (625,) or not np.isfinite(x).all():
            raise ValueError('Finite 625-dimensional sensor observation required')
        mask = masks_from_observation(x[:282])
        if type(nominal) is not int or not 0 <= nominal < 12 or not mask[nominal]:
            raise ValueError('Nominal action must be in the recorded current mask')
        info = dict(nominal=nominal, choice=nominal, teacher_controlled=False,
            trainable=bool(trainable), armed_before=self.armed, reason='unchanged_model')

        def result(reason):
            info.update(reason=reason, armed_after=self.armed)
            return info

        if nominal:
            self.armed = False
            return result('model_motion')
        if not trainable or not mask[1:].any() or np.linalg.norm(x[218:220]*12) <= .3:
            self.armed = False
            return result('outside_trainable_motion_support')
        if not self.armed and x[434]*5 < 2.-1e-6:
            return result('wait_not_sustained')
        if x[435] <= .5 or x[438] <= .5:
            return result('invalid_current_prediction_or_scan')
        history = x[439:481].reshape(3, 14)
        valid = history[:, 12] > .5
        if not valid[-1] or valid.sum() < 2:
            return result('insufficient_supported_history')
        # A stop predicted to be hit also needs a different training problem;
        # this teacher only checks the specific earlier-release hypothesis.
        if np.any(history[valid, 0] > 0.):
            self.armed = False
            return result('stopping_itself_has_forecast_contact')
        hit, no_return = x[481:].reshape(2, 72)
        unknown = 1.-hit-no_return
        if (min(hit.min(), no_return.min(), unknown.min()) < -2e-6
                or max(hit.max(), no_return.max(), unknown.max()) > 1.+2e-6):
            raise ValueError('Scan-quality fractions do not match their contract')
        forward_hit = float(hit[self.forward_sectors].mean())
        forward_unknown = float(unknown[self.forward_sectors].mean())
        info.update(forward_finite_hit_fraction=forward_hit, forward_unknown_fraction=forward_unknown)
        if forward_hit < .25 or forward_unknown > .15:
            return result('insufficient_measured_forward_scan')
        waypoint = x[216:218].astype(float)*2.
        initial_distance = float(np.linalg.norm(waypoint))
        if initial_distance < .05:
            return result('no_meaningful_waypoint')
        initial_bearing = float(np.arctan2(waypoint[1], waypoint[0]))
        velocity = x[220:222].astype(float)*[.2,.5]
        candidates = []
        for choice in np.flatnonzero(mask[1:])+1:
            if np.any(history[valid, choice] > 0.):
                continue
            pose = force_response_trajectory(velocity, COMMANDS[choice], 1., 20., [.5,1.])[-1]
            delta = waypoint-pose[:2]
            bearing = float(np.arctan2(np.sin(np.arctan2(delta[1],delta[0])-pose[2]),
                np.cos(np.arctan2(delta[1],delta[0])-pose[2])))
            progress = initial_distance-float(np.linalg.norm(delta))
            score = progress+.03*(abs(initial_bearing)-abs(bearing))-.005*abs(COMMANDS[choice,1])
            candidates.append((float(score), int(choice), float(progress)))
        info['admitted_motion_choices'] = [choice for _, choice, _ in candidates]
        if not candidates:
            self.armed = False
            return result('all_motion_candidates_have_forecast_contact')
        score, choice, progress = max(candidates)
        if score <= .005:
            return result('no_measured_waypoint_improvement')
        self.armed = True
        info.update(choice=choice, teacher_controlled=True, selected_score=score,
            predicted_waypoint_progress_m=progress)
        return result('supported_sensor_checked_earlier_release')
