"""Do not activate recovery merely because a zero-risk prediction was valid.

Near geometry, restricted actions, actual forecast risk and executed waiting
remain recovery contexts. A valid zero-contact history alone is not a reason
to replace the frozen normal-navigation actor. No permission is changed.
"""
import torch

from experiments.guard_reason_release_actor import GuardReasonReleaseActor


def contract():
    return dict(schema='risk_relevant_release_support_v1', observation_dimensions=625,
        history_trigger='supported_nonzero_contact_score_in_last_three_frames',
        zero_contact_prediction_validity_alone_does_not_activate=True,
        other_triggers=['previous_executed_wait', 'nearest_scan_below0.8m',
            'restricted_motion_mask', 'current_guard_contact_forecast'],
        valid_prediction_and_scan_motion_legal_goal_far_required=True,
        parent_exact_in_clear_zero_risk_unblocked_nonwaiting_states=True,
        near_invalid_stop_only_terminal_rules_and_permissions_unchanged=True,
        no_runtime_teacher_or_forced_motion=True)


class RiskRelevantReleaseActor(GuardReasonReleaseActor):
    def support(self, observations):
        masks = self.masks(observations)
        history = observations[:, 439:481].reshape(-1, 3, 14)
        supported_risk = ((history[:, :, 12] > .5)
            & (history[:, :, :12].amax(2) > 0.)).any(1)
        waiting = (observations[:, 433:435] > 0.).any(1)
        near = observations[:, 144:216].amin(1) * 12. < .8
        restricted = ~masks[:, 1:].all(1)
        forecast = observations[:, 419:431].amax(1) > 0.
        valid = ((observations[:, 435] > .5) & (observations[:, 438] > .5)
            & masks[:, 1:].any(1)
            & (torch.linalg.vector_norm(observations[:, 218:220] * 12., dim=1) > .3))
        return valid & (supported_risk | waiting | near | restricted | forecast)
