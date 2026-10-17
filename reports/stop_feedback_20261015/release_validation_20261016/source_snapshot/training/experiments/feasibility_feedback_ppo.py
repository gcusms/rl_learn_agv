"""Train the continuous actor to use sensor-derived rejection feedback.

An auxiliary loss attracts rejected stationary proposals toward the nearest
currently accepted probe. It never executes a probe or changes the guard.
Probe acceptance is instantaneous geometry, not future collision assurance.
"""
import torch
from experiments.prior_anchored_ppo import PriorAnchoredPPO
from neural_navigation.safety_context import (
    BASE_DIM, OBSERVATION_DIM, REASON_START, REASONS, RECOVERY_PROBES)


FEEDBACK_REASONS = ('reverse_without_rear_visibility', 'reverse_duration_limit',
    'static_braking_footprint', 'laser_braking_footprint',
    'unsafe_translation_blocked', 'unsafe_turn_blocked',
    'escape_not_increasing_clearance')


def feasibility_loss(observations, mean):
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM or mean.shape != (len(observations), 2):
        raise ValueError('V3 navigation observations and two continuous means required')
    probes = torch.as_tensor(RECOVERY_PROBES, device=mean.device, dtype=mean.dtype)
    probes = probes / probes.new_tensor([.2, .5])
    accepted = observations[:, BASE_DIM:BASE_DIM+9] > .5
    reasons = [BASE_DIM+REASON_START+REASONS.index(x) for x in FEEDBACK_REASONS]
    # Only a measured, overridden, stationary command is eligible. Missing
    # scans, uncertain localization, incomplete routes, no-alternative states
    # and voluntary waiting retain the existing PPO objective.
    eligible = (accepted.any(dim=1)
        & (observations[:, reasons] > .5).any(dim=1)
        & (observations[:, BASE_DIM+14] > .5)
        & (observations[:, BASE_DIM+12].abs() < .05)
        & (observations[:, BASE_DIM+13].abs() < .1)
        & (observations[:, -1] >= .2))  # one second of stationary rejection
    # Use the deployed clipped proposal. The straight-through derivative
    # avoids zero gradients when the raw Gaussian mean exceeds action bounds.
    proposal = mean + (mean.clamp(-1, 1)-mean).detach()
    distances = ((proposal[:, None, :]-probes[None, :, :])**2).mean(dim=2)
    # Replace ineligible rows before min, so all-rejected states cannot create
    # infinity*zero or NaN in the optimizer.
    candidates = accepted | ~eligible[:, None]
    distances = distances.masked_fill(~candidates, torch.inf)
    nearest = distances.min(dim=1).values
    loss = (nearest*eligible.to(mean.dtype)).mean()
    return loss, eligible.float().mean()


class FeasibilityFeedbackPPO(PriorAnchoredPPO):
    """Reuse the audited PPO update with a different auxiliary objective."""
    def reference_loss(self, observations):
        # Parent's update invokes this hook; no reference actor is consumed.
        mean = self.policy.get_distribution(observations).distribution.mean
        return feasibility_loss(observations, mean)

    def train(self):
        super().train()
        self.last_feedback_statistics = dict(
            feasibility_loss=self.last_anchor_statistics.get('reference_mean_loss', 0.),
            eligible_state_fraction=self.last_anchor_statistics.get('eligible_state_fraction', 0.),
            updates=self._n_updates)
        self.logger.record('train/feasibility_loss', self.last_feedback_statistics['feasibility_loss'])
        self.logger.record('train/feasibility_state_fraction', self.last_feedback_statistics['eligible_state_fraction'])
