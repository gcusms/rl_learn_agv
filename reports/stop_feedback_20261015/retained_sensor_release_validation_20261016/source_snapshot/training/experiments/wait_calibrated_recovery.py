"""Calibrate exploration after an observed legal wait, inside recovery support.

The frozen base determines a bounded temperature. Its parameters and every
learned parameter are unchanged on migration. Dividing all legal logits by the
same positive value preserves greedy decisions but allows PPO to sample and
evaluate alternatives using one distribution. No action is replaced afterward.

This is an experiment, not proof that a currently legal maneuver can escape a
stall. Physical feasibility and normal-navigation regression remain separate.
"""
import torch
import torch.nn.functional as F

from experiments.proactive_recovery import (
    ProactiveRecoveryPolicy, contract as parent_contract, obstacle_context,
)
from experiments.steady_base_recovery import EXPLORATION_FRACTION
from experiments.stop_recovery_policy import masks_tensor, usable
from neural_navigation.decision_feedback import STAT_SLICE


SCHEMA = 'stop_feedback_legal_wait_calibration_v1'
MINIMUM_LEGAL_WAIT_S = 2.
MAXIMUM_TEMPERATURE = 8.
TARGET_BASE_LOGIT_GAP = 2.
LEGAL_WAIT_INDEX = 300 + STAT_SLICE.start + 3


def contract():
    return dict(parent_contract(), schema=SCHEMA,
        inside_support_distribution_unchanged=False,
        inside_existing_stop_support_distribution_unchanged=False,
        deterministic_initial_choices_preserved=True,
        stochastic_initial_parity_claimed=False,
        minimum_observed_legal_wait_s=MINIMUM_LEGAL_WAIT_S,
        maximum_temperature=MAXIMUM_TEMPERATURE,
        target_frozen_stop_minus_best_motion_logit_gap=TARGET_BASE_LOGIT_GAP,
        temperature_source='unchanged_frozen_base_logits_and_current_sensor_mask',
        calibration_gate='existing_proactive_support_and_observed_legal_wait',
        temperature_is_independent_of_trainable_residual=True,
        stop_remains_selectable=True,
        sampling_likelihood_and_export_share_head=True,
        no_post_sampling_action_replacement=True,
        current_legal_motion_is_not_a_usefulness_label=True)


class WaitCalibratedFeatures(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.base, self.residual = source.base, source.residual

    def forward(self, observations):
        return torch.cat([
            self.base(observations[:, :300]), self.residual(observations),
            masks_tensor(observations).to(observations.dtype),
            usable(observations)[:, None].to(observations.dtype),
            obstacle_context(observations)[:, None].to(observations.dtype),
            observations[:, LEGAL_WAIT_INDEX:LEGAL_WAIT_INDEX + 1] * 5.,
        ], dim=-1)


class WaitCalibratedHead(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        # Reuse Adam's existing parameter objects and state-dict names/shapes.
        self.base, self.residual = source.base, source.residual
        self.base_width = source.base_width

    def parts(self, features):
        baseline = self.base(features[:, :self.base_width])
        masks = features[:, -15:-3] > .5
        choice = baseline.masked_fill(~masks, -1e8).argmax(-1)
        active = (features[:, -3] > .5) & ((choice == 0) | (features[:, -2] > .5))
        return baseline, active

    def temperature(self, features, baseline, active):
        masks = features[:, -15:-3] > .5
        # Only the frozen base defines the scale. Scaling by a detached span
        # of trainable logits would give the wrong derivative during PPO.
        motion = baseline[:, 1:].masked_fill(~masks[:, 1:], -1e8).amax(-1)
        gap = (baseline[:, 0] - motion).clamp_min(0.)
        scale = (gap / TARGET_BASE_LOGIT_GAP).clamp(1., MAXIMUM_TEMPERATURE)
        waiting = features[:, -1] >= MINIMUM_LEGAL_WAIT_S
        return torch.where(active & waiting, scale, torch.ones_like(scale))

    def forward(self, features):
        baseline, active = self.parts(features)
        masks = features[:, -15:-3] > .5
        learned = baseline + active[:, None].to(features.dtype) * self.residual(
            features[:, self.base_width:-15])
        learned = learned / self.temperature(features, baseline, active)[:, None]
        choice = baseline.masked_fill(~masks, -1e8).argmax(-1)
        probabilities = (
            EXPLORATION_FRACTION * masks.to(features.dtype) / masks.sum(-1, keepdim=True)
            + (1. - EXPLORATION_FRACTION) * F.one_hot(choice, 12).to(features.dtype))
        frozen = probabilities.clamp_min(torch.finfo(features.dtype).tiny).log().masked_fill(~masks, -1e8)
        return torch.where(active[:, None], learned, frozen)


class WaitCalibratedRecoveryPolicy(ProactiveRecoveryPolicy):
    recovery_contract = staticmethod(contract)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.mlp_extractor.policy_net = WaitCalibratedFeatures(self.mlp_extractor.policy_net)
        self.action_net = WaitCalibratedHead(self.action_net)
        self.mlp_extractor.latent_dim_pi += 1
