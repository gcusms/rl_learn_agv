"""Bound exploration where the recovery actor has no trainable influence.

The unchanged frozen base's masked preference gets 98% of probability outside
recovery support. The other 2% is uniform over every legal action, including
stop. Inside support the learned recovery distribution is unchanged. Sampling,
PPO likelihoods and export use the same head; no action is replaced afterward.
This changes stochastic training semantics, not deterministic initial choices,
the masks, guard, physical model, rewards or available sensor information.
"""
import torch
import torch.nn.functional as F

from experiments.stop_recovery_policy import RecoveryHead, StopRecoveryPolicy, contract as parent_contract


SCHEMA = 'stop_feedback_recovery_steady_base_v1'
EXPLORATION_FRACTION = .02


def contract():
    return dict(parent_contract(), schema=SCHEMA,
        frozen_base_sampling='98_percent_masked_greedy_plus_2_percent_uniform_legal',
        outside_support_uniform_exploration=EXPLORATION_FRACTION,
        inside_support_distribution_unchanged=True,
        deterministic_initial_choices_preserved=True,
        stochastic_initial_parity_claimed=False,
        sampling_likelihood_and_export_share_head=True,
        no_post_sampling_action_replacement=True)


class SteadyBaseHead(RecoveryHead):
    def __init__(self, source):
        # Preserve the exact parameter objects already registered with Adam.
        torch.nn.Module.__init__(self)
        self.base = source.base
        self.base_width = source.base_width
        self.residual = source.residual

    def forward(self, features):
        baseline, supported = self.parts(features)
        residual = self.residual(features[:, self.base_width:-13])
        recovery = baseline+supported[:, None].to(features.dtype)*residual
        masks = features[:, -13:-1] > .5
        chosen = baseline.masked_fill(~masks, -1e8).argmax(-1)
        uniform = masks.to(features.dtype)/masks.sum(-1, keepdim=True)
        probabilities = (EXPLORATION_FRACTION*uniform
            +(1.-EXPLORATION_FRACTION)*F.one_hot(chosen, 12).to(features.dtype))
        stable = probabilities.clamp_min(torch.finfo(features.dtype).tiny).log().masked_fill(~masks, -1e8)
        return torch.where(supported[:, None], recovery, stable)


class SteadyBaseRecoveryPolicy(StopRecoveryPolicy):
    recovery_contract = staticmethod(contract)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = SteadyBaseHead(self.action_net)
