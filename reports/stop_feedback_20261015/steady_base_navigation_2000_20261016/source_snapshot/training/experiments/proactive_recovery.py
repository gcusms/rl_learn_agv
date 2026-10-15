"""Let the learned residual act before a near-obstacle approach becomes a stop.

This version expands learnable support, not motion permission. All commands
still come from the masked learned policy, including stop. Clear navigation,
invalid inputs and terminal docking retain the frozen learned base. No episode
clock, simulator pose, human trajectory or timed maneuver enters this policy.
"""
import torch
import torch.nn.functional as F

from experiments.steady_base_recovery import EXPLORATION_FRACTION, contract as parent_contract
from experiments.stop_recovery_policy import StopRecoveryPolicy, masks_tensor, usable
from neural_navigation.decision_feedback import RISK_SLICE, STAT_SLICE


SCHEMA = 'stop_feedback_proactive_recovery_v1'


def contract():
    return dict(parent_contract(), schema=SCHEMA,
        support='valid_inputs_goal_far_and_base_stop_or_near_lidar_or_restricted_motion_or_guard_forecast',
        residual_can_act_before_base_stops=True,
        latest_lidar_history_slice=[144,216],lidar_normalization_m=12.,
        learning_support_nearest_lidar_range_m=.8,
        context_is_learning_support_not_motion_permission=True)


def obstacle_context(observations):
    masks = masks_tensor(observations)
    near = observations[:,144:216].amin(-1)*12. < .8
    constrained = ~masks[:,1:].all(-1)
    risk = observations[:,300+RISK_SLICE.start:300+RISK_SLICE.stop].amax(-1) > 0.
    tracks = observations[:,300+STAT_SLICE.start+5] > 0.
    return near | constrained | risk | tracks


class ProactiveFeatures(torch.nn.Module):
    def __init__(self,source):
        super().__init__()
        self.base,self.residual = source.base,source.residual

    def forward(self,observations):
        return torch.cat([self.base(observations[:,:300]),self.residual(observations),
            masks_tensor(observations).to(observations.dtype),
            usable(observations)[:,None].to(observations.dtype),
            obstacle_context(observations)[:,None].to(observations.dtype)],dim=-1)


class ProactiveHead(torch.nn.Module):
    def __init__(self,source):
        super().__init__()
        self.base,self.residual = source.base,source.residual
        self.base_width = source.base_width

    def parts(self,features):
        baseline = self.base(features[:,:self.base_width])
        masks = features[:,-14:-2] > .5
        choice = baseline.masked_fill(~masks,-1e8).argmax(-1)
        active = (features[:,-2]>.5) & ((choice==0) | (features[:,-1]>.5))
        return baseline,active

    def forward(self,features):
        baseline,active = self.parts(features)
        learned = baseline+active[:,None].to(features.dtype)*self.residual(features[:,self.base_width:-14])
        masks = features[:,-14:-2] > .5
        choice = baseline.masked_fill(~masks,-1e8).argmax(-1)
        probabilities = (EXPLORATION_FRACTION*masks.to(features.dtype)/masks.sum(-1,keepdim=True)
            +(1.-EXPLORATION_FRACTION)*F.one_hot(choice,12).to(features.dtype))
        frozen = probabilities.clamp_min(torch.finfo(features.dtype).tiny).log().masked_fill(~masks,-1e8)
        return torch.where(active[:,None],learned,frozen)


class ProactiveRecoveryPolicy(StopRecoveryPolicy):
    recovery_contract = staticmethod(contract)

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.mlp_extractor.policy_net = ProactiveFeatures(self.mlp_extractor.policy_net)
        self.action_net = ProactiveHead(self.action_net)
        self.mlp_extractor.latent_dim_pi += 1
