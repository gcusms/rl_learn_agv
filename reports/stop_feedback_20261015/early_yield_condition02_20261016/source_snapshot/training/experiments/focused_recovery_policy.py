"""Recovery residual support for trusted reorientation or observed stopping.

Does not choose a maneuver. The frozen learned actor retains other navigation
states, including ordinary moving-obstacle interaction. Recovery logits can
still choose wait, any valid turn, or any valid translation under native masks.
"""
import copy
import torch
from experiments.gated_recovery_policy import GatedRecoveryPolicy,GatedRecoveryPPO,RecoveryFeatures,RecoveryHead,warm_start_gated


def recovery_support(observations):
    alternatives=(observations[:,235:244]>.5).any(-1)|(observations[:,267:269]>.5).any(-1)
    behind=((observations[:,294]>.5)&(observations[:,229]>=.5)&(observations[:,232]<-.2)
        &(torch.linalg.vector_norm(observations[:,216:218],dim=-1)>.025))
    rejected=(observations[:,250]<.5)&(observations[:,264]<.5)
    waiting=(observations[:,266]>=.04)&(observations[:,264]<.5)
    # Low-quality pose does not freeze a bad wait when the native guard still
    # offers a maneuver. The actor sees the unsupported prediction flag and
    # may still choose zero; masks never certify future dynamic safety.
    return alternatives&(behind|rejected|waiting)


class FocusedRecoveryFeatures(RecoveryFeatures):
    def __init__(self,source):
        torch.nn.Module.__init__(self)
        self.base=source.base;self.residual=source.residual

    def forward(self,observations):
        return torch.cat([self.base(observations),self.residual(observations),
            recovery_support(observations)[:,None].to(observations.dtype)],dim=-1)


class FocusedRecoveryHead(RecoveryHead):
    def __init__(self,source):
        torch.nn.Module.__init__(self)
        self.base=source.base;self.base_width=source.base_width;self.residual=source.residual

    def forward(self,features):
        return self.base(features[:,:self.base_width])+features[:,-1:]*self.residual(features[:,self.base_width:-1])


class FocusedRecoveryPolicy(GatedRecoveryPolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        # Reuse the exact parameter objects already registered with Adam.
        self.mlp_extractor.policy_net=FocusedRecoveryFeatures(self.mlp_extractor.policy_net)
        self.action_net=FocusedRecoveryHead(self.action_net)


def warm_start_focused(checkpoint,env,seed=92611):
    model=warm_start_gated(checkpoint,env,seed)
    previous=model.policy;rng=torch.get_rng_state()
    try:
        policy=FocusedRecoveryPolicy(model.observation_space,model.action_space,model.lr_schedule,
            **copy.deepcopy(model.policy_kwargs)).to(model.device)
        policy.load_state_dict(previous.state_dict())
        policy.optimizer.load_state_dict(copy.deepcopy(previous.optimizer.state_dict()))
    finally:torch.set_rng_state(rng)
    model.policy=policy;model.policy_class=FocusedRecoveryPolicy
    # Compatible PPO subclass changes diagnostics only, retaining all already
    # initialized rollout/optimizer state and the seed's sampling RNG.
    model.__class__=FocusedRecoveryPPO
    model.gated_migration['residual_support']='trusted behind waypoint or observed previous stop/wait with a legal alternative; low-pose stop contexts remain trainable'
    return model


class FocusedRecoveryPPO(GatedRecoveryPPO):
    def snapshot_reference(self):
        super().snapshot_reference()
        self.last_reference_statistics['protection']='frozen_base_zero_residual_outside_recovery_support'

    def train(self):
        super().train()
        self.last_reference_statistics['protection']='frozen_base_zero_residual_outside_recovery_support'
