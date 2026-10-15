"""Train earlier sensor-supported decisions while preserving clear navigation.

The residual can act before a stop if a supported moving scan track is present
or native safety probes have already narrowed the moving choices. No truth
labels, new actions, weaker mask, or steering controller enter the policy.
"""
import torch
from experiments.focused_recovery_policy import recovery_support,FocusedRecoveryFeatures
from experiments.soft_prior_recovery_policy import SoftPriorRecoveryPolicy,warm_start_soft_prior
from experiments.conditional_recovery_ppo import ConditionalRecoveryPPO


def anticipatory_support(observations):
    choices=torch.cat([observations[:,235:244],observations[:,267:269]],dim=-1)>.5
    alternatives=choices.any(-1)
    moving=observations[:,281]>.5
    narrowed=~choices.all(-1)
    return alternatives&(recovery_support(observations)|moving|narrowed)


class AnticipatoryFeatures(FocusedRecoveryFeatures):
    def forward(self,observations):
        return torch.cat([self.base(observations),self.residual(observations),
            anticipatory_support(observations)[:,None].to(observations.dtype)],dim=-1)


class AnticipatoryRecoveryPolicy(SoftPriorRecoveryPolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        # Reuse the optimizer's exact parameter objects; only the fixed sensor
        # support predicate changes, identically for sampling and evaluation.
        self.mlp_extractor.policy_net=AnticipatoryFeatures(self.mlp_extractor.policy_net)


class AnticipatoryRecoveryPPO(ConditionalRecoveryPPO):
    actor_support=staticmethod(anticipatory_support)


def warm_start_anticipatory(checkpoint,env,seed):
    model=warm_start_soft_prior(checkpoint,env,seed)
    model.policy.mlp_extractor.policy_net=AnticipatoryFeatures(model.policy.mlp_extractor.policy_net)
    model.policy.__class__=AnticipatoryRecoveryPolicy
    model.policy_class=AnticipatoryRecoveryPolicy
    model.__class__=AnticipatoryRecoveryPPO
    model.gated_migration.update(residual_support='legal alternative and previous recovery, supported moving scan track, or narrowed native choices',
        sensor_contract_unchanged=True)
    return model
