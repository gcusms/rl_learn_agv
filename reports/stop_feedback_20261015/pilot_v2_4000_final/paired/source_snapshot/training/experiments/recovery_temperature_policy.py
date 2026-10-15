"""Masked PPO exploration temperature for states with few accepted choices.

The same distribution supplies sampling, log probabilities and entropy. It
does not sample outside the mask or change deterministic command ordering.
"""
import copy
import numpy as np
import torch
from sb3_contrib.common.maskable.distributions import MaskableCategorical,MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from experiments.greedy_stable_quality_ppo import GreedyStableQualityPPO


class RecoveryCategorical(MaskableCategorical):
    def __init__(self,logits,temperature=3.,maximum_choices=3):
        if not np.isfinite(temperature) or temperature<1 or maximum_choices<1:
            raise ValueError('Temperature >=1 and positive choice count required')
        self.recovery_temperature=float(temperature);self.maximum_choices=int(maximum_choices)
        self.unscaled_logits=None
        super().__init__(logits=logits)

    def apply_masking(self,masks):
        if self.unscaled_logits is None:self.unscaled_logits=self._original_logits
        if masks is None:
            self._original_logits=self.unscaled_logits
        else:
            allowed=torch.as_tensor(masks,dtype=torch.bool,device=self.unscaled_logits.device).reshape(self.unscaled_logits.shape)
            count=allowed.sum(dim=-1,keepdim=True)
            if (count<1).any():raise ValueError('At least one legal action required')
            temperature=torch.where(count<=self.maximum_choices,self.recovery_temperature,1.)
            self._original_logits=self.unscaled_logits/temperature
        super().apply_masking(masks)


class RecoveryDistribution(MaskableCategoricalDistribution):
    def __init__(self,action_dim,temperature=3.,maximum_choices=3):
        super().__init__(action_dim);self.temperature=temperature;self.maximum_choices=maximum_choices

    def proba_distribution(self,action_logits):
        self.distribution=RecoveryCategorical(action_logits.view(-1,self.action_dim),self.temperature,self.maximum_choices)
        return self

    def mode(self):
        logits=self.distribution.unscaled_logits
        if self.distribution.masks is not None:logits=logits.masked_fill(~self.distribution.masks,-torch.inf)
        return logits.argmax(dim=-1)


class RecoveryTemperaturePolicy(MaskableActorCriticPolicy):
    def __init__(self,*args,recovery_temperature=3.,recovery_maximum_choices=3,**kwargs):
        super().__init__(*args,**kwargs)
        if self.action_space.n!=12:raise ValueError('Twelve native velocity choices required')
        self.recovery_temperature=float(recovery_temperature);self.recovery_maximum_choices=int(recovery_maximum_choices)
        self.action_dist=RecoveryDistribution(12,self.recovery_temperature,self.recovery_maximum_choices)

    def _get_constructor_parameters(self):
        parameters=super()._get_constructor_parameters()
        parameters.update(recovery_temperature=self.recovery_temperature,recovery_maximum_choices=self.recovery_maximum_choices)
        return parameters


def migrate_policy(model):
    if isinstance(model.policy,RecoveryTemperaturePolicy):return
    old=model.policy;kwargs=copy.deepcopy(model.policy_kwargs)
    kwargs.update(recovery_temperature=3.,recovery_maximum_choices=3)
    rng=torch.get_rng_state()
    try:
        policy=RecoveryTemperaturePolicy(model.observation_space,model.action_space,model.lr_schedule,**kwargs).to(model.device)
        policy.load_state_dict(old.state_dict())
        policy.optimizer.load_state_dict(copy.deepcopy(old.optimizer.state_dict()))
    finally:torch.set_rng_state(rng)
    model.policy=policy;model.policy_class=RecoveryTemperaturePolicy;model.policy_kwargs=kwargs


class RecoveryTemperaturePPO(GreedyStableQualityPPO):
    def snapshot_reference(self):
        migrate_policy(self)
        super().snapshot_reference()
