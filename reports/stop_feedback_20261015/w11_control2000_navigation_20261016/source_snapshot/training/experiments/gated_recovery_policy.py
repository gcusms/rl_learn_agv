"""Frozen learned base plus trainable masked recovery logits.

The clear-state predicate is the existing anchor's sensor predicate. It only
selects parameter support; it never selects a driving command. Outside those
states the residual and frozen learned base jointly choose all twelve actions.
No truth, route teacher, evaluation replay, or replacement controller is used.
"""
import copy
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.recovery_temperature_policy import RecoveryTemperaturePolicy,migrate_policy


def sensor_clear(observations):
    return ((observations[:,235:244]>.5).all(-1)
        &(observations[:,267:269]>.5).all(-1)
        &(observations[:,250]>.5)&(observations[:,266]<.04)
        &(observations[:,234]>.4)&(observations[:,232]>.2)
        &(observations[:,281]<.5)&(observations[:,294]>.5)
        &(observations[:,295:299]<.5).all(-1))


class RecoveryFeatures(torch.nn.Module):
    def __init__(self,base,residual_width):
        super().__init__();self.base=base
        self.base.requires_grad_(False)
        self.residual=torch.nn.Sequential(torch.nn.Linear(300,residual_width),torch.nn.Tanh(),
            torch.nn.Linear(residual_width,residual_width),torch.nn.Tanh())

    def forward(self,observations):
        return torch.cat([self.base(observations),self.residual(observations),
            sensor_clear(observations)[:,None].to(observations.dtype)],dim=-1)


class RecoveryHead(torch.nn.Module):
    def __init__(self,base,base_width,residual_width):
        super().__init__();self.base=base;self.base_width=base_width
        self.base.requires_grad_(False)
        self.residual=torch.nn.Linear(residual_width,12)
        torch.nn.init.zeros_(self.residual.weight);torch.nn.init.zeros_(self.residual.bias)

    def forward(self,features):
        baseline=self.base(features[:,:self.base_width])
        # The original anchor excludes a frozen base's preferred stop. Keep
        # those states trainable, even if the sensor predicate says clear.
        protected=features[:,-1]*(baseline.argmax(-1)!=0).to(features.dtype)
        residual=self.residual(features[:,self.base_width:-1])
        return baseline+(1-protected[:,None])*residual


class GatedRecoveryPolicy(RecoveryTemperaturePolicy):
    def __init__(self,*args,residual_width=64,critic_learning_rate_ratio=.1,**kwargs):
        self.residual_width=int(residual_width)
        self.critic_learning_rate_ratio=float(critic_learning_rate_ratio)
        if self.residual_width<=0:raise ValueError('Positive residual width required')
        if not np.isfinite(self.critic_learning_rate_ratio) or not 0<self.critic_learning_rate_ratio<=1:
            raise ValueError('Critic learning-rate ratio must be in (0,1]')
        super().__init__(*args,**kwargs)

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        if self.observation_space.shape!=(300,) or self.action_space.n!=12:
            raise ValueError('Native300-field twelve-action policy required')
        if any(True for _ in self.features_extractor.parameters()):
            raise ValueError('Shared trainable feature extractor is unsupported')
        width=self.mlp_extractor.latent_dim_pi
        self.mlp_extractor.policy_net=RecoveryFeatures(self.mlp_extractor.policy_net,self.residual_width)
        self.action_net=RecoveryHead(self.action_net,width,self.residual_width)
        self.mlp_extractor.latent_dim_pi=width+self.residual_width+1
        recovery=list(self.mlp_extractor.policy_net.residual.parameters())+list(self.action_net.residual.parameters())
        critic=list(self.mlp_extractor.value_net.parameters())+list(self.value_net.parameters())
        self.optimizer=self.optimizer_class([
            dict(params=recovery,lr=lr_schedule(1),role='recovery'),
            dict(params=critic,lr=lr_schedule(1)*self.critic_learning_rate_ratio,role='critic')],
            lr=lr_schedule(1),**self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),residual_width=self.residual_width,
            critic_learning_rate_ratio=self.critic_learning_rate_ratio)


def warm_start_gated(checkpoint,env,seed=92611):
    old=MaskablePPO.load(checkpoint,device='cpu');migrate_policy(old)
    if isinstance(old.policy,GatedRecoveryPolicy):raise ValueError('Use native parent for an explicit migration')
    options=copy.deepcopy(old.policy_kwargs)
    new=GatedRecoveryPPO(GatedRecoveryPolicy,env,seed=seed,device='cpu',
        n_steps=old.n_steps,batch_size=old.batch_size,n_epochs=old.n_epochs,
        learning_rate=3e-4,target_kl=old.target_kl,ent_coef=old.ent_coef,
        gamma=old.gamma,gae_lambda=old.gae_lambda,vf_coef=old.vf_coef,
        max_grad_norm=old.max_grad_norm,clip_range=old.clip_range,
        clip_range_vf=old.clip_range_vf,normalize_advantage=old.normalize_advantage,
        policy_kwargs=options)
    policy=new.policy
    policy.mlp_extractor.policy_net.base.load_state_dict(old.policy.mlp_extractor.policy_net.state_dict())
    policy.action_net.base.load_state_dict(old.policy.action_net.state_dict())
    policy.mlp_extractor.value_net.load_state_dict(old.policy.mlp_extractor.value_net.state_dict())
    policy.value_net.load_state_dict(old.policy.value_net.state_dict())
    previous=dict(old.policy.named_parameters())
    for name,parameter in policy.named_parameters():
        if not parameter.requires_grad:continue
        source=previous.get(name)
        if source is not None and source in old.policy.optimizer.state:
            policy.optimizer.state[parameter]=copy.deepcopy(old.policy.optimizer.state[source])
    new.training_reward_scale=float(getattr(old,'training_reward_scale',1.))
    scales=env.get_attr('reward_scale') if hasattr(env,'get_attr') else [env.reward_scale]
    if any(scale!=new.training_reward_scale for scale in scales):raise ValueError('Reward units differ')
    new.gated_migration=dict(base_weights_frozen=True,residual_head_initially_zero=True,
        critic_weights_and_adam_retained=True,base_actor_adam_not_used=True,
        exact_old_training_resume=False,reference_rank_rollback_removed=True)
    return new


class GatedRecoveryPPO(MaskablePPO):
    def _update_learning_rate(self,optimizers):
        super()._update_learning_rate(optimizers)
        groups=self.policy.optimizer.param_groups
        for group in groups:
            if group['role']=='critic':group['lr']*=self.policy.critic_learning_rate_ratio

    def snapshot_reference(self):
        self.last_reference_statistics=dict(protection='frozen_base_and_clear_state_zero_residual',
            limitation='Only the existing clear predicate with nonzero base winner; not all navigation states')

    def train(self):
        super().train()
        self.last_reference_statistics=dict(protection='frozen_base_and_clear_state_zero_residual',
            reference_rank_rollback=False,trainable_residual=True,
            learning_rates={g['role']:g['lr'] for g in self.policy.optimizer.param_groups},
            approximate_kl=float(self.logger.name_to_value.get('train/approx_kl',0.)))
