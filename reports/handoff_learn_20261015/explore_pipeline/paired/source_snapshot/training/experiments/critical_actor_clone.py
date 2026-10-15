"""Fine-tune a pretrained actor copy only at sensor-gated decision states.

Unlike a fresh small residual, the trainable branch starts with all the
navigation features and head. The clear-state actor remains byte-for-byte
frozen. Both branches use the same observations, masks and PPO probabilities.
"""
import copy
import torch
from experiments.anticipatory_recovery_ppo import (
    AnticipatoryRecoveryPolicy,anticipatory_support,warm_start_anticipatory)


class CloneFeatures(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.base.requires_grad_(False)
        self.adapted=copy.deepcopy(base);self.adapted.requires_grad_(True)

    def forward(self,observations):
        return torch.cat([self.base(observations),self.adapted(observations),
            anticipatory_support(observations)[:,None].to(observations.dtype)],dim=-1)


class CloneHead(torch.nn.Module):
    def __init__(self,base,prior_scale):
        super().__init__();self.base=base;self.base.requires_grad_(False)
        self.base_width=base.in_features;self.prior_scale=float(prior_scale)
        self.adapted=copy.deepcopy(base);self.adapted.requires_grad_(True)

    def forward(self,features):
        baseline=self.base(features[:,:self.base_width])
        adapted=self.adapted(features[:,self.base_width:-1])*self.prior_scale
        return torch.where(features[:,-1:]>.5,adapted,baseline)


class CriticalClonePolicy(AnticipatoryRecoveryPolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.mlp_extractor.policy_net=CloneFeatures(self.mlp_extractor.policy_net.base)
        self.action_net=CloneHead(self.action_net.base,self.recovery_prior_scale)
        self.mlp_extractor.latent_dim_pi=2*self.action_net.base_width+1
        actor=list(self.mlp_extractor.policy_net.adapted.parameters())+list(self.action_net.adapted.parameters())
        critic=list(self.mlp_extractor.value_net.parameters())+list(self.value_net.parameters())
        self.optimizer=self.optimizer_class([
            dict(params=actor,lr=lr_schedule(1),role='recovery'),
            dict(params=critic,lr=lr_schedule(1)*self.critic_learning_rate_ratio,role='critic')],
            lr=lr_schedule(1),**self.optimizer_kwargs)


def warm_start_clone(checkpoint,env,seed):
    model=warm_start_anticipatory(checkpoint,env,seed);previous=model.policy
    rng=torch.get_rng_state()
    try:
        policy=CriticalClonePolicy(model.observation_space,model.action_space,model.lr_schedule,
            **copy.deepcopy(model.policy_kwargs)).to(model.device)
        base_features=previous.mlp_extractor.policy_net.base.state_dict()
        base_head=previous.action_net.base.state_dict()
        for branch in (policy.mlp_extractor.policy_net.base,policy.mlp_extractor.policy_net.adapted):
            branch.load_state_dict(base_features)
        for branch in (policy.action_net.base,policy.action_net.adapted):branch.load_state_dict(base_head)
        policy.mlp_extractor.value_net.load_state_dict(previous.mlp_extractor.value_net.state_dict())
        policy.value_net.load_state_dict(previous.value_net.state_dict())
        parameters=dict(previous.named_parameters())
        for name,parameter in policy.named_parameters():
            if name.startswith(('mlp_extractor.value_net.','value_net.')):
                state=previous.optimizer.state.get(parameters[name])
                if state:policy.optimizer.state[parameter]=copy.deepcopy(state)
    finally:torch.set_rng_state(rng)
    model.policy=policy;model.policy_class=CriticalClonePolicy
    model.gated_migration.update(trainable_component='pretrained actor copy inside sensor support',
        adapted_actor_initialized_from_native_parent=True,base_navigation_weights_frozen=True,
        observation_and_mask_unchanged=True,initial_greedy_commands_retained=True)
    return model
