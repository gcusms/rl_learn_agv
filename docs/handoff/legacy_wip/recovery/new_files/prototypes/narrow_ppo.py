"""Versioned 318D/20-action actor, retaining a frozen old actor as a prior."""
import copy
import numpy as np
import torch
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from prototypes.narrow_observation import ACTIONS

class FineLogits(nn.Module):
    def __init__(self,source,fine_prior_penalty=1.):
        super().__init__()
        self.teacher=copy.deepcopy(source.mlp_extractor.policy_net)
        self.teacher_head=copy.deepcopy(source.action_net)
        self.teacher.requires_grad_(False);self.teacher_head.requires_grad_(False)
        self.fine_prior_penalty=float(fine_prior_penalty)
        self.residual=nn.Sequential(nn.Linear(318,64),nn.Tanh(),nn.Linear(64,20))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,x):
        old=self.teacher_head(self.teacher(x[:,:300]))
        # New fine-action priors interpolate related old logits. They are not
        # pretrained fine maneuvers and no old optimizer state is claimed.
        extra=torch.stack([.8*old[:,0]+.2*old[:,4],.8*old[:,0]+.2*old[:,5],
            .6*old[:,0]+.4*old[:,4],.6*old[:,0]+.4*old[:,5],
            .8*old[:,3]+.2*old[:,6],.8*old[:,3]+.2*old[:,7],
            .8*old[:,2]+.2*old[:,6],.8*old[:,2]+.2*old[:,7]],dim=1)-self.fine_prior_penalty
        # Blocked/poorly recognized states need trainable recovery decisions.
        # Safety admission belongs to the action mask, never to this gradient.
        return torch.cat([old,extra],dim=1)+self.residual(x)

class FineExtractor(nn.Module):
    latent_dim_pi=20;latent_dim_vf=128
    def __init__(self,source,fine_prior_penalty=1.):
        super().__init__();self.policy_net=FineLogits(source,fine_prior_penalty)
        self.value_net=nn.Sequential(nn.Linear(318,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
    def forward_actor(self,x):return self.policy_net(x)
    def forward_critic(self,x):return self.value_net(x)
    def forward(self,x):return self.forward_actor(x),self.forward_critic(x)

class NarrowPolicy(MaskableActorCriticPolicy):
    def __init__(self,*args,teacher_checkpoint='models/recovery_20261012/candidate/policy.zip',fine_prior_penalty=1.,**kwargs):
        if not np.isfinite(fine_prior_penalty) or fine_prior_penalty<0:raise ValueError('Nonnegative finite fine prior penalty required')
        self.fine_prior_penalty=float(fine_prior_penalty)
        self.teacher_checkpoint=str(teacher_checkpoint);super().__init__(*args,**kwargs)
    def _build(self,lr_schedule):
        if self.observation_space.shape!=(318,) or self.action_space.n!=20:raise ValueError('318D/20-action policy required')
        super()._build(lr_schedule)
        source=MaskablePPO.load(self.teacher_checkpoint,device='cpu').policy
        self.mlp_extractor=FineExtractor(source,self.fine_prior_penalty);self.action_net=nn.Identity()
        self.value_net=nn.Linear(128,1)
        self.optimizer=self.optimizer_class([p for p in self.parameters() if p.requires_grad],lr=lr_schedule(1),**self.optimizer_kwargs)
    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),teacher_checkpoint=self.teacher_checkpoint,fine_prior_penalty=self.fine_prior_penalty)

class PortableFineActor(nn.Module):
    def __init__(self,policy):
        super().__init__();self.net=copy.deepcopy(policy.mlp_extractor.policy_net)
        self.register_buffer('actions',torch.tensor(ACTIONS.copy()))
    def forward(self,x):
        mask=torch.cat([torch.ones_like(x[:,:1],dtype=torch.bool),x[:,235:244]>.5,x[:,267:269]>.5,x[:,300:308]>.5],dim=1)
        logits=self.net(x);index=torch.where(mask,logits,torch.full_like(logits,-1e8)).argmax(1)
        return self.actions[index]
