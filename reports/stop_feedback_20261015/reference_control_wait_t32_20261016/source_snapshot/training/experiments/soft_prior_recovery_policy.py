"""Reduce a frozen actor's extreme prior only inside recovery support.

The same logits feed sampling, PPO likelihood/entropy and portable argmax.
Positive prior scaling preserves initial greedy commands, while increasing
exploration of currently masked-in actions. No maneuver is selected here.
"""
import copy
import numpy as np
import torch
from experiments.focused_recovery_policy import FocusedRecoveryPolicy,FocusedRecoveryHead,warm_start_focused


class SoftPriorHead(FocusedRecoveryHead):
    def __init__(self,source,prior_scale):
        super().__init__(source);self.prior_scale=float(prior_scale)

    def forward(self,features):
        support=features[:,-1:]
        prior=self.base(features[:,:self.base_width])
        scale=1+(self.prior_scale-1)*support
        return scale*prior+support*self.residual(features[:,self.base_width:-1])


class SoftPriorRecoveryPolicy(FocusedRecoveryPolicy):
    def __init__(self,*args,recovery_prior_scale=1/6,**kwargs):
        self.recovery_prior_scale=float(recovery_prior_scale)
        if not np.isfinite(self.recovery_prior_scale) or not 0<self.recovery_prior_scale<=1:
            raise ValueError('Prior scale must be finite and in (0,1]')
        super().__init__(*args,**kwargs)

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.action_net=SoftPriorHead(self.action_net,self.recovery_prior_scale)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),recovery_prior_scale=self.recovery_prior_scale)


def warm_start_soft_prior(checkpoint,env,seed=92611,prior_scale=1/6):
    model=warm_start_focused(checkpoint,env,seed)
    previous=model.policy;options=copy.deepcopy(model.policy_kwargs)
    options['recovery_prior_scale']=prior_scale
    rng=torch.get_rng_state()
    try:
        policy=SoftPriorRecoveryPolicy(model.observation_space,model.action_space,model.lr_schedule,**options).to(model.device)
        policy.load_state_dict(previous.state_dict())
        policy.optimizer.load_state_dict(copy.deepcopy(previous.optimizer.state_dict()))
    finally:torch.set_rng_state(rng)
    model.policy=policy;model.policy_class=SoftPriorRecoveryPolicy;model.policy_kwargs=options
    model.gated_migration.update(recovery_frozen_prior_scale=prior_scale,
        initial_greedy_commands_retained=True,initial_recovery_sampling_deliberately_changed=True)
    return model
