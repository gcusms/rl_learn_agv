"""Full masked actor adaptation with causal guard explanations.

All states with a nonzero legal alternative can update the actor, including
an initially preferred wait. Forced-only-stop states still train the critic.
No teacher command, minimum velocity or forbidden-stop rule is introduced.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from experiments.conditional_recovery_ppo import actor_terms
from neural_navigation.decision_feedback import TRAIN_DIM, RISK_SLICE, LEGACY_SCHEMA, contract


def decision_support(observations):
    return (observations[:,235:244]>.5).any(-1)|(observations[:,267:269]>.5).any(-1)


class PrefixLinear(torch.nn.Module):
    """Separate the new zero-initialized inputs from the parent's dot product.

    A larger GEMM can round the old summation differently despite appended
    zero weights. Keep the old contiguous 300-wide calculation unchanged.
    The full affine weights and their migrated Adam moments remain trainable.
    """
    def __init__(self,source):
        super().__init__();self.weight=source.weight;self.bias=source.bias
    def forward(self,x):
        return (F.linear(x[:,:300],self.weight[:,:300].contiguous(),self.bias)
                +F.linear(x[:,300:],self.weight[:,300:].contiguous()))


class StopFeedbackPolicy(MaskableActorCriticPolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        if self.observation_space.shape!=(TRAIN_DIM,):
            raise ValueError('Explicit appended stop-feedback policy required')
        self.mlp_extractor.policy_net[0]=PrefixLinear(self.mlp_extractor.policy_net[0])
        self.mlp_extractor.value_net[0]=PrefixLinear(self.mlp_extractor.value_net[0])


class StopFeedbackPPO(MaskablePPO):
    def train(self):
        self.policy.set_training_mode(True); self._update_learning_rate(self.policy.optimizer)
        clip=self.clip_range(self._current_progress_remaining)
        logs={key:[] for key in ('entropy_loss','policy_gradient_loss','value_loss',
            'approx_kl','clip_fraction','actor_decision_fraction','clear_reference_kl')}
        running=True; loss=None
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                values,log_prob,entropy=self.policy.evaluate_actions(batch.observations,
                    batch.actions.long().flatten(),action_masks=batch.action_masks)
                pg,ent,kl,fraction,count=actor_terms(batch.observations,log_prob,batch.old_log_prob,
                    batch.advantages,entropy,clip,decision_support)
                values=values.flatten()
                vf=F.mse_loss(batch.returns,values)
                with torch.no_grad():
                    ref_logits=self.reference_net(batch.observations[:,:300])
                    mask=batch.action_masks.bool()
                    ref_log=F.log_softmax(ref_logits.masked_fill(~mask,-1e8),dim=-1)
                    risk=batch.observations[:,300+RISK_SLICE.start:300+RISK_SLICE.stop]
                    # Protect ordinary clear nonzero navigation. A reference
                    # stop never prevents the new actor from learning to move.
                    clear=mask[:,1:].all(-1)&(risk.max(-1).values==0)&(ref_log.argmax(-1)!=0)
                distribution=self.policy.get_distribution(batch.observations,action_masks=batch.action_masks)
                new_log=distribution.distribution.logits
                anchor=(ref_log.exp()*(ref_log-new_log)).sum(-1)[clear].mean() if clear.any() else log_prob.sum()*0.
                loss=pg+self.ent_coef*ent+self.vf_coef*vf+self.anchor_strength*anchor
                for key,value in [('entropy_loss',ent),('policy_gradient_loss',pg),('value_loss',vf),
                                  ('approx_kl',kl),('clip_fraction',fraction),('clear_reference_kl',anchor)]:
                    logs[key].append(float(value.detach()))
                logs['actor_decision_fraction'].append(count/len(batch.advantages))
                if count and self.target_kl is not None and kl>1.5*self.target_kl:
                    running=False; break
                self.policy.optimizer.zero_grad(); loss.backward()
                if not count:
                    # Zero actor gradients would still decay inherited Adam
                    # momentum. No-choice batches must update only the critic.
                    for name,parameter in self.policy.named_parameters():
                        if name.startswith(('mlp_extractor.policy_net.', 'action_net.')):
                            parameter.grad=None
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates+=1
            if not running:break
        for key,values in logs.items():self.logger.record('train/'+key,float(np.mean(values)))
        self.logger.record('train/explained_variance',explained_variance(
            self.rollout_buffer.values.flatten(),self.rollout_buffer.returns.flatten()))
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        self.logger.record('train/clip_range',clip)
        if loss is not None:self.logger.record('train/loss',float(loss.detach()))


def warm_start(checkpoint,env,seed=20261015,learning_rate=1e-4):
    old=MaskablePPO.load(checkpoint,device='cpu')
    if old.observation_space.shape!=(300,) or old.action_space.n!=12:
        raise ValueError('Native 300D quality PPO checkpoint required')
    if env.observation_space.shape!=(TRAIN_DIM,):
        raise ValueError('Explicit new feedback observation required')
    feedback_schema=getattr(env,'feedback_schema',LEGACY_SCHEMA)
    contract(feedback_schema)
    scale=float(old.training_reward_scale)
    if env.reward_scale!=scale:raise ValueError('Training reward units differ from checkpoint')
    new=StopFeedbackPPO(StopFeedbackPolicy,env,device='cpu',seed=seed,n_steps=250,batch_size=125,
        n_epochs=4,learning_rate=learning_rate,target_kl=.02,ent_coef=.01,
        gamma=old.gamma,gae_lambda=old.gae_lambda,vf_coef=old.vf_coef,max_grad_norm=old.max_grad_norm,
        policy_kwargs=copy.deepcopy(old.policy_kwargs))
    source=old.policy.state_dict();target=new.policy.state_dict()
    inputs={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
    for name,value in target.items():
        if name in inputs:
            value.zero_();value[:,:300]=source[name]
        else:
            if value.shape!=source[name].shape:raise ValueError('Parent policy architecture differs')
            value.copy_(source[name])
    new.policy.load_state_dict(target)
    previous=dict(old.policy.named_parameters())
    for name,parameter in new.policy.named_parameters():
        state=old.policy.optimizer.state.get(previous[name])
        if not state:continue
        migrated={}
        for key,value in state.items():
            if torch.is_tensor(value) and name in inputs and value.shape==previous[name].shape:
                expanded=torch.zeros_like(parameter);expanded[:,:300]=value;migrated[key]=expanded
            else:migrated[key]=value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        new.policy.optimizer.state[parameter]=migrated
    new.reference_net=torch.nn.Sequential(copy.deepcopy(old.policy.mlp_extractor.policy_net),
        copy.deepcopy(old.policy.action_net)).eval().requires_grad_(False)
    new.anchor_strength=.5
    new.training_reward_scale=scale
    new.feedback_schema=feedback_schema
    new.feedback_migration=dict(parent_checkpoint_timesteps=old.num_timesteps,
        dimensions=TRAIN_DIM,legacy_prefix_dimensions=300,new_input_weights_initially_zero=True,
        all_actor_parameters_trainable=True,critic_uses_all_transitions=True,
        actor_support='every state with at least one nonzero legal choice',
        optimizer_moments_retained=True,reward_unchanged=True,actions_masks_guard_unchanged=True,
        reference_stop_excluded_from_clear_anchor=True,exact_rollout_resume=False,
        feedback_schema=feedback_schema)
    return new
