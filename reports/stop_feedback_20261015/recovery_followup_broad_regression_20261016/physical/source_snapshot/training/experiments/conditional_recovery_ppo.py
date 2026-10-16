"""PPO actor loss conditioned on the residual's fixed sensor support.

Forced stops and frozen base states still train the value function. They do
not dilute the recovery actor loss or its advantage normalization. Sampling,
stored actions, old probabilities, masks and clipped PPO ratios are unchanged.
Training loop follows sb3-contrib MaskablePPO (MIT).
"""
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.utils import explained_variance
from experiments.gated_recovery_policy import GatedRecoveryPPO
from experiments.focused_recovery_policy import recovery_support


def actor_terms(observations,log_prob,old_log_prob,advantages,entropy,clip,support=recovery_support):
    keep=support(observations)
    if not keep.any():
        zero=log_prob.sum()*0.
        return zero,zero,zero,zero,0
    advantage=advantages[keep]
    if len(advantage)>1:advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
    log_ratio=log_prob[keep]-old_log_prob[keep];ratio=torch.exp(log_ratio)
    pg=-torch.minimum(advantage*ratio,advantage*torch.clamp(ratio,1-clip,1+clip)).mean()
    ent=-entropy[keep].mean() if entropy is not None else log_prob[keep].mean()
    with torch.no_grad():
        kl=((ratio-1)-log_ratio).mean();fraction=(abs(ratio-1)>clip).float().mean()
    return pg,ent,kl,fraction,int(keep.sum())


class ConditionalRecoveryPPO(GatedRecoveryPPO):
    actor_support=staticmethod(recovery_support)

    def train(self):
        self.policy.set_training_mode(True);self._update_learning_rate(self.policy.optimizer)
        clip=self.clip_range(self._current_progress_remaining)
        clip_vf=self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None
        logs={key:[] for key in ('entropy_loss','policy_gradient_loss','value_loss','approx_kl','clip_fraction','recovery_fraction')}
        running=True;loss=None
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                values,log_prob,entropy=self.policy.evaluate_actions(batch.observations,batch.actions.long().flatten(),action_masks=batch.action_masks)
                values=values.flatten()
                pg,ent,kl,fraction,count=actor_terms(batch.observations,log_prob,batch.old_log_prob,batch.advantages,entropy,clip,self.actor_support)
                prediction=values if clip_vf is None else batch.old_values+torch.clamp(values-batch.old_values,-clip_vf,clip_vf)
                vf=F.mse_loss(batch.returns,prediction);loss=pg+self.ent_coef*ent+self.vf_coef*vf
                for key,value in [('entropy_loss',ent),('policy_gradient_loss',pg),('value_loss',vf),
                                  ('approx_kl',kl),('clip_fraction',fraction)]:logs[key].append(float(value.detach()))
                logs['recovery_fraction'].append(count/len(batch.advantages))
                if count and self.target_kl is not None and kl>1.5*self.target_kl:
                    running=False;break
                self.policy.optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates+=1
            if not running:break
        for key,values in logs.items():self.logger.record('train/'+key,float(np.mean(values)))
        self.logger.record('train/explained_variance',explained_variance(self.rollout_buffer.values.flatten(),self.rollout_buffer.returns.flatten()))
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        self.logger.record('train/clip_range',clip)
        if loss is not None:self.logger.record('train/loss',float(loss.detach()))
        self.last_reference_statistics=dict(protection='frozen_base_zero_residual_outside_recovery_support',
            actor_advantages_normalized_within_support=True,critic_uses_all_transitions=True,
            conditional_approx_kl=float(np.mean(logs['approx_kl'])),
            recovery_fraction=float(np.mean(logs['recovery_fraction'])))
