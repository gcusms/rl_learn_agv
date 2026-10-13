"""Masked PPO with a frozen reference on observed clear navigation states.

The reference never drives the robot. A small training-only observation cache
protects earlier clear states while recovery states remain free to learn.
KL budgets constrain sampled/cache states, not every possible deployment state.
Train loop adapted from sb3-contrib 2.7.0 MaskablePPO (MIT).
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance


class AnchoredMaskedPPO(MaskablePPO):
    def __init__(self,*args,reference_strength=0.,reference_kl_budget=.02,**kwargs):
        if not np.isfinite([reference_strength,reference_kl_budget]).all() or reference_strength<0 or reference_kl_budget<=0:
            raise ValueError('Nonnegative reference strength and positive finite KL budget required')
        self.reference_strength=float(reference_strength)
        self.reference_kl_budget=float(reference_kl_budget)
        self.reference_policy=None;self.reference_cache=None
        self.last_reference_statistics={}
        super().__init__(*args,**kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['reference_policy','reference_cache']

    def snapshot_reference(self):
        if self.observation_space.shape!=(282,) or self.action_space.n!=12:
            raise ValueError('Native masked navigation policy required')
        self.reference_policy=copy.deepcopy(self.policy)
        self.reference_policy.set_training_mode(False);self.reference_policy.requires_grad_(False)
        self.reference_cache=None

    def eligible(self,observations,masks):
        if self.reference_policy is None:raise ValueError('Freeze the reference before learning')
        with torch.no_grad():
            choices=self.reference_policy.get_distribution(observations,action_masks=masks).distribution.probs.argmax(dim=1)
            # Existing sensor fields only: clear forward corridor, waypoint
            # ahead, no last rejection/wait, and no confirmed moving track.
            return (masks.bool().all(dim=1)&(observations[:,250]>.5)
                &(observations[:,266]<.04)&(observations[:,234]>.4)
                &(observations[:,232]>.2)&(observations[:,281]<.5)&(choices!=0))

    def remember_clear_states(self):
        observations=torch.as_tensor(self.rollout_buffer.observations.reshape(-1,282),device=self.device)
        masks=torch.as_tensor(self.rollout_buffer.action_masks.reshape(-1,12),device=self.device)
        keep=self.eligible(observations,masks)
        observations=observations[keep];masks=masks[keep]
        if len(observations)>128:
            index=torch.as_tensor(np.linspace(0,len(observations)-1,128).astype(int),device=self.device)
            observations=observations[index];masks=masks[index]
        if len(observations):
            if self.reference_cache is not None:
                observations=torch.cat([self.reference_cache[0],observations])
                masks=torch.cat([self.reference_cache[1],masks])
            self.reference_cache=(observations[-1024:].detach().clone(),masks[-1024:].detach().clone())

    def reference_loss(self,observations,masks):
        keep=self.eligible(observations,masks);fraction=keep.float().mean()
        protected=observations[keep];protected_masks=masks[keep]
        if self.reference_cache is not None:
            protected=torch.cat([self.reference_cache[0],protected])
            protected_masks=torch.cat([self.reference_cache[1],protected_masks])
        if not len(protected):return observations.sum()*0.,fraction
        with torch.no_grad():reference=self.reference_policy.get_distribution(protected,action_masks=protected_masks).distribution
        current=self.policy.get_distribution(protected,action_masks=protected_masks).distribution
        return torch.distributions.kl_divergence(reference,current).mean(),fraction

    def train(self):
        if self.reference_strength==0:return super().train()
        self.policy.set_training_mode(True);self._update_learning_rate(self.policy.optimizer)
        clip=self.clip_range(self._current_progress_remaining)
        clip_vf=self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None
        self.remember_clear_states()
        logs={name:[] for name in ('entropy_loss','policy_gradient_loss','value_loss','clip_fraction','approx_kl','reference_kl','reference_fraction')}
        stopped=False;accepted=0;rollbacks=0;loss=None
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                values,log_prob,entropy=self.policy.evaluate_actions(batch.observations,batch.actions.long().flatten(),action_masks=batch.action_masks)
                values=values.flatten();advantages=batch.advantages
                if self.normalize_advantage:advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
                ratio=torch.exp(log_prob-batch.old_log_prob)
                pg=-torch.min(advantages*ratio,advantages*torch.clamp(ratio,1-clip,1+clip)).mean()
                prediction=values if clip_vf is None else batch.old_values+torch.clamp(values-batch.old_values,-clip_vf,clip_vf)
                vf=F.mse_loss(batch.returns,prediction)
                ent=-torch.mean(-log_prob) if entropy is None else -entropy.mean()
                reference,fraction=self.reference_loss(batch.observations,batch.action_masks)
                loss=pg+self.ent_coef*ent+self.vf_coef*vf+self.reference_strength*reference
                with torch.no_grad():
                    log_ratio=log_prob-batch.old_log_prob
                    approx=((torch.exp(log_ratio)-1)-log_ratio).mean().item()
                for name,value in [('entropy_loss',ent),('policy_gradient_loss',pg),('value_loss',vf),
                    ('clip_fraction',(torch.abs(ratio-1)>clip).float().mean()),('reference_kl',reference),('reference_fraction',fraction)]:
                    logs[name].append(float(value.detach()))
                logs['approx_kl'].append(approx)
                if (self.target_kl is not None and approx>1.5*self.target_kl) or reference.detach().item()>self.reference_kl_budget:
                    stopped=True;break
                if not self.constrained_optimizer_step(loss,batch.observations,batch.action_masks):
                    rollbacks+=1;stopped=True;break
                accepted+=1
            if stopped:break
        self._n_updates+=self.n_epochs  # Native contrib bookkeeping convention.
        for name,values in logs.items():self.logger.record('train/'+name,float(np.mean(values)) if values else 0.)
        self.logger.record('train/loss',float(loss.detach()) if loss is not None else 0.)
        self.logger.record('train/explained_variance',explained_variance(self.rollout_buffer.values.flatten(),self.rollout_buffer.returns.flatten()))
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        self.logger.record('train/clip_range',clip)
        self.last_reference_statistics=dict(reference_kl=float(np.mean(logs['reference_kl'])) if logs['reference_kl'] else 0.,
            eligible_fraction=float(np.mean(logs['reference_fraction'])) if logs['reference_fraction'] else 0.,
            cache_states=0 if self.reference_cache is None else len(self.reference_cache[0]),
            accepted_optimizer_steps=accepted,rolled_back_optimizer_steps=rollbacks,
            update_stopped=stopped,reference_kl_budget=self.reference_kl_budget)

    def constrained_optimizer_step(self,loss,observations,masks):
        """Default behavior retains the original whole-update rollback."""
        weights={k:v.detach().clone() for k,v in self.policy.state_dict().items()}
        optimizer=copy.deepcopy(self.policy.optimizer.state_dict())
        self.policy.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
        self.policy.optimizer.step()
        with torch.no_grad():post,_=self.reference_loss(observations,masks)
        if not torch.isfinite(post) or post.item()>self.reference_kl_budget:
            self.policy.load_state_dict(weights);self.policy.optimizer.load_state_dict(optimizer)
            return False
        return True
