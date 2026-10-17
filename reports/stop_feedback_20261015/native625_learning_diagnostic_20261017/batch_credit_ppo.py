"""Complete on-policy episodes with one actor-advantage normalization per batch.

This experimental learner retains the original sensor/action/permission
contract. It deliberately rejects logged samples from different policies.
Complete finite-task returns carry success/failure to earlier decisions;
minibatch shuffling does not recenter those decisions a second time.
"""
import numpy as np
import torch
import torch.nn.functional as F

from experiments.native625_onpolicy import Native625OnPolicy
from retained_stop_ppo import RetainedStopOnPolicy


@torch.no_grad()
def prepare_completed_batch(learner, episodes, gamma=.99, check_batch_size=128):
    if len(episodes) < 2 or not 0. < gamma <= 1.:
        raise ValueError('At least two complete same-policy episodes are required')
    keys=('observations','masks','actions','old_log_probability','old_values')
    joined={key:[] for key in keys}
    returns=[]
    offsets=[0]
    for episode in episodes:
        n=len(episode['actions'])
        expected=np.zeros(n,bool)
        if not n:
            raise ValueError('Empty episodes cannot supply on-policy experience')
        expected[-1]=True
        if not np.array_equal(episode['terminated'],expected) or not np.array_equal(episode['episode_ends'],expected):
            raise ValueError('A complete terminal task is required; do not bridge resets')
        if float(episode['temperature'])!=learner.temperature:
            raise ValueError('All episodes must use this fixed sampling temperature')
        following=0.
        discounted=np.zeros(n,np.float32)
        for index in range(n-1,-1,-1):
            following=float(episode['rewards'][index])+gamma*following
            discounted[index]=following
        if not np.isfinite(discounted).all():
            raise ValueError('Finite complete-task returns are required')
        for key in keys:
            value=np.asarray(episode[key])
            if len(value)!=n:
                raise ValueError('One stored observation and likelihood per actual decision required')
            joined[key].append(value)
        returns.append(discounted)
        offsets.append(offsets[-1]+n)
    batch={key:np.concatenate(value) for key,value in joined.items()}
    eligible=[]
    for start in range(0,len(batch['actions']),check_batch_size):
        end=start+check_batch_size
        terms=learner.evaluate(batch['observations'][start:end],batch['masks'][start:end],batch['actions'][start:end])
        if not np.allclose(terms['log_probability'].numpy(),batch['old_log_probability'][start:end],rtol=0,atol=2e-5):
            raise ValueError('Episodes were not collected with this unchanged behavior policy')
        if not np.allclose(terms['values'].numpy(),batch['old_values'][start:end],rtol=0,atol=2e-5):
            raise ValueError('Critic changed while the collection batch was in progress')
        eligible.extend(terms['eligible'].numpy())
    keep=np.asarray(eligible,bool)
    batch['returns']=np.concatenate(returns)
    raw=batch['returns']-batch['old_values']
    normalized=np.zeros(len(raw),np.float32)
    mean=float(raw[keep].mean()) if keep.any() else 0.
    std=float(raw[keep].std()) if keep.any() else 0.
    if keep.any():
        normalized[keep]=(raw[keep]-mean)/(std+1e-8) if keep.sum()>1 else raw[keep]
    batch.update(advantages=normalized,raw_advantages=raw,eligible=keep)
    return batch,dict(normalization_scope='all_actor_eligible_samples_in_one_frozen_policy_collection_batch',
        complete_episodes=len(episodes),episode_offsets=offsets,eligible_samples=int(keep.sum()),
        raw_advantage_mean=mean,raw_advantage_std=std,gamma=gamma,gae_lambda=1.,
        no_minibatch_recentering=True,unchanged_behavior_likelihoods_verified=True,
        no_critic_or_policy_update_during_collection=True)


class BatchCreditKernel(Native625OnPolicy):
    def update(self,observations,masks,actions,old_log_probability,advantages,returns,
            *,sampled_temperature,rollout_advantages_normalized=False,anchors=None,clip=.2,
            entropy_coefficient=.01,anchor_strength=.05,target_kl=.02):
        if not rollout_advantages_normalized:
            raise ValueError('Normalize once over the complete frozen-policy collection batch')
        if sampled_temperature!=self.temperature:
            raise ValueError('Sampling and PPO likelihoods must use the same temperature')
        terms=self.evaluate(observations,masks,actions)
        old=torch.as_tensor(old_log_probability,dtype=torch.float32).detach()
        advantage=torch.as_tensor(advantages,dtype=torch.float32).detach()
        returns=torch.as_tensor(returns,dtype=torch.float32).detach()
        if any(x.shape!=terms['values'].shape or not torch.isfinite(x).all() for x in (old,advantage,returns)):
            raise ValueError('Finite likelihood, pre-normalized advantage and return per action required')
        keep=terms['eligible']
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_updated=False
        kl=entropy=actor_loss=anchor=fraction=0.
        if keep.any():
            selected=advantage[keep]  # Already normalized across the collection batch.
            log_ratio=terms['log_probability'][keep]-old[keep]
            ratio=log_ratio.exp()
            kl=float((((ratio-1.)-log_ratio).mean()).detach())
            fraction=float((abs(ratio.detach()-1.)>clip).float().mean())
            entropy_term=terms['entropy'][keep].mean()
            pg=-torch.minimum(selected*ratio,selected*ratio.clamp(1.-clip,1.+clip)).mean()
            anchoring=self.anchor_loss(anchors)
            loss=pg-entropy_coefficient*entropy_term+anchor_strength*anchoring
            entropy,actor_loss,anchor=[float(x.detach()) for x in (entropy_term,pg,anchoring)]
            if kl<=1.5*target_kl:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.head.parameters(),1.)
                self.actor_optimizer.step()
                actor_updated=True
        value_loss=F.mse_loss(terms['values'],returns)
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1.)
        self.critic_optimizer.step()
        return dict(actor_updated=actor_updated,eligible_samples=int(keep.sum()),total_samples=len(keep),
            approx_kl=kl,clip_fraction=fraction,entropy=entropy,actor_loss=actor_loss,anchor_kl=anchor,
            value_loss=float(value_loss.detach()),actor_optimizer_untouched_on_no_choice=not bool(keep.any()),
            advantages_normalized_once_before_minibatching=True)


class BatchCreditOnPolicy(RetainedStopOnPolicy,BatchCreditKernel):
    """The STOP guard wraps the batch-normalized kernel through cooperative MRO."""
