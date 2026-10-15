"""Training-only rank protection on the existing clear-state reference set.

Mean KL permits argmax flips when two commands have similar probabilities.
Rollback also checks a fraction of the frozen reference's winning margin.
Recovery/uncertain states retain the parent's exclusion rules. This protects
sampled/cache states only; it is not a safety certificate or runtime controller.
"""
import torch
from experiments.quality_masked_navigation import QualityAnchoredPPO


def rank_violation(reference,current,margin_fraction=.25):
    if reference.shape!=current.shape or reference.ndim!=2 or reference.shape[1]!=12:
        raise ValueError('Aligned twelve-choice probability batches required')
    if not 0<=margin_fraction<=1:raise ValueError('Margin fraction must be in [0,1]')
    winner=reference.argmax(dim=1)
    chosen=reference.gather(1,winner[:,None]).flatten()
    other=reference.clone();other.scatter_(1,winner[:,None],-1.)
    required=(chosen-other.max(dim=1).values)*margin_fraction
    current_chosen=current.gather(1,winner[:,None]).flatten()
    alternatives=current.clone();alternatives.scatter_(1,winner[:,None],-1.)
    return torch.clamp(required+alternatives.max(dim=1).values-current_chosen,min=0.)


class GreedyStableQualityPPO(QualityAnchoredPPO):
    def train(self):
        super().train()
        stats=self.last_reference_statistics
        if 'reference_kl' in stats:
            stats['reference_constraint_loss']=stats.pop('reference_kl')
            stats['reference_constraint_budget']=stats.pop('reference_kl_budget')
            stats['rank_guard_margin_fraction']=.25
            stats['constraint_loss_is_not_pure_kl']=True

    def reference_loss(self,observations,masks):
        keep=self.eligible(observations,masks);fraction=keep.float().mean()
        protected=observations[keep];protected_masks=masks[keep]
        if self.reference_cache is not None:
            protected=torch.cat([self.reference_cache[0],protected])
            protected_masks=torch.cat([self.reference_cache[1],protected_masks])
        if not len(protected):return observations.sum()*0.,fraction
        with torch.no_grad():
            reference=self.reference_policy.get_distribution(protected,action_masks=protected_masks).distribution
        current=self.policy.get_distribution(protected,action_masks=protected_masks).distribution
        mean_kl=torch.distributions.kl_divergence(reference,current).mean()
        worst=rank_violation(reference.probs,current.probs).max()
        # Parent train checks this before and after Adam, restoring both
        # weights and optimizer state on a rejected post-update candidate.
        loss=mean_kl+torch.where(worst>0,self.reference_kl_budget+worst,worst)
        return loss,fraction
