"""Repair newly discovered clear-state drift before constrained PPO updates.

No runtime action substitution. Repair uses only sampled clear observations
and the frozen learned reference, never development evaluation observations.
"""
import torch
from experiments.greedy_stable_quality_ppo import rank_violation
from experiments.recovery_temperature_policy import RecoveryTemperaturePPO


def repair_rank_drift(policy,reference,observations,masks,budget=.02,max_steps=64):
    parameters=[p for name,p in policy.named_parameters()
        if name.startswith(('mlp_extractor.policy_net.','action_net.')) and p.requires_grad]
    originals=[p.detach().clone() for p in parameters]
    with torch.no_grad():
        teacher=reference.get_distribution(observations,action_masks=masks).distribution.probs.clone()

    def metrics():
        current=policy.get_distribution(observations,action_masks=masks).distribution.probs
        kl=(teacher*(teacher.clamp_min(1e-20).log()-current.clamp_min(1e-20).log())).sum(-1).mean()
        violations=rank_violation(teacher,current)
        return kl,violations,kl+2*violations.mean()

    accepted=0
    for _ in range(max_steps+1):
        kl,violations,objective=metrics()
        if not torch.isfinite(objective):break
        if float(violations.max().detach())==0 and float(kl.detach())<=budget:
            return dict(repaired=True,steps=accepted,mean_kl=float(kl.detach()),
                max_rank_violation=0.,restored_original=False)
        if accepted==max_steps:break
        gradients=torch.autograd.grad(objective,parameters)
        if not all(torch.isfinite(g).all() for g in gradients):break
        before=[p.detach().clone() for p in parameters]
        improved=False
        for exponent in range(14):
            step=.1*(.5**exponent)
            with torch.no_grad():
                for parameter,value,gradient in zip(parameters,before,gradients):parameter.copy_(value-step*gradient)
                _,_,after=metrics()
            if torch.isfinite(after) and after<float(objective.detach()):improved=True;break
        if not improved:break
        accepted+=1
    with torch.no_grad():
        for parameter,value in zip(parameters,originals):parameter.copy_(value)
    kl,violations,_=metrics()
    return dict(repaired=False,steps=accepted,mean_kl=float(kl.detach()),
        max_rank_violation=float(violations.max().detach()),restored_original=True)


class RepairableRecoveryPPO(RecoveryTemperaturePPO):
    def remember_clear_states(self):
        super().remember_clear_states()
        if self.reference_cache is None:
            self.last_rank_repair=dict(repaired=True,steps=0,empty_cache=True)
            return
        self.last_rank_repair=repair_rank_drift(self.policy,self.reference_policy,
            *self.reference_cache,budget=self.reference_kl_budget)

    def train(self):
        super().train()
        self.last_reference_statistics['clear_state_rank_repair']=self.last_rank_repair
