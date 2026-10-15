"""Opt-in actor backtracking with independent critic/Adam updates retained.

No command substitution, new labels, relaxed rank limits or evaluation cache.
Only policies with disjoint, parameter-free feature extraction are supported.
"""
import copy
import torch
from experiments.repairable_recovery_ppo import RepairableRecoveryPPO


def parameter_partition(policy):
    actor={};critic={}
    for name,parameter in policy.named_parameters():
        if name.startswith(('mlp_extractor.policy_net.','action_net.')):actor[name]=parameter
        elif name.startswith(('mlp_extractor.value_net.','value_net.')):critic[name]=parameter
        else:raise ValueError('Backtracking requires disjoint actor and critic parameters: '+name)
    if not actor or not critic:raise ValueError('Both actor and critic required')
    return actor,critic


def commit_constrained_candidate(policy,before_weights,before_optimizer,constraint,budget,max_backtracks=12):
    """Called after Adam. Test smaller actor deltas; keep full critic delta.

    Accepted scaled steps retain Adam moments from the same gradient, as a
    smaller actor learning rate would. On complete rejection restore actor
    weights AND its moments, retaining the independent critic's update.
    """
    actor,_=parameter_partition(policy)
    candidate={name:p.detach().clone() for name,p in actor.items()}
    for attempt in range(max_backtracks+1):
        fraction=2.**(-attempt)
        with torch.no_grad():
            for name,p in actor.items():p.copy_(before_weights[name]+fraction*(candidate[name]-before_weights[name]))
            value=float(constraint())
        if torch.isfinite(torch.tensor(value)) and value<=budget:
            return dict(actor_step_accepted=True,actor_step_fraction=fraction,
                backtracks=attempt,critic_step_retained=True,constraint_loss=value)
    with torch.no_grad():
        for name,p in actor.items():p.copy_(before_weights[name])
    # Match optimizer parameter IDs by actual group order, not state-dict names.
    original_ids={id(p):key for group,saved in zip(policy.optimizer.param_groups,before_optimizer['param_groups'])
        for p,key in zip(group['params'],saved['params'])}
    for p in actor.values():
        state=before_optimizer['state'].get(original_ids[id(p)])
        if state is None:policy.optimizer.state.pop(p,None)
        else:policy.optimizer.state[p]=copy.deepcopy(state)
    with torch.no_grad():value=float(constraint())
    if not torch.isfinite(torch.tensor(value)) or value>budget:
        # No actor change can repair a pre-existing violation. Restore fully
        # and stop rather than continuing under an unverified constraint.
        policy.load_state_dict(before_weights);policy.optimizer.load_state_dict(before_optimizer)
        return dict(actor_step_accepted=False,actor_step_fraction=0.,backtracks=max_backtracks,
            critic_step_retained=False,constraint_loss=value)
    return dict(actor_step_accepted=False,actor_step_fraction=0.,backtracks=max_backtracks,
        critic_step_retained=True,constraint_loss=value)


class BacktrackingRecoveryPPO(RepairableRecoveryPPO):
    def constrained_optimizer_step(self,loss,observations,masks):
        parameter_partition(self.policy)  # Fail before mutation for shared networks.
        weights={k:v.detach().clone() for k,v in self.policy.state_dict().items()}
        optimizer=copy.deepcopy(self.policy.optimizer.state_dict())
        self.policy.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
        self.policy.optimizer.step()
        result=commit_constrained_candidate(self.policy,weights,optimizer,
            lambda:self.reference_loss(observations,masks)[0],self.reference_kl_budget)
        self.backtracking_steps.append(result)
        return result['critic_step_retained']

    def train(self):
        self.backtracking_steps=[]
        super().train()
        self.last_reference_statistics['backtracking_updates']=self.backtracking_steps
        self.last_reference_statistics['accepted_optimizer_steps_include_critic_only']=True
