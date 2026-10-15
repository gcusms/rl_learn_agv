"""Discarded repair diagnostic on newly generated training-only observations."""
import copy
import json
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO
from experiments.train_quality_navigation import make_environment
from experiments.fixed_training_replay import reset_training_episode
from experiments.time_consistent_tracking import TimeConsistentCore
from experiments.repairable_recovery_ppo import RepairableRecoveryPPO,repair_rank_drift
from experiments.check_recovery_temperature_migration import identical


def main():
    root=Path(__file__).resolve().parents[2]/'training/runs/safety_context_20261011'
    run=root/'recovery_temperature_3000';torch.set_num_threads(1)
    model=RepairableRecoveryPPO.load(run/'candidate/policy.zip',device='cpu')
    initial=MaskablePPO.load(run/'candidate/policy_initial.zip',device='cpu')
    model.reference_policy=copy.deepcopy(initial.policy);model.reference_policy.requires_grad_(False)
    pool=json.loads((root/'masked_velocity_ppo_4000/candidate/failure_pool.json').read_text())
    records=[json.loads(line) for line in (root/'plain_recovery_control_12000/candidate/episodes.jsonl').read_text().splitlines()]
    record=[r for r in records if r.get('step')==10458 and 'worker' in r and 'training_case_json' in r]
    assert len(record)==1;case=json.loads(record[0]['training_case_json']);assert case['stage']=='robust'
    env=make_environment(run/'candidate',pool['templates'],(),0,core_factory=TimeConsistentCore)()
    observations=[];masks=[]
    try:
        prefix,_=reset_training_episode(env.env,case)
        obs=env.prediction.augment(prefix,env.env.core)
        for _ in range(250):
            mask=env.action_masks();observations.append(obs.copy());masks.append(mask.copy())
            choice=model.predict(obs,action_masks=mask,deterministic=True)[0]
            obs,_,done,truncated,info=env.step(choice)
            if done or truncated:break
    finally:env.close()
    observations=torch.as_tensor(__import__('numpy').asarray(observations),dtype=torch.float32)
    masks=torch.as_tensor(__import__('numpy').asarray(masks),dtype=torch.bool)
    keep=model.eligible(observations,masks);observations=observations[keep];masks=masks[keep]
    if not len(observations):raise ValueError('No sampled clear training states')
    before={k:v.clone() for k,v in model.policy.state_dict().items() if 'value' in k}
    adam=copy.deepcopy(model.policy.optimizer.state_dict())
    reference=copy.deepcopy(model.reference_policy.state_dict())
    model.reference_cache=(observations,masks)
    before_loss,_=model.reference_loss(observations,masks)
    repair=repair_rank_drift(model.policy,model.reference_policy,observations,masks)
    after_loss,_=model.reference_loss(observations,masks)
    identical(before,{k:v for k,v in model.policy.state_dict().items() if 'value' in k})
    identical(adam,model.policy.optimizer.state_dict());identical(reference,model.reference_policy.state_dict())
    report=dict(training_step=10458,sampled_clear_states=len(observations),
        constraint_before=float(before_loss.detach()),constraint_after=float(after_loss.detach()),repair=repair,
        critic_unchanged=True,adam_unchanged=True,reference_unchanged=True,modified_policy_saved=False,
        limitation='Fresh replay of a training-log case, no frozen evaluation cases used. Local constraint repair does not prove safe recovery or deployment readiness.')
    output=run/'ACTUAL_TRAINING_STATE_RANK_REPAIR.json'
    if output.exists():raise ValueError('Fresh diagnostic output required')
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
