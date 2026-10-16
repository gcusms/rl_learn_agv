"""Frozen stochastic policy probes on one recorded training task.

Seeded sampling respects current masks. Simulator labels only score outcomes;
no branch is used as a deployment controller or selected using truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.masked_navigation import MaskedNavigationCurriculum
from experiments.quality_masked_navigation import QualityMaskedWrapper
from experiments.fixed_training_replay import reset_training_episode
from experiments.audit_stop_feedback import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--training-log',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--seed',type=int,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    records=[json.loads(x) for x in a.training_log.read_text().splitlines()]
    selected=[r for r in records if r.get('step')==a.step and 'worker' in r and 'training_case_json' in r]
    if len(selected)!=1:p.error('Unique training episode required')
    case=json.loads(selected[0]['training_case_json']);a.output.mkdir();torch.set_num_threads(1)
    fingerprint=hashlib.sha256(a.policy.read_bytes()).hexdigest();model=MaskablePPO.load(a.policy,device='cpu')
    if model.observation_space.shape!=(300,):raise ValueError('Quality actor required')
    results=[];status=dict(status='running',completed_trials=0,started_unix=time.time())
    def save():(a.output/'status.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        for seed in a.seed:
            base=MaskedNavigationCurriculum(a.policy.parent,stage='robust',motion_profiles=True,footprint_clearance_cost=1.)
            env=QualityMaskedWrapper(base);rows=[]
            try:
                env.prediction.reset();prefix,_=reset_training_episode(base,case)
                obs=env.prediction.augment(prefix,base.core);torch.manual_seed(seed)
                for _ in range(base.max_steps):
                    mask=env.action_masks()
                    with torch.no_grad():probs=model.policy.get_distribution(torch.from_numpy(obs[None]),action_masks=mask[None]).distribution.probs.numpy()[0].copy()
                    choice=int(model.predict(obs,action_masks=mask,deterministic=False)[0])
                    if not mask[choice]:raise ValueError('Masked command selected')
                    row=dict(timestamp=base.active.elapsed,policy_observation=obs.tolist(),mask=mask.tolist(),
                        choice=choice,choice_probabilities=probs.tolist())
                    obs,_,done,truncated,info=env.step(choice)
                    if info['nonzero_choice_changed_by_guard']:raise ValueError('Nonzero command mismatch')
                    row.update(command=info['commanded_velocity'].tolist(),stop_reason=info['execution_guard_reason']);rows.append(row)
                    if done or truncated:break
                feedback=audit(dict(case=case,samples=[dict(r,policy_observation=r['policy_observation'][:282]) for r in rows]))
                if feedback['counts']['feedback_mismatch'] or feedback['counts']['missing']:raise ValueError('Feedback mismatch')
                result=dict(seed=seed,safe_success=bool(info['verified_mission_success']),collisions=int(info['collision_events']),
                    timeout=bool(truncated),seconds=base.active.elapsed,travel_m=base.active.travel,frames=len(rows))
                (a.output/f'trial_{seed}.json').write_text(json.dumps(dict(case=case,result=result,samples=rows,feedback=feedback),allow_nan=False))
                results.append(result);status['completed_trials']=len(results);save();print(json.dumps(result),flush=True)
            finally:env.close()
        if hashlib.sha256(a.policy.read_bytes()).hexdigest()!=fingerprint:raise ValueError('Frozen actor changed')
        (a.output/'SUMMARY.json').write_text(json.dumps(dict(policy_sha256=fingerprint,results=results,
            limitation='Seeded stochastic diagnostic on reused training task. No generalization or deployment claim; no model updates.'),indent=2))
        status.update(status='complete',finished_unix=time.time());save()
    except Exception as error:
        status.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
