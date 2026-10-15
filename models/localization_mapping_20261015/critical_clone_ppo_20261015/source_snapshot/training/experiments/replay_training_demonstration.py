"""Reproduce recorded training actions to capture native rewards/returns.

This is offline simulator data collection, never a deployment controller.
Truth-based simulator rewards do not enter actor observations or masks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.masked_navigation import MaskedNavigationCurriculum
from experiments.quality_masked_navigation import QualityMaskedWrapper
from experiments.fixed_training_replay import reset_training_episode


def discounted_returns(rewards,gamma):
    rewards=np.asarray(rewards,float)
    if rewards.ndim!=1 or not np.isfinite(rewards).all() or not np.isfinite(gamma) or not 0<gamma<=1:
        raise ValueError('Finite rewards and discount in (0,1] required')
    output=np.zeros(len(rewards));value=0.
    for i in range(len(rewards)-1,-1,-1):value=float(rewards[i])+gamma*value;output[i]=value
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True);p.add_argument('--training-log',type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gamma',type=float,default=.995)
    p.add_argument('--allow-failed-training-episode',action='store_true');a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    trace=json.loads(a.trace.read_text());result=trace['result']
    if result['timeout']:p.error('Complete terminal episode required; truncated returns need a separate treatment')
    if not a.allow_failed_training_episode and (not result['safe_success'] or result['collisions']):
        p.error('Verified safe training trace required unless failure collection explicitly enabled')
    records=[json.loads(x) for x in a.training_log.read_text().splitlines()]
    matches=[r for r in records if r.get('step')==a.step and 'worker' in r and 'training_case_json' in r]
    if len(matches)!=1 or json.loads(matches[0]['training_case_json'])!=trace['case']:
        p.error('Trace must match the original recorded training task')
    torch.set_num_threads(1);base=MaskedNavigationCurriculum(a.components,stage='robust',motion_profiles=True,footprint_clearance_cost=1.)
    env=QualityMaskedWrapper(base);rows=[];rewards=[]
    try:
        env.prediction.reset();prefix,_=reset_training_episode(base,trace['case']);obs=env.prediction.augment(prefix,base.core)
        for index,recorded in enumerate(trace['samples']):
            np.testing.assert_allclose(obs,recorded['policy_observation'],atol=2e-6,rtol=0)
            if abs(base.active.elapsed-recorded['timestamp'])>1e-8:raise ValueError('Replay timestamp diverged')
            masks=env.action_masks();np.testing.assert_array_equal(masks,recorded['mask']);choice=recorded['choice']
            if not masks[choice]:raise ValueError('Demonstration action is not accepted')
            row=dict(observation=obs.tolist(),action=choice,action_mask=masks.tolist())
            obs,reward,done,truncated,info=env.step(choice)
            np.testing.assert_allclose(info['commanded_velocity'],recorded['command'],atol=2e-8,rtol=0)
            if info['execution_guard_reason']!=recorded['stop_reason']:raise ValueError('Stop feedback diverged')
            row.update(reward=float(reward),terminated=bool(done),truncated=bool(truncated));rows.append(row);rewards.append(reward)
            if done or truncated:
                if index!=len(trace['samples'])-1:raise ValueError('Replay ended early')
                break
        if not done or truncated or bool(info['verified_mission_success'])!=result['safe_success'] or int(info['collision_events'])!=result['collisions']:
            raise ValueError('Original terminal training outcome did not reproduce')
        returns=discounted_returns(rewards,a.gamma)
        for row,value in zip(rows,returns):row['return']=float(value)
        kind='reproduced_safe_training_demonstration' if result['safe_success'] else 'reproduced_failed_training_episode'
        a.output.write_text(json.dumps(dict(kind=kind,case=trace['case'],
            trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
            training_log_sha256=hashlib.sha256(a.training_log.read_bytes()).hexdigest(),training_step=a.step,
            gamma=a.gamma,observations_and_masks_reproduced=True,rewards_source='native training environment',
            transitions=rows,safe_success=result['safe_success'],collisions=result['collisions'],timeout=False,
            limitation='One reused training task, not generalization. Recorded actions only replayed offline; no controller/model changed.'),allow_nan=False))
        print(json.dumps(dict(reproduced=True,transitions=len(rows),seconds=base.active.elapsed,total_reward=sum(rewards),initial_return=float(returns[0]))))
    finally:env.close()


if __name__=='__main__':main()
