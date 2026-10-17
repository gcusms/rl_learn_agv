"""Replay actual waiting trajectory and audit discounted native reward costs.

Reduced clearance cost is a shadow calculation on identical executed actions,
not a policy improvement or training update. Truth remains reward scoring only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from experiments.masked_navigation import MaskedNavigationCurriculum
from experiments.fixed_training_replay import reset_training_episode
from experiments.fresh_perception_runtime import FreshPerceptionCore
from experiments.quality_masked_navigation import QualityMaskedWrapper
from experiments.train_quality_navigation import digest


def discounted(rewards,gamma):
    output=np.empty(len(rewards));value=0.
    for index in range(len(rewards)-1,-1,-1):
        value=float(rewards[index])+gamma*value;output[index]=value
    return output


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    torch.set_num_threads(1);trace=json.loads(args.trace.read_text());case=trace['case']
    training_case=dict(scene=str(ROOT/case['scene']),seed=case['seed'],stage='robust',
        options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case},interventions=case)
    base=MaskedNavigationCurriculum(args.bundle,stage='robust',route_mode='shortcut',motion_profiles=True,
        footprint_clearance_cost=1.,core_factory=FreshPerceptionCore)
    env=QualityMaskedWrapper(base);rows=[];streak=0;wait_start=None
    try:
        prefix,_=reset_training_episode(base,training_case);env.prediction.reset()
        obs=env.prediction.augment(prefix,base.core)
        for index,row in enumerate(trace['samples']):
            np.testing.assert_allclose(obs,row['policy_observation'],atol=2e-6,rtol=0)
            np.testing.assert_array_equal(env.action_masks(),row['mask'])
            streak=streak+1 if row['choice']==0 and any(row['mask'][1:]) else 0
            if streak==10 and wait_start is None:wait_start=index-9
            obs,reward,done,truncated,info=env.step(row['choice'])
            np.testing.assert_allclose(info['commanded_velocity'],row['command'],atol=2e-8,rtol=0)
            if info['execution_guard_reason']!=row['stop_reason']:raise ValueError('Parking reason replay differs')
            delta=.8*info['footprint_clearance_penalty']
            shadow=min(reward+delta,-100.) if info['safety_termination'] else reward+delta
            rows.append(dict(timestamp=row['timestamp'],native_reward=reward,shadow_clearance_0_2_reward=shadow,
                body_gap_m=info['measured_body_gap_m'],body_cost=info['footprint_clearance_penalty']))
            if done or truncated:
                if index!=len(trace['samples'])-1:raise ValueError('Replay ended earlier than trace')
                break
        if len(rows)!=len(trace['samples']) or wait_start is None:raise ValueError('Complete sustained-wait trace required')
        gamma=.995
        old=discounted([r['native_reward'] for r in rows],gamma)
        new=discounted([r['shadow_clearance_0_2_reward'] for r in rows],gamma)
        report=dict(reproduced_observation_mask_command_and_reason=True,frames=len(rows),gamma=gamma,
            trace_sha256=digest(args.trace),wait_start_frame=wait_start,
            native_return_from_wait=float(old[wait_start]),shadow_return_from_wait=float(new[wait_start]),
            immediate_collision_terminal_penalty=-100.,rows=rows,deployment_approved=False,
            limitation='Identical observed waiting trajectory and shadow reward coefficient only. Collision comparison excludes the trajectory required to cause a collision. No improvement or causal training attribution. Gamma .995 matches current training checkpoint.')
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
