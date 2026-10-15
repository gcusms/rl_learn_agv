"""Probe a finer current-command grid at actual masked training-replay stops.

Read-only cloned guards, no simulator truth, no command replacement or model
training. A finite probe acceptance is not a future dynamic safety guarantee.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.masked_navigation import MaskedNavigationCurriculum
from experiments.fixed_training_replay import reset_training_episode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--training-log',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    records=[json.loads(line) for line in args.training_log.read_text().splitlines()]
    selected=[x for x in records if x.get('step')==args.step and 'worker' in x and 'training_case_json' in x]
    if len(selected)!=1:parser.error('Unique recorded training task required')
    torch.set_num_threads(1);case=json.loads(selected[0]['training_case_json'])
    model=MaskablePPO.load(args.policy,device='cpu')
    env=MaskedNavigationCurriculum(args.policy.parent,stage='robust',motion_profiles=True,footprint_clearance_cost=1.)
    commands=np.array([[v,w] for v in (0.,.01,.03,.05,.08,.12,.16,.2)
        for w in (-.5,-.35,-.2,0.,.2,.35,.5) if v!=0 or w!=0])
    samples=[];last=-100.;fingerprint=hashlib.sha256(args.policy.read_bytes()).hexdigest()
    try:
        obs,_=reset_training_episode(env,case)
        for _ in range(env.max_steps):
            masks=env.action_masks();choice=int(model.predict(obs,action_masks=masks,deterministic=True)[0])
            if not masks[1:].any() and env.active.elapsed-last>=2. and len(samples)<8:
                accepted=[];without_memory=[];reason_counts={}
                class EmptyMemory:
                    def update(self,unused):return np.zeros((0,2)),np.zeros(0)
                    def reset(self):pass
                for command in commands:
                    guard=copy.deepcopy(env.guard)
                    executed,reason=guard.check(env.core,command,env.active.elapsed,env.active.elapsed)
                    if np.allclose(executed,command,atol=1e-9,rtol=0):accepted.append(command.tolist())
                    else:reason_counts[reason]=reason_counts.get(reason,0)+1
                    # Preserve all preflight/escape rules, remove only the
                    # remembered endpoints in a shadow guard. Never execute.
                    shadow=copy.deepcopy(env.guard);shadow.memory=EmptyMemory()
                    executed,_=shadow.check(env.core,command,env.active.elapsed,env.active.elapsed)
                    if np.allclose(executed,command,atol=1e-9,rtol=0):without_memory.append(command.tolist())
                valid=np.isfinite(env.core.scan)&(env.core.scan>=env.core.lidar.range_min)&(env.core.scan<env.core.lidar.range_max)
                mount=env.core.lidar.mount;angle=env.core.lidar.angles[valid]+mount[3]
                points=env.core.scan[valid,None]*np.c_[np.cos(angle),np.sin(angle)]+mount[:2]
                gaps=np.linalg.norm(points-np.clip(points,-env.guard.half,env.guard.half),axis=1)
                samples.append(dict(timestamp=env.active.elapsed,estimated_pose=env.core.estimate.tolist(),
                    measured_velocity=env.core.measured_velocity.tolist(),
                    current_scan_body_gap_m=float(gaps.min()) if len(gaps) else None,
                    original_nonzero_choices=int(masks[1:].sum()),finer_accepted_commands=accepted,
                    shadow_without_memory_accepted_commands=without_memory,
                    rejected_reason_counts=reason_counts))
                last=env.active.elapsed
            obs,_,done,truncated,info=env.step(choice)
            if done or truncated:break
        if hashlib.sha256(args.policy.read_bytes()).hexdigest()!=fingerprint:
            raise ValueError('Frozen actor changed during audit')
        result=dict(case=case,policy_sha256=fingerprint,probed_commands=commands.tolist(),
            snapshots=samples,original_grid_empty_but_finer_grid_nonempty=sum(bool(x['finer_accepted_commands']) for x in samples),
            outcome=dict(safe_success=bool(info['verified_mission_success']),collisions=int(info['collision_events']),timeout=bool(truncated)),
            actor_inputs_and_probes='native scan/estimated-pose core and cloned guard only',
            commands_replaced=False,limitation='Sampled current-command feasibility only; no future safety certificate or policy improvement claim.')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(snapshots=len(samples),missed_alternative_snapshots=result['original_grid_empty_but_finer_grid_nonempty'],outcome=result['outcome'])))
    finally:env.close()


if __name__=='__main__':main()
