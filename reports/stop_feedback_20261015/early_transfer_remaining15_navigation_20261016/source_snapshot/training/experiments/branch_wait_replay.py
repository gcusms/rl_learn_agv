"""Single-action interventions on a recorded training failure; diagnostic only.

No branch selects commands using simulator labels. Outcomes use physics only
for scoring. This does not change the actor or deployment behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.masked_navigation import MaskedNavigationCurriculum
from experiments.fixed_training_replay import reset_training_episode
from neural_navigation.masked_actions import COMMANDS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--training-log',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--timestamp',type=float,required=True)
    p.add_argument('--duration',type=float,default=.2)
    p.add_argument('--choice',type=int,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    selected=[r for r in map(json.loads,a.training_log.read_text().splitlines())
              if r.get('step')==a.step and 'training_case_json' in r]
    if len(selected)!=1:p.error('Unique recorded training task required')
    if any(i<0 or i>=len(COMMANDS) for i in a.choice):p.error('Existing choice indices required')
    if not np.isfinite(a.duration) or not .2<=a.duration<=2.4:p.error('Diagnostic duration must be 0.2..2.4 seconds')
    case=json.loads(selected[0]['training_case_json']);torch.set_num_threads(1)
    digest=hashlib.sha256(a.policy.read_bytes()).hexdigest()
    model=MaskablePPO.load(a.policy,device='cpu');a.output.mkdir(parents=True)
    summaries=[]
    for choice in a.choice:
        env=MaskedNavigationCurriculum(a.policy.parent,stage='robust',motion_profiles=True,footprint_clearance_cost=1.)
        intervention=None;changed=0;wait_alternatives=0;only_stop=0;branch_closed=False;branch_frames=0
        try:
            obs,_=reset_training_episode(env,case)
            for step in range(env.max_steps):
                mask=env.action_masks();actor=int(model.predict(obs,action_masks=mask,deterministic=True)[0])
                action=actor
                if intervention is None and env.active.elapsed>=a.timestamp-1e-6:
                    if not mask[choice]:raise ValueError('Branch command rejected by current sensor guard')
                    if actor!=0:raise ValueError('Selected timestamp is not an actor wait')
                    intervention=dict(timestamp=env.active.elapsed,actor_choice=actor,
                        diagnostic_choice=choice,command=COMMANDS[choice].tolist(),
                        masks=mask.tolist(),risk_features=obs[269:282].tolist(),
                        estimate=env.core.estimate.tolist(),measured_velocity=env.core.measured_velocity.tolist())
                if intervention is not None and not branch_closed and env.active.elapsed<intervention['timestamp']+a.duration-1e-6:
                    if mask[choice]:
                        action=choice;branch_frames+=1
                    else:
                        branch_closed=True
                        intervention['discontinued_reason']='command_no_longer_currently_accepted'
                wait_alternatives+=int(action==0 and mask[1:].any())
                only_stop+=int(not mask[1:].any())
                obs,reward,done,truncated,info=env.step(action)
                changed+=int(info['nonzero_choice_changed_by_guard'])
                if done or truncated:break
            if intervention is None:raise ValueError('Episode terminated before branch timestamp')
            result=dict(choice=choice,intervention=intervention,steps=step+1,
                requested_intervention_duration_s=a.duration,intervention_frames=branch_frames,
                seconds=env.active.elapsed,travel_m=env.active.travel,
                safe_success=bool(info['verified_mission_success']),collisions=int(info['collision_events']),
                timeout=bool(truncated),nonzero_choices_changed_by_guard=changed,
                wait_with_alternative_frames=wait_alternatives,only_stop_frames=only_stop)
            summaries.append(result)
            (a.output/f'choice_{choice}.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)
        finally:env.close()
    assert hashlib.sha256(a.policy.read_bytes()).hexdigest()==digest
    (a.output/'SUMMARY.json').write_text(json.dumps(dict(completed=True,kind='training_counterfactual_diagnostic',
        policy_sha256=digest,case=case,results=summaries,deployment_approved=False,
        limitation='One-step interventions on one reused training task; immediate feasibility is not future safety. No deployment controller or training-data change.'),indent=2))


if __name__=='__main__':main()
