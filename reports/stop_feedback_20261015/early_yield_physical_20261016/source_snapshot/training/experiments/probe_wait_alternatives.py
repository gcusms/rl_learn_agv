"""Offline diagnostic branches, never a deployment recovery controller.

Replay a recorded sensor-policy trajectory exactly. At sustained voluntary
waiting, test a currently masked-in action for at most one second, rechecking
the mask every frame, then return control to the original actor. Truth is
scoring only. Results do not become training labels or deployment commands.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.masked_actions import ACTIONS,masks_from_observation
from experiments.fresh_perception_runtime import FreshPerceptionRuntime
from experiments.train_quality_navigation import digest


class DiagnosticPolicy:
    def __init__(self,policy,choice):
        self.policy=policy;self.choice=choice;self.frames_left=5
        self.original_proposal=None;self.interventions=0;self.rejected=0

    def __getattr__(self,name):return getattr(self.policy,name)

    def predict(self,prefix):
        original=self.policy.predict(prefix)
        self.original_proposal=np.asarray(original).copy()
        masks=masks_from_observation(self.policy.last_observation[:282])
        if self.frames_left:
            self.frames_left-=1
            if self.choice is not None and masks[self.choice]:
                self.interventions+=1
                return ACTIONS[self.choice].copy()
            if self.choice is not None:self.rejected+=1
        return original


def run(case,recorded,bundle,target,choice):
    scene=ROOT/case['scene']
    env=StressNavigationEnv(case=case,components=bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    try:
        env.reset(seed=case['seed'],options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case})
        prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0]))
        runtime=FreshPerceptionRuntime(bundle,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
        policy=None;rows=[];start_score=None
        for index in range(min(700,target+105)):
            if index==target:
                policy=DiagnosticPolicy(runtime.policy,choice);runtime.policy=policy
                start_score=env.truth[:2].copy()
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            if index<target:
                old=recorded[index]
                np.testing.assert_allclose(result['policy_observation'],old['policy_observation'],atol=2e-6,rtol=0)
                np.testing.assert_allclose(result['command'],old['command'],atol=2e-8,rtol=0)
                if result['stop_reason']!=old['stop_reason']:raise ValueError('Replay reason differs')
            else:
                obs=np.asarray(result['policy_observation']);mask=masks_from_observation(obs[:282])
                actual=int(np.max(abs(ACTIONS-np.asarray(result['proposed_action'])),axis=1).argmin())
                rows.append(dict(timestamp=float(env.elapsed),diagnostic_proposed_choice=actual,
                    original_actor_choice=int(np.max(abs(ACTIONS-policy.original_proposal),axis=1).argmin()),
                    feasible_choices=np.flatnonzero(mask).tolist(),command=result['command'],
                    stop_reason=result['stop_reason'],pose_supported=float(obs[294]),
                    route_probability=float(obs[229]),waypoint_heading_cosine=float(obs[232]),
                    remaining_m=float(runtime.core.remaining)))
            if result['stop_reason']=='estimated_goal_reached':break
            v,w=result['command'];env.step(np.array([v/(.2 if v>=0 else .08),w/.5]))
            if env.collision_events or env.hard_guard_activations:break
        return dict(diagnostic_choice=choice,replayed_frames=target,branch_frames=len(rows),
            interventions=policy.interventions,mask_rejections=policy.rejected,
            collisions=env.collision_events,physical_guards=env.hard_guard_activations,
            goal_distance_score_only=float(np.linalg.norm(env.truth[:2]-env.goal)),
            displacement_score_only_m=float(np.linalg.norm(env.truth[:2]-start_score)),
            rows=rows)
    finally:env.close()


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    trace=json.loads(args.trace.read_text());streak=0;target=None
    for i,row in enumerate(trace['samples']):
        streak=streak+1 if row['choice']==0 and any(row['mask'][1:]) else 0
        if streak>=10 and row['policy_observation'][294]>.5:
            target=i;break
    if target is None:raise ValueError('No sustained pose-supported voluntary wait')
    choices=[int(x) for x in np.flatnonzero(trace['samples'][target]['mask']) if x]
    if len(choices)>3:raise ValueError('Bounded targeted probe expects at most three alternatives')
    torch.set_num_threads(1)
    report=dict(completed=False,target_frame=target,target_time=trace['samples'][target]['timestamp'],
        trace_sha256=digest(args.trace),bundle_manifest_sha256=digest(args.bundle/'manifest.json'),
        results=[],deployment_approved=False,
        limitation='Diagnostic action interventions in simulation, not RL policy success. Current feasibility is not future safety. No training or truth inputs to actor. Each branch returns to original actor after five frames.')
    def save():args.output.write_text(json.dumps(report,indent=2))
    save()
    for choice in [None]+choices:
        result=run(trace['case'],trace['samples'],args.bundle,target,choice)
        report['results'].append(result);save()
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    report['completed']=True;save()


if __name__=='__main__':main()
