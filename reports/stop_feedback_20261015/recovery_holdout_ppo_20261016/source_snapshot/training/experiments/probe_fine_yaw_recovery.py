"""Continuous fine-yaw simulator branches with native per-frame guard checks.

Diagnostic commands are never deployed or used as RL successes/training labels.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.masked_actions import masks_from_observation
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.train_quality_navigation import digest


class FineYawDiagnostic:
    def __init__(self,original,runtime,yaw,frames):
        self.original=original;self.runtime=runtime;self.yaw=yaw;self.frames_left=frames
        self.accepted=0;self.rejected=0;self.used=False;self.original_proposal=None

    def __getattr__(self,name):return getattr(self.original,name)

    def predict(self,prefix):
        proposed=self.original.predict(prefix);self.original_proposal=np.asarray(proposed).copy();self.used=False
        if self.frames_left:
            self.frames_left-=1
            if self.yaw is not None:
                command=np.array([0.,self.yaw]);runtime=self.runtime
                checked,_=copy.deepcopy(runtime.guard).check(runtime.core,command,runtime.core.elapsed,runtime.core.elapsed)
                if np.allclose(checked,command,atol=1e-9,rtol=0):
                    self.accepted+=1;self.used=True;return command/np.array([.2,.5])
                self.rejected+=1
        return proposed


def branch(trace,bundle,target,yaw,intervention_frames,followup_frames=100):
    case=trace['case'];scene=ROOT/case['scene']
    env=StressNavigationEnv(case=case,components=bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    rows=[];wrapper=None
    try:
        env.reset(seed=case['seed'],options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case})
        runtime=TimeConsistentFreshRuntime(bundle,scene/'decision_map.yaml',
            np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0])),case['goal'],'shortcut')
        for index in range(min(700,target+intervention_frames+followup_frames)):
            if index==target:
                wrapper=FineYawDiagnostic(runtime.policy,runtime,yaw,intervention_frames);runtime.policy=wrapper
                start_score=env.truth.copy()
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            if index<=target or yaw is None:
                np.testing.assert_allclose(result['policy_observation'],trace['samples'][index]['policy_observation'],atol=2e-6,rtol=0)
            if index<target or yaw is None:
                np.testing.assert_allclose(result['command'],trace['samples'][index]['command'],atol=2e-8,rtol=0)
                assert result['stop_reason']==trace['samples'][index]['stop_reason']
            if index>=target:
                if wrapper.used and result['stop_reason']!='estimated_goal_reached':
                    np.testing.assert_allclose(result['command'],[0.,yaw],atol=2e-8,rtol=0)
                obs=np.asarray(result['policy_observation'])
                rows.append(dict(timestamp=float(env.elapsed),diagnostic_used=wrapper.used,
                    original_proposal=wrapper.original_proposal.tolist(),command=result['command'],stop_reason=result['stop_reason'],
                    native_feasible_choices=np.flatnonzero(masks_from_observation(obs[:282])).tolist(),
                    prediction_supported=float(obs[294]),waypoint_heading_cosine=float(obs[232]),
                    route_probability=float(obs[229]),remaining_m=float(runtime.core.remaining)))
            if result['stop_reason']=='estimated_goal_reached':break
            v,w=result['command'];env.step(np.array([v/(.2 if v>=0 else .08),w/.5]))
            if env.collision_events or env.hard_guard_activations:break
        return dict(yaw_rad_s=yaw,replayed_frames=target,branch_frames=len(rows),
            interventions=wrapper.accepted,guard_rejections=wrapper.rejected,
            collisions=env.collision_events,physical_guards=env.hard_guard_activations,
            displacement_score_only_m=float(np.linalg.norm(env.truth[:2]-start_score[:2])),
            heading_change_score_only_rad=float(env.truth[2]-start_score[2]),
            goal_distance_score_only_m=float(np.linalg.norm(env.truth[:2]-env.goal)),rows=rows)
    finally:env.close()


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frame',type=int,default=56);p.add_argument('--intervention-frames',type=int,default=105)
    a=p.parse_args()
    if a.output.exists() or a.frame<0 or not 1<=a.intervention_frames<=105:p.error('Fresh output, valid frame and bounded intervention required')
    trace=json.loads(a.trace.read_text())
    if a.frame>=len(trace['samples']) or trace['samples'][a.frame]['choice']!=0:p.error('Recorded wait frame required')
    torch.set_num_threads(1)
    report=dict(completed=False,target_frame=a.frame,intervention_frames=a.intervention_frames,
        trace_sha256=digest(a.trace),bundle_manifest_sha256=digest(a.bundle/'manifest.json'),results=[],deployment_approved=False,
        limitation='Manual simulator diagnostic using native physical guard each frame. No trained actor action-contract change. Small-yaw motor response not calibrated on real hardware. Truth only scores outcomes, not actor/guard input.')
    def save():a.output.write_text(json.dumps(report,indent=2))
    save()
    for yaw in (None,.05,.1,-.1):
        result=branch(trace,a.bundle,a.frame,yaw,a.intervention_frames);report['results'].append(result);save()
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    report['completed']=True;save()


if __name__=='__main__':main()
