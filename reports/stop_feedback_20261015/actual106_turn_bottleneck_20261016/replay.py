"""Reproduce the completed long-turn failure and separate geometry vetoes.

No physical rollout, training, teacher or relaxed guard is executed. Shadow
geometry checks only attribute a veto to current returns or retained returns.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import ACTIONS, COMMANDS, masks_from_observation


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'reports/stop_feedback_20261015'
SOURCE=BASE/'actual106_confirmed_free_probe_verified_20261016'
BUNDLE=ROOT/'models/stop_feedback_20261015/actual95_transfer_bc400_s231016/bc_portable'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,data):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)


def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def inspect(guard,core,command,points,margins,raw_count):
    initial=np.linalg.norm(points-np.clip(points,-guard.half,guard.half),axis=1)
    rotating=bool(guard.clearance_increasing_rotation and abs(command[0])<1e-8
        and abs(command[1])>.005 and np.all(initial>0))
    escaping=bool((command[0]>.005 or rotating) and np.any(initial<margins))
    required=np.minimum(initial,margins) if escaping else margins
    worst=None;reason=None
    for index,(local,world,horizon) in enumerate(guard._trajectory(core,command,0.)):
        if not guard._static_pose_clear(core,world,command,horizon):
            reason='static_braking_footprint';break
        c,s=np.cos(local[2]),np.sin(local[2])
        body=(points-local[:2])@np.array([[c,-s],[s,c]])
        gap=np.linalg.norm(body-np.clip(body,-guard.half,guard.half),axis=1)
        if len(gap):
            j=int((gap-required).argmin())
            record=dict(sample=index,local_pose=local.tolist(),point_initial=points[j].tolist(),
                point_relative=body[j].tolist(),point_source='current_laser' if j<raw_count else 'memory',
                initial_gap_m=float(initial[j]),gap_m=float(gap[j]),required_m=float(required[j]),
                nominal_margin_m=float(margins[j]),margin_excess_m=float(gap[j]-required[j]))
            if worst is None or record['margin_excess_m']<worst['margin_excess_m']:worst=record
            if np.any(gap<required-1e-4):reason='laser_braking_footprint';break
    if reason is None and escaping and np.any(gap[initial<margins]<initial[initial<margins]+.005):
        reason='escape_not_increasing_clearance'
    actual=guard._trajectory_check(core,command,points,margins,0.,allow_escape=True)
    assert reason==actual,(reason,actual)
    return dict(reason=reason,escaping=escaping,worst_sample=worst)


def main():
    if (OUT/'PLAN.json').exists():raise FileExistsError('Fresh diagnostic output required')
    summary=json.loads((SOURCE/'RESULTS.json').read_text())
    assert summary['completed'] and not any(r['safe_success'] for r in summary['rows'])
    row=next(r for r in summary['rows'] if r['case_id']=='expert_right_3_4')
    trace=SOURCE/'physical/expert_right_3_4.npz'
    assert digest(trace)==row['trace_sha256']
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    paths=[Path(__file__),SOURCE/'RESULTS.json',SOURCE/'PLAN.json',SOURCE/'prefix.json',trace,BUNDLE/'manifest.json']
    paths += [BUNDLE/name for name in manifest['files']]
    for folder in ['navrl','neural_navigation','experiments']:
        paths+=list((ROOT/'training'/folder).rglob('*.py'))
    case=json.loads((SOURCE/'prefix.json').read_text())['case']
    paths += list((ROOT/'training'/case['scene']).glob('*'))
    inputs={str(p.relative_to(ROOT)):digest(p) for p in set(paths) if p.is_file()}
    write('PLAN.json',dict(at=stamp(),inputs=inputs,new_physical_steps=0,new_optimizer_updates=0,
        new_ppo_steps=0,source_failed_probe_not_a_positive_example=True,
        same_source_experimental_memory_activation_s=106.,soft_cpu_cores=.35,
        no_shadow_action_executed=True,goal_achieved=False,deployment_approved=False))
    torch.set_num_threads(1)
    budget=ExperimentBudget(.35,memory_mib=3072,directory=OUT)
    with np.load(trace,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    times=data['labels'][:,0]
    selected={int(np.argmin(abs(times-t))) for t in [110.,111.6,112.,113.4,116.,120.]}
    for choice,start in [(5,110.),(1,113.4)]:
        blocked=np.flatnonzero((times>=start-1e-8)&~data['masks'][:,choice])
        if len(blocked):selected.add(int(blocked[0]))
    scene=ROOT/'training'/case['scene']
    runtime=StopFeedbackRuntime(BUNDLE,scene/'decision_map.yaml',case['start'],case['goal'],'shortcut')
    class RecordedActor:
        cursor=0
        def predict(self,observation):
            i=self.cursor
            np.testing.assert_allclose(observation,data['observations'][i],atol=2e-6,rtol=0)
            choice=int(data['choices'][i])
            assert masks_from_observation(np.asarray(observation)[:282])[choice]
            self.cursor+=1
            return ACTIONS[choice].copy()
    recorded=RecordedActor();runtime.policy.policy=recorded
    records=[]
    for i in range(max(selected)+1):
        if i==530:
            runtime.guard.memory.config=dict(runtime.guard.memory.config,
                free_evidence_margin_m=.03,free_evidence_frames=3,free_evidence_max_gap_s=.4)
            runtime.guard.memory.free_confirmations.clear()
        result=runtime.step(data['scans'][i],data['odometry'][i],data['measured_velocity'][i],
                            times[i],capture_observation=True)
        np.testing.assert_allclose(result['command'],data['commands'][i],atol=2e-8,rtol=0)
        np.testing.assert_allclose(runtime.core.estimate,data['labels'][i,4:7],atol=2e-8,rtol=0)
        if i in selected:
            core,guard=runtime.core,runtime.guard
            valid=np.isfinite(core.scan)&(core.scan>=core.lidar.range_min)&(core.scan<core.lidar.range_max)
            angles=core.lidar.angles[valid]+core.lidar.mount[3]
            raw=core.scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]+core.lidar.mount[:2]
            entries=list(guard.memory.entries.values())
            c,s=np.cos(core.odom[2]),np.sin(core.odom[2])
            memory=(np.array([e[0] for e in entries])-core.odom[:2])@np.array([[c,-s],[s,c]])
            ages=np.maximum(times[i]-np.array([e[1] for e in entries]),0.)
            uncertainty=np.minimum(guard.memory.config.get('max_uncertainty_m',.04),
                ages*guard.memory.config.get('uncertainty_m_s',.002))
            points=np.vstack([raw,memory]);margins=np.r_[np.full(len(raw),.04),.04+uncertainty]
            probes=[]
            for choice in [0,1,3,4,5,6,7,8,9]:
                command=COMMANDS[choice]
                combined=inspect(guard,core,command,points,margins,len(raw))
                raw_only=inspect(guard,core,command,raw,np.full(len(raw),.04),len(raw))
                memory_only=inspect(guard,core,command,memory,.04+uncertainty,0)
                probes.append(dict(choice=choice,command=command.tolist(),legal=bool(data['masks'][i,choice]),
                    original_probe_reason=runtime.context.guard_probe_frame['reasons'][choice],
                    combined=combined,current_laser_same_escape_shadow=raw_only,
                    memory_same_escape_shadow=memory_only))
            records.append(dict(frame=i,time_s=float(times[i]),recorded_command=data['commands'][i].tolist(),
                recorded_choice=int(data['choices'][i]),legal_choices=np.flatnonzero(data['masks'][i]).tolist(),
                probes=probes))
            write('PARTIAL.json',dict(completed=False,reproduced_frames=i+1,records=records))
        if i%50==0:write('STATUS.json',dict(at=stamp(),status='causal_sensor_replay',frames=i+1,target=max(selected)+1))
        budget.tick()
    assert recorded.cursor==max(selected)+1
    for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    write('RESULTS.json',dict(completed=True,at=stamp(),reproduced_frames=recorded.cursor,
        observations_commands_estimates_reproduced=True,records=records,resource=budget.statistics(),
        no_shadow_actions_executed=True,new_physical_steps=0,new_optimizer_updates=0,new_ppo_steps=0,
        limitations=['Shadow geometry only; deleting memory here is never a running command.',
            'Geometric acceptance alone is not complete recovery or safe physical motion.',
            'Source expert trial failed and is not admitted as a positive demonstration.'],
        goal_achieved=False,deployment_approved=False))
    write('STATUS.json',dict(at=stamp(),status='complete',frames=recorded.cursor))


if __name__=='__main__':
    os.chdir(ROOT)
    try:main()
    except Exception as error:
        write('FAILURE.json',dict(at=stamp(),error=repr(error),automatically_restarted=False,
            new_physical_steps=0,goal_achieved=False,deployment_approved=False))
        raise
