"""Compact sensor-only regression on the prior run's fixed development cases."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.masked_actions import ACTIONS,COMMANDS,masks_from_observation
from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime
from experiments.audit_stop_feedback import audit
from experiments.train_quality_navigation import digest


def evaluate(case,bundle):
    if case['category']=='continuous_mission':raise ValueError('Separate continuous-goal protocol required')
    scene=ROOT/case.get('scene','assets/room_navigation_scripted')
    env=StressNavigationEnv(case=case,components=bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    try:
        options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case}
        env.reset(seed=case['seed'],options=options)
        prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0]))
        runtime=QualitySensorNavigationRuntime(bundle,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
        rows=[];errors=[];stops=Counter();arrived=False;false_arrival=False
        for _ in range(700):
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            obs=np.asarray(result['policy_observation'],np.float32);masks=masks_from_observation(obs[:282])
            distances=np.max(np.abs(ACTIONS-np.asarray(result['proposed_action'])),axis=1);choice=int(distances.argmin())
            if distances[choice]>2e-6 or not masks[choice]:raise ValueError('Unknown or rejected actor command')
            if choice and result['stop_reason']!='estimated_goal_reached':
                np.testing.assert_allclose(result['command'],COMMANDS[choice],atol=2e-8,rtol=0)
            rows.append(dict(timestamp=float(env.elapsed),policy_observation=obs.tolist(),choice=choice,
                mask=masks.tolist(),proposed_action=result['proposed_action'],command=result['command'],
                stop_reason=result['stop_reason'],estimate=runtime.core.estimate.tolist()))
            errors.append(float(np.linalg.norm(runtime.core.estimate[:2]-env.truth[:2])))
            if result['safety_override']:stops[result['stop_reason']]+=1
            if result['stop_reason']=='estimated_goal_reached':
                arrived=bool(np.linalg.norm(env.truth[:2]-env.goal)<=.2);false_arrival=not arrived;break
            command=result['command'];env.step(np.array([command[0]/(.2 if command[0]>=0 else .08),command[1]/.5]))
            if env.collision_events or env.hard_guard_activations:break
        valid=not case.get('obstacle') or env.intervention is not None
        if 'reverse_after_event_s' in case:valid=valid and env.intervention is not None and 'reversal_time_s' in env.intervention
        if 'actor_pause_after_event_s' in case:valid=valid and env.intervention is not None and 'pause_completed_time_s' in env.intervention
        feedback=audit(dict(case=case,samples=[dict(r,policy_observation=r['policy_observation'][:282]) for r in rows]))
        if feedback['counts']['missing'] or feedback['counts']['feedback_mismatch']:raise ValueError('Parking feedback mismatch')
        result=dict(case_id=case['id'],safe_success=bool(arrived and valid and not env.collision_events and not env.hard_guard_activations),
            arrived=arrived,protocol_valid=bool(valid),false_arrival=false_arrival,collisions=env.collision_events,
            physical_guards=env.hard_guard_activations,timeout=bool(not arrived and not false_arrival and not env.collision_events and not env.hard_guard_activations),
            seconds=float(env.elapsed),travel_m=float(env.travel),localization_error_p95_m=float(np.quantile(errors,.95)),
            frames=len(rows),voluntary_wait_frames=sum(r['choice']==0 and any(r['mask'][1:]) for r in rows),
            only_stop_frames=sum(not any(r['mask'][1:]) for r in rows),emergency_stop_counts=dict(stops))
        return dict(case=case,samples=rows,result=result,feedback=feedback)
    finally:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    status=json.loads((args.reference_run/'status.json').read_text())
    if status['status']!='complete':parser.error('Completed fixed development reference required')
    cases={};sources={}
    for index in range(len(status['summaries'])):
        report=json.loads((args.reference_run/f'runtime_{index}.json').read_text())
        if not report['completed']:parser.error('Incomplete reference case collection')
        for result in report['cases']:
            path=args.reference_run/f'traces_{index}'/(result['case_id']+'.json')
            data=json.loads(path.read_text());case=data['case']
            if data['result']!=result:parser.error('Reference trace/result mismatch')
            if case['id'] in cases and cases[case['id']]!=case:parser.error('Conflicting fixed case')
            cases[case['id']]=case;sources[str(path)]=digest(path)
    if len(cases)!=48:parser.error('Expected the existing 48-task development regression')
    args.output.mkdir();torch.set_num_threads(1);results=[]
    manifest=json.loads((args.bundle/'manifest.json').read_text())
    report=dict(completed=False,cases=results,tasks=len(cases),reference_case_sha256=sources,
        bundle_manifest=manifest,started_unix=time.time(),deployment_approved=False,
        limitation='48 fixed development cases; final 200-case acceptance remains sealed. Simulated sensor streams, not calibrated real hardware or Orin timing.')
    def save():(args.output/'RESULTS.json').write_text(json.dumps(report,indent=2))
    save()
    for case in cases.values():
        trace=evaluate(case,args.bundle);path=args.output/(case['id']+'.json')
        path.write_text(json.dumps(trace,allow_nan=False));results.append(dict(trace['result'],trace_sha256=digest(path)))
        report.update(safe_success=sum(r['safe_success'] for r in results),collisions=sum(r['collisions'] for r in results),
            timeouts=sum(r['timeout'] for r in results),completed=len(results)==len(cases));save()
        print(json.dumps(trace['result']),flush=True)
    report['finished_unix']=time.time();save()


if __name__=='__main__':main()
