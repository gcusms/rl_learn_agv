"""Frozen development evaluation of learned terminal actor and localization."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.masked_actions import ACTIONS,COMMANDS,masks_from_observation
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose import contract
from experiments.docking_navigation import DockingRuntime
from experiments.goal_pose_navigation import GoalPoseRuntime


def additional_cases():
    cases=[];rng=np.random.default_rng(114980)
    for i in range(16):
        scene=ROOT/f'assets/navigation_stage2_dataset/train_{1+i%5:02d}'
        m=StaticMap(scene/'decision_map.yaml',radius=.3);cells=np.argwhere(m.clearance>.8)
        while True:
            xy=m.xy(cells[int(rng.integers(len(cells)))])
            yaw=float(rng.uniform(-np.pi,np.pi));distance=0. if i<8 else float(rng.uniform(.3,.7))
            goal=xy+distance*np.array([np.cos(yaw),np.sin(yaw)])
            if m.lookup(goal,m.clearance,outside=0)>.65 and m.visible(xy,goal):break
        offset=[0.,.12,-.12,.5,-.5,np.pi/2,-np.pi/2,np.pi][i%8]
        case=dict(id=f'docking_dev_{i:02d}',category='static',pilot_category='new_alignment' if i<8 else 'new_approach',
            seed=114980+i,scene=str(scene.relative_to(ROOT)),start=[*xy,yaw],
            goal=[*goal,float(wrap(yaw+offset))],odom_scale=1.,max_steps=150)
        if i in (6,14):case['extra_scan_dropout']=.08
        if i in (7,15):case['dropout_windows']=[[1.,1.4]]
        cases.append(case)
    return cases


def evaluate(case,bundle,budget):
    from collections import Counter
    import time
    scene=ROOT/case['scene'];env=StressNavigationEnv(case=case,components=bundle,
        scene_dir=scene,route_mode='shortcut',max_steps=case['max_steps'])
    try:
        options=dict(start=case['start'],goal=case['goal'][:2])
        if 'initial_prior_offset' in case:options['initial_prior_offset']=case['initial_prior_offset']
        env.reset(seed=case['seed'],options=options)
        from experiments.command_odometry_input import make_sensor_odometry,advance_sensor_odometry
        command_input=make_sensor_odometry(case)
        manifest=json.loads((bundle/'manifest.json').read_text())
        from neural_navigation.goal_pose_runtime import resolve_runtime
        runtime_name=manifest.get('required_runtime')
        # The archived 300D quality actor predates the runtime-name field.
        # Identify it by its exact feature contract, never assume 303D input.
        if runtime_name is None and manifest.get('requires_quality_feature_generator') and manifest.get('observation_contract',{}).get('schema')=='quality_prediction_append_v1':
            runtime_name='experiments.quality_sensor_runtime.QualitySensorNavigationRuntime'
        runtime_class=resolve_runtime(runtime_name)
        if runtime_class is None:
            if runtime_name!='experiments.quality_sensor_runtime.QualitySensorNavigationRuntime':
                raise ValueError('Unsupported evaluation runtime')
            from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime
            runtime_class=QualitySensorNavigationRuntime
        prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0]))
        runtime=runtime_class(bundle,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
        labels=[];observations=[];masks=[];choices=[];commands=[];scans=[];humans=[];stops=Counter();latencies=[]
        odometry=[];velocities=[];route_diagnostics=[];route_reasons=[];constraint_diagnostics=[]
        arrived=False;false_arrival=False;physical_since=None;c=contract();required=len(case['goal'])==3
        parity_frames=0;last_progress=time.monotonic()
        for step_index in range(case['max_steps']):
            start=time.perf_counter()
            odom=env.observed_odometry if command_input is None else command_input.pose
            velocity_input=env.measured_velocity if command_input is None else command_input.velocity
            result=runtime.step(env.scan,odom,velocity_input,env.elapsed,capture_observation=True)
            latencies.append((time.perf_counter()-start)*1000)
            obs=np.asarray(result['policy_observation'],np.float32);mask=masks_from_observation(obs[:282])
            distances=np.max(abs(ACTIONS-np.asarray(result['proposed_action'])),axis=1);choice=int(distances.argmin())
            if distances[choice]>2e-6 or not mask[choice]:raise ValueError('Portable action/mask mismatch')
            if choice and result['stop_reason']!='estimated_goal_reached':np.testing.assert_allclose(result['command'],COMMANDS[choice],rtol=0,atol=2e-8)
            xy=float(np.linalg.norm(env.truth[:2]-case['goal'][:2]));yaw=float(abs(wrap(case['goal'][2]-env.truth[2]))) if required else 0.
            velocity=env._info('running')['actual_velocity']
            good=xy<(c['position_tolerance_m'] if required else .2) and (not required or (
                yaw<c['yaw_tolerance_rad'] and abs(velocity[0])<c['stopped_linear_m_s'] and abs(velocity[1])<c['stopped_angular_rad_s']))
            if not good:physical_since=None
            elif physical_since is None:physical_since=env.elapsed
            settled=not required or physical_since is not None and env.elapsed-physical_since>=c['settle_seconds']-1e-9
            labels.append([env.elapsed,*env.truth,*runtime.core.estimate,xy,yaw,*velocity])
            observations.append(obs);masks.append(mask);choices.append(choice);commands.append(result['command']);stops[str(result['stop_reason'])]+=1
            scans.append(env.scan.copy());humans.append(env._humans().copy())
            # Record the actual runtime inputs, not simulator truth substituted
            # for odometry. This permits causal sensor replay after a failure.
            odometry.append(np.asarray(odom).copy());velocities.append(np.asarray(velocity_input).copy())
            core=runtime.core;loc=core.localizer
            route_diagnostics.append([core.plan_revision,len(core.path),
                *core.path[-1],float(getattr(loc,'observable_rank',-1)),
                float(loc.map_agreement),*np.asarray(loc.std)])
            route_reasons.append(core.replan_reason)
            constraint=result.get('predictive_constraint',{})
            constraint_diagnostics.append([constraint.get('supported_tracks',0),constraint.get('rejected_nonzero_choices',0),
                constraint.get('compute_ms',0.),*[np.nan if v is None else v for v in constraint.get('predicted_contact_seconds',[None]*12)]])
            if time.monotonic()-last_progress>=30.:
                print(json.dumps(dict(evaluation_progress=True,case_id=case['id'],bundle=str(bundle),
                    frame=step_index+1,max_frames=case['max_steps'],simulation_seconds=float(env.elapsed),
                    travel_m=float(env.travel),nonzero_choices=int(mask[1:].sum()),
                    stop_reason=result['stop_reason'])),flush=True)
                last_progress=time.monotonic()
            if result['stop_reason']=='estimated_goal_reached':
                if good and settled:arrived=True;break
                if not good:false_arrival=True;break
            previous_measured_yaw=env.observed_odometry[2]
            env.step(np.asarray(result['command'])/[.2,.5]);budget.tick()
            if command_input is not None:advance_sensor_odometry(command_input,result['command'],env,previous_measured_yaw,case)
            if env.collision_events or env.hard_guard_activations:break
        protocol=not case.get('obstacle') or env.intervention is not None
        if 'reverse_after_event_s' in case:protocol=protocol and 'reversal_time_s' in (env.intervention or {})
        if 'actor_pause_after_event_s' in case:protocol=protocol and 'pause_completed_time_s' in (env.intervention or {})
        labels=np.array(labels);masks=np.array(masks);choices=np.array(choices)
        result=dict(case_id=case['id'],category=case['pilot_category'],safe_success=bool(arrived and protocol and not env.collision_events and not env.hard_guard_activations),
            physical_arrival=arrived,false_arrival=false_arrival,protocol_valid=bool(protocol),collisions=int(env.collision_events),
            input_odometry=case.get('input_odometry','noisy_motion_feedback'),
            physical_guards=int(env.hard_guard_activations),timeout=bool(not arrived and not false_arrival and not env.collision_events and not env.hard_guard_activations),
            seconds=float(env.elapsed),travel_m=float(env.travel),frames=len(choices),
            final_xy_error_m=float(labels[-1,7]),final_yaw_error_rad=float(labels[-1,8]) if required else None,
            localization_xy_p95_m=float(np.percentile(np.linalg.norm(labels[:,1:3]-labels[:,4:6],axis=1),95)),
            localization_yaw_p95_rad=float(np.percentile(abs(wrap(labels[:,3]-labels[:,6])),95)),
            stop_choices=int((choices==0).sum()),only_stop_frames=int((~masks[:,1:].any(axis=1)).sum()),stop_reasons=dict(stops),
            voluntary_wait_frames=int(((choices==0)&masks[:,1:].any(axis=1)).sum()),
            predictive_constraint_frames=int(np.count_nonzero(np.asarray(constraint_diagnostics)[:,1])),
            runtime_ms=dict(p95=float(np.percentile(latencies,95)),p99=float(np.percentile(latencies,99)),maximum=float(max(latencies))))
        return result,dict(labels=labels,observations=np.array(observations),masks=masks,choices=choices,commands=np.array(commands),
            scans=np.array(scans),human_positions=np.array(humans),odometry=np.array(odometry),
            measured_velocity=np.array(velocities),route_diagnostics=np.array(route_diagnostics),route_reasons=np.array(route_reasons),
            constraint_diagnostics=np.array(constraint_diagnostics))
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path);p.add_argument('--bc',type=Path);p.add_argument('--localized',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=Path)
    p.add_argument('--cpu-cores',type=float,default=.5);p.add_argument('--additional',action='store_true')
    p.add_argument('--candidate-name',default='trained');p.add_argument('--baseline-name',default='previous_3k')
    p.add_argument('--localized-name',default='trained_v4_localization')
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    a.output.mkdir(parents=True);torch.set_num_threads(1);budget=ExperimentBudget(a.cpu_cores,directory=a.output)
    from experiments.experiment_provenance import record_provenance
    record_provenance(a.output,a.candidate)
    cases=json.loads(a.cases.read_text()) if a.cases else additional_cases()
    if a.additional:cases+=additional_cases()
    case_file=a.output/'cases.json';case_file.write_text(json.dumps(cases,indent=2))
    report=dict(completed=False,scope='fixed development tasks; independent physical labels; not real deployment acceptance',
        deployment_approved=False,cases_sha256=hashlib.sha256(case_file.read_bytes()).hexdigest(),rows=[])
    for case in cases:
        variants=[(a.candidate_name,a.candidate)]
        if a.baseline:variants.insert(0,(a.baseline_name,a.baseline))
        if a.bc and case['pilot_category']!='position_regression':variants.insert(0,('bc_only',a.bc))
        if a.localized:variants.append((a.localized_name,a.localized))
        for variant,bundle in variants:
            result,trace=evaluate(case,bundle,budget);result['variant']=variant
            trace_path=a.output/f"{case['id']}_{variant}.npz";np.savez_compressed(trace_path,**trace)
            result['trace_sha256']=hashlib.sha256(trace_path.read_bytes()).hexdigest();report['rows'].append(result)
            report['resource']=budget.statistics();(a.output/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(result),flush=True)
    report['completed']=True;report['groups']={}
    for category in sorted(set(r['category'] for r in report['rows'])):
        report['groups'][category]={}
        for variant in sorted(set(r['variant'] for r in report['rows'])):
            rows=[r for r in report['rows'] if r['category']==category and r['variant']==variant]
            if rows:report['groups'][category][variant]=dict(tasks=len(rows),success=sum(r['safe_success'] for r in rows),
                collisions=sum(r['collisions'] for r in rows),timeouts=sum(r['timeout'] for r in rows),false_arrivals=sum(r['false_arrival'] for r in rows))
    (a.output/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['groups'],indent=2))


if __name__=='__main__':main()
