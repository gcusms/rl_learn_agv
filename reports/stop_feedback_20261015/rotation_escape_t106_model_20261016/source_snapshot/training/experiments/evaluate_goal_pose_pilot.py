"""Frozen paired goal-pose development pilot, with physical arrival labels.

No truth enters the sensor runtime. This is a small diagnostic, not final
acceptance or a claim of unseen-map generalization.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.masked_actions import ACTIONS,COMMANDS,masks_from_observation
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose import contract
from experiments.goal_pose_navigation import GoalPoseRuntime
from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime


def frozen_cases(reference):
    source=json.loads((reference/'cases.json').read_text())['cases']
    old=json.loads((reference/'RESULTS.json').read_text())['cases']
    failures={r['case_id'] for r in old if not r['safe_success']}
    selected=[c for c in source if c['id'] in failures]
    selected += [c for c in source if c['id'] not in failures][:2]
    cases=[dict(c,pilot_category='position_regression',max_steps=700) for c in selected]
    m=StaticMap(ROOT/'assets/room_navigation_scripted/decision_map.yaml',radius=.3)
    cells=np.argwhere(m.clearance>.9);rng=np.random.default_rng(114900)
    for i in range(8):
        while True:
            xy=m.xy(cells[int(rng.integers(len(cells)))])
            yaw=float(rng.uniform(-np.pi,np.pi))
            goal=xy.copy() if i<4 else xy+.7*np.array([np.cos(yaw),np.sin(yaw)])
            if m.lookup(goal,m.clearance,outside=0)>.65 and m.visible(xy,goal):break
        cases.append(dict(id=f'goal_pose_{i:02d}',category='static',
            pilot_category='alignment' if i<4 else 'short_approach',seed=114900+i,
            scene='assets/room_navigation_scripted',start=[*xy,yaw],
            goal=[*goal,float(wrap(yaw+[-np.pi/2,np.pi/2,np.pi,-np.pi/2][i%4]))],
            odom_scale=1.,max_steps=120 if i<4 else 220))
    return cases


def evaluate(case,bundle,budget,legacy=False):
    scene=ROOT/case['scene'];env=StressNavigationEnv(case=case,components=bundle,
        scene_dir=scene,route_mode='shortcut',max_steps=case['max_steps'])
    try:
        options=dict(start=case['start'],goal=case['goal'][:2])
        if 'initial_prior_offset' in case:options['initial_prior_offset']=case['initial_prior_offset']
        env.reset(seed=case['seed'],options=options)
        runtime_class=QualitySensorNavigationRuntime if legacy else GoalPoseRuntime
        prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0]))
        runtime=runtime_class(bundle,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
        labels=[];observations=[];masks=[];choices=[];commands=[];stops=Counter();latencies=[]
        claimed=False;arrived=False;physical_since=None;false_arrival=False
        c=contract();required=len(case['goal'])==3
        for _ in range(case['max_steps']):
            started=time.perf_counter()
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            latencies.append((time.perf_counter()-started)*1000)
            obs=np.asarray(result['policy_observation'],np.float32);mask=masks_from_observation(obs[:282])
            distance=np.max(abs(ACTIONS-np.asarray(result['proposed_action'])),axis=1)
            choice=int(distance.argmin())
            if distance[choice]>2e-6 or not mask[choice]:raise ValueError('Masked actor parity failure')
            if choice and result['stop_reason']!='estimated_goal_reached':
                np.testing.assert_allclose(result['command'],COMMANDS[choice],rtol=0,atol=2e-8)
            xy=float(np.linalg.norm(env.truth[:2]-np.asarray(case['goal'][:2])))
            yaw=float(abs(wrap(case['goal'][2]-env.truth[2]))) if required else 0.
            velocity=env._info('running')['actual_velocity']
            physically_good=(xy<(c['position_tolerance_m'] if required else .2) and
                (not required or (yaw<c['yaw_tolerance_rad'] and
                abs(velocity[0])<c['stopped_linear_m_s'] and abs(velocity[1])<c['stopped_angular_rad_s'])))
            if not physically_good:physical_since=None
            elif physical_since is None:physical_since=env.elapsed
            settled=not required or (physical_since is not None and env.elapsed-physical_since>=c['settle_seconds']-1e-9)
            labels.append([env.elapsed,*env.truth,*runtime.core.estimate,xy,yaw,*velocity])
            observations.append(obs);masks.append(mask);choices.append(choice);commands.append(result['command'])
            stops[str(result['stop_reason'])]+=1
            claimed=result['stop_reason']=='estimated_goal_reached'
            if claimed:
                if physically_good and settled:arrived=True;break
                # A physically correct but unsettled claim continues with zero
                # output until both physical and estimated dwell are satisfied.
                if not physically_good:false_arrival=True;break
            env.step(np.array(result['command'])/[.2,.5]);budget.tick()
            if env.collision_events or env.hard_guard_activations:break
        protocol=not case.get('obstacle') or env.intervention is not None
        if 'reverse_after_event_s' in case:protocol=protocol and 'reversal_time_s' in (env.intervention or {})
        if 'actor_pause_after_event_s' in case:protocol=protocol and 'pause_completed_time_s' in (env.intervention or {})
        labels=np.array(labels);masks=np.array(masks);choices=np.array(choices)
        result=dict(case_id=case['id'],category=case['pilot_category'],
            safe_success=bool(arrived and protocol and not env.collision_events and not env.hard_guard_activations),
            claimed_arrival=claimed,physical_arrival=arrived,false_arrival=false_arrival,protocol_valid=bool(protocol),
            collisions=int(env.collision_events),physical_guards=int(env.hard_guard_activations),
            timeout=bool(not arrived and not false_arrival and not env.collision_events and not env.hard_guard_activations),
            seconds=float(env.elapsed),travel_m=float(env.travel),frames=len(choices),
            final_xy_error_m=float(labels[-1,7]),final_yaw_error_rad=float(labels[-1,8]) if required else None,
            best_yaw_error_rad=float(labels[:,8].min()) if required else None,
            localization_xy_p95_m=float(np.percentile(np.linalg.norm(labels[:,1:3]-labels[:,4:6],axis=1),95)),
            wait_with_motion_frames=int(((choices==0)&masks[:,1:].any(axis=1)).sum()),
            only_stop_frames=int((~masks[:,1:].any(axis=1)).sum()),stop_reasons=dict(stops),
            runtime_ms=dict(p95=float(np.percentile(latencies,95)),p99=float(np.percentile(latencies,99)),maximum=float(max(latencies))))
        trace=dict(labels=labels,observations=np.array(observations),masks=masks,choices=choices,commands=np.array(commands))
        return result,trace
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-cores',type=float,default=.5)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    a.output.mkdir(parents=True);torch.set_num_threads(1);budget=ExperimentBudget(a.cpu_cores,directory=a.output)
    cases=frozen_cases(a.reference);case_file=a.output/'cases.json'
    case_file.write_text(json.dumps(cases,indent=2));rows=[]
    report=dict(completed=False,scope='12 fixed development tasks, not final acceptance; physical MuJoCo labels; heading-only PPO ablation',
        deployment_approved=False,cases_sha256=hashlib.sha256(case_file.read_bytes()).hexdigest(),rows=rows)
    for case in cases:
        variants=[('initial',a.run/'initial_portable',False),('trained',a.run/'portable',False)]
        if case['pilot_category']=='position_regression':variants.insert(0,('legacy',a.legacy,True))
        for variant,bundle,legacy in variants:
            result,trace=evaluate(case,bundle,budget,legacy);result['variant']=variant
            path=a.output/f"{case['id']}_{variant}.npz";np.savez_compressed(path,**trace)
            result['trace_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(result)
            report['resource']=budget.statistics();(a.output/'RESULTS.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(result),flush=True)
    report['completed']=True;report['resource']=budget.statistics()
    report['groups']={category:{variant:dict(tasks=len(subset),success=sum(r['safe_success'] for r in subset),
        collisions=sum(r['collisions'] for r in subset),timeouts=sum(r['timeout'] for r in subset),
        false_arrivals=sum(r['false_arrival'] for r in subset))
        for variant in ('legacy','initial','trained')
        if (subset:=[r for r in rows if r['category']==category and r['variant']==variant])}
        for category in ('position_regression','alignment','short_approach')}
    (a.output/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['groups'],indent=2))


if __name__=='__main__':main()
