"""Closed-loop test of the exported sensor runtime, including its stop guard.

MuJoCo supplies sensors and collision labels only. The portable runtime emits
every command. No simulator pose, actors, or planned path enter that runtime.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT,wrap
from neural_navigation.runtime import SensorNavigationRuntime
from neural_navigation.stress_env import StressNavigationEnv


def run_case(task):
    case,bundle,max_steps,trace_dir=task;torch.set_num_threads(1)
    if case['category']=='continuous_mission':raise ValueError('Continuous goals need a separate protocol')
    scene=ROOT/case.get('scene','assets/room_navigation_scripted')
    env=StressNavigationEnv(case=case,components=bundle,scene_dir=scene,route_mode='shortcut',max_steps=max_steps)
    try:
        options={k:case[k] for k in ['start','goal','initial_prior_offset'] if k in case}
        env.reset(seed=case['seed'],options=options)
        prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0]))
        runtime=SensorNavigationRuntime(bundle,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
        stops=Counter();errors=[];executed_reverse=0.;proposed_reverse=0.;arrived=False;false_arrival=False;trace=[]
        feasible_steps=0;feasible_stops=0;context_times=[]
        for step in range(max_steps):
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,
                                capture_observation=trace_dir is not None)
            if 'action_feasibility' in result:
                feasible=bool(any(result['action_feasibility']))
                feasible_steps+=int(feasible)
                feasible_stops+=int(feasible and result['safety_override'] and np.linalg.norm(result['command'])<1e-8)
                context_times.append(result['safety_context_ms'])
            errors.append(float(np.linalg.norm(runtime.core.estimate[:2]-env.truth[:2])))
            if result['safety_override']:stops[result['stop_reason']]+=1
            if result['model_command'][0]<-.001:proposed_reverse+=env.dt
            cmd=np.asarray(result['command']);executed_reverse+=env.dt if cmd[0]<-.001 else 0.
            if trace_dir:
                trace.append(dict(timestamp=float(env.elapsed),truth=env.truth.tolist(),estimate=runtime.core.estimate.tolist(),
                                  odometry=runtime.core.odom.tolist(),
                                  guard_memory=[dict(point=point.tolist(),last_seen=float(seen))
                                                for point,seen in runtime.guard.memory.entries.values()] if runtime.guard.memory else [],
                                  proposed_action=result['proposed_action'],model_command=result['model_command'],command=result['command'],
                                  policy_observation=result['policy_observation'],
                                  stop_reason=result['stop_reason'],scan=[float(v) if np.isfinite(v) else None for v in env.scan],
                                  actors=[p.tolist() for p in env._humans()],world_min_gap_m=float(env._minimum_gap()),
                                  path=runtime.core.path.tolist(),diagnostics=result['diagnostics'],
                                  safety_context={k:result[k] for k in ['action_feasibility','policy_previous_stop_reason','policy_previous_executed_action','policy_wait_fraction','safety_context_ms'] if k in result},
                                  hard_cells=np.argwhere(runtime.core.laser_fusion.hard).tolist()))
            if result['stop_reason']=='estimated_goal_reached':
                arrived=np.linalg.norm(env.truth[:2]-env.goal)<=.2;false_arrival=not arrived;break
            action=np.array([cmd[0]/(.2 if cmd[0]>=0 else .08),cmd[1]/.5])
            env.step(action)
            if env.collision_events or env.hard_guard_activations:break
        valid=not case.get('obstacle') or env.intervention is not None
        if 'reverse_after_event_s' in case:valid=valid and env.intervention is not None and 'reversal_time_s' in env.intervention
        if 'actor_pause_after_event_s' in case:valid=valid and env.intervention is not None and 'pause_completed_time_s' in env.intervention
        result=dict(case_id=case['id'],arrived=bool(arrived),safe_success=bool(arrived and valid and not env.collision_events and not env.hard_guard_activations),
                    protocol_valid=bool(valid),false_arrival=bool(false_arrival),timeout=bool(not arrived and not false_arrival and not env.collision_events and not env.hard_guard_activations),
                    seconds=float(env.elapsed),travel_m=float(env.travel),collisions=env.collision_events,physical_guards=env.hard_guard_activations,
                    emergency_stop_counts=dict(stops),emergency_override_steps=sum(stops.values()),proposed_reverse_seconds=proposed_reverse,executed_reverse_seconds=executed_reverse,
                    localization_error_p95_m=float(np.quantile(errors,.95)),final_true_goal_distance_m=float(np.linalg.norm(env.truth[:2]-env.goal)),
                    stress_intervention=env.intervention,missing_scan_steps=env.sensor_fault_steps)
        if context_times:
            result.update(feasible_action_steps=feasible_steps,stopped_despite_feasible_steps=feasible_stops,
                          safety_context_p95_ms=float(np.quantile(context_times,.95)))
        if trace_dir:
            trace_dir.mkdir(parents=True,exist_ok=True)
            (trace_dir/(case['id']+'.json')).write_text(json.dumps(dict(case=case,samples=trace,result=result)))
        return result
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=7);p.add_argument('--max-steps',type=int,default=700)
    p.add_argument('--trace-dir',type=Path);a=p.parse_args()
    if a.output.exists():p.error('Use a fresh output')
    manifest=json.loads(a.manifest.read_text());tasks=[(case,a.bundle.resolve(),a.max_steps,a.trace_dir) for case in manifest['cases']]
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(run_case,tasks):results.append(result);print('CASE',json.dumps(result),flush=True)
    report=dict(completed=len(results)==len(tasks),cases=results,safe_success=sum(c['safe_success'] for c in results),tasks=len(tasks),
                collisions=sum(c['collisions'] for c in results),physical_guards=sum(c['physical_guards'] for c in results),
                manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),bundle_manifest=json.loads((a.bundle/'manifest.json').read_text()),
                limitation='Closed-loop simulated sensors with exported runtime and emergency guard. Instantaneous scans and simulated measured velocity; not calibrated hardware, real Orin or final acceptance. Rear clearance remains unknown.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print('SUMMARY',report['safe_success'],len(tasks),flush=True)


if __name__=='__main__':main()
