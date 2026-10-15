"""Unknown-map MuJoCo navigation through measured scan/odom SLAM.

The simulator map is used solely to construct physics and task fixtures. It is
never passed to SLAM, the actor, the guard or the frontier selector. Bootstrap
is an explicitly recorded, supervised sensor sweep, not learned cold start.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose import contract
from neural_navigation.online_mapping_runtime import OnlineMappingRuntime
from experiments.probe_online_slam import relative_pose,OdometryProfile
from experiments.slam_toolbox_backend import SlamBackend


def fixtures(count,exploration=False,dynamic=False,seed_base=115710,scenes=None):
    cases=[]
    for i in range(count):
        scene=ROOT/scenes[i%len(scenes)] if scenes else ROOT/f'assets/navigation_stage2_dataset/validation_{i%2:02d}'
        m=StaticMap(scene/'decision_map.yaml',radius=.30);rng=np.random.default_rng(seed_base+i)
        cells=np.argwhere(m.clearance>1.2)
        for _ in range(500):
            xy=m.xy(cells[int(rng.integers(len(cells)))])
            heading=float(rng.uniform(-np.pi,np.pi));length=[1.,2.,3.][i%3]
            goal=m.xy(cells[int(rng.integers(len(cells)))]) if exploration and i%2 else xy+length*np.array([np.cos(heading),np.sin(heading)])
            if not m.lookup(goal,m.clearance,outside=0)>.7:continue
            if exploration and i%2:
                if m.visible(xy,goal) or np.linalg.norm(goal-xy)<2.:continue
                try:path=m.plan(xy,goal)
                except ValueError:continue
                if np.linalg.norm(np.diff(path,axis=0),axis=1).sum()>8.:continue
            elif not m.visible(xy,goal):continue
            break
        else:raise RuntimeError('Fixture sampling failed')
        case=dict(id=f'online_map_{i:02d}',seed=seed_base+i,scene=str(scene.relative_to(ROOT)),
            start=[*xy,heading],goal=[*goal,heading+(.5 if i%2 else -.5)],
            odom_scale=1.,extra_scan_dropout=.08,sensor_max_range_m=10.,max_steps=500 if exploration else 350,
            task_kind='occluded_goal_exploration' if exploration and i%2 else 'visible_goal')
        if dynamic and i%3==1:case.update(obstacle='crossing',obstacle_speed_m_s=.08,trigger_travel_m=.4)
        cases.append(case)
    return cases


def evaluate(case,bundle,output,budget,domain,rf2o_executable=None,laser_increments=False,ray_free_layer=False,coarse_subgoals=False,known_free_dynamics=False,distance_prior_variance=None):
    output.mkdir(parents=True,exist_ok=False)
    env=StressNavigationEnv(case=case,components=bundle,scene_dir=ROOT/case['scene'],
        route_mode='shortcut',max_steps=case['max_steps']+90)
    backend=None;trace=[];stops=Counter();latencies=[];physical_since=None;arrived=False;false_arrival=False
    from experiments.command_odometry_input import make_sensor_odometry,advance_sensor_odometry
    command_input=make_sensor_odometry(case)
    try:
        env.reset(seed=case['seed'],options=dict(start=case['start'],goal=case['goal'][:2]))
        reference=env.truth.copy();odom_reference=env.observed_odometry.copy();profile=OdometryProfile(odom_reference)
        goal=relative_pose(np.asarray(case['goal']),reference)
        sensor_lidar=copy.copy(env.lidar);sensor_lidar.range_max=case.get('sensor_max_range_m',env.lidar.range_max)
        backend=SlamBackend(sensor_lidar,output/'slam',domain=domain,rf2o_executable=rf2o_executable,laser_increments=laser_increments,distance_prior_variance=distance_prior_variance)
        def sensors():
            odom=profile.update(env.observed_odometry,env.elapsed) if command_input is None else command_input.pose
            estimate=backend.push(env.scan,odom,env.elapsed)
            return backend.last_odometry.copy(),estimate
        # Supervised 360-degree sweep validates initialization separately from
        # the retained policy; its motion and any collision count are recorded.
        sweep=[]
        for _ in range(66):
            odom,estimate=sensors();sweep.append(dict(seconds=env.elapsed,pose=relative_pose(env.truth,reference).tolist()))
            previous_measured_yaw=env.observed_odometry[2]
            env.step(np.array([0.,1.]));budget.tick()
            if command_input is not None:advance_sensor_odometry(command_input,[0.,.5],env,previous_measured_yaw,case)
            if env.collision_events or env.hard_guard_activations:raise RuntimeError('Bootstrap sweep not physically safe')
        for _ in range(5):
            previous_measured_yaw=env.observed_odometry[2]
            env.step(np.array([0.,0.]))
            if command_input is not None:advance_sensor_odometry(command_input,[0.,0.],env,previous_measured_yaw,case)
            odom,estimate=sensors();budget.tick()
        backend.spin_for(.5);backend.save();observed=backend.current_map()
        estimate=backend.estimated_pose(odom)
        (output/'initialization.json').write_text(json.dumps(dict(estimate=None if estimate is None else estimate.tolist(),
            goal=goal.tolist(),map_version=backend.map_version,pose_stamp=backend.pose_metadata()[0],
            pose_free=bool(observed.lookup(estimate[:2],observed.free,outside=False)) if observed is not None and estimate is not None else False),indent=2))
        if observed is None or estimate is None:raise RuntimeError('No observed map and pose after bootstrap')
        initial_known=int((~observed.blocked).sum())
        runtime=OnlineMappingRuntime(bundle,observed,estimate,goal,output/'slam/map.yaml',ray_free_layer=ray_free_layer,coarse_subgoals=coarse_subgoals,known_free_dynamics=known_free_dynamics)
        start_time=env.elapsed;c=contract()
        for step in range(case['max_steps']):
            odom,estimate=sensors();stamp,std=backend.pose_metadata()
            observed=backend.current_map()
            if estimate is None or stamp is None:result=dict(command=[0.,0.],stop_reason='waiting_slam',task_goal_achieved=False)
            else:
                started=time.perf_counter()
                velocity_input=env.measured_velocity if command_input is None else command_input.velocity
                result=runtime.step(env.scan,odom,velocity_input,env.elapsed,observed,estimate,std,pose_timestamp=stamp)
                latencies.append((time.perf_counter()-started)*1000)
            truth=relative_pose(env.truth,reference);xy=float(np.linalg.norm(truth[:2]-goal[:2]));yaw=float(abs(wrap(truth[2]-goal[2])))
            velocity=env._info('running')['actual_velocity']
            good=(xy<c['position_tolerance_m'] and yaw<c['yaw_tolerance_rad']
                and abs(velocity[0])<c['stopped_linear_m_s'] and abs(velocity[1])<c['stopped_angular_rad_s'])
            if not good:physical_since=None
            elif physical_since is None:physical_since=env.elapsed
            settled=physical_since is not None and env.elapsed-physical_since>=c['settle_seconds']-1e-9
            stops[str(result['stop_reason'])]+=1
            trace.append(dict(seconds=env.elapsed-start_time,truth=truth.tolist(),estimate=runtime.core.estimate.tolist(),
                odometry=odom.tolist(),goal=goal.tolist(),command=np.asarray(result['command']).tolist(),
                subgoal=runtime.core.goal.tolist(),stop_reason=result['stop_reason'],
                localization=runtime.core.diagnostics(),map_version=backend.map_version,xy_error=xy,yaw_error=yaw,
                scan=[float(v) if np.isfinite(v) else None for v in env.scan],
                humans=[relative_pose(np.r_[xy,0.],reference)[:2].tolist() for xy in env._humans()]))
            if result.get('task_goal_achieved'):
                arrived=bool(good and settled);false_arrival=not good
                if arrived or false_arrival:break
            previous_measured_yaw=env.observed_odometry[2]
            env.step(np.asarray(result['command'])/[.2,.5]);budget.tick()
            if command_input is not None:advance_sensor_odometry(command_input,result['command'],env,previous_measured_yaw,case)
            if env.collision_events or env.hard_guard_activations:break
            if step%20==0:(output/'progress.json').write_text(json.dumps(dict(steps=step+1,total=case['max_steps'],stop_reasons=dict(stops),**budget.statistics()),indent=2))
        saved=backend.save();truths=np.array([r['truth'] for r in trace]);ests=np.array([r['estimate'] for r in trace])
        result=dict(case_id=case['id'],safe_success=bool(arrived and not env.collision_events and not env.hard_guard_activations),
            collisions=int(env.collision_events),physical_guards=int(env.hard_guard_activations),false_arrival=false_arrival,
            timeout=not arrived and not false_arrival and not env.collision_events,seconds=env.elapsed-start_time,
            travel_including_bootstrap_m=float(env.travel),bootstrap_frames=len(sweep),frames=len(trace),
            final_xy_error_m=trace[-1]['xy_error'],final_yaw_error_rad=trace[-1]['yaw_error'],
            localization_xy_p95_m=float(np.percentile(np.linalg.norm(truths[:,:2]-ests[:,:2],axis=1),95)),
            localization_yaw_p95_rad=float(np.percentile(abs(wrap(truths[:,2]-ests[:,2])),95)),
            map_updates=runtime.map_updates,subgoal_updates=runtime.subgoal_updates,
            initial_known_cells=initial_known,final_known_cells=int((~backend.current_map().blocked).sum()),
            stop_reasons=dict(stops),serialization=saved,
            actor_ms_p95=float(np.percentile(latencies,95)) if latencies else None)
        result['task_kind']=case['task_kind'];result['dynamic_obstacle_introduced']=env.intervention is not None
        result['input_odometry']=case.get('input_odometry','biased_motion_feedback')
        (output/'RESULTS.json').write_text(json.dumps(result,indent=2))
        (output/'trace.json').write_text(json.dumps(trace));(output/'bootstrap.json').write_text(json.dumps(sweep))
        return result
    finally:
        if backend:backend.close()
        env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=int,default=6);p.add_argument('--cpu-cores',type=float,default=.4);p.add_argument('--domain',type=int,default=198)
    p.add_argument('--exploration',action='store_true');p.add_argument('--dynamic',action='store_true')
    p.add_argument('--rf2o-executable',type=Path)
    p.add_argument('--laser-increments',action='store_true',help='Experimental measured scan-to-scan incremental odometry')
    p.add_argument('--ray-free-layer',action='store_true',help='Experimental finite-ray free evidence in local planning map')
    p.add_argument('--coarse-subgoals',action='store_true',help='Restrict online subgoals to the actual neural routing grid component')
    p.add_argument('--known-free-dynamics',action='store_true',help='Experimental: do not duplicate unknown-space inflation in the dynamic routing layer')
    p.add_argument('--slam-distance-prior-variance',type=float,help='Experimental Karto odometry-distance penalty; system default is unchanged when omitted')
    p.add_argument('--seed-base',type=int,default=115710)
    p.add_argument('--scenes',nargs='+',help='Physics/fixture maps only; never supplied to the policy or SLAM')
    p.add_argument('--case-file',type=Path,help='Frozen fixture subset for explicit regression or repair diagnosis')
    p.add_argument('--resume',action='store_true',help='Continue incomplete frozen cases; preserve attempts and record a new source epoch')
    a=p.parse_args()
    if a.resume:
        if not a.output.is_dir():p.error('Resume output does not exist')
    else:a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    from experiments.experiment_provenance import record_provenance
    root=Path(__file__).resolve().parents[2]
    extra=list((root/'.research/rf2o_laser_odometry-ros2/src').glob('*.cpp')) if a.rf2o_executable else []
    budget=ExperimentBudget(a.cpu_cores,directory=a.output)
    cases=json.loads(a.case_file.read_text()) if a.case_file else fixtures(a.cases,a.exploration,a.dynamic,a.seed_base,a.scenes)
    if a.resume:
        import hashlib
        report=json.loads((a.output/'RESULTS.json').read_text())
        if report['completed'] or cases!=json.loads((a.output/'cases.json').read_text()):p.error('Frozen cases changed or already completed')
        flags=dict(laser_increments=a.laser_increments,ray_free_layer=a.ray_free_layer,
            coarse_subgoals=a.coarse_subgoals,known_free_dynamics=a.known_free_dynamics,
            distance_prior_variance=a.slam_distance_prior_variance)
        previous_flags=report.get('optional_flags',{})
        if any(previous_flags.get(key,False if key!='distance_prior_variance' else None)!=value for key,value in flags.items()):
            p.error('Resume cannot change optional mapping flags')
        provenance=json.loads((a.output/'PROVENANCE.json').read_text())
        if hashlib.sha256((a.bundle/'manifest.json').read_bytes()).hexdigest()!=provenance['bundle_manifest_sha256']:p.error('Resume model changed')
        epoch=a.output/f"continuation_{len(report.get('continuations',[]))+1:02d}";epoch.mkdir()
        record_provenance(epoch,a.bundle,extra)
        report.setdefault('continuations',[]).append(dict(source_epoch=epoch.name,completed_cases=len(report['rows']),
            previous_resource=report.get('resource_parent_only')))
    else:
        record_provenance(a.output,a.bundle,extra)
        (a.output/'cases.json').write_text(json.dumps(cases,indent=2))
        report=dict(completed=False,scope='validation maps; supervised scan bootstrap then retained RL control; no simulator truth input',
            deployment_approved=False,rows=[],optional_flags=dict(laser_increments=a.laser_increments,
                ray_free_layer=a.ray_free_layer,coarse_subgoals=a.coarse_subgoals,known_free_dynamics=a.known_free_dynamics,
                distance_prior_variance=a.slam_distance_prior_variance))
    try:
        for case in cases:
            if any(row['case_id']==case['id'] for row in report['rows']):continue
            folder=a.output/case['id']
            if folder.exists():
                attempt=a.output/(case['id']+'_incomplete_attempt_'+str(len(report.get('continuations',[]))))
                folder.rename(attempt)
            row=evaluate(case,a.bundle,a.output/case['id'],budget,a.domain,a.rf2o_executable,a.laser_increments,a.ray_free_layer,a.coarse_subgoals,a.known_free_dynamics,a.slam_distance_prior_variance);report['rows'].append(row)
            report['resource_parent_only']=budget.statistics();(a.output/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
        report['completed']=True
    finally:
        report['resource_parent_only']=budget.statistics();(a.output/'RESULTS.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
