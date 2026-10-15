"""Long sensor trajectories in initially unknown maps, with separate labels.

Commands are scripted measurement probes, not autonomous RL navigation. SLAM
receives only degraded lidar and command translation / simulated gyro input.
The simulator's map and reference pose never enter the SLAM backend.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT,wrap
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.experiment_budget import ExperimentBudget
from experiments.command_odometry_input import make_sensor_odometry,advance_sensor_odometry
from experiments.probe_online_slam import relative_pose
from experiments.slam_toolbox_backend import SlamBackend
from experiments.experiment_provenance import record_provenance


def pose_metrics(rows,name):
    valid=[r for r in rows if r[name] is not None]
    if not valid:return dict(frames=0)
    truth=np.asarray([r['truth'] for r in valid]);estimate=np.asarray([r[name] for r in valid])
    delta=estimate[:,:2]-truth[:,:2]
    return dict(frames=len(valid),xy_p95_m=float(np.percentile(np.linalg.norm(delta,axis=1),95)),
        x_p95_m=float(np.percentile(abs(delta[:,0]),95)),y_p95_m=float(np.percentile(abs(delta[:,1]),95)),
        yaw_p95_rad=float(np.percentile(abs(wrap(estimate[:,2]-truth[:,2])),95)),
        final_xy_m=float(np.linalg.norm(delta[-1])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-cores',type=float,default=.25)
    p.add_argument('--domain',type=int,default=194)
    p.add_argument('--slam-distance-prior-variance',type=float)
    p.add_argument('--scenes',nargs='+',default=['single_long_wall','feature_to_open'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    budget=ExperimentBudget(a.cpu_cores,directory=a.output);record_provenance(a.output,a.bundle)
    report=dict(completed=False,scope='manual 12m sensor trajectories; not autonomous RL navigation or real-world accuracy',
        input='command translation plus noisy simulated gyro, 8% translation scale error, 0.003rad/s gyro bias',
        experimental_distance_prior_variance=a.slam_distance_prior_variance,rows=[])
    try:
        for name in a.scenes:
            folder=a.output/name;folder.mkdir();backend=None
            case=dict(odom_scale=1.,sensor_max_range_m=10.,extra_scan_dropout=.08,
                input_odometry='command_linear_imu_yaw',command_translation_scale=1.08,
                gyro_fusion_weight=.8,gyro_bias_rad_s=.003)
            env=StressNavigationEnv(case=case,components=a.bundle,
                scene_dir=ROOT/'assets/localization_geometry_20261015'/name,route_mode='shortcut',max_steps=500)
            try:
                env.reset(seed=116210,options=dict(start=[1.2,6.5,0.],goal=[1.2,6.5]))
                reference=env.truth.copy();observer=make_sensor_odometry(case)
                lidar=copy.copy(env.lidar);lidar.range_max=10.
                backend=SlamBackend(lidar,folder/'slam',domain=a.domain,distance_prior_variance=a.slam_distance_prior_variance);traces=[]
                for frame in range(465):
                    estimate=backend.push(env.scan,observer.pose,env.elapsed)
                    stamp,std=backend.pose_metadata();age=None if stamp is None else env.elapsed-stamp
                    traces.append(dict(seconds=env.elapsed,truth=relative_pose(env.truth,reference).tolist(),
                        odometry=observer.pose.tolist(),slam=None if estimate is None else estimate.tolist(),
                        pose_age_s=age,std=std.tolist(),finite_returns=int(np.isfinite(env.scan).sum()),
                        map_version=backend.map_version))
                    if frame==464:break
                    command=np.array([.15,0.]) if frame<200 or frame>=264 else np.array([0.,.25])
                    previous=env.observed_odometry[2];env.step(command/[.2,.5])
                    advance_sensor_odometry(observer,command,env,previous,case);budget.tick()
                    if env.collision_events or env.hard_guard_activations:break
                    if frame%20==0:
                        (folder/'progress.json').write_text(json.dumps(dict(frames=frame+1,total=465,**budget.statistics()),indent=2))
                backend.spin_for(.5);saved=backend.save()
                fresh=[t for t in traces if t['pose_age_s'] is not None and -.05<=t['pose_age_s']<=1.]
                row=dict(scene=name,frames=len(traces),seconds=env.elapsed,travel_m=env.travel,
                    collisions=env.collision_events,physical_guards=env.hard_guard_activations,
                    metrics=dict(odometry=pose_metrics(traces,'odometry'),slam_all=pose_metrics(traces,'slam'),
                        slam_fresh=pose_metrics(fresh,'slam')),serialization=saved,
                    loop_closure_enabled=True,loop_closure_causal_effect_isolated=False)
                report['rows'].append(row);(folder/'trace.json').write_text(json.dumps(traces))
                (folder/'RESULTS.json').write_text(json.dumps(row,indent=2))
                report['resource_parent_only']=budget.statistics()
                (a.output/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
            finally:
                if backend:backend.close()
                env.close()
        report['completed']=True
    finally:
        report['resource_parent_only']=budget.statistics();(a.output/'RESULTS.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
