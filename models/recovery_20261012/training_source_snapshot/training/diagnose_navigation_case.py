"""Sensor/action/route traces for a specific frozen development failure."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from evaluate_navigation_suite import FixedCase


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--components',type=Path,required=True)
    parser.add_argument('--route-mode',choices=['raw','shortcut'],default='raw')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stride',type=int,default=5)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh trace file')
    if args.stride<=0:parser.error('Trace stride must be positive')
    cases=json.loads(args.manifest.read_text())['cases']
    case=next(c for c in cases if c['id']==args.case)
    torch.set_num_threads(1);policy=PPO.load(args.policy,device='cpu')
    env=FixedCase(case,args.components,700,args.route_mode);rows=[]
    try:
        observation,_=env.reset();steps=int(case.get('max_steps',700))
        for step in range(steps):
            action,_=policy.predict(observation,deterministic=True)
            observation,_,done,truncated,info=env.step(action);e=env.env
            if step%args.stride==0 or done or truncated:
                row=dict(step=step+1,time_s=e.elapsed,truth=e.truth.tolist(),estimate=e.estimate.tolist(),
                         action=np.asarray(action).tolist(),velocity=np.asarray(info['actual_velocity']).tolist(),
                         waypoint_relative=(observation[216:218]*2).tolist(),heading=observation[231:233].tolist(),
                         front_clearance_m=float(observation[234]*2),recent_progress=float(observation[233]),
                         path=e.path.tolist(),goal=e.goal.tolist(),remaining=float(e.remaining),
                         route_reason=info['route_switch_reason'],reachability=float(e.current_probability),
                         scan=[float(v) if np.isfinite(v) else None for v in e.scan],
                         actors=[p.tolist() for p in e._humans()],hard_cells=np.argwhere(e.laser_fusion.hard).tolist(),
                         localization_error_m=float(info['localization_error_m']),qpos=e.data.qpos.tolist())
                row['localization_std']=np.asarray(e.localizer.std).tolist()
                row['valid_scan_fraction']=float(e.localizer.valid_scan_fraction)
                row['map_agreement']=float(e.localizer.map_agreement)
                row['posterior_entropy']=float(e.localizer.posterior_entropy)
                row['recovery_attempts']=int(e.localizer.recovery_attempts)
                row['recovery_accepted']=int(e.localizer.recovery_accepted)
                row['odometry']=np.asarray(getattr(e,'observed_odometry',e.odom)).tolist()
                row['measured_velocity']=np.asarray(e.measured_velocity).tolist()
                row['collisions']=int(e.collision_events)
                row['guards']=int(e.hard_guard_activations)
                row['current_gap_m']=float(e._minimum_gap())
                row['forecast_cells']=e.laser_fusion.forecast_cells
                predictor=e.laser_fusion.predictor
                row['scan_tracks']=[] if predictor is None else [dict(center=t['center'].tolist(),velocity=t['velocity'].tolist(),hits=t['hits']) for t in predictor.tracks]
                rows.append(row)
                if step%100==0:print(json.dumps({k:row[k] for k in ['step','time_s','estimate','action','waypoint_relative','front_clearance_m','recent_progress','route_reason']}),flush=True)
            if done or truncated:break
        report=dict(case=case,route_mode=args.route_mode,samples=rows,result=info.get('diagnostic_result'))
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report))
        print('RESULT',json.dumps(report['result']),flush=True)
    finally:env.close()


if __name__=='__main__':main()
