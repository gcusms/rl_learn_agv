"""Paired sensor replay with independent MuJoCo pose labels, not navigation.

Manual fixed velocity sequences generate the same physical trajectory for both
localizers. Physics geometry remains intact when corrupting localization scans.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT,wrap,integrate
from neural_navigation.models import LearnedLocalizer
from neural_navigation.odom_anchored_localizer import OdometryAnchoredLocalizer,contract
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    p=argparse.ArgumentParser();p.add_argument('--components',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-cores',type=float,default=.5)
    p.add_argument('--cases',type=int,default=24);a=p.parse_args()
    if a.output.exists() or a.cases<=0:p.error('Fresh output and positive case count required')
    a.output.mkdir(parents=True);torch.set_num_threads(1)
    budget=ExperimentBudget(a.cpu_cores,directory=a.output);rows=[]
    conditions=['normal_feedback','small_initial_offset','unmapped_obstacle','missing_sector','command_odometry','command_and_unmapped']
    for case_id in range(a.cases):
        mode=conditions[case_id%len(conditions)];seed=114000+case_id;rng=np.random.default_rng(seed)
        scene=ROOT/('assets/room_navigation_scripted' if case_id//len(conditions)%2==0 else 'assets/navigation_stage2_dataset/train_00')
        physical_case=dict(odom_scale=1.)
        if 'unmapped' in mode:physical_case.update(obstacle='sudden_block',trigger_travel_m=0.,hold_s=100.)
        e=StressNavigationEnv(case=physical_case,components=a.components,scene_dir=scene,route_mode='shortcut',max_steps=50)
        try:
            # Starting state is a specified simulator reset, not a per-frame
            # localization input. Labels below come solely from physical pose.
            from navrl.map import StaticMap
            m=StaticMap(scene/'decision_map.yaml',radius=.3);cells=np.argwhere(m.clearance>.9)
            start=np.r_[m.xy(cells[int(rng.integers(len(cells)))]),rng.uniform(-np.pi,np.pi)]
            e.reset(seed=seed,options=dict(start=start.tolist(),goal=start[:2].tolist()))
            prior=start.copy()
            if mode=='small_initial_offset':prior+=[.06,-.04,.04]
            localizers=[LearnedLocalizer(e.static_map,e.lidar,a.components/'localizer.pt'),
                        OdometryAnchoredLocalizer(e.static_map,e.lidar,a.components/'localizer.pt',contract())]
            for loc in localizers:loc.reset(prior,None)
            traces=[];command_odom=np.zeros(3)
            for step in range(41):
                scan=e.scan.copy()
                if mode=='missing_sector':
                    scan[rng.random(len(scan))<.08]=np.nan
                    scan[(e.lidar.angles>-.12)&(e.lidar.angles<.12)]=np.nan
                odom=command_odom.copy() if mode.startswith('command') else e.observed_odometry.copy()
                estimates=[localizers[0].update(odom,scan),localizers[1].update(odom,scan,e.elapsed)]
                traces.append(dict(seconds=float(e.elapsed),truth=e.truth.tolist(),odometry=odom.tolist(),
                    estimates=[v.tolist() for v in estimates],
                    candidate_reason=localizers[1].correction_reason,candidate_std=localizers[1].std.tolist()))
                budget.tick()
                if step==40:break
                command=np.array([.08 if step<24 else 0.,.15 if step<24 else (.3 if case_id%2==0 else -.3)])
                e.step(command/[.2,.5]);command_odom=integrate(command_odom,*command,e.dt)
            errors=[]
            for index in (0,1):
                est=np.array([r['estimates'][index] for r in traces]);truth=np.array([r['truth'] for r in traces])
                errors.append(dict(xy_p95_m=float(np.percentile(np.linalg.norm(est[:,:2]-truth[:,:2],axis=1),95)),
                    final_xy_m=float(np.linalg.norm(est[-1,:2]-truth[-1,:2])),
                    yaw_p95_rad=float(np.percentile(abs(wrap(est[:,2]-truth[:,2])),95))))
            result=dict(case_id=case_id,seed=seed,condition=mode,scene=str(scene),initial_pose=start.tolist(),
                collision_events=int(e.collision_events),unmapped_obstacle_introduced=e.intervention is not None,
                baseline=errors[0],candidate=errors[1],frames=len(traces),
                rejected_corrections=sum(r['candidate_reason'] not in ('accepted_static_correction','validated_static_prior') for r in traces))
            (a.output/f'case_{case_id:02d}.json').write_text(json.dumps(dict(result=result,samples=traces),indent=2))
            rows.append(result)
            (a.output/'progress.json').write_text(json.dumps(dict(completed_cases=len(rows),total_cases=a.cases,**budget.statistics()),indent=2))
        finally:e.close()
    groups={}
    for condition in conditions:
        subset=[r for r in rows if r['condition']==condition]
        groups[condition]={label:dict(cases=len(subset),median_case_xy_p95_m=float(np.median([r[label]['xy_p95_m'] for r in subset])),
            maximum_case_xy_p95_m=float(max(r[label]['xy_p95_m'] for r in subset)),
            median_case_yaw_p95_rad=float(np.median([r[label]['yaw_p95_rad'] for r in subset]))) for label in ('baseline','candidate')}
    result=dict(scope='fixed manual MuJoCo trajectories; pose labels independent of localization; NOT closed-loop navigation acceptance',
        completed=True,cases=rows,groups=groups,resource=budget.statistics(),contract=contract())
    (a.output/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
