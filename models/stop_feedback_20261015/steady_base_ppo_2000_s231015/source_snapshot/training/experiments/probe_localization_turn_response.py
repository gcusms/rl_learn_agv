"""Parameter ablation on independently labelled trajectories, not a release.

Tests whether bounded yaw correction response explains command-odometry turn
error. No pose labels enter either localizer. Baseline v3 stays unchanged.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT,wrap,integrate
from navrl.map import StaticMap
from neural_navigation.models import LearnedLocalizer
from neural_navigation.odom_anchored_localizer import OdometryAnchoredLocalizer,contract,turn_response_contract,precision_contract
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    p=argparse.ArgumentParser();p.add_argument('--components',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=24)
    p.add_argument('--cpu-cores',type=float,default=.25);p.add_argument('--seed-base',type=int,default=114000)
    p.add_argument('--precision',action='store_true');p.add_argument('--validation-maps',action='store_true')
    p.add_argument('--scenes',type=Path,help='Fresh geometry list; only sensor inputs go to localizers')
    p.add_argument('--steps',type=int,default=41);a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    a.output.mkdir(parents=True);torch.set_num_threads(1);budget=ExperimentBudget(a.cpu_cores,directory=a.output)
    conditions=['normal_feedback','small_initial_offset','unmapped_obstacle','missing_sector','command_odometry','command_and_unmapped']
    scenes=json.loads(a.scenes.read_text()) if a.scenes else None
    rows=[];labels=['legacy','v4','v5'] if a.precision else ['legacy','v3','yaw_rate_ablation']
    for case_id in range(a.cases):
        mode=conditions[case_id%6];seed=a.seed_base+case_id;rng=np.random.default_rng(seed)
        scene=ROOT/('assets/room_navigation_scripted' if case_id//6%2==0 else 'assets/navigation_stage2_dataset/train_00')
        if a.validation_maps:scene=ROOT/f'assets/navigation_stage2_dataset/validation_{case_id//6%2:02d}'
        if scenes:scene=ROOT/scenes[case_id//6%len(scenes)]
        physical_case=dict(odom_scale=1.)
        if 'unmapped' in mode:physical_case.update(obstacle='sudden_block',trigger_travel_m=0.,hold_s=100.)
        e=StressNavigationEnv(case=physical_case,components=a.components,scene_dir=scene,route_mode='shortcut',max_steps=a.steps+1)
        try:
            m=StaticMap(scene/'decision_map.yaml',radius=.3);cells=np.argwhere(m.clearance>.9)
            start=np.r_[m.xy(cells[int(rng.integers(len(cells)))]),rng.uniform(-np.pi,np.pi)]
            e.reset(seed=seed,options=dict(start=start.tolist(),goal=start[:2].tolist()))
            prior=start.copy()
            if mode=='small_initial_offset':prior+=[.06,-.04,.04]
            localizers=([LearnedLocalizer(e.static_map,e.lidar,a.components/'localizer.pt')]+
                [OdometryAnchoredLocalizer(e.static_map,e.lidar,a.components/'localizer.pt',contract()) for _ in range(2)])
            # Explicitly experimental parameter mutation. Not a supported
            # portable contract and cannot silently become a deployment bundle.
            localizers[2].config=dict(contract(),maximum_yaw_rate_rad_s=.15)
            if a.precision:
                localizers[1].config=turn_response_contract();localizers[2].config=precision_contract()
            for loc in localizers:loc.reset(prior,None)
            traces=[];command_odom=np.zeros(3)
            for step in range(a.steps):
                scan=e.scan.copy()
                if mode=='missing_sector':
                    scan[rng.random(len(scan))<.08]=np.nan
                    scan[(e.lidar.angles>-.12)&(e.lidar.angles<.12)]=np.nan
                odom=command_odom.copy() if mode.startswith('command') else e.observed_odometry.copy()
                estimates=[loc.update(odom,scan) if i==0 else loc.update(odom,scan,e.elapsed) for i,loc in enumerate(localizers)]
                traces.append(dict(seconds=float(e.elapsed),truth=e.truth.tolist(),odometry=odom.tolist(),
                    estimates=[v.tolist() for v in estimates],reasons=[getattr(loc,'correction_reason','legacy') for loc in localizers]))
                budget.tick()
                if step==a.steps-1:break
                translation_steps=int(a.steps*24/41)
                command=np.array([.08 if step<translation_steps else 0.,.15 if step<translation_steps else (.3 if case_id%2==0 else -.3)])
                e.step(command/[.2,.5]);command_odom=integrate(command_odom,*command,e.dt)
            errors={};truth=np.array([r['truth'] for r in traces])
            for i,label in enumerate(labels):
                est=np.array([r['estimates'][i] for r in traces])
                errors[label]=dict(xy_p95_m=float(np.percentile(np.linalg.norm(est[:,:2]-truth[:,:2],axis=1),95)),
                    final_xy_m=float(np.linalg.norm(est[-1,:2]-truth[-1,:2])),
                    yaw_p95_rad=float(np.percentile(abs(wrap(est[:,2]-truth[:,2])),95)))
            row=dict(case_id=case_id,seed=seed,condition=mode,scene=str(scene),errors=errors,
                collision_events=int(e.collision_events),unmapped_obstacle_introduced=e.intervention is not None)
            rows.append(row);(a.output/f'case_{case_id:02d}.json').write_text(json.dumps(dict(result=row,samples=traces),indent=2))
            (a.output/'RESULTS.json').write_text(json.dumps(dict(completed=False,rows=rows,resource=budget.statistics()),indent=2))
        finally:e.close()
    groups={mode:{label:{metric:float(np.median([r['errors'][label][metric] for r in rows if r['condition']==mode]))
        for metric in ('xy_p95_m','final_xy_m','yaw_p95_rad')} for label in labels} for mode in conditions}
    result=dict(completed=True,scope='paired localization development comparison; physical labels; not deployment approval',
        precision_comparison=a.precision,validation_maps=a.validation_maps,scenes=scenes,steps_per_case=a.steps,
        seed_base=a.seed_base,yaw_rate_baseline_rad_s=.06,yaw_rate_ablation_rad_s=.15,rows=rows,groups=groups,resource=budget.statistics())
    (a.output/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
