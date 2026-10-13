"""Freeze development or untouched acceptance cases before training selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from navrl.geometry import ROOT
from navrl.map import StaticMap


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['development','acceptance'],required=True)
    parser.add_argument('--dataset',type=Path,default=ROOT/'assets/navigation_stage2_dataset')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Frozen manifests must not be overwritten')
    rng=np.random.default_rng(53000 if args.kind=='development' else 93000)
    maps={};sources={}
    def sample(scene):
        path=ROOT/scene
        if scene not in maps:maps[scene]=StaticMap(path/'decision_map.yaml',radius=.3)
        m=maps[scene];cells=np.argwhere(m.clearance>.65)
        for _ in range(500):
            a,b=[m.xy(cells[int(rng.integers(len(cells)))]) for _ in range(2)]
            if np.linalg.norm(a-b)<3.:continue
            try:route=m.plan(a,b)
            except ValueError:continue
            if np.linalg.norm(np.diff(route,axis=0),axis=1).sum()>12.:continue
            return [*a,float(rng.uniform(-np.pi,np.pi))],b.tolist()
        raise RuntimeError('Unable to sample a feasible task')
    def task(category,index,scene='assets/room_navigation_scripted',**extra):
        start,goal=sample(scene)
        return dict(id=f'{category}_{index:03d}',category=category,scene=scene,
                    seed=int(rng.integers(100000,900000)),start=start,goal=goal,**extra)
    relative=str(args.dataset.resolve().relative_to(ROOT))
    cases=[]
    if args.kind=='development':
        original=json.loads((ROOT/'config/navigation_diagnostic_v1.json').read_text())
        cases.extend(c for c in original['cases'] if c['category']=='static')
        for i in range(8):cases.append(task('new_layout',i,f'{relative}/validation_{i%2:02d}'))
        starts=[([1.2,1.2,0.],[8.6,1.2]),([8.6,1.2,np.pi],[1.2,1.2]),
                ([1.2,1.2,np.pi/2],[1.2,6.6]),([8.6,1.2,np.pi/2],[8.6,6.6])]
        for kind in ['sudden_block','crossing']:
            for i,(start,goal) in enumerate(starts):
                cases.append(dict(id=f'{kind}_{i:03d}',category=kind,seed=54000+i,start=start,goal=goal,
                                  obstacle=kind,obstacle_speed_m_s=.12,hold_s=3.))
        offsets=[[.16,-.14,.08],[-.2,.15,-.1],[.3,-.18,.16],[-.3,.2,-.16]]
        for i,offset in enumerate(offsets):cases.append(task('initial_pose',i,initial_prior_offset=offset))
        for i,scale in enumerate([1.1,.9,1.25,.75]):
            cases.append(task('odometry_drift',i,odom_scale=scale,yaw_odom_scale=scale,odom_bias_m_s=[.005,-.005]))
        for i in range(4):cases.append(task('scan_dropout',i,dropout_windows=[[8.,9.],[22.,22.6]],extra_scan_dropout=.05))
        for i in range(4):cases.append(dict(id=f'continuous_{i:03d}',category='continuous_mission',seed=55000+i,max_steps=2800))
        limitation='40 development cases: known-map static and new validation layouts, physically separated sudden/crossing actors, pose/odometry/scan faults, and continuous four-goal missions. Not final acceptance.'
    else:
        for i in range(60):cases.append(task('static',i))
        for i in range(50):
            kind='crossing' if i%2 else 'sudden_block'
            cases.append(task('dynamic',i,obstacle=kind,obstacle_speed_m_s=float(rng.uniform(.08,.18)),hold_s=float(rng.uniform(2.,5.))))
        for i in range(30):
            extra=dict(dropout_windows=[[8.,9.],[22.,22.6]],extra_scan_dropout=.1)
            if i%2:extra.update(obstacle='crossing',dropout_after_event_s=[[.4,1.0]],obstacle_speed_m_s=.12)
            cases.append(task('sensor_fault',i,**extra))
        for i in range(30):
            cases.append(task('localization_fault',i,initial_prior_offset=[float(rng.uniform(-.3,.3)),float(rng.uniform(-.3,.3)),float(rng.uniform(-.16,.16))],
                              odom_scale=float(rng.uniform(.8,1.2)),yaw_odom_scale=float(rng.uniform(.85,1.15)),odom_bias_m_s=[.005,-.005]))
        for i in range(30):cases.append(task('unseen_layout',i,f'{relative}/test_{i%3:02d}'))
        limitation='200 frozen final acceptance cases. Do not use for repeated development/model selection. Orin Nano hardware and real-robot behavior are separate acceptance requirements.'
    for case in cases:
        scene=case.get('scene','assets/navigation_comparison' if case['category']=='continuous_mission' else 'assets/room_navigation_scripted')
        for filename in ['decision_map.pgm','decision_map.yaml','physical_scene.xml','physical_scene.json']:
            path=ROOT/scene/filename;sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(version=2,kind=args.kind,limitation=limitation,cases=cases,scene_sha256=sources,
                actor_protocol='Introduced at >0.35m footprint gap after 0.6m robot travel, <=0.18m/s, one-way crossing or hold-and-leave. Actor truth is WORLD control only.',
                reference='Offline inflated-grid paths validate sampling only; no reference path enters runtime policy')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print('Frozen',len(cases),args.kind,'cases:',args.output)


if __name__=='__main__':main()
