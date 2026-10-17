"""New validation tasks; offline reachability is a fixture label only."""
import argparse
import json
from pathlib import Path
import numpy as np
from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=48);a=p.parse_args()
    if a.output.exists():p.error('Fresh validation task file required')
    cases=[]
    for i in range(a.cases):
        scene=ROOT/f'assets/navigation_stage2_dataset/validation_{i%2:02d}'
        m=StaticMap(scene/'decision_map.yaml',radius=.30);rng=np.random.default_rng(116010+i)
        cells=np.argwhere(m.clearance>.85)
        category=['boundary_alignment','approach_and_stop','navigation_regression'][i//16%3]
        for _ in range(500):
            xy=m.xy(cells[int(rng.integers(len(cells)))]);heading=float(rng.uniform(-np.pi,np.pi))
            if category=='boundary_alignment':goal=xy
            elif category=='approach_and_stop':goal=xy+rng.uniform(.3,.8)*np.array([np.cos(heading),np.sin(heading)])
            else:goal=m.xy(cells[int(rng.integers(len(cells)))])
            if not m.lookup(goal,m.clearance,outside=0)>.7:continue
            if category=='approach_and_stop' and not m.visible(xy,goal):continue
            if category=='navigation_regression':
                if np.linalg.norm(goal-xy)<2.:continue
                try:path=m.plan(xy,goal)
                except ValueError:continue
                length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                if length>7.:continue
            break
        else:raise RuntimeError('Task sampling failed')
        offset=[0.,.10,-.10,.145,-.145,.155,-.155,.17,-.17,.25,-.25,.5,-.5,np.pi/2,-np.pi/2,np.pi][i%16]
        case=dict(id=f'localization_nav_{i:02d}',pilot_category=category,category='static',scene=str(scene.relative_to(ROOT)),
            seed=116010+i,start=[*xy,heading],goal=[*goal,float(wrap(heading+offset))],odom_scale=1.,
            sensor_max_range_m=10.,
            max_steps=150 if category!='navigation_regression' else 500)
        if i%8==6:case['extra_scan_dropout']=.08
        if i%8==7:case['dropout_windows']=[[1.,1.4]]
        if i%8==4:case['initial_prior_offset']=[.06,-.04,.04]
        if category=='navigation_regression' and i%4==1:
            case.update(obstacle='crossing',obstacle_speed_m_s=.08,trigger_travel_m=.4)
        cases.append(case)
    a.output.write_text(json.dumps(cases,indent=2));print('Saved',len(cases),'fresh validation tasks')


if __name__=='__main__':main()
