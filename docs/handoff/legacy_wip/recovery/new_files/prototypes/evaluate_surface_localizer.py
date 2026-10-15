"""Compare localizers on identical MuJoCo sensor streams, truth for scoring only.

Commands follow a fixed diagnostic schedule, not a learned navigation result.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.models import LearnedLocalizer
from prototypes.surface_localizer import SurfaceLocalizer
from navrl.geometry import wrap


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Fresh output required')
    args.output.mkdir(parents=True);torch.set_num_threads(1)
    bundle=Path('models/recovery_20261012/portable');scene=Path('training/runs/narrow_long_50000_v2_20261013/scenes/gap_150')
    cases=[dict(id='stationary_entrance',start=[1.9,2.,0.],seed=94120),
           dict(id='moving_corridor',start=[2.3,2.,0.],seed=94121),
           dict(id='scan_loss_recovery',start=[2.3,2.,0.],seed=94122,dropout_windows=[[4.,6.]])]
    summary=[]
    for case in cases:
        case.update(goal=[5.,2.]);env=StressNavigationEnv(case=case,components=bundle,scene_dir=scene,route_mode='shortcut',max_steps=180)
        try:
            env.reset(seed=case['seed'],options=dict(start=case['start'],goal=case['goal']))
            old=LearnedLocalizer(env.static_map,env.lidar,bundle/'localizer.pt')
            new=SurfaceLocalizer(env.static_map,env.lidar,bundle/'localizer.pt')
            for loc in (old,new):loc.reset(case['start'],None)
            rows=[]
            for i in range(150):
                row=dict(frame=i,truth_for_scoring_only=env.truth.tolist())
                for label,loc in [('old',old),('surface',new)]:
                    begin=time.perf_counter();estimate=loc.update(env.observed_odometry,env.scan);cost=time.perf_counter()-begin
                    error=estimate-env.truth;error[2]=wrap(error[2])
                    row[label]=dict(pose=estimate.tolist(),error=error.tolist(),accepted=bool(loc.scan_update_accepted),cpu_wall_ms=cost*1000)
                row['surface'].update(rank=new.observable_rank,support=new.surface_support,residual=new.surface_residual_m)
                rows.append(row)
                # Small forward speed with initial/final stationary intervals.
                speed=.05 if case['id']!='stationary_entrance' and 20<=i<100 else 0.
                env.step(np.array([speed/.2,0.]))
            result=dict(case=case,collision_events=env.collision_events,physical_guards=env.hard_guard_activations,frames=len(rows))
            for label in ['old','surface']:
                error=np.asarray([r[label]['error'] for r in rows]);latency=[r[label]['cpu_wall_ms'] for r in rows]
                result[label]=dict(max_position_m=float(np.linalg.norm(error[:,:2],axis=1).max()),
                    p95_position_m=float(np.quantile(np.linalg.norm(error[:,:2],axis=1),.95)),
                    max_lateral_m=float(np.abs(error[:,1]).max()),max_yaw_rad=float(np.abs(error[:,2]).max()),
                    median_update_ms=float(np.median(latency)),accepted=sum(r[label]['accepted'] for r in rows))
            (args.output/(case['id']+'.json')).write_text(json.dumps(dict(result=result,samples=rows),indent=2))
            summary.append(result);(args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(result),flush=True)
        finally:env.close()

if __name__=='__main__':main()
