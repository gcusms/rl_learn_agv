"""~12m physical out-and-back trajectories with independent pose labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT,wrap
from neural_navigation.models import LearnedLocalizer
from neural_navigation.odom_anchored_localizer import OdometryAnchoredLocalizer,turn_response_contract,precision_contract
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.experiment_budget import ExperimentBudget
from experiments.probe_online_slam import OdometryProfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--components',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-cores',type=float,default=.25);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);budget=ExperimentBudget(a.cpu_cores,directory=a.output);rows=[]
    from experiments.experiment_provenance import record_provenance
    record_provenance(a.output,a.components)
    for name in ['single_long_wall','parallel_corridor','sparse_asymmetric','feature_to_open']:
        scene=ROOT/'assets/localization_geometry_20261015'/name
        env=StressNavigationEnv(case=dict(odom_scale=1.,sensor_max_range_m=10.,extra_scan_dropout=.08),
            components=a.components,scene_dir=scene,route_mode='shortcut',max_steps=465)
        try:
            initial=np.array([1.2,6.5,0.]);env.reset(seed=116210,options=dict(start=initial.tolist(),goal=initial[:2].tolist()))
            profile=OdometryProfile(env.observed_odometry.copy(),translation_scale=1.08,yaw_scale=1.025)
            localizers=[LearnedLocalizer(env.static_map,env.lidar,a.components/'localizer.pt'),
                OdometryAnchoredLocalizer(env.static_map,env.lidar,a.components/'localizer.pt',turn_response_contract()),
                OdometryAnchoredLocalizer(env.static_map,env.lidar,a.components/'localizer.pt',precision_contract())]
            for loc in localizers:loc.reset(initial,None)
            traces=[]
            for frame in range(465):
                odom=profile.update(env.observed_odometry,env.elapsed)
                estimates=[loc.update(odom,env.scan) if i==0 else loc.update(odom,env.scan,env.elapsed) for i,loc in enumerate(localizers)]
                traces.append(dict(seconds=env.elapsed,truth=env.truth.tolist(),odometry=odom.tolist(),
                    estimates=[x.tolist() for x in estimates],stds=[loc.std.tolist() for loc in localizers]))
                if frame==464:break
                command=np.array([.15,0.]) if frame<200 or frame>=264 else np.array([0.,.25])
                env.step(command/[.2,.5]);budget.tick()
                if env.collision_events or env.hard_guard_activations:break
            truth=np.array([r['truth'] for r in traces]);errors={}
            for i,label in enumerate(['legacy','v4','v5']):
                est=np.array([r['estimates'][i] for r in traces]);delta=est[:,:2]-truth[:,:2]
                errors[label]=dict(xy_p95_m=float(np.percentile(np.linalg.norm(delta,axis=1),95)),
                    x_p95_m=float(np.percentile(abs(delta[:,0]),95)),y_p95_m=float(np.percentile(abs(delta[:,1]),95)),
                    final_xy_m=float(np.linalg.norm(delta[-1])),yaw_p95_rad=float(np.percentile(abs(wrap(est[:,2]-truth[:,2])),95)))
            row=dict(scene=name,frames=len(traces),seconds=env.elapsed,travel_m=env.travel,
                collisions=env.collision_events,physical_guards=env.hard_guard_activations,errors=errors)
            rows.append(row);(a.output/(name+'.json')).write_text(json.dumps(dict(result=row,samples=traces)))
            (a.output/'RESULTS.json').write_text(json.dumps(dict(completed=len(rows)==4,
                scope='manual ~12m sensor trajectories, NOT autonomous navigation or real-world accuracy',rows=rows,resource=budget.statistics()),indent=2));print(json.dumps(row),flush=True)
        finally:env.close()


if __name__=='__main__':main()
