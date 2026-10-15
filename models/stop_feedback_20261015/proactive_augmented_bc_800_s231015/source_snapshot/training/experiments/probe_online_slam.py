"""MuJoCo sensor replay into real SLAM; independent labels never enter backend."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.experiment_budget import ExperimentBudget
from experiments.slam_toolbox_backend import SlamBackend


def relative_pose(pose,reference):
    c,s=np.cos(reference[2]),np.sin(reference[2]);delta=pose[:2]-reference[:2]
    return np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],wrap(pose[2]-reference[2])])


class OdometryProfile:
    """Scale yaw increments; multiplying a wrapped angle creates fake jumps."""
    def __init__(self,reference,translation_scale=1.08,yaw_scale=1.025):
        self.reference=reference;self.translation_scale=translation_scale;self.yaw_scale=yaw_scale
        self.previous_yaw=0.;self.yaw=0.
    def update(self,pose,seconds):
        value=relative_pose(pose,self.reference)
        self.yaw+=float(wrap(value[2]-self.previous_yaw))*self.yaw_scale
        self.previous_yaw=value[2]
        value[:2]*=self.translation_scale;value[1]+=.001*seconds;value[2]=wrap(self.yaw)
        return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=160);p.add_argument('--scene',default='assets/navigation_stage2_dataset/validation_00')
    p.add_argument('--cpu-cores',type=float,default=.5);p.add_argument('--domain',type=int,default=197)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    budget=ExperimentBudget(a.cpu_cores,directory=a.output);scene=ROOT/a.scene
    m=StaticMap(scene/'decision_map.yaml',radius=.3);cells=np.argwhere(m.clearance>1.2)
    if not len(cells):raise RuntimeError('No safe sensor-loop fixture')
    rng=np.random.default_rng(115601);xy=m.xy(cells[int(rng.integers(len(cells)))])
    env=StressNavigationEnv(case=dict(odom_scale=1.),components=a.bundle,scene_dir=scene,route_mode='shortcut',max_steps=a.steps+1)
    backend=None;traces=[]
    try:
        env.reset(seed=115601,options=dict(start=[*xy,0.],goal=xy.tolist()))
        truth_reference=env.truth.copy();odom_reference=env.observed_odometry.copy();profile=OdometryProfile(odom_reference)
        backend=SlamBackend(env.lidar,a.output/'slam',domain=a.domain)
        for step in range(a.steps):
            scan=env.scan.copy();scan[rng.random(len(scan))<.08]=np.nan
            truth=relative_pose(env.truth,truth_reference)
            odom=profile.update(env.observed_odometry,env.elapsed)
            estimate=backend.push(scan,odom,env.elapsed)
            traces.append(dict(seconds=float(env.elapsed),truth=truth.tolist(),odometry=odom.tolist(),
                estimate=None if estimate is None else estimate.tolist(),map_version=backend.map_version))
            if step%10==0:
                (a.output/'progress.json').write_text(json.dumps(dict(steps=step+1,total=a.steps,map_messages=backend.map_version,**budget.statistics()),indent=2))
            command=np.array([0.,.25]) if step<32 else np.array([.08,.25])
            env.step(command/[.2,.5]);budget.tick()
            if env.collision_events or env.hard_guard_activations:break
        backend.spin_for(.6);saved=backend.save();known=backend.current_map()
        valid=[r for r in traces if r['estimate'] is not None]
        truth=np.array([r['truth'] for r in valid]);odom=np.array([r['odometry'] for r in valid]);est=np.array([r['estimate'] for r in valid])
        metrics={}
        for name,value in [('odometry',odom),('slam',est)]:
            metrics[name]=dict(xy_p95_m=float(np.percentile(np.linalg.norm(value[:,:2]-truth[:,:2],axis=1),95)),
                yaw_p95_rad=float(np.percentile(abs(wrap(value[:,2]-truth[:,2])),95)),
                final_xy_m=float(np.linalg.norm(value[-1,:2]-truth[-1,:2])))
        result=dict(completed=True,scope='manual physical sensor-loop replay, not RL closed-loop navigation',
            scene=a.scene,valid_pose_frames=len(valid),frames=len(traces),metrics=metrics,serialization=saved,
            known_free_cells=int(np.count_nonzero(known.free)),origin_planning_free=bool(known.lookup([0,0],known.free,outside=False)),
            collisions=int(env.collision_events),resource=budget.statistics())
        (a.output/'RESULTS.json').write_text(json.dumps(result,indent=2));(a.output/'trace.json').write_text(json.dumps(traces))
        print(json.dumps(result,indent=2),flush=True)
    finally:
        if backend is not None:backend.close()
        env.close()


if __name__=='__main__':main()
