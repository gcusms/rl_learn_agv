"""Verify saved graph reload in a fresh private ROS process at its first node."""
import argparse,json,copy
from pathlib import Path
import numpy as np
from navrl.geometry import ROOT,wrap
from navrl.map import StaticMap
from neural_navigation.stress_env import StressNavigationEnv
from experiments.probe_online_slam import OdometryProfile,relative_pose
from experiments.slam_toolbox_backend import SlamBackend


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--domain',type=int,default=189);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    from experiments.experiment_provenance import record_provenance
    record_provenance(a.output,a.bundle)
    case=json.loads((a.source/'cases.json').read_text())[0];original=StaticMap(a.source/'online_map_00/slam/map.yaml')
    env=StressNavigationEnv(case=case,components=a.bundle,scene_dir=ROOT/case['scene'],route_mode='shortcut',max_steps=25)
    backend=None
    try:
        env.reset(seed=case['seed'],options=dict(start=case['start'],goal=case['goal'][:2]))
        reference=env.truth.copy();profile=OdometryProfile(env.observed_odometry.copy())
        lidar=copy.copy(env.lidar);lidar.range_max=case.get('sensor_max_range_m',12.)
        backend=SlamBackend(lidar,a.output/'slam',domain=a.domain)
        reload=backend.load_graph(a.source/'online_map_00/slam/persistent_map')
        restored=backend.current_map();cells=np.argwhere(original.occupied)
        points=np.array([original.xy(cell) for cell in cells])
        recovered=float(restored.lookup(points,restored.occupied,outside=False).mean()) if len(points) else None
        errors=[]
        for i in range(20):
            odom=profile.update(env.observed_odometry,env.elapsed)
            estimate=backend.push(env.scan,odom,env.elapsed)
            if estimate is not None:
                truth=relative_pose(env.truth,reference)
                errors.append([float(np.linalg.norm(estimate[:2]-truth[:2])),float(abs(wrap(estimate[2]-truth[2])))])
            env.step(np.zeros(2))
        if not errors:raise RuntimeError('Reloaded map never localized the stationary first-node scan')
        result=dict(completed=True,scope='fresh-process graph persistence at original first node; not kidnapped/global relocalization',
            reload=reload,original_occupied_cells=len(cells),restored_occupied_recall=recovered,
            matched_pose_frames=len(errors),xy_p95_m=float(np.percentile(np.array(errors)[:,0],95)),
            yaw_p95_rad=float(np.percentile(np.array(errors)[:,1],95)),collisions=int(env.collision_events),
            graph_resaved=backend.save())
        (a.output/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    finally:
        if backend:backend.close()
        env.close()


if __name__=='__main__':main()
