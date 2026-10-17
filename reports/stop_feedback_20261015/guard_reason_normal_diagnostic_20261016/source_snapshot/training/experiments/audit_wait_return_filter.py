"""Exact replay: score which obstacle returns survive the static-map filter.

Simulator ray IDs are diagnostic labels computed only after the actor decides.
They never enter observation, tracking, guard decisions or training.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.train_quality_navigation import digest


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    torch.set_num_threads(1);trace=json.loads(a.trace.read_text());case=trace['case'];scene=ROOT/case['scene']
    env=StressNavigationEnv(case=case,components=a.bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    samples=[]
    try:
        env.reset(seed=case['seed'],options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case})
        runtime=TimeConsistentFreshRuntime(a.bundle,scene/'decision_map.yaml',
            np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0])),case['goal'],'shortcut')
        for index,recorded in enumerate(trace['samples'][:120]):
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            np.testing.assert_allclose(result['policy_observation'],recorded['policy_observation'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(result['command'],recorded['command'],atol=2e-8,rtol=0)
            assert result['stop_reason']==recorded['stop_reason']
            if index in (35,40,47,50,56,60,80,100,119):
                core=runtime.core;mount=core.lidar.mount;pose=core.estimate
                valid=np.isfinite(core.scan)&(core.scan>=core.lidar.range_min)&(core.scan<core.lidar.range_max)
                origin=pose[:2]+np.array([np.cos(pose[2])*mount[0]-np.sin(pose[2])*mount[1],
                    np.sin(pose[2])*mount[0]+np.cos(pose[2])*mount[1]])
                angles=core.lidar.angles+pose[2]+mount[3]
                endpoints=origin+np.where(valid,core.scan,0.)[:,None]*np.c_[np.cos(angles),np.sin(angles)]
                distances=core.static_map.lookup(endpoints,core.static_map.distance,outside=0.)
                yaw=env.truth[2]
                score_origin=np.array([env.truth[0]+np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1],
                    env.truth[1]+np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1],mount[2]])
                score_angles=core.lidar.angles+yaw+mount[3]
                directions=np.c_[np.cos(score_angles),np.sin(score_angles),np.zeros(len(score_angles))].reshape(-1)
                ids=np.zeros(len(score_angles),np.int32);ranges=np.zeros(len(score_angles))
                mujoco.mj_multiRay(env.model,env.data,score_origin,directions,np.array([1,0,0,0,0,0],np.uint8),
                    1,env.robot,ids,ranges,len(score_angles),core.lidar.range_max)
                scored_actor=np.array([mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(g))=='crossing_obstacle_geom'
                    if g>=0 else False for g in ids])
                observed=valid&scored_actor;kept=observed&(distances>.22)
                samples.append(dict(frame=index,timestamp=float(env.elapsed),choice=recorded['choice'],
                    actual_unmatched_returns=int(core.unmatched_returns),tracks=len(core.laser_fusion.predictor.tracks),
                    finite_actor_returns_score_only=int(observed.sum()),actor_returns_kept_score_only=int(kept.sum()),
                    actor_return_map_distances_score_only=distances[observed].tolist(),
                    measured_actor_ranges_score_only=core.scan[observed].tolist()))
            v,w=result['command'];env.step(np.array([v/(.2 if v>=0 else .08),w/.5]))
            if env.collision_events or env.hard_guard_activations:raise ValueError('Replay physical failure')
    finally:env.close()
    report=dict(trace_sha256=digest(a.trace),bundle_manifest_sha256=digest(a.bundle/'manifest.json'),
        actual_observation_and_command_reproduced=True,samples=samples,
        limitation='Ray IDs are simulation scoring labels after decisions. Endpoints and static distance use actual scan/estimated pose. Not a deployable detector, training signal or proof for every missing track.')
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps([{k:v for k,v in s.items() if not isinstance(v,list)} for s in samples]),flush=True)


if __name__=='__main__':main()
