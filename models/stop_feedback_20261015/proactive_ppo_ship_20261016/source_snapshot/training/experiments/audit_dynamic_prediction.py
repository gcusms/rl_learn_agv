"""Exact sensor replay with read-only tracking diagnostics; truth is scoring only."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime
from experiments.train_quality_navigation import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    data=json.loads(args.trace.read_text());case=data['case'];scene=ROOT/case['scene']
    torch.set_num_threads(1);env=StressNavigationEnv(case=case,components=args.bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    rows=[];previous_humans=None;previous_time=None
    try:
        options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case};env.reset(seed=case['seed'],options=options)
        prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0]))
        runtime=QualitySensorNavigationRuntime(args.bundle,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
        for expected in data['samples']:
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            np.testing.assert_allclose(result['policy_observation'],expected['policy_observation'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(result['command'],expected['command'],atol=2e-8,rtol=0)
            assert result['stop_reason']==expected['stop_reason']
            core=runtime.core;prediction=runtime.policy.prediction;raw=core.laser_fusion.predictor.tracks
            # Work on copies: repeated diagnostics must not update live anchors.
            processed=copy.deepcopy(prediction.quality).update(raw,float(env.elapsed)) if result['quality_prediction_features'][12]>.5 else []
            humans=np.asarray(env._humans()).reshape(-1,2);velocities=np.zeros_like(humans)
            if previous_humans is not None and len(previous_humans)==len(humans):velocities=(humans-previous_humans)/(env.elapsed-previous_time)
            tracks=[]
            for track in processed:
                points=np.asarray(track['points']);distance=np.linalg.norm(humans-track['center'],axis=1) if len(humans) else np.array([])
                nearest=int(distance.argmin()) if len(distance) else None
                score_match=nearest is not None and distance[nearest]<.45
                tracks.append(dict(id=int(track['id']),time=float(track['time']),age=float(env.elapsed-track['time']),
                    center=np.asarray(track['center']).tolist(),velocity=np.asarray(track['velocity']).tolist(),hits=int(track['hits']),
                    count=len(points),diameter=prediction.quality.geometry(points)[0],
                    source=track['contour_velocity_source'],degraded=track['contour_degraded'],points=points.tolist(),
                    nearest_actor_score_only=nearest if score_match else None,
                    actor_velocity_score_only=velocities[nearest].tolist() if score_match else None,
                    velocity_error_score_only_m_s=float(np.linalg.norm(track['velocity']-velocities[nearest])) if score_match else None))
            rows.append(dict(timestamp=float(env.elapsed),choice=expected['choice'],command=result['command'],stop_reason=result['stop_reason'],
                actual_risk_scores=result['quality_prediction_features'][:12],quality_flags=result['quality_prediction_features'][12:],
                estimate=core.estimate.tolist(),measured_velocity=core.measured_velocity.tolist(),tracks=tracks,
                actor_positions_score_only=humans.tolist(),world_gap_score_only_m=float(env._minimum_gap())))
            previous_humans=humans.copy();previous_time=env.elapsed
            command=result['command'];env.step(np.array([command[0]/(.2 if command[0]>=0 else .08),command[1]/.5]))
            if env.collision_events:break
        report=dict(trace_sha256=digest(args.trace),bundle_manifest_sha256=digest(args.bundle/'manifest.json'),
            observations_commands_reproduced=True,collision_events=env.collision_events,samples=rows,
            limitation='Single simulated development failure. Nearest actor association is diagnostic, not identity proof. Truth never enters actor, tracking or prediction; no changed commands or training.')
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({'frames':len(rows),'collisions':env.collision_events}))
    finally:env.close()


if __name__=='__main__':main()
