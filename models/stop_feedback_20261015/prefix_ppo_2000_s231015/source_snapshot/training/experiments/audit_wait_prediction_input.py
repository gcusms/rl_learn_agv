"""Exact sensor replay with prediction-support diagnostics; no action edits."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from experiments.time_consistent_tracking import TimeConsistentFreshRuntime
from experiments.train_quality_navigation import digest


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames',type=int,default=120)
    a=p.parse_args()
    if a.output.exists() or a.frames<1:p.error('Fresh output and positive frame count required')
    torch.set_num_threads(1);trace=json.loads(a.trace.read_text());case=trace['case'];scene=ROOT/case['scene']
    env=StressNavigationEnv(case=case,components=a.bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    rows=[]
    try:
        env.reset(seed=case['seed'],options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case})
        runtime=TimeConsistentFreshRuntime(a.bundle,scene/'decision_map.yaml',
            np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0])),case['goal'],'shortcut')
        for index,recorded in enumerate(trace['samples'][:a.frames]):
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            np.testing.assert_allclose(result['policy_observation'],recorded['policy_observation'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(result['command'],recorded['command'],atol=2e-8,rtol=0)
            assert result['stop_reason']==recorded['stop_reason']
            core=runtime.core;loc=core.localizer;predictor=core.laser_fusion.predictor
            obs=np.asarray(result['policy_observation'])
            rows.append(dict(frame=index,timestamp=float(env.elapsed),choice=recorded['choice'],
                feasible_choices=np.flatnonzero(recorded['mask']).tolist(),stop_reason=result['stop_reason'],
                map_agreement=float(loc.map_agreement),scan_update_accepted=bool(loc.scan_update_accepted),
                localization_std=loc.std.tolist(),odometry_sanity_failure=bool(getattr(core,'odometry_sanity_failure',False)),
                prediction_alignment_supported=bool(core.laser_fusion.last_alignment_supported),
                unmatched_returns=int(core.unmatched_returns),quality_features=obs[282:].tolist(),
                route_probability=float(obs[229]),waypoint_heading_cosine=float(obs[232]),
                remembered_endpoints=len(runtime.guard.memory.entries),
                tracks=[dict(id=t['id'],hits=t['hits'],age_s=float(env.elapsed-t['time']),
                    speed_m_s=float(np.linalg.norm(t['velocity'])),velocity=t['velocity'].tolist(),
                    contour_points=len(t['points']),confirmation_age_s=float(t['time']-t.get('confirmation_started',t['time'])))
                    for t in predictor.tracks]))
            v,w=result['command'];env.step(np.array([v/(.2 if v>=0 else .08),w/.5]))
            if env.collision_events or env.hard_guard_activations:raise ValueError('Unexpected replay physical failure')
    finally:env.close()
    report=dict(trace_sha256=digest(a.trace),bundle_manifest_sha256=digest(a.bundle/'manifest.json'),
        actual_observation_command_and_reason_reproduced=True,frames=len(rows),samples=rows,
        limitation='Diagnostic prediction-state logs, no action replacement, no simulator truth actor input, no training labels. Does not establish cause of all waits.')
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    waits=[r for r in rows if r['choice']==0 and len(r['feasible_choices'])>1]
    print(json.dumps(dict(frames=len(rows),voluntary_wait_frames=len(waits),
        wait_without_tracks=sum(not r['tracks'] for r in waits),
        wait_prediction_supported=sum(r['quality_features'][12]>.5 for r in waits))),flush=True)


if __name__=='__main__':main()
