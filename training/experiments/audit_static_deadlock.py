"""Reproduce first all-rejected frame; simulator truth is scoring only."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT
from neural_navigation.stress_env import StressNavigationEnv
from neural_navigation.static_footprint import static_footprint_clear
from neural_navigation.masked_actions import masks_from_observation,COMMANDS
from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime
from experiments.train_quality_navigation import digest
from neural_navigation.models import alignment_features,scan_map_agreement


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','trace','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--refine-shadow',action='store_true',help='Score finer learned pose hypotheses without applying them')
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh output required')
    recorded=json.loads(args.trace.read_text());case=recorded['case'];scene=ROOT/case['scene']
    torch.set_num_threads(1);env=StressNavigationEnv(case=case,components=args.bundle,scene_dir=scene,route_mode='shortcut',max_steps=700)
    try:
        options={k:case[k] for k in ('start','goal','initial_prior_offset') if k in case}
        env.reset(seed=case['seed'],options=options)
        prior=np.asarray(case['start'])+np.asarray(case.get('initial_prior_offset',[0,0,0]))
        runtime=QualitySensorNavigationRuntime(args.bundle,scene/'decision_map.yaml',prior,case['goal'],'shortcut')
        history=[];diagnosis=None
        for row in recorded['samples']:
            result=runtime.step(env.scan,env.observed_odometry,env.measured_velocity,env.elapsed,capture_observation=True)
            np.testing.assert_allclose(result['policy_observation'],row['policy_observation'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(result['command'],row['command'],atol=2e-8,rtol=0)
            assert result['stop_reason']==row['stop_reason']
            core=runtime.core;guard=runtime.guard;obs=np.asarray(result['policy_observation']);masks=masks_from_observation(obs[:282])
            score=dict(timestamp=float(env.elapsed),choice=row['choice'],estimate=core.estimate.tolist(),truth_score_only=env.truth.tolist(),
                pose_error_m=float(np.linalg.norm(core.estimate[:2]-env.truth[:2])),localizer_std=core.localizer.std.tolist(),
                map_agreement=float(core.localizer.map_agreement),scan_update_accepted=bool(core.localizer.scan_update_accepted),
                masks=masks.tolist(),stop_reason=result['stop_reason'])
            history.append(score)
            if not masks[1:].any():
                footprint=guard.half*2;static=core.static_map
                checks={label:{str(m):bool(static_footprint_clear(static,pose,footprint,m)) for m in (0.,.02,.04)}
                    for label,pose in [('estimated',core.estimate),('simulator_truth_score_only',env.truth)]}
                probes=[]
                for command in COMMANDS[1:]:
                    poses=list(guard._trajectory(core,command,0.))
                    first=next((i for i,(_,world,horizon) in enumerate(poses) if not guard._static_pose_clear(core,world,command,horizon)),None)
                    executed,reason=copy.deepcopy(guard).check(core,command,env.elapsed,env.elapsed)
                    probes.append(dict(command=command.tolist(),rejected_reason=reason,executed=executed.tolist(),
                        first_static_rejection_sample=first,predicted_world_poses=[world.tolist() for _,world,_ in poses]))
                diagnosis=dict(first_all_rejected=score,body_map_checks=checks,footprint_m=footprint.tolist(),
                    current_estimated_map_clearance_m=float(static.lookup(core.estimate[:2])),
                    current_true_map_clearance_score_only_m=float(static.lookup(env.truth[:2])),
                    probes=probes,preceding_frames=history[-12:])
                if args.refine_shadow:
                    candidates=[]
                    for spans in ((.12,.12,.08),(.06,.06,.04)):
                        features,hypotheses=alignment_features(static,core.lidar,core.estimate,core.scan,
                            spans=spans,points=7,directional=core.localizer.feature_dim==15)
                        with torch.no_grad():weights=torch.softmax(core.localizer.network(torch.from_numpy(features)),dim=0).numpy()
                        proposed=np.r_[np.sum(weights[:,None]*hypotheses[:,:2],axis=0),
                            np.arctan2(weights@np.sin(hypotheses[:,2]),weights@np.cos(hypotheses[:,2]))]
                        candidates.append(dict(spans=list(spans),pose=proposed.tolist(),
                            scan_map_agreement=scan_map_agreement(static,core.lidar,proposed,core.scan),
                            pose_error_truth_score_only_m=float(np.linalg.norm(proposed[:2]-env.truth[:2])),
                            body_clear_at_original_margin=bool(static_footprint_clear(static,proposed,footprint,.04))))
                    diagnosis['learned_refinement_shadow']=candidates
                    diagnosis['refinement_applied']=False
                break
            command=result['command'];env.step(np.array([command[0]/(.2 if command[0]>=0 else .08),command[1]/.5]))
            if env.collision_events:raise ValueError('Collision before the recorded all-rejected frame')
        if diagnosis is None:raise ValueError('No recorded all-rejected frame reproduced')
        report=dict(trace_sha256=digest(args.trace),bundle_manifest_sha256=digest(args.bundle/'manifest.json'),
            reproduced_observation_command_and_reason=True,diagnosis=diagnosis,
            limitation='One simulated fixed development case, no guard or actor changes. Simulator truth used only after decisions for diagnostic scoring; not actor input or training.')
        args.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in diagnosis.items() if k not in ('probes','preceding_frames')}))
    finally:env.close()


if __name__=='__main__':main()
