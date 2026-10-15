"""Frozen masked actors on the same recorded training task, never eval replay learning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from experiments.fixed_training_replay import reset_training_episode
from experiments.masked_navigation import MaskedNavigationCurriculum
from neural_navigation.masked_actions import ACTIONS
from experiments.audit_stop_feedback import audit
from neural_navigation.safety_context import REASONS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-log',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--policy',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sensor-details-until',type=float,default=0.,
        help='Capture native sensors/tracks through this time; diagnostic only')
    args=parser.parse_args()
    if args.output.exists():parser.error('Fresh trace output required')
    records=[json.loads(line) for line in args.training_log.read_text().splitlines()]
    selected=[row for row in records if row.get('step')==args.step and 'training_case_json' in row and 'worker' in row]
    if len(selected)!=1:parser.error('Unique recorded training episode required')
    case=json.loads(selected[0]['training_case_json']);torch.set_num_threads(1)
    args.output.mkdir(parents=True);summaries=[]
    for index,path in enumerate(args.policy):
        fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
        model=MaskablePPO.load(path,device='cpu')
        env=MaskedNavigationCurriculum(path.parent,stage='robust',motion_profiles=True,footprint_clearance_cost=1.)
        rows=[];changed=0
        try:
            obs,_=reset_training_episode(env,case)
            for _ in range(env.max_steps):
                masks=env.action_masks();choice=int(model.predict(obs,action_masks=masks,deterministic=True)[0])
                if not masks[choice]:raise ValueError('Frozen actor selected a masked-out command')
                row=dict(timestamp=env.active.elapsed,estimate=env.core.estimate.tolist(),
                    policy_observation=obs.tolist(),action_mask=masks.tolist(),masked_choice=choice,
                    proposed_action=ACTIONS[choice].tolist(),safety_context=dict(
                        action_feasibility=obs[235:244].tolist(),
                        policy_previous_stop_reason=REASONS[int(np.argmax(obs[250:266]))]))
                if args.sensor_details_until>0 and env.active.elapsed<=args.sensor_details_until:
                    core=env.core; predictor=getattr(core.laser_fusion,'predictor',None)
                    tracks=predictor.tracks if predictor is not None else []
                    valid=np.isfinite(core.scan)&(core.scan>=core.lidar.range_min)&(core.scan<core.lidar.range_max)
                    mount=core.lidar.mount;angles=core.lidar.angles[valid]+mount[3]
                    points=core.scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]+mount[:2]
                    gaps=np.linalg.norm(points-np.clip(points,-env.guard.half,env.guard.half),axis=1)
                    row['sensor_diagnostic']=dict(
                        scan=[float(v) if np.isfinite(v) else None for v in core.scan],
                        measured_velocity=core.measured_velocity.tolist(),
                        odometry=core.odom.tolist(),path=core.path.tolist(),
                        scan_body_gap_m=float(gaps.min()) if len(gaps) else None,
                        pose_std=core.localizer.std.tolist(),
                        map_agreement=float(core.localizer.map_agreement),
                        scan_update_accepted=bool(core.localizer.scan_update_accepted),
                        tracks=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in t.items()} for t in tracks])
                    # Physics labels are kept separate and are never fed back
                    # to the core, action mask, actor or tracker.
                    row['scoring_only']=dict(truth=env.active.truth.tolist(),
                        actors=[p.tolist() for p in env.active._humans()],
                        world_min_gap_m=float(env.active._minimum_gap()))
                obs,_,done,truncated,info=env.step(choice)
                row.update(command=info['commanded_velocity'].tolist(),stop_reason=info['execution_guard_reason'],
                    nonzero_choice_changed_by_guard=info['nonzero_choice_changed_by_guard'])
                changed+=int(info['nonzero_choice_changed_by_guard']);rows.append(row)
                if done or truncated:break
            result=dict(policy=str(path.resolve()),policy_sha256=fingerprint,steps=len(rows),
                seconds=env.active.elapsed,travel_m=env.active.travel,
                safe_success=bool(info['verified_mission_success']),collisions=int(info['collision_events']),
                timeout=bool(truncated),nonzero_choices_changed_by_guard=changed,
                wait_with_current_alternative_frames=sum(r['masked_choice']==0 and any(r['action_mask'][1:]) for r in rows))
            trace=dict(kind='training_masked_replay',case=dict(case,id=f'training_step_{args.step}'),
                samples=rows,result=result,limitation='One reused training task; not independent evaluation. '
                    'Scans/estimated pose define actor inputs and masks; simulator labels score the outcome only.')
            trace['sensor_details_until_s']=args.sensor_details_until
            trace['sensor_metadata']=dict(angles=env.core.lidar.angles.tolist(),
                mount=env.core.lidar.mount.tolist(),footprint=(env.guard.half*2).tolist())
            feedback=audit(trace)
            if feedback['counts']['missing'] or feedback['counts']['feedback_mismatch']:
                raise ValueError('Recorded stop feedback does not match execution')
            trace['feedback_audit']=feedback
            if hashlib.sha256(path.read_bytes()).hexdigest()!=fingerprint:
                raise ValueError('Actor checkpoint changed during replay')
            (args.output/f'trial_{index}.json').write_text(json.dumps(trace,allow_nan=False))
            summaries.append(result);print(json.dumps(result),flush=True)
        finally:env.close()
    (args.output/'SUMMARY.json').write_text(json.dumps(dict(completed=True,kind='training_diagnostic',
        training_log_sha256=hashlib.sha256(args.training_log.read_bytes()).hexdigest(),
        case=case,trials=summaries,deployment_approved=False),indent=2))


if __name__=='__main__':main()
