"""Replay measured near-field memory using recorded scans and input odometry.

Alternative clearing configurations are evaluated only at the fixed recorded
poses. No guard is changed, no action is authorized and no physics is run.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from neural_navigation.decision_feedback import FEEDBACK_REASONS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.motion_response import force_response_trajectory
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.sensors import LidarSpec


ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'reports/stop_feedback_20261015'
HERE=Path(__file__).resolve().parent
EARLY_REASONS={'nonfinite_input','stale_or_future_scan','scan_unavailable','odometry_jump',
    'localization_uncertain','predicted_dynamic_contact','predictive_command_contract'}


def main():
    output=HERE/'MEMORY_GEOMETRY_AUDIT.json'
    assert not output.exists()
    budget=ExperimentBudget(.1,memory_mib=650,directory=HERE)
    hardware_path=ROOT/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable/hardware.json'
    hardware=json.loads(hardware_path.read_text());lidar=LidarSpec(hardware['lidar'])
    half=np.asarray(hardware['robot_footprint_m'])/2
    prefix_path=BASE/'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    with np.load(prefix_path,allow_pickle=False) as saved:
        prefix={k:saved[k].copy() for k in ['scans','odometry','labels','observations']}
    snapshot_path=HERE/'FIRST3_GEOMETRY_ROWS.json';rows=json.loads(snapshot_path.read_text())
    configs={'original':hardware['near_field_memory'],
        'finite_free_3cm_3frames':dict(hardware['near_field_memory'],free_evidence_margin_m=.03,
            free_evidence_frames=3,free_evidence_max_gap_s=.4)}
    initial={name:NearFieldMemory(config) for name,config in configs.items()}
    for index in range(450):
        core=SimpleNamespace(elapsed=float(prefix['labels'][index,0]),odom=prefix['odometry'][index],
            scan=prefix['scans'][index],lidar=lidar)
        reason=FEEDBACK_REASONS[int(prefix['observations'][index+1,399:419].argmax())]
        for memory in initial.values():
            if reason=='odometry_jump':memory.reset()
            if reason not in EARLY_REASONS:memory.update(core)
        if index%25==0:budget.tick()
    results=[];files=[Path(__file__),hardware_path,prefix_path,snapshot_path,
        ROOT/'training/neural_navigation/safety_memory.py',ROOT/'training/neural_navigation/motion_response.py']
    for row in rows:
        path=ROOT/row['rollout_path'];files.append(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['rollout_sha256']
        with np.load(path,allow_pickle=False) as saved:
            data={k:saved[k].copy() for k in saved.files}
        memories=copy.deepcopy(initial)
        counts={name:dict(reconstructed_rejections=0,current_raw_clear_rejections=0) for name in configs}
        actual=0;original_misses=0;examples=[]
        for index,scan in enumerate(data['raw_scans']):
            core=SimpleNamespace(elapsed=float(data['labels'][index,0]),odom=data['input_odometry'][index],scan=scan,lidar=lidar)
            executed_reason=str(data['stop_reasons'][index])
            remembered={}
            for name,memory in memories.items():
                shadow=copy.deepcopy(memory) if executed_reason in EARLY_REASONS else memory
                remembered[name]=shadow.update(core)
                if executed_reason=='odometry_jump':memory.reset()
            if data['masks'][index,1:].any():continue
            valid=np.isfinite(scan)&(scan>=lidar.range_min)&(scan<lidar.range_max)
            angles=lidar.angles[valid]+lidar.mount[3]
            current=scan[valid,None]*np.c_[np.cos(angles),np.sin(angles)]+lidar.mount[:2]
            velocity=data['input_measured_velocity'][index]
            for action,command in [(4,np.array([0.,.5])),(5,np.array([0.,-.5]))]:
                if data['per_action_guard_reasons'][index,action]!='laser_braking_footprint':continue
                actual+=1
                horizon=max(.2,abs(velocity[0])/.3+.2,abs(velocity[1]-command[1])+.2)
                poses=force_response_trajectory(velocity,command,horizon,20.,(.5,1.))
                raw_min=np.inf;failures={}
                for name,(old,uncertainty) in remembered.items():
                    points=np.vstack([current,old]);margins=np.r_[np.full(len(current),.04),.04+uncertainty]
                    failed=None
                    for local in poses:
                        c,s=np.cos(local[2]),np.sin(local[2])
                        relative=(points-local[:2])@np.array([[c,-s],[s,c]])
                        gaps=np.linalg.norm(relative-np.clip(relative,-half,half),axis=1)
                        if name=='original' and len(current):raw_min=min(raw_min,float(gaps[:len(current)].min()))
                        hit=np.flatnonzero(gaps<margins-1e-4)
                        if len(hit) and failed is None:failed=int(hit[0])
                    failures[name]=failed
                    counts[name]['reconstructed_rejections']+=int(failed is not None)
                if failures['original'] is None:original_misses+=1
                if raw_min>=.04-1e-4:
                    for name,failed in failures.items():
                        counts[name]['current_raw_clear_rejections']+=int(failed is not None)
                    culprit=failures['original']
                    if culprit is not None and culprit>=len(current) and len(examples)<8:
                        key=list(memories['original'].entries)[culprit-len(current)]
                        point,seen=memories['original'].entries[key]
                        examples.append(dict(time_s=core.elapsed,action=action,
                            remembered_point_world=point.tolist(),last_seen_s=float(seen),age_s=core.elapsed-float(seen),
                            current_raw_trajectory_minimum_gap_m=raw_min,
                            three_frame_finite_free_variant_still_rejects=failures['finite_free_3cm_3frames'] is not None))
            if index%25==0:budget.tick()
        results.append(dict(episode=row['episode'],actual_laser_rejected_turns=actual,
            original_reconstruction_misses=original_misses,variants=counts,examples=examples))
    budget.tick()
    result=dict(completed=True,episodes=results,initialization_sensor_frames_replayed=450,
        prefix_actual_stop_reasons_decoded_from_next_observation=True,
        original_rejections_reproduced=all(r['original_reconstruction_misses']==0 for r in results),
        alternative_is_fixed_trajectory_diagnostic_not_closed_loop=True,
        independent_static_or_dynamic_obstacle_truth_not_used_to_clear_memory=True,
        uncertainty_4cm_body_masks_and_runtime_not_changed=True,
        no_claim_that_removed_points_are_physically_safe_without_closed_loop_validation=True,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        resource=budget.statistics(),no_new_physics=True,no_optimizer_updates=True,new_ppo_steps=0,
        goal_achieved=False,deployment_approved=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
