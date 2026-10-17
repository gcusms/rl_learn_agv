"""Separate current-return geometry from unrecorded remembered obstacles.

This never authorizes an action: static geometry, memory and prediction still
apply. A clear current-ray trajectory does not prove true-world safety.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.motion_response import force_response_trajectory
from neural_navigation.sensors import LidarSpec


ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'reports/stop_feedback_20261015'
HERE=Path(__file__).resolve().parent


def main():
    output=HERE/'FIRST3_RAW_GEOMETRY_AUDIT.json'
    assert not output.exists()
    budget=ExperimentBudget(.1,memory_mib=600,directory=HERE)
    report=json.loads((BASE/'native625_batch_credit_pilot_20261017/RESULTS.json').read_text())
    rows=report['episodes'][:3]
    assert len(rows)==3
    snapshot=HERE/'FIRST3_GEOMETRY_ROWS.json';assert not snapshot.exists()
    snapshot.write_text(json.dumps(rows,indent=2)+'\n')
    hardware_path=ROOT/'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable/hardware.json'
    hardware=json.loads(hardware_path.read_text());lidar=LidarSpec(hardware['lidar'])
    half=np.asarray(hardware['robot_footprint_m'],float)/2
    files=[Path(__file__),snapshot,hardware_path,ROOT/'training/neural_navigation/motion_response.py']
    results=[]
    for row in rows:
        path=ROOT/row['rollout_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['rollout_sha256']
        files.append(path)
        with np.load(path,allow_pickle=False) as archive:
            data={k:archive[k].copy() for k in archive.files}
        tested=clear=0;examples=[]
        for index,scan in enumerate(data['raw_scans']):
            if data['masks'][index,1:].any():
                continue
            valid=np.isfinite(scan)&(scan>=lidar.range_min)&(scan<lidar.range_max)
            angle=lidar.angles[valid]+lidar.mount[3]
            points=scan[valid,None]*np.c_[np.cos(angle),np.sin(angle)]+lidar.mount[:2]
            if not len(points):
                continue
            velocity=data['input_measured_velocity'][index]
            initial=float(np.linalg.norm(points-np.clip(points,-half,half),axis=1).min())
            for action,command in [(4,np.array([0.,.5])),(5,np.array([0.,-.5]))]:
                reason=str(data['per_action_guard_reasons'][index,action])
                if reason!='laser_braking_footprint':
                    continue
                tested+=1
                horizon=max(.2,abs(velocity[0])/.3+.2,abs(velocity[1]-command[1])+.2)
                poses=force_response_trajectory(velocity,command,horizon,20.,(.5,1.))
                minimum=np.inf
                for local in poses:
                    c,s=np.cos(local[2]),np.sin(local[2])
                    relative=(points-local[:2])@np.array([[c,-s],[s,c]])
                    minimum=min(minimum,float(np.linalg.norm(relative-np.clip(relative,-half,half),axis=1).min()))
                if minimum>=.04-1e-4:
                    clear+=1
                    if len(examples)<5:
                        examples.append(dict(time_s=float(data['labels'][index,0]),action=action,
                            raw_initial_minimum_gap_m=initial,raw_trajectory_minimum_gap_m=minimum,
                            horizon_s=horizon,reported_reason=reason))
            if index%25==0:
                budget.tick()
        results.append(dict(episode=row['episode'],training_safe_success=row['safe_success'],
            only_stop_frames=row['only_stop_frames'],laser_rejected_pure_turn_candidates=tested,
            candidates_clear_of_current_raw_returns_only=clear,examples=examples))
    budget.tick()
    result=dict(completed=True,episodes=results,
        identical_declared_measured_response_and_4cm_margin_used=True,
        static_map_memory_uncertainty_and_dynamic_predictions_not_removed_from_runtime=True,
        remembered_points_not_logged_so_their_accuracy_remains_unverified=True,
        raw_return_clearance_does_not_certify_an_action_safe=True,
        no_guard_or_action_permission_changes=True,
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        resource=budget.statistics(),no_new_physics=True,no_optimizer_updates=True,
        new_ppo_steps=0,goal_achieved=False,deployment_approved=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
