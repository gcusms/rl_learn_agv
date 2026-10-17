"""Correct offline circle labels into the predictor's odometry coordinates.

The first audit compared map-frame circles directly with odom-frame tracks.
Those circle mismatch labels are invalid. Neither predictor nor model consumed
them; this correction leaves the exact contact and action replays unchanged.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from neural_navigation.predictive_constraint import PredictiveConstraint
from neural_navigation.experiment_budget import ExperimentBudget
from audit import ROOT, HERE, digest, write


def main():
    output=HERE/'CIRCLE_FRAME_CORRECTION.json'
    assert not output.exists()
    replay_path=HERE/'PREDICTOR_REPLAY.json'
    replay=json.loads(replay_path.read_text())
    expected=next(r for r in replay['rows'] if r['source']=='dynamic01_v2_ppo1993')
    selected={r['frame'] for r in expected['examples']}
    result_path=HERE/'RESULTS.json'
    report=json.loads(result_path.read_text())
    path=next(ROOT/name for name in report['inputs'] if name.endswith('observed6_pre_ppo_holdout_dynamic_01_candidate.npz'))
    assert digest(path)==report['inputs'][str(path.relative_to(ROOT))]
    hardware_path=ROOT/'models/stop_feedback_20261015/guard_aligned_migration_s231017/portable/hardware.json'
    hardware=json.loads(hardware_path.read_text())
    lidar=SimpleNamespace(**hardware['lidar']);lidar.mount=np.asarray(lidar.mount)
    lidar.angles=np.linspace(lidar.angle_min,lidar.angle_max,lidar.beams)
    predictor=PredictiveConstraint(hardware['near_field_memory']['predictive_action_constraint'],hardware['robot_footprint_m'],.3)
    rows=[];budget=ExperimentBudget(.35,550,HERE)
    with np.load(path,allow_pickle=False) as z:
        for i in range(max(selected)+1):
            core=SimpleNamespace(elapsed=float(z['labels'][i,0]),odom=z['odometry'][i],scan=z['scans'][i],
                measured_velocity=z['measured_velocity'][i],lidar=lidar,odometry_sanity_failure=False)
            predictor.prepare(core)
            if i in selected:
                truth=z['labels'][i,1:4];odom=z['odometry'][i]
                c,s=np.cos(truth[2]),np.sin(truth[2]);co,so=np.cos(odom[2]),np.sin(odom[2])
                circles=(z['human_positions'][i]-truth[:2])@np.array([[c,-s],[s,c]])
                circles=circles@np.array([[co,so],[-so,co]])+odom[:2]
                tracks=[]
                for track in predictor.tracks:
                    radial=np.abs(np.linalg.norm(track['points'][:,None]-circles[None],axis=2)-.25)
                    tracks.append(dict(id=int(track['id']),source=track['source'],age_s=float(track['age']),
                        velocity_m_s=track['velocity'].tolist(),speed_m_s=float(np.linalg.norm(track['velocity'])),
                        mean_circle_surface_mismatch_m=radial.mean(axis=0).tolist(),
                        contour_position_uncertainty_m=float(track['contour_position_uncertainty_m']),
                        anchor_measurement_age_s=(float(core.elapsed-predictor.contours.states[track['id']]['anchor'][0])
                            if predictor.contours.states[track['id']]['anchor'] is not None else None)))
                rows.append(dict(frame=i,sensor_time_s=core.elapsed,tracks=tracks))
            budget.tick()
    write(output,dict(completed=True,invalid_previous_fields='PREDICTOR_REPLAY.json rows[].examples[].original_tracks[].mean_circle_surface_mismatch_m',
        reason='world circles were compared to odometry track points without frame transformation',
        corrected_transform='world circle to true body to recorded odometry frame, offline labels only',
        contact_and_action_replay_results_unchanged=True,reference_world_pose_used_only_for_offline_labels=True,
        rows=rows,new_physical_episodes=0,new_ppo_steps=0,resource=budget.statistics(),
        inputs={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),replay_path,result_path,path,hardware_path]},
        goal_achieved=False,deployment_approved=False))
    print(json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
