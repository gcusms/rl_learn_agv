"""Inspect the previously observed collision pair, without a new rollout.

Contacts are the recorded sensor predictor's outputs. Object positions are used
only for explanatory physical labels and never as proposed policy inputs.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / 'KNOWN_COLLISION_AUDIT.json'
    assert not output.exists()
    report_path = HERE.parent / 'native625_local_speedup_retention_20261017/RESULTS.json'
    raw = report_path.read_bytes()
    source = json.loads(raw)
    hardware = ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable/hardware.json'
    spec = json.loads(hardware.read_text())
    inputs = [Path(__file__), report_path, hardware,
              ROOT / 'training/experiments/evaluate_docking_navigation.py',
              ROOT / 'training/navrl/physical_env.py',
              ROOT / 'training/neural_navigation/stress_env.py']
    rows = []
    for row in source['frozen_rows']:
        if row['case_id'] != 'pre_ppo_holdout_sensor_01':
            continue
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        inputs.append(path)
        with np.load(path, allow_pickle=False) as z:
            times = z['labels'][:, 0]
            x, choices, masks = z['observations'], z['choices'], z['masks']
            short = z['constraint_diagnostics'][:, 3:15]
            valid = (x[:,435] > .5) & (x[:,438] > .5)
            finite = np.isfinite(short) & valid[:,None]
            lower = np.zeros_like(short, np.float32)
            lower[finite] = np.clip(1.-short[finite]/5.2, 0., 1.)
            missing = finite & (x[:,467:479] < lower-1e-7)
            only_stop = ~masks[:,1:].any(1)
            suffix = len(choices)
            while suffix and only_stop[suffix-1]:
                suffix -= 1
            stop_commands = np.max(abs(z['commands']), axis=1) < 1e-8
            stop_begin = len(choices)
            while stop_begin and stop_commands[stop_begin-1]:
                stop_begin -= 1
            diagnostics = []
            for i in range(max(0, len(choices)-13), len(choices)):
                pose = z['labels'][i,1:4]
                delta = z['human_positions'][i,0]-pose[:2]
                c, s = np.cos(pose[2]), np.sin(pose[2])
                body = delta @ np.array([[c,-s],[s,c]])
                half = np.asarray(spec['robot_footprint_m'])/2
                # Same rectangle/circle formula as PhysicalNavigationEnv.robot_gap.
                gap = np.linalg.norm(body-np.clip(body,-half,half))-.25
                lidar_delta = body-np.asarray(spec['lidar']['mount'][:2])
                bearing = float(np.arctan2(lidar_delta[1],lidar_delta[0]))
                angular_radius = float(np.arcsin(min(1., .25/np.linalg.norm(lidar_delta))))
                possible_fov_overlap = (bearing+angular_radius >= spec['lidar']['angle_min']
                    and bearing-angular_radius <= spec['lidar']['angle_max'])
                diagnostics.append(dict(sensor_time_s=float(times[i]), action=int(choices[i]),
                    legal_actions=np.flatnonzero(masks[i]).tolist(),
                    supported_tracks=int(z['constraint_diagnostics'][i,0]),
                    stop_short_ttc_s=float(short[i,0]) if np.isfinite(short[i,0]) else None,
                    old_stop_long_risk=float(x[i,467]),
                    stop_guard_risk_lower_bound=float(lower[i,0]),
                    actual_human_rectangle_gap_before_step_m=float(gap),
                    human_center_bearing_rad=bearing,
                    ideal_circle_can_overlap_fov=bool(possible_fov_overlap),
                    possible_overlap_is_not_measured_visibility=True,
                    physical_forward_velocity_m_s=float(z['labels'][i,9]),
                    physical_yaw_velocity_rad_s=float(z['labels'][i,10])))
            rows.append(dict(variant=row['variant'], source=row['trace_path'],
                collisions=row['collisions'], terminal_seconds=row['seconds'], recorded_frames=len(choices),
                frames_with_valid_short_contact=int(finite.any(1).sum()),
                frames_with_omitted_short_contact_in_old_history=int(missing.any(1).sum()),
                stop_contact_frames=int(finite[:,0].sum()),
                omitted_stop_contact_frames=int(missing[:,0].sum()),
                final_only_stop_suffix_frames=len(choices)-suffix,
                final_only_stop_suffix_begin_s=float(times[suffix]) if suffix<len(choices) else None,
                final_zero_command_suffix_frames=len(choices)-stop_begin,
                final_zero_command_suffix_begin_s=float(times[stop_begin]) if stop_begin<len(choices) else None,
                final_nonstop_rejection_groups=dict(Counter(
                    x[-1,303:399].reshape(12,8)[1:].argmax(1).tolist())),
                late_frames=diagnostics))
    assert len(rows) == 2 and all(row['collisions'] == 1 for row in rows)
    (HERE / 'KNOWN_COLLISION_SOURCE_SNAPSHOT.json').write_bytes(raw)
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), rows=rows,
        same_known_map_previously_observed_development_pair=True,
        no_new_physics=True, new_ppo_steps=0, new_optimizer_updates=0,
        old_v1_policy_did_not_receive_all_recorded_short_guard_contact_risk=True,
        v2_lower_bound_is_input_repair_not_evidence_of_collision_prevention=True,
        stopped_at_collision_does_not_waive_original_safety_failure=True,
        collision_happened_after_last_recorded_pre_step_state=True,
        all_used_object_positions_are_offline_labels_only=True,
        remaining_requires_actual_v2_closed_loop_validation=True,
        inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        goal_achieved=False,deployment_approved=False)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='late_frames'} for row in rows],ensure_ascii=False))


if __name__ == '__main__':
    main()
