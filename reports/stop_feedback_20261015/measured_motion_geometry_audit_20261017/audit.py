"""Compare nominal guard geometry with measured-velocity finite-response shadows.

Read-only: no neural/physics model loads, permission changes or positive labels.
An accepted instantaneous probe does not establish a complete feasible route.
"""
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.decision_feedback import FEEDBACK_REASONS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.sensors import LidarSpec
from neural_navigation.motion_response import force_response_trajectory
from navrl.geometry import integrate, wrap


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EARLY = {'nonfinite_input', 'stale_or_future_scan', 'scan_unavailable',
         'odometry_jump', 'localization_uncertain', 'predicted_dynamic_contact',
         'predictive_command_contract'}
FRAMES = (480, 490, 500, 520, 530, 531, 532, 533, 535, 537, 540, 550, 580, 583, 588, 620, 660)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = HERE / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def response_report(name, data, budget):
    nominal_errors, finite_errors, rows = [], [], []
    for frame in range(len(data['labels']) - 1):
        dt = float(data['labels'][frame + 1, 0] - data['labels'][frame, 0])
        assert abs(dt - .2) < 1e-6
        command = data['commands'][frame]
        velocity = data['measured_velocity'][frame]
        yaw = float(data['labels'][frame, 3])
        c, sn = np.cos(yaw), np.sin(yaw)
        world_delta = data['labels'][frame + 1, 1:3] - data['labels'][frame, 1:3]
        actual = np.r_[world_delta @ np.array([[c, -sn], [sn, c]]),
                       float(wrap(data['labels'][frame + 1, 3] - yaw))]
        nominal = integrate(np.zeros(3), *command, dt)
        finite = force_response_trajectory(velocity, command, dt)[-1]
        nominal_errors.append(np.abs(nominal - actual))
        finite_errors.append(np.abs(finite - actual))
        if command[1] * velocity[1] < -.005:
            rows.append(dict(frame=frame, sensor_time_s=float(data['labels'][frame, 0]),
                actual_executed_command=command.tolist(), measured_velocity=velocity.tolist(),
                actual_next_body_pose=actual.tolist(), nominal_next_body_pose=nominal.tolist(),
                finite_next_body_pose=finite.tolist(),
                actual_rotation_still_opposes_command=bool(command[1] * actual[2] < -1e-5)))
        if frame % 25 == 0:
            budget.tick()
    a, f = np.asarray(nominal_errors), np.asarray(finite_errors)
    return dict(trace=name, actual_executed_transition_frames=len(a),
        error_coordinates=["body_x_m", "body_y_m", "yaw_rad"],
        nominal_mean_absolute_xy_yaw_error=a.mean(0).tolist(),
        finite_mean_absolute_xy_yaw_error=f.mean(0).tolist(),
        nominal_p95_absolute_xy_yaw_error=np.quantile(a, .95, axis=0).tolist(),
        finite_p95_absolute_xy_yaw_error=np.quantile(f, .95, axis=0).tolist(),
        reversal_frames=len(rows), actual_opposite_rotation_frames=sum(r['actual_rotation_still_opposes_command'] for r in rows),
        reversal_records=rows, truth_only_for_historical_error_measurement=True,
        finite_prediction_uses_only_measured_velocity_and_executed_command=True,
        approximate_body_response_not_calibrated_real_car_model=True)


def main():
    assert not (HERE / 'RESULTS.json').exists()
    geometry_path = BASE / 'actual106_resolution_audit_20261017/audit.py'
    spec = importlib.util.spec_from_file_location('unchanged_guard_geometry_audit', geometry_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    geometry = module.geometry
    hardware_path = ROOT / 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017/portable/hardware.json'
    hardware = json.loads(hardware_path.read_text())
    config = hardware['near_field_memory']
    assert config.get('motion_response_schema') is None
    assert hardware['robot_footprint_m'] == [.393924904, .329460022]
    assert config['static_footprint_margin_m'] == .04
    traces = {
        'current_model_106': BASE / 'full_navigation_joint_linear_validation_20261017/physical/hard3_w11_training_stall_t106_candidate.npz',
        'creep_04_teacher': BASE / 'late106_creep_teacher_20261017/probe_0/trace.npz',
        'original96_success': BASE / 'hybrid_memory_pair_preflight_20261017/physical/legal_wait_parent_t90_control.npz',
    }
    paths = [Path(__file__), geometry_path, hardware_path, *traces.values(),
        ROOT / 'training/neural_navigation/motion_response.py',
        ROOT / 'training/navrl/physical_env.py', ROOT / 'training/navrl/geometry.py']
    paths += [ROOT / 'training/neural_navigation' / name for name in
              ('safety_memory.py', 'sensors.py', 'runtime.py', 'decision_feedback.py', 'masked_actions.py')]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    write('PLAN.json', dict(inputs=hashes, selected_frames=FRAMES,
        original_recorded_body_margin_memory_unknown_and_actions_retained=True,
        counterfactual_finite_response_only_not_new_mask_permissions=True,
        no_runtime_model_or_physical_changes=True,
        actual_guard_geometry_schema='nominal_constant_command',
        uses_recorded_sensor_odometry_and_execution_history=True,
        raw_only_checks_are_not_action_permission=True,
        no_model_or_physics_loads=True, no_positive_training_labels=True,
        new_physical_steps=0, new_training_updates=0, new_ppo_steps=0))
    budget = ExperimentBudget(.1, 550, HERE)
    lidar = LidarSpec(hardware['lidar'])
    half = np.asarray(hardware['robot_footprint_m']) / 2
    records = []
    checked = matched = 0
    summaries = []
    response_tracking = []
    finite_config = dict(config, motion_response_schema='force_response_v1',
        response_gain_s_inv=20., response_linear_acceleration_m_s2=.5,
        response_angular_acceleration_rad_s2=1.)
    for name, path in traces.items():
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        response_tracking.append(response_report(name, data, budget))
        memory = NearFieldMemory(config)
        start = len(records)
        for frame in range(max(FRAMES) + 1):
            now = float(data['labels'][frame, 0])
            core = SimpleNamespace(elapsed=now, odom=data['odometry'][frame],
                scan=data['scans'][frame], lidar=lidar)
            execution_reason = (str(data['executed_stop_reason'][frame])
                if 'executed_stop_reason' in data else
                FEEDBACK_REASONS[int(data['observations'][frame + 1, 399:419].argmax())])
            shadow = copy.deepcopy(memory)
            old, uncertainty = shadow.update(core)
            if frame in FRAMES:
                valid = np.isfinite(core.scan) & (core.scan >= lidar.range_min) & (core.scan < lidar.range_max)
                angle = lidar.angles[valid] + lidar.mount[3]
                current = core.scan[valid, None] * np.c_[np.cos(angle), np.sin(angle)] + lidar.mount[:2]
                seen = np.array([entry[1] for entry in shadow.entries.values()])
                assert len(seen) == len(old)
                all_points = np.vstack([current, old])
                margins = np.r_[np.full(len(current), .04), .04 + uncertainty]
                groups = data['observations'][frame, 303:399].reshape(12, 8).argmax(1)
                probes = []
                for action in range(1, 12):
                    combined = geometry(all_points, margins, data['measured_velocity'][frame],
                        COMMANDS[action], half, len(current), seen, now, config)
                    raw = geometry(current, np.full(len(current), .04), data['measured_velocity'][frame],
                        COMMANDS[action], half, len(current), np.empty(0), now, config)
                    finite = geometry(all_points, margins, data['measured_velocity'][frame],
                        COMMANDS[action], half, len(current), seen, now, finite_config)
                    if int(groups[action]) in (0, 4):
                        expected = int(groups[action]) == 0
                        checked += 1
                        matched += int(combined['laser_geometry_clear'] == expected)
                        assert combined['laser_geometry_clear'] == expected, (name, frame, action, int(groups[action]), combined)
                    worst = combined['worst']
                    if worst is not None and len(current):
                        worst['nearest_current_return_distance_m'] = float(np.linalg.norm(
                            current - np.asarray(worst['point_in_initial_body']), axis=1).min())
                    probes.append(dict(action=action, command=COMMANDS[action].tolist(),
                        recorded_allowed=bool(data['masks'][frame, action]),
                        recorded_cause_group=int(groups[action]),
                        raw_returns_alone_clear=raw['laser_geometry_clear'],
                        finite_response_geometry=finite,
                        nominal_passes_finite_rejects=combined['laser_geometry_clear'] and not finite['laser_geometry_clear'],
                        nominal_rejects_finite_passes=not combined['laser_geometry_clear'] and finite['laser_geometry_clear'],
                        **combined))
                records.append(dict(trace=name, frame=frame, sensor_time_s=now,
                    recorded_choice=int(data['choices'][frame]),
                    recorded_execution_reason=execution_reason,
                    legal_motion_actions=(np.flatnonzero(data['masks'][frame, 1:]) + 1).tolist(),
                    remembered_points=len(old), probes=probes))
            if execution_reason not in EARLY:
                memory = shadow
            if execution_reason == 'odometry_jump':
                memory.reset()
            if frame % 25 == 0:
                budget.tick()
        release = 480 if name == 'original96_success' else 530
        suffix = data['masks'][release:]
        chosen = data['choices'][release:]
        rows = records[start:]
        summaries.append(dict(trace=name, controlled_frames=len(chosen),
            only_stop_frames=int((~suffix[:, 1:].any(1)).sum()),
            stop_with_motion_frames=int(((chosen == 0) & suffix[:, 1:].any(1)).sum()),
            nonzero_selected_frames=int((chosen != 0).sum()),
            sampled_nominal_pass_finite_reject=sum(probe['nominal_passes_finite_rejects'] for row in rows for probe in row['probes']),
            sampled_nominal_reject_finite_pass=sum(probe['nominal_rejects_finite_passes'] for row in rows for probe in row['probes']),
            sampled_geometry_checks=sum(len(row['probes']) for row in rows),
            sampled_raw_clear_but_memory_geometry_blocks=sum(
                probe['raw_returns_alone_clear'] and not probe['laser_geometry_clear']
                for row in rows for probe in row['probes'])))
    for path, sha in hashes.items():
        assert digest(ROOT / path) == sha, path
    budget.tick()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        model='models/stop_feedback_20261015/full_navigation_joint_linear_s231017',
        geometry_checks_matched=matched, geometry_checks_tested=checked,
        summaries=summaries, response_tracking=response_tracking,
        counterfactual_config=finite_config, records=records, inputs=hashes, resource=budget.statistics(),
        no_runtime_or_permission_changes=True, no_success_or_infeasibility_claim=True,
        no_new_localization_or_generalization_claim=True,
        new_physical_steps=0, new_training_updates=0, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)
    write('RESULTS.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('records', 'inputs')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write('FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
