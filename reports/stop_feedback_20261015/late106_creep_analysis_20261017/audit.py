"""Reconstruct unchanged measured geometry behind failed recovery decisions.

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


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EARLY = {'nonfinite_input', 'stale_or_future_scan', 'scan_unavailable',
         'odometry_jump', 'localization_uncertain', 'predicted_dynamic_contact',
         'predictive_command_contract'}
FRAMES = (530, 531, 532, 533, 535, 537, 540, 550, 580, 583, 588, 620, 698)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = HERE / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


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
    }
    paths = [Path(__file__), geometry_path, hardware_path, *traces.values()]
    paths += [ROOT / 'training/neural_navigation' / name for name in
              ('safety_memory.py', 'sensors.py', 'runtime.py', 'decision_feedback.py', 'masked_actions.py')]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    write('PLAN.json', dict(inputs=hashes, selected_frames=FRAMES,
        original_body_margin_memory_unknown_and_action_permissions_unchanged=True,
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
    for name, path in traces.items():
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
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
                        raw_returns_alone_clear=raw['laser_geometry_clear'], **combined))
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
        suffix = data['masks'][530:]
        chosen = data['choices'][530:]
        rows = records[start:]
        summaries.append(dict(trace=name, controlled_frames=len(chosen),
            only_stop_frames=int((~suffix[:, 1:].any(1)).sum()),
            stop_with_motion_frames=int(((chosen == 0) & suffix[:, 1:].any(1)).sum()),
            nonzero_selected_frames=int((chosen != 0).sum()),
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
        summaries=summaries, records=records, inputs=hashes, resource=budget.statistics(),
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
