"""Test whether finer yaw choices address the measured late-state veto.

This is a fixed-sensor-trajectory geometry diagnostic, not an action mask,
counterfactual rollout, new demonstration, or permission to move. The original
memory is reconstructed using the actually executed stop reasons. Alternative
memory and angular speeds never affect the recorded commands or any runtime.
"""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from navrl.geometry import integrate
from neural_navigation.decision_feedback import FEEDBACK_REASONS
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.masked_actions import COMMANDS
from neural_navigation.motion_response import force_response_trajectory
from neural_navigation.safety_memory import NearFieldMemory
from neural_navigation.sensors import LidarSpec


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
EARLY = {'nonfinite_input', 'stale_or_future_scan', 'scan_unavailable',
         'odometry_jump', 'localization_uncertain', 'predicted_dynamic_contact',
         'predictive_command_contract'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def geometry(points, margins, velocity, command, half, raw_count, seen, now, motion_config):
    initial = np.linalg.norm(points - np.clip(points, -half, half), axis=1)
    escaping = bool(command[0] > .005 and np.any(initial < margins))
    required = np.minimum(initial, margins) if escaping else margins
    horizon = abs(command[0])/.3 + .2
    if motion_config.get('motion_response_schema') is None:
        poses = np.array([integrate(np.zeros(3), *command, t) for t in np.linspace(0., horizon, 9)])
    else:
        acceleration = np.array([motion_config.get('response_linear_acceleration_m_s2', .5),
                                 motion_config.get('response_angular_acceleration_rad_s2', 1.)])
        horizon = max(horizon, abs(velocity[0])/.3+.2,
                      abs(velocity[1]-command[1])/acceleration[1]+.2)
        poses = force_response_trajectory(velocity, command, horizon,
            motion_config.get('response_gain_s_inv', 20.), acceleration)
    minimum = np.inf
    worst = None
    raw_minimum = np.inf
    for index, pose in enumerate(poses):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        relative = (points - pose[:2]) @ np.array([[c, -s], [s, c]])
        gaps = np.linalg.norm(relative - np.clip(relative, -half, half), axis=1)
        excess = gaps - required
        if raw_count:
            raw_minimum = min(raw_minimum, float(excess[:raw_count].min()))
        if len(excess) and float(excess.min()) < minimum:
            j = int(excess.argmin())
            minimum = float(excess[j])
            worst = dict(sample=index, local_pose=pose.tolist(),
                point_in_initial_body=points[j].tolist(),
                source='current_scan' if j < raw_count else 'retained_measurement',
                retained_age_s=None if j < raw_count else float(now-seen[j-raw_count]),
                initial_gap_m=float(initial[j]), sampled_gap_m=float(gaps[j]),
                nominal_margin_m=float(margins[j]), required_gap_m=float(required[j]),
                gap_minus_required_m=minimum)
    escape_incomplete = bool(escaping and np.any(gaps[initial < margins] < initial[initial < margins]+.005))
    return dict(laser_geometry_clear=bool(minimum >= -1e-4 and not escape_incomplete),
        minimum_margin_excess_m=None if not np.isfinite(minimum) else minimum,
        current_scan_minimum_excess_m=None if not np.isfinite(raw_minimum) else raw_minimum,
        escape_incomplete=escape_incomplete, horizon_s=horizon, worst=worst)


def main():
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('An audit must not overwrite its earlier evidence')
    hardware_path = ROOT / 'models/stop_feedback_20261015/normal_suite_wait_pair_v2_s231016/wait_continuation/portable/hardware.json'
    trace_path = BASE / 'context_broad_diagnostic_20261016/physical/hard3_w11_training_stall_t106.npz'
    hardware = json.loads(hardware_path.read_text())
    with np.load(trace_path, allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in ['labels', 'observations', 'scans',
                'odometry', 'measured_velocity', 'masks', 'choices']}
    assert data['observations'].shape == (700, 625)
    assert tuple(hardware['robot_footprint_m']) == (.393924904, .329460022)
    assert hardware['near_field_memory']['static_footprint_margin_m'] == .04
    assert hardware['near_field_memory'].get('motion_response_schema') in (None, 'force_response_v1')
    files = [Path(__file__), hardware_path, trace_path]
    for name in ['decision_feedback.py', 'safety_context.py', 'safety_memory.py',
                 'sensors.py', 'masked_actions.py', 'motion_response.py', 'experiment_budget.py', 'runtime.py']:
        files.append(ROOT / 'training/neural_navigation' / name)
    files += [OUT / 'INITIAL_FAILURE.json', OUT / 'audit_initial.py', ROOT / 'training/navrl/geometry.py']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    selected = [530, 535, 550, 565, 580, 600, 625, 650, 675, 698]
    speeds = [.05, .1, .15, .2, .3, .5]
    hypotheses = [(f'pure_yaw_{sign*w:+.2f}', np.array([0., sign*w]))
                  for sign in [-1, 1] for w in speeds]
    hypotheses += [(f'original_{i}', COMMANDS[i]) for i in [4, 5, 8, 9]]
    write('PLAN.json', dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), inputs=hashes,
        selected_frames=selected, sampled_yaw_speeds_rad_s=speeds,
        hypothesis='Finer yaw may expose an admissible continuation unavailable at0.5rad/s',
        geometry_only=True, no_runtime_or_contract_changes=True, no_true_obstacle_positions_used=True,
        original_recorded_trajectory_unchanged=True, new_ppo_steps=0, new_physical_steps=0,
        initial_diagnostic_precondition_failure_preserved='INITIAL_FAILURE.json',
        initial_failed_unit='rl-actual106-resolution-audit-20261017.service',
        corrected_diagnostic_reads_actual_guard_trajectory_schema=True,
        guard_trajectory_schema=hardware['near_field_memory'].get('motion_response_schema', 'nominal_constant_command'),
        predictive_response_model_not_substituted_for_geometry_guard=True,
        soft_cpu_cores=.1, memory_max_mib=650, goal_achieved=False, deployment_approved=False))
    budget = ExperimentBudget(.1, memory_mib=550, directory=OUT)
    lidar = LidarSpec(hardware['lidar'])
    half = np.asarray(hardware['robot_footprint_m'])/2
    configs = dict(original=hardware['near_field_memory'],
        finite_free_3cm_3frames=dict(hardware['near_field_memory'],
            free_evidence_margin_m=.03, free_evidence_frames=3, free_evidence_max_gap_s=.4))
    memories = {key: NearFieldMemory(value) for key, value in configs.items()}
    records = []
    matched = tested = 0
    for frame in range(max(selected)+1):
        now = float(data['labels'][frame, 0])
        core = SimpleNamespace(elapsed=now, odom=data['odometry'][frame],
            scan=data['scans'][frame], lidar=lidar)
        reason = FEEDBACK_REASONS[int(data['observations'][frame+1, 399:419].argmax())]
        for name, memory in memories.items():
            # Probes use a copy before the actual execution decides whether
            # memory is committed. Unknown/early-stop histories are retained.
            if frame in selected:
                shadow = copy.deepcopy(memory)
                old, uncertainty = shadow.update(core)
                valid = np.isfinite(core.scan) & (core.scan >= lidar.range_min) & (core.scan < lidar.range_max)
                angle = lidar.angles[valid] + lidar.mount[3]
                current = core.scan[valid, None]*np.c_[np.cos(angle), np.sin(angle)] + lidar.mount[:2]
                seen = np.array([entry[1] for entry in shadow.entries.values()])
                assert len(seen) == len(old)
                points = np.vstack([current, old])
                margins = np.r_[np.full(len(current), .04), .04+uncertainty]
                probes = []
                for label, command in hypotheses:
                    result = geometry(points, margins, data['measured_velocity'][frame],
                                      command, half, len(current), seen, now, configs[name])
                    probes.append(dict(label=label, hypothetical_command=command.tolist(), **result))
                if name == 'original':
                    groups = data['observations'][frame, 303:399].reshape(12, 8).argmax(1)
                    for action in [4, 5]:
                        if groups[action] in (0, 4):
                            actual = bool(data['masks'][frame, action])
                            measured = next(p for p in probes if p['label'] == f'original_{action}')['laser_geometry_clear']
                            tested += 1
                            matched += int(actual == measured)
                            assert actual == measured, (frame, action, actual, measured)
                new_clear = []
                for sign, standard in [(-1, 'original_5'), (1, 'original_4')]:
                    regular = next(p for p in probes if p['label'] == standard)
                    if not regular['laser_geometry_clear']:
                        new_clear += [p['label'] for p in probes if p['label'].startswith('pure_yaw_')
                            and p['hypothetical_command'][1]*sign > 0 and p['laser_geometry_clear']]
                records.append(dict(frame=frame, time_s=now, memory_variant=name,
                    recorded_choices=np.flatnonzero(data['masks'][frame]).tolist(),
                    remembered_points=len(old), finer_pure_yaws_clear_when_standard_blocked=new_clear,
                    probes=probes))
                if reason not in EARLY:
                    memories[name] = shadow
            elif reason not in EARLY:
                memory.update(core)
            if reason == 'odometry_jump':
                memories[name].reset()
        if frame % 25 == 0:
            write('STATUS.json', dict(status='measured_memory_replay', frame=frame, target=max(selected)))
            budget.tick()
    for path, expected in hashes.items():
        assert digest(ROOT / path) == expected, path
    summary = {name: dict(sampled_states=sum(r['memory_variant'] == name for r in records),
        states_with_finer_clear_turn=sum(bool(r['finer_pure_yaws_clear_when_standard_blocked'])
            for r in records if r['memory_variant'] == name)) for name in configs}
    budget.tick()
    write('RESULTS.json', dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        original_turn_geometry_checks=tested, original_turn_geometry_matches=matched,
        summaries=summary, records=records, inputs=hashes, resource=budget.statistics(),
        original_4cm_body_margin_aging_uncertainty_and_unknown_memory_retained=True,
        no_static_map_or_dynamic_forecast_admissibility_claim=True,
        no_full_reachability_or_safety_proof=True, no_successful_teacher_labels=True,
        new_physical_steps=0, new_optimizer_updates=0, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False,
        limitations=['Same saved poses; no hypothetical command is physically executed.',
            'A clear laser geometry check is not full guard acceptance or successful recovery.',
            'Alternative angular speeds require a separate versioned action contract and training.',
            'Alternative memory is only a shadow history on the fixed recorded trajectory.']))
    write('STATUS.json', dict(status='complete', original_turn_geometry_checks=tested, summaries=summary))
    print(json.dumps(dict(completed=True, summaries=summary, original_turn_geometry_checks=tested,
        resource=budget.statistics()), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write('FAILURE.json', dict(error=repr(error), automatically_restarted=False,
            goal_achieved=False, deployment_approved=False))
        raise
