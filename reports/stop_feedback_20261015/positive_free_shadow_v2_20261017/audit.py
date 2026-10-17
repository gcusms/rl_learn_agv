"""No-physics safety-evidence checks and recorded-trajectory shadow replay."""
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from evidence_memory import EvidenceMemory
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = HERE / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def check_evidence(config, lidar_config):
    config = dict(config)
    config['resolution_m'] = .01
    toy_lidar = LidarSpec(dict(lidar_config, beams=9, angle_min=-.4, angle_max=.4, mount=[0., 0., 0., 0.]))
    key = (50, 0)
    def memory():
        value = EvidenceMemory(config)
        value.entries[key] = (np.array([.5, 0.]), 0.)
        return value
    def core(t, ranges):
        return SimpleNamespace(elapsed=t, odom=np.zeros(3), scan=np.full(9, ranges), lidar=toy_lidar)
    value = memory()
    for t in [.2, .4]:
        value.update(core(t, .54)); assert key in value.entries
    value.update(core(.6, .54)); assert key not in value.entries
    value = memory()
    value.update(core(.2, .54)); value.update(core(.4, np.inf))
    value.update(core(.6, .54)); value.update(core(.8, .54)); assert key in value.entries
    value.update(core(1., .54)); assert key not in value.entries
    value = memory()
    value.update(core(.2, .54)); value.update(core(.8, .54))
    value.update(core(1., .54)); assert key in value.entries
    value.update(core(1.2, .54)); assert key not in value.entries
    value = memory(); value.update(core(.2, .7)); assert key not in value.entries
    value = memory(); value.entries[key] = (np.array([-.5, 0.]), 0.)
    for t in [.2, 10., 100.]:
        value.update(core(t, .7)); assert key in value.entries
    value = memory()
    for t in [.2, .4, 10., 100.]:
        value.update(core(t, np.nan)); assert key in value.entries
    result = dict(completed=True, stable_three_finite_frames_required=True,
        invalid_scan_breaks_confirmation=True, excessive_gap_breaks_confirmation=True,
        original_strong_ray_clearing_preserved=True, blind_point_retained=True,
        unavailable_scan_and_time_alone_never_clear=True)
    write('EVIDENCE_CHECKS.json', result)
    return result


def main():
    assert not (HERE / 'RESULTS.json').exists()
    hardware_path = ROOT / 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017/portable/hardware.json'
    hardware = json.loads(hardware_path.read_text())
    config = hardware['near_field_memory']
    assert config.get('free_evidence_frames', 1) == 1 and config.get('free_evidence_margin_m', .1) == .1
    check_evidence(config, hardware['lidar'])
    geometry_path = BASE / 'actual106_resolution_audit_20261017/audit.py'
    spec = importlib.util.spec_from_file_location('original_guard_geometry_shadow', geometry_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    geometry = module.geometry
    traces = {
        'current106': BASE / 'full_navigation_joint_linear_validation_20261017/physical/hard3_w11_training_stall_t106_candidate.npz',
        'creep04_failed': BASE / 'late106_creep_teacher_20261017/probe_0/trace.npz',
        'original96_success': BASE / 'hybrid_memory_pair_preflight_20261017/physical/legal_wait_parent_t90_control.npz',
    }
    paths = [Path(__file__), HERE / 'evidence_memory.py', hardware_path, geometry_path, *traces.values(),
        ROOT / 'training/neural_navigation/safety_memory.py', ROOT / 'training/neural_navigation/runtime.py']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    write('PLAN.json', dict(inputs=hashes, added_finite_free_margin_m=.03,
        added_consecutive_frames=3, added_maximum_gap_s=.4,
        original_fast_free_rule_preserved=True, no_runtime_or_bundle_change=True,
        positive_free_only_not_time_eviction=True, shadow_fixed_recorded_trajectories_only=True,
        no_actual_mask_permission_or_physical_success_claim=True,
        from_initialization_and_full_retention_required_before_adoption=True,
        new_physical_steps=0, new_training_updates=0, new_ppo_steps=0))
    budget = ExperimentBudget(.1, 550, HERE)
    lidar = LidarSpec(hardware['lidar']); half = np.asarray(hardware['robot_footprint_m']) / 2
    summaries, records, clearances = [], [], []
    for name, path in traces.items():
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        memories = {'original': NearFieldMemory(config), 'positive_free': EvidenceMemory(config)}
        compared = retained_checked = max_original = max_shadow = 0
        changed_checks = 0
        for frame in range(len(data['labels']) - 1):
            core = SimpleNamespace(elapsed=float(data['labels'][frame, 0]),
                odom=data['odometry'][frame], scan=data['scans'][frame], lidar=lidar)
            reason = (str(data['executed_stop_reason'][frame]) if 'executed_stop_reason' in data else
                FEEDBACK_REASONS[int(data['observations'][frame + 1, 399:419].argmax())])
            shadows = {key: copy.deepcopy(value) for key, value in memories.items()}
            returned = {key: memory.update(core) for key, memory in shadows.items()}
            original, shadow = shadows['original'], shadows['positive_free']
            assert set(shadow.entries) <= set(original.entries), (name, frame, 'Not an original-point subset')
            assert max(len(original.entries), len(shadow.entries)) < config['max_points'], 'Capacity changes need independent analysis'
            keys = list(shadow.entries)
            # Exact comparison of every coordinate and timestamp, in batches.
            np.testing.assert_array_equal(
                np.array([shadow.entries[key][0] for key in keys]),
                np.array([original.entries[key][0] for key in keys]))
            np.testing.assert_array_equal(
                np.array([shadow.entries[key][1] for key in keys]),
                np.array([original.entries[key][1] for key in keys]))
            compared += 1; retained_checked += len(shadow.entries)
            max_original = max(max_original, len(original.entries)); max_shadow = max(max_shadow, len(shadow.entries))
            for clear in shadow.last_positive_clears:
                assert clear['confirmations'] >= 3 and clear['range_excess_m'] > .03
                clearances.append(dict(trace=name, frame=frame,
                    committed_in_recorded_history=reason not in EARLY, **clear))
            if frame >= 480 and frame % 10 == 0:
                valid = np.isfinite(core.scan) & (core.scan >= lidar.range_min) & (core.scan < lidar.range_max)
                angle = lidar.angles[valid] + lidar.mount[3]
                raw = core.scan[valid, None] * np.c_[np.cos(angle), np.sin(angle)] + lidar.mount[:2]
                row = dict(trace=name, frame=frame, sensor_time_s=core.elapsed, probes=[])
                for action in (4, 5, 8, 9):
                    checks = {}
                    for key, (old, uncertainty) in returned.items():
                        seen = np.array([entry[1] for entry in shadows[key].entries.values()])
                        checks[key] = geometry(np.vstack([raw, old]), np.r_[np.full(len(raw), .04), .04 + uncertainty],
                            data['measured_velocity'][frame], COMMANDS[action], half,
                            len(raw), seen, core.elapsed, config)['laser_geometry_clear']
                    assert not checks['original'] or checks['positive_free'], (name, frame, action, 'Existing geometry allowance lost')
                    changed_checks += int(checks['positive_free'] and not checks['original'])
                    row['probes'].append(dict(action=action, **checks))
                records.append(row)
            if reason not in EARLY:
                memories = shadows
            if reason == 'odometry_jump':
                for memory in memories.values():memory.reset()
            if frame % 15 == 0:
                budget.tick()
        summaries.append(dict(trace=name, compared_frames=compared,
            retained_original_point_timestamp_checks=retained_checked,
            maximum_original_points=max_original, maximum_shadow_points=max_shadow,
            no_capacity_eviction=True, all_retained_points_original=True,
            sampled_extra_geometry_allowances=changed_checks))
    for path, sha in hashes.items():assert digest(ROOT / path) == sha, path
    budget.tick()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        summaries=summaries, records=records, positive_clearances=clearances, inputs=hashes,
        original_fast_clearing_preserved=True, original_body4cm_margins_aging_unknown_blind_retained=True,
        all_recorded_retained_point_sets_are_original_subsets=True,
        no_runtime_permission_model_or_physics_change=True,
        not_from_initialization_closed_loop_or_full_retention_proof=True,
        rejected_previous_three_frame_hybrid_not_adopted=True,
        new_physical_steps=0, new_training_updates=0, new_ppo_steps=0,
        resource=budget.statistics(), goal_achieved=False, deployment_approved=False)
    write('RESULTS.json', result)
    print(json.dumps({key: value for key, value in result.items()
        if key not in ('records', 'positive_clearances', 'inputs')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:main()
    except Exception as error:
        write('FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
