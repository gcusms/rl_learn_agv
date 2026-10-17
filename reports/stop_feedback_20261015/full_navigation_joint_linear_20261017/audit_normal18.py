"""Independently check all original normal18 gates and matched actual traces."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def same(a, b):
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)


def safe(row):
    return (row['safe_success'] and row['protocol_valid']
        and not any(row[k] for k in ('collisions', 'false_arrival', 'physical_guards')))


def gates(reference, candidate):
    old = {row['case_id']: row for row in reference}
    new = {row['case_id']: row for row in candidate}
    assert len(old) == len(new) == 18 and old.keys() == new.keys()
    checks = []
    for key, row in new.items():
        previous = old[key]
        checks.append(dict(case_id=key, safety=bool(safe(row)),
            seconds_within_original_20pct=bool(row['seconds'] <= previous['seconds'] * 1.2 + 1e-8),
            travel_within_original_20pct=bool(row['travel_m'] <= previous['travel_m'] * 1.2 + 1e-8)))
    old_time, new_time = (sum(row['seconds'] for row in rows) for rows in (reference, candidate))
    old_path, new_path = (sum(row['travel_m'] for row in rows) for rows in (reference, candidate))
    time_ok = new_time <= old_time * 1.05 + 1e-8
    path_ok = new_path <= old_path * 1.02 + 1e-8
    return dict(all_passed=bool(time_ok and path_ok
        and all(all(row[k] for k in ('safety', 'seconds_within_original_20pct',
            'travel_within_original_20pct')) for row in checks)),
        checks=checks, seconds=[old_time, new_time], travel_m=[old_path, new_path],
        aggregate_seconds_within_original_5pct=bool(time_ok),
        aggregate_travel_within_original_2pct=bool(path_ok))


def main():
    assert not (HERE / 'NORMAL18_AUDIT.json').exists()
    budget = ExperimentBudget(.1, 550, HERE)
    validation_path = BASE / 'full_navigation_joint_linear_validation_20261017/RESULTS.json'
    raw = validation_path.read_bytes()
    validation = json.loads(raw)
    candidate = validation['groups']['normal18']['rows']
    assert len(candidate) == 18
    assert validation['model'] == 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'
    fit_path = HERE / 'FIT_RESULTS.json'
    fit = read(fit_path)
    model_path = ROOT / validation['model'] / 'checkpoint.pt'
    assert fit['model'] == validation['model'] and fit['fit_passed']
    assert digest(model_path) == fit['checkpoint_sha256']
    snapshot = HERE / 'NORMAL18_VALIDATION_SNAPSHOT.json'
    assert not snapshot.exists()
    snapshot.write_bytes(raw)
    fixed_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    declared_path = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    direct_path = BASE / 'v2_behavior_retention_validation_20261017/RESULTS.json'
    fixed = read(fixed_path)['reference']['normal18']
    all_declared = {row['case_id']:row for group in read(declared_path)['groups'].values()
        for row in group['rows']}
    declared = [all_declared[row['case_id']] for row in candidate]
    direct = read(direct_path)['groups']['normal18']['rows']
    comparisons = dict(original_fixed=gates(fixed, candidate),
        declared_parent=gates(declared, candidate), direct_auxiliary119=gates(direct, candidate))
    assert all(value['all_passed'] for value in comparisons.values())
    paths = [Path(__file__), snapshot, fixed_path, declared_path, direct_path, fit_path, model_path]
    previous_rows = {row['case_id']:row for row in direct}
    parity = []
    frames = 0
    for row in candidate:
        previous = previous_rows[row['case_id']]
        pair = [ROOT / r['trace_path'] for r in (previous, row)]
        for r, path in zip((previous, row), pair):
            assert digest(path) == r['trace_sha256']
            paths.append(path)
        with np.load(pair[0], allow_pickle=False) as old, np.load(pair[1], allow_pickle=False) as new:
            fields = ['observations', 'masks', 'choices', 'commands', 'scans',
                'odometry', 'measured_velocity', 'labels', 'human_positions',
                'route_diagnostics', 'route_reasons']
            matched = {name:same(old[name], new[name]) for name in fields}
            matched['predictive_constraints_excluding_compute_ms'] = same(
                np.delete(old['constraint_diagnostics'], 2, axis=1),
                np.delete(new['constraint_diagnostics'], 2, axis=1))
            frames += len(new['choices'])
        parity.append(dict(case_id=row['case_id'], frames=row['frames'], fields=matched,
            all_non_timing_trace_fields_exact=all(matched.values())))
        budget.tick()
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in paths}
    budget.tick()
    assert not {'torch', 'mujoco', 'rclpy'} & sys.modules.keys()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        model=validation['model'], normal_cases=18, safe_cases=sum(safe(row) for row in candidate),
        comparisons=comparisons, matched_trace_cases=parity, recorded_frames=frames,
        all18_auxiliary119_recorded_non_timing_traces_exact=all(row['all_non_timing_trace_fields_exact'] for row in parity),
        zero_motion_initially_arrived_cases=[row['case_id'] for row in candidate
            if row['seconds'] <= 1e-9 and row['travel_m'] <= 1e-9],
        fixed_development_regression_not_generalization=True,
        known_sensor01_gain_is_separate_training_development_case=True,
        remaining_sensor5_source3_heading2_known_hard97_6_106_and_conditional_fresh_gates_not_waived=True,
        input_hashes=hashes, resource=budget.statistics(), model_or_physics_not_loaded=True,
        new_training_updates=0, new_ppo_steps=0, new_physical_episodes=0,
        goal_achieved=False, deployment_approved=False)
    (HERE / 'NORMAL18_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key:result[key] for key in ('completed', 'normal_cases', 'safe_cases',
        'recorded_frames', 'all18_auxiliary119_recorded_non_timing_traces_exact',
        'zero_motion_initially_arrived_cases', 'resource')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
