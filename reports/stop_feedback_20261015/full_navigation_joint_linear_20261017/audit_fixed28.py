"""Independently verify the unchanged fixed28 gates and controlled suffixes.

Loads saved arrays only; no model, physics, training or control commands.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose import contract
from neural_navigation.masked_actions import COMMANDS


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)


def safe(row):
    return (row['safe_success'] and row['protocol_valid']
        and not any(row[name] for name in ('collisions', 'false_arrival', 'physical_guards')))


def main():
    assert not (HERE / 'FIXED28_AUDIT.json').exists()
    budget = ExperimentBudget(.1, 550, HERE)
    validation_path = BASE / 'full_navigation_joint_linear_validation_20261017/RESULTS.json'
    raw = validation_path.read_bytes()
    validation = json.loads(raw)
    assert validation['model'] == 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'
    fit_path = HERE / 'FIT_RESULTS.json'
    fit = read(fit_path)
    model_path = ROOT / validation['model'] / 'checkpoint.pt'
    assert fit['model'] == validation['model'] and fit['fit_passed']
    assert digest(model_path) == fit['checkpoint_sha256']
    assert all(len(validation.get('groups', {}).get(group, {}).get('rows', [])) == count
        for group, count in (('normal18', 18), ('sensor5', 5), ('source_recovery3', 3), ('heading2', 2))), 'Wait for all original28 to finish'
    snapshot = HERE / 'FIXED28_VALIDATION_SNAPSHOT.json'
    assert not snapshot.exists()
    snapshot.write_bytes(raw)
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    parent_path = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    direct_path = BASE / 'v2_behavior_retention_validation_20261017/RESULTS.json'
    tasks = read(tasks_path)
    parent = {row['case_id']:row for g in read(parent_path)['groups'].values() for row in g['rows']}
    direct = {row['case_id']:row for g in read(direct_path)['groups'].values() for row in g['rows']}
    reference = dict(normal18=tasks['reference']['normal18'], sensor5=tasks['reference']['sensor5'],
        source_recovery3=tasks['reference']['actual95_actual90'] + tasks['reference']['delayed_wait'],
        heading2=tasks['reference']['heading2'])
    specifications = {case['id']:case for group in ('normal18', 'sensor5', 'heading2') for case in tasks[group]}
    specifications.update({s['id']:s for s in [*tasks['actual95_actual90'], tasks['delayed_wait']]})
    paths = [Path(__file__), snapshot, tasks_path, parent_path, direct_path, fit_path, model_path]
    output, recoveries = [], []
    all_parity, frames, controlled_frames = True, 0, 0
    c = contract()
    for group, expected in (('normal18', 18), ('sensor5', 5), ('source_recovery3', 3), ('heading2', 2)):
        rows = validation['groups'][group]['rows']
        fixed = {r['case_id']:r for r in reference[group]}
        assert len(rows) == expected and {r['case_id'] for r in rows} == fixed.keys()
        for row in rows:
            key = row['case_id']
            assert safe(row)
            metric = 'post_release_seconds' if group == 'source_recovery3' else 'seconds'
            comparisons = {}
            for name, previous in (('original_fixed', fixed[key]), ('declared_parent', parent[key]), ('direct_aux119', direct[key])):
                comparisons[name] = {field:bool(row[field] <= previous[field] * 1.2 + 1e-8)
                    for field in (metric, 'travel_m')}
                assert all(comparisons[name].values()), (key, name, comparisons[name])
            path = ROOT / row['trace_path']
            prior_path = ROOT / direct[key]['trace_path']
            assert digest(path) == row['trace_sha256']
            assert digest(prior_path) == direct[key]['trace_sha256']
            paths += [path, prior_path]
            with np.load(path, allow_pickle=False) as data, np.load(prior_path, allow_pickle=False) as old:
                x, masks, choices = data['observations'], data['masks'], data['choices']
                labels, commands = data['labels'], data['commands']
                assert x.shape == (row['frames'], 625) and masks.shape == (len(x), 12)
                np.testing.assert_array_equal(masks, np.c_[np.ones(len(x), bool), x[:, 235:244] > .5, x[:, 267:269] > .5])
                assert masks[np.arange(len(x)), choices].all()
                np.testing.assert_allclose(commands[:-1], COMMANDS[choices[:-1]], rtol=0, atol=2e-8)
                assert labels[-1, 7] < (.12 if group == 'heading2' else .2)
                if group == 'heading2':
                    assert labels[-1, 8] < c['yaw_tolerance_rad']
                    assert abs(labels[-1, 9]) < c['stopped_linear_m_s']
                    assert abs(labels[-1, 10]) < c['stopped_angular_rad_s']
                    held = ((labels[:, 7] < .12) & (labels[:, 8] < .15)
                        & (abs(labels[:, 9]) < .03) & (abs(labels[:, 10]) < .08))
                    beginning = len(held) - 1
                    while beginning and held[beginning - 1]:
                        beginning -= 1
                    assert labels[-1, 0] - labels[beginning, 0] >= .4 - 1e-9
                controlled = data['model_controlled'] if 'model_controlled' in data else np.ones(len(x), bool)
                if group == 'source_recovery3':
                    n = row['frozen_prefix_frames']
                    np.testing.assert_array_equal(controlled, np.arange(len(x)) >= n)
                    assert abs(labels[n, 0] - row['release_sensor_timestamp_s']) < 2e-8
                    spec = specifications[key]
                    original = ROOT / spec['trace']
                    assert digest(original) == row['prefix_trace_sha256']
                    paths.append(original)
                    with np.load(original, allow_pickle=False) as source:
                        for name in ('scans', 'odometry', 'measured_velocity', 'commands', 'choices'):
                            np.testing.assert_allclose(data[name][:n], source[name][:n], rtol=0, atol=2e-8, equal_nan=True)
                        np.testing.assert_allclose(labels[:n + 1, 1:4], source['labels'][:n + 1, 1:4], rtol=0, atol=2e-8)
                    waits = (choices[n:] == 0) & masks[n:, 1:].any(1)
                    only = ~masks[n:, 1:].any(1)
                    assert int(waits.sum()) == row['post_release_wait_with_choices_frames']
                    assert int(only.sum()) == row['post_release_only_stop_frames']
                    assert abs(row['seconds'] - spec['release_s'] - row['post_release_seconds']) < 2e-8
                    recoveries.append(dict(case_id=key, initialization_frames=n,
                        learned_frames=int(controlled.sum()), post_release_seconds=row['post_release_seconds'],
                        voluntary_wait_frames=int(waits.sum()), only_stop_frames=int(only.sum()),
                        original_raw_sensor_physics_and_command_prefix_verified=True))
                fields = ['observations', 'masks', 'choices', 'commands', 'scans', 'odometry',
                    'measured_velocity', 'labels', 'human_positions', 'route_diagnostics', 'route_reasons']
                exact = all(same(data[name], old[name]) for name in fields)
                exact &= same(np.delete(data['constraint_diagnostics'], 2, axis=1),
                    np.delete(old['constraint_diagnostics'], 2, axis=1))
                frames += len(x)
                controlled_frames += int(controlled.sum())
                all_parity &= bool(exact)
            output.append(dict(case_id=key, group=group, safe=True, comparisons=comparisons,
                direct_aux119_non_timing_trace_exact=bool(exact)))
            budget.tick()
    normal = validation['groups']['normal18']['rows']
    aggregates = {}
    for name, previous in (('original_fixed', reference['normal18']),
        ('declared_parent', [parent[r['case_id']] for r in normal]),
        ('direct_aux119', [direct[r['case_id']] for r in normal])):
        aggregates[name] = dict(time_passed=sum(r['seconds'] for r in normal) <= sum(r['seconds'] for r in previous) * 1.05 + 1e-8,
            path_passed=sum(r['travel_m'] for r in normal) <= sum(r['travel_m'] for r in previous) * 1.02 + 1e-8)
        assert all(aggregates[name].values())
    assert not {'torch', 'mujoco', 'rclpy'} & sys.modules.keys()
    result = dict(completed=True, at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        model=validation['model'], fixed_cases=28, safe_cases=28,
        rows=output, normal_aggregate_original_gates=aggregates, recovery_suffix_audits=recoveries,
        all28_direct_aux119_non_timing_traces_exact=all_parity,
        recorded_frames=frames, model_controlled_frames=controlled_frames,
        input_hashes={str(path.relative_to(ROOT)):digest(path) for path in set(paths)},
        resource=budget.statistics(), model_or_physics_not_loaded=True,
        original97_6_106_known_full_missions_and_conditional_fresh_not_waived=True,
        goal_achieved=False, deployment_approved=False)
    (HERE / 'FIXED28_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key:result[key] for key in ('completed', 'fixed_cases', 'safe_cases',
        'recorded_frames', 'model_controlled_frames', 'all28_direct_aux119_non_timing_traces_exact', 'resource')}))


if __name__ == '__main__':
    main()
