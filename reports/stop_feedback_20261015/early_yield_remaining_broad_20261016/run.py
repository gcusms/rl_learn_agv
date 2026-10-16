"""Conditional broad regression after the bounded learned-only16 checks.

Reuse exactly one normal and five extra cases; no duplicate success counts.
At most29 new serial physical episodes, including one added-delay transfer.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
MODEL = ROOT / 'models/stop_feedback_20261015/early_yield_remaining_bc600_s231016'
KNOWN = BASE / 'early_yield_remaining_validation_20261016'
DEPENDENCY = 'rl-early-yield-remaining-validation-20261016.service'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def state():
    raw = subprocess.check_output(['systemctl', '--user', 'show', DEPENDENCY,
        '-p', 'MainPID', '-p', 'ActiveState', '-p', 'Result', '-p', 'InvocationID'], text=True)
    return dict(line.split('=', 1) for line in raw.splitlines())


def main():
    if (OUT / 'WAIT_PLAN.json').exists():
        raise FileExistsError('Use a fresh regression; no restart')
    predecessor = state()
    write('WAIT_PLAN.json', dict(at=stamp(), predecessor=DEPENDENCY,
        predecessor_invocation=predecessor['InvocationID'], maximum_new_physical_cases=29,
        driver_sha256=digest(Path(__file__)),
        requires_bounded_gates_passed=True, no_physics_while_waiting=True,
        no_optimizer_updates=True, goal_achieved=False, deployment_approved=False))
    while True:
        current = state()
        if current['InvocationID'] not in ('', predecessor['InvocationID']):
            raise RuntimeError('Bounded predecessor restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            if current['ActiveState'] != 'inactive' or current['Result'] != 'success' or current['MainPID'] != '0':
                write('RESULTS.json', dict(completed=True, skipped=True,
                    reason='Bounded predecessor did not terminate successfully',
                    new_physical_steps=0, goal_achieved=False, deployment_approved=False))
                return
            break
        write('STATUS.json', dict(at=stamp(), status='waiting_for_bounded_learned_validation',
            physical_simulation_started=False, predecessor=current))
        time.sleep(5)
    known_path = KNOWN / 'RESULTS.json'
    known = json.loads(known_path.read_text())
    assert known['completed']
    assert digest(Path(__file__)) == json.loads((OUT / 'WAIT_PLAN.json').read_text())['driver_sha256']
    if not known.get('bounded_gates_passed'):
        write('RESULTS.json', dict(completed=True, skipped=True,
            reason='Bounded learned-only safety/efficiency gates did not pass',
            predecessor_report_sha256=digest(known_path), new_physical_steps=0,
            goal_achieved=False, deployment_approved=False))
        write('STATUS.json', dict(at=stamp(), status='complete_without_physics'))
        return
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.evaluate_feedback_recovery import evaluate_recovery
    from experiments.experiment_provenance import record_provenance
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    direct_path = BASE / 'recovery_followup_broad_regression_20261016/RESULTS.json'
    diagnostic_path = BASE / 'early_transfer_regression_20261016/RESULTS.json'
    diagnostic_extra_path = BASE / 'actual_stop_validation_20261016/RESULTS_main.json'
    wait_parent_path = BASE / 'actual95_wait_verified_20261016/RESULTS.json'
    direct = json.loads(direct_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    diagnostic_extra = json.loads(diagnostic_extra_path.read_text())
    wait_parent = json.loads(wait_parent_path.read_text())
    assert all(value['completed'] for value in (direct, diagnostic, diagnostic_extra, wait_parent))
    normal = [row for row in known['rows'] if row['group'] == 'normal_w11']
    extras = [row for row in known['rows'] if row['group'] in ('original_collision_conditions', 'retained_fresh33')]
    assert len(normal) == 1 and len(extras) == 5
    assert all(row['safe_success'] for row in [*normal, *extras])
    full_path = BASE / 'w11_recovery_bc800_navigation_20261016/cases.json'
    fresh_path = BASE / 'proactive_augmented_protocol_20261016/fresh_cases.json'
    prefixes_path = BASE / 'w11_recovery_bc800_prefixes_20261016/specifications.json'
    headings_path = BASE / 'heading_probe_cases.json'
    wait_spec_path = BASE / 'actual_stop_validation_20261016/actual_prefix.json'
    full_cases = json.loads(full_path.read_text())
    fresh_cases = json.loads(fresh_path.read_text())
    prefixes = json.loads(prefixes_path.read_text())
    headings = json.loads(headings_path.read_text())
    pending_full = [case for case in full_cases if case['id'] not in {row['case_id'] for row in normal}]
    pending_fresh = [case for case in fresh_cases if case['id'] not in {row['case_id'] for row in extras}]
    assert (len(pending_full), len(pending_fresh), len(prefixes), len(headings)) == (17, 7, 2, 2)
    bundle = MODEL / 'bc_portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    paths = [Path(__file__), known_path, direct_path, diagnostic_path, diagnostic_extra_path,
        wait_parent_path, full_path, fresh_path, prefixes_path, headings_path, wait_spec_path,
        MODEL / 'policy_bc.zip', MODEL / 'RESULTS.json', bundle / 'manifest.json',
        *[bundle / name for name in manifest['files']]]
    for folder in ('neural_navigation', 'navrl', 'experiments'):
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    paths.append(ROOT / 'training/neural_navigation/static_tracking_kernel.cpp')
    for case in [*full_cases, *fresh_cases, *headings, *[item['case'] for item in prefixes]]:
        paths += [path for path in (ROOT / 'training' / case['scene']).iterdir() if path.is_file()]
    for row in [*normal, *extras]:
        path = ROOT / row['trace_path']
        assert digest(path) == row['trace_sha256']
        paths.append(path)
    paths += [ROOT / item['trace'] for item in prefixes]
    old_wait_trace = BASE / 'actual95_wait_verified_20261016/physical/new_data.npz'
    paths.append(old_wait_trace)
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write('PLAN.json', dict(at=stamp(), inputs=inputs,
        checkpoint_sha256=digest(MODEL / 'policy_bc.zip'), maximum_new_physical_cases=29,
        broad_reference_model='recovery_followup_bc400_s231016',
        actual_parent_model='early_yield_bc400_s231016',
        reused_normal_cases=[row['case_id'] for row in normal],
        reused_extra_cases=[row['case_id'] for row in extras],
        new_groups=dict(normal=17, extra=7, prefix=2, heading=2, added_wait=1),
        full18_safety_and_efficiency_required_before_extra_groups=True,
        added_wait_requires_all_preceding_safety_efficiency_and_heading=True,
        unchanged_body_actions_masks_guard_memory_deadlines=True,
        added_zero_commands_are_only_common_test_initialization=True,
        no_runtime_teacher_after_model_release=True, no_parallel_physics=True,
        new_ppo_steps=0, new_optimizer_updates=0, goal_achieved=False, deployment_approved=False))

    def verify():
        for name, expected in inputs.items():
            assert digest(ROOT / name) == expected, name

    verify()
    physical = OUT / 'physical'
    physical.mkdir()
    record_provenance(physical, bundle, [Path(__file__)])
    budget = ExperimentBudget(.35, memory_mib=3072, directory=physical)
    report = dict(completed=False, full18_rows=list(normal), groups=dict(fresh12=dict(rows=list(extras))),
        new_ppo_steps=0, new_optimizer_updates=0, actual106_not_claimed_fixed=True,
        goal_achieved=False, deployment_approved=False)
    new_count = 0

    def save():
        report['resource'] = budget.statistics()
        write('RESULTS.json', report)

    def finish(reason):
        verify()
        report.update(completed=True, at=stamp(), reason=reason)
        save()
        write('STATUS.json', dict(at=stamp(), status='complete', reason=reason))

    def run(case, group, recovery=False):
        nonlocal new_count
        verify()
        write('STATUS.json', dict(at=stamp(), status='physical_regression', group=group,
            case_id=case['id'], finished_new_cases=new_count))
        row, trace = (evaluate_recovery if recovery else evaluate)(case, bundle, budget)
        path = physical / (group + '_' + case['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(variant='early_yield_remaining_bc600', trace_path=str(path.relative_to(ROOT)),
            trace_sha256=digest(path), learned_model_only_after_initialization=True)
        report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        new_count += 1
        save()
        return row

    for case in pending_full:
        report['full18_rows'].append(run(case, 'remaining_full17'))
    assert len(report['full18_rows']) == len({row['case_id'] for row in report['full18_rows']}) == 18
    report['full18_direct_pair'] = compare(direct['full18_rows'], report['full18_rows'])
    report['full18_diagnostic_pair'] = compare(diagnostic['candidate_rows'], report['full18_rows'])
    full_pass = all(report[name]['all_regression_checks_passed'] for name in ('full18_direct_pair', 'full18_diagnostic_pair'))
    report['full18_completed'] = True
    report['subsequent_groups_admitted'] = full_pass
    save()
    if not full_pass:
        finish('Normal full18 safety/efficiency gate failed; no extra expansion')
        return
    for case in pending_fresh:
        run(case, 'fresh12')
    fresh = report['groups']['fresh12']
    assert len(fresh['rows']) == len({row['case_id'] for row in fresh['rows']}) == 12
    fresh['direct_pair'] = compare(direct['groups']['fresh12']['rows'], fresh['rows'])
    fresh['diagnostic_pair'] = compare(diagnostic_extra['groups']['fresh']['rows'], fresh['rows'])
    for prefix in prefixes:
        run(prefix, 'prefixes2', recovery=True)
    report['groups']['prefixes2']['direct_pair'] = compare(
        direct['groups']['original_prefixes2']['rows'], report['groups']['prefixes2']['rows'])
    for case in headings:
        run(case, 'heading2')
    report['groups']['heading2']['direct_pair'] = compare(
        direct['groups']['heading2']['rows'], report['groups']['heading2']['rows'])
    ready = (all(row['safe_success'] for row in fresh['rows'])
        and fresh['direct_pair']['all_regression_checks_passed']
        and fresh['diagnostic_pair']['all_regression_checks_passed']
        and all(report['groups'][name]['direct_pair']['all_regression_checks_passed'] for name in ('prefixes2', 'heading2')))
    report['added_wait_admitted'] = ready
    save()
    if not ready:
        finish('Broader regression recorded; safety/efficiency not fully retained, added wait skipped')
        return
    # One added2.4s stop initializes the delayed-release test. It is not
    # deployed logic and contributes no learned-control or training claims.
    from unittest.mock import patch
    import experiments.evaluate_feedback_recovery as recovery_module
    from neural_navigation.decision_feedback_runtime import StopFeedbackRuntime
    from neural_navigation.masked_actions import ACTIONS
    prefix = json.loads(wait_spec_path.read_text())
    prefix.update(id='w11_actual95_wait2_4', release_s=95.)
    actors = []

    class HoldActor:
        def __init__(self, actor, core):
            self.actor, self.core, self.held = actor, core, 0

        def predict(self, observation):
            if 95. - 1e-8 <= self.core.elapsed < 97.4 - 1e-8:
                self.held += 1
                return ACTIONS[0].copy()
            return self.actor.predict(observation)

    class Runtime(StopFeedbackRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            actor = HoldActor(self.policy.policy, self.core)
            self.policy.policy = actor
            actors.append(actor)

    verify()
    write('STATUS.json', dict(at=stamp(), status='added_wait_transfer', finished_new_cases=new_count))
    with patch.object(recovery_module, 'StopFeedbackRuntime', Runtime):
        row, trace = recovery_module.evaluate_recovery(prefix, bundle, budget)
    assert len(actors) == 1 and actors[0].held == 12
    times = trace['labels'][:, 0]
    hold = (times >= 95. - 1e-8) & (times < 97.4 - 1e-8)
    assert int(hold.sum()) == 12
    np.testing.assert_array_equal(trace['commands'][hold], np.zeros((12, 2)))
    trace['initialization_hold'] = hold
    trace['model_controlled'] &= ~hold
    n = int(np.flatnonzero(trace['model_controlled'])[0])
    assert n == 487 and abs(times[n] - 97.4) < 1e-8
    with np.load(old_wait_trace, allow_pickle=False) as archive:
        for key in ('labels', 'scans', 'odometry', 'measured_velocity', 'commands'):
            a = trace[key][:n + (key == 'labels')]
            b = archive[key][:n + (key == 'labels')]
            if key == 'labels':
                a, b = a[:, :4], b[:, :4]
            np.testing.assert_allclose(a, b, rtol=0, atol=2e-8, equal_nan=True)
    path = physical / 'added_wait2_4.npz'
    np.savez_compressed(path, **trace)
    waits = (trace['choices'][n:] == 0) & trace['masks'][n:, 1:].any(-1)
    row.update(variant='early_yield_remaining_bc600', trace_path=str(path.relative_to(ROOT)),
        trace_sha256=digest(path), release_sensor_timestamp_s=97.4,
        source_frozen_prefix_frames=475, additional_initialization_hold_frames=12,
        model_controlled_frames=int(trace['model_controlled'].sum()),
        post_release_wait_with_choices_frames=int(waits.sum()),
        physical_initialization_matches_parent=True, all_post_initialization_commands_from_model=True)
    report['groups']['added_wait'] = dict(rows=[row], direct_pair=compare(
        [next(item for item in wait_parent['rows'] if item['variant'] == 'new_data')], [row]))
    new_count += 1
    assert new_count <= 29
    report['broader_gates_passed'] = report['groups']['added_wait']['direct_pair']['all_regression_checks_passed']
    finish('Broader checks completed; actual106 recovery/prevention and hardware transfer remain unproven')


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('FAILURE.json', dict(at=stamp(), error=repr(error), automatically_restarted=False))
        raise
