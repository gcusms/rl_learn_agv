"""Conditional serial regressions after all twelve measured-context checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'
DEPENDENCY = 'rl-release-validation-20261016.service'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def state():
    raw = subprocess.check_output(['systemctl', '--user', 'show', DEPENDENCY,
        '-p', 'MainPID', '-p', 'ActiveState', '-p', 'Result', '-p', 'InvocationID'], text=True)
    return dict(line.split('=', 1) for line in raw.splitlines())


def main(model, output):
    model, output = model.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    first = state()
    driver_hash = digest(Path(__file__))
    write(output / 'WAIT_PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(),
        predecessor=DEPENDENCY, predecessor_state=first, driver_sha256=driver_hash,
        maximum_new_physical_episodes=34, no_physics_while_waiting=True,
        requires_all_twelve_bounded_cases_safe_and_retained_efficiency=True,
        phases=['actual95_actual90', 'normal18', 'remaining_extra8', 'original_prefixes2',
            'heading2', 'added_wait2_4', 'actual106_diagnostic'],
        new_ppo_steps=0, new_optimizer_updates=0, deployment_approved=False))
    while True:
        current = state()
        if current['InvocationID'] not in ('', first['InvocationID']):
            raise RuntimeError('The pinned validation process was restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            break
        write(output / 'STATUS.json', dict(status='waiting_for_bounded_validation',
            physical_started=False, predecessor=current))
        time.sleep(5)
    known_path = BASE / 'release_validation_20261016/RESULTS.json'
    known = json.loads(known_path.read_text())
    passed = (current['ActiveState'] == 'inactive' and current['Result'] == 'success'
        and current['MainPID'] == '0' and known['completed'] and len(known['rows']) == 12
        and len(known['gates']) == 3 and all(gate['passed'] for gate in known['gates'].values()))
    assert digest(Path(__file__)) == driver_hash
    if not passed:
        write(output / 'RESULTS.json', dict(completed=True, skipped=True,
            reason='Bounded safety/efficiency checks did not all pass',
            predecessor_report_sha256=digest(known_path), new_physical_steps=0,
            goal_achieved=False, deployment_approved=False))
        write(output / 'STATUS.json', dict(status='complete', skipped=True,
            physical_started=False, new_physical_steps=0))
        return
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.experiment_provenance import record_provenance
    from experiments.release_recovery import evaluate_recovery
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    bundle = model / 'portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    full_path = BASE / 'w11_recovery_bc800_navigation_20261016/cases.json'
    fresh_path = BASE / 'proactive_augmented_protocol_20261016/fresh_cases.json'
    prefix_path = BASE / 'w11_recovery_bc800_prefixes_20261016/specifications.json'
    w11_path = BASE / 'w11_recovery_bc800_w11_20261016/specifications.json'
    heading_path = BASE / 'heading_probe_cases.json'
    direct_path = BASE / 'recovery_followup_broad_regression_20261016/RESULTS.json'
    direct_first_path = BASE / 'recovery_followup_validation_20261016/RESULTS.json'
    retained_path = BASE / 'w11_retained2000_fresh_20261016/RESULTS.json'
    wait_parent_path = BASE / 'actual95_wait_verified_20261016/RESULTS.json'
    wait_trace_path = BASE / 'actual95_wait_verified_20261016/physical/new_data.npz'
    full = json.loads(full_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    prefixes = json.loads(prefix_path.read_text())
    w11 = json.loads(w11_path.read_text())
    headings = json.loads(heading_path.read_text())
    direct = json.loads(direct_path.read_text())
    direct_first = json.loads(direct_first_path.read_text())
    retained = json.loads(retained_path.read_text())
    wait_parent = json.loads(wait_parent_path.read_text())
    assert all(report['completed'] for report in (direct, direct_first, retained, wait_parent))
    reused = [row for row in known['rows'] if row['case_id'] in {
        'proactive_fresh_condition_' + suffix for suffix in ('01', '02', '04', '05')}]
    pending_fresh = [case for case in fresh if case['id'] not in {row['case_id'] for row in reused}]
    assert (len(full), len(fresh), len(pending_fresh), len(prefixes), len(headings)) == (18, 12, 8, 2, 2)
    actual = [next(item for item in w11 if item['id'] == 'w11_training_stall_t' + suffix) for suffix in ('95', '90')]
    actual106 = next(item for item in w11 if item['id'] == 'w11_training_stall_t106')
    wait = dict(actual[0], id='w11_actual95_wait2_4', release_s=97.4, trace=str(wait_trace_path))
    paths = [Path(__file__), known_path, full_path, fresh_path, prefix_path, w11_path,
        heading_path, direct_path, direct_first_path, retained_path, wait_parent_path, wait_trace_path,
        model / 'checkpoint.pt', model / 'RESULTS.json', bundle / 'manifest.json']
    paths += [bundle / name for name in manifest['files']]
    paths += [ROOT / item['trace'] for item in [*prefixes, *w11]]
    paths += [ROOT / row['trace_path'] for row in reused]
    for folder in ('neural_navigation', 'navrl', 'experiments'):
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    paths.append(ROOT / 'training/neural_navigation/static_tracking_kernel.cpp')
    for case in [*full, *fresh, *headings, *[item['case'] for item in [*prefixes, *w11]]]:
        paths += [path for path in (ROOT / 'training' / case['scene']).iterdir() if path.is_file()]
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in set(paths)}
    write(output / 'PLAN.json', dict(inputs=inputs, reused_extra_cases=[row['case_id'] for row in reused],
        maximum_new_physical_episodes=34, recovery_and_full18_gates_before_expansion=True,
        compare_extra_to_both_full18_followup_and_retained2k=True,
        prefix_commands_are_common_initialization_only=True,
        no_teacher_after_release=True, no_parallel_physical_main=True,
        body_margin_masks_guard_memory_and_deadlines_unchanged=True,
        new_ppo_steps=0, new_optimizer_updates=0, goal_achieved=False, deployment_approved=False))
    record_provenance(output, bundle, [Path(__file__)])
    physical = output / 'physical'
    physical.mkdir()
    report = dict(completed=False, groups={'extra12': dict(rows=list(reused))}, new_physical_episodes=0,
        new_ppo_steps=0, new_optimizer_updates=0, goal_achieved=False, deployment_approved=False)

    def verify():
        for name, expected in inputs.items():
            assert digest(ROOT / name) == expected, name

    def save():
        budget.tick()
        report['at'] = datetime.now(timezone.utc).isoformat()
        report['resource'] = budget.statistics()
        write(output / 'RESULTS.json', report)

    def finish(reason):
        verify()
        report.update(completed=True, reason=reason)
        save()
        write(output / 'STATUS.json', dict(status='complete', reason=reason))

    def run(case, group, recovery=False):
        verify()
        write(output / 'STATUS.json', dict(status='physical_regression', group=group,
            case_id=case['id'], finished_new_episodes=report['new_physical_episodes']))
        row, trace = (evaluate_recovery if recovery else evaluate)(case, bundle, budget)
        file = physical / (group + '_' + case['id'] + '.npz')
        np.savez_compressed(file, **trace)
        row.update(variant='measured_release_context_bc50', trace_path=str(file.relative_to(ROOT)),
            trace_sha256=digest(file), model_only_after_initialization=True)
        report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        return row

    for item in actual:
        run(item, 'actual95_actual90', recovery=True)
    actual_rows = report['groups']['actual95_actual90']['rows']
    old_actual = [next(row for row in direct_first['rows'] if row['case_id'] == item['id']) for item in actual]
    report['groups']['actual95_actual90']['paired'] = compare(old_actual, actual_rows)
    if not all(row['safe_success'] for row in actual_rows) or not report['groups']['actual95_actual90']['paired']['all_regression_checks_passed']:
        finish('Retained90/95 recovery gate failed; no wider expansion')
        return
    for case in full:
        run(case, 'normal18')
    normal = report['groups']['normal18']
    normal['paired'] = compare(direct['full18_rows'], normal['rows'])
    if not all(row['safe_success'] for row in normal['rows']) or not normal['paired']['all_regression_checks_passed']:
        finish('Normal18 safety/efficiency gate failed; no extra expansion')
        return
    for case in pending_fresh:
        run(case, 'extra12')
    extra = report['groups']['extra12']
    extra['followup_pair'] = compare(direct['groups']['fresh12']['rows'], extra['rows'])
    extra['retained_2k_pair'] = compare(retained['rows'], extra['rows'])
    for item in prefixes:
        run(item, 'prefixes2', recovery=True)
    report['groups']['prefixes2']['paired'] = compare(direct['groups']['original_prefixes2']['rows'], report['groups']['prefixes2']['rows'])
    for case in headings:
        run(case, 'heading2')
    report['groups']['heading2']['paired'] = compare(direct['groups']['heading2']['rows'], report['groups']['heading2']['rows'])
    ready = (all(row['safe_success'] for row in extra['rows'])
        and all(extra[name]['all_regression_checks_passed'] for name in ('followup_pair', 'retained_2k_pair'))
        and all(report['groups'][group]['paired']['all_regression_checks_passed'] for group in ('prefixes2', 'heading2')))
    report['added_wait_admitted'] = ready
    save()
    if not ready:
        finish('Extra/prefix/heading results recorded but gates failed; delayed wait not expanded')
        return
    delayed = run(wait, 'added_wait2_4', recovery=True)
    old_wait = next(row for row in wait_parent['rows'] if row['variant'] == 'new_data')
    report['groups']['added_wait2_4']['paired'] = compare([old_wait], [delayed])
    if delayed['safe_success'] and report['groups']['added_wait2_4']['paired']['all_regression_checks_passed']:
        run(actual106, 'actual106_diagnostic', recovery=True)
    report['all_broad_gates_passed'] = (delayed['safe_success']
        and report['groups']['added_wait2_4']['paired']['all_regression_checks_passed'])
    assert report['new_physical_episodes'] <= 34
    finish('Declared broad checks completed; actual106 and real hardware need their own completion evidence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.model, args.output)
    except Exception as error:
        if args.output.exists():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
