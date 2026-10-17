"""Prospective retention study of an observed recovery-time improvement.

The earlier protocol required a gained successful case and correctly stopped.
This separate study tests whether the measured96s speedup preserves navigation.
It does not revise that decision or waive the unresolved97.6s/106s recoveries.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'reports/stop_feedback_20261015'
HERE = Path(__file__).resolve().parent
DEPENDENCY = 'rl-earlier-memory-evidence-20261017.service'
INVOCATION = 'ebb63ba4a5a246ba8dd53aaecd904392'
MODEL = ROOT / 'models/stop_feedback_20261015/native625_batch_credit_s231017/portable'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    parent_path = BASE / 'normal_suite_wait_validation_v2_20261016/RESULTS.json'
    candidate_path = BASE / 'native625_batch_credit_validation_20261017/RESULTS.json'
    training_path = BASE / 'native625_batch_credit_pilot_20261017/RESULTS.json'
    cases_path = BASE / 'native625_frozen_holdout_20261016/CASES.json'
    frozen_path = BASE / 'native625_frozen_holdout_20261016/PLAN.json'
    training_plan = BASE / 'native625_batch_credit_pilot_20261017/PLAN.json'
    tasks, parent, candidate, training = [read(p) for p in [task_path, parent_path, candidate_path, training_path]]
    assert all(x['completed'] for x in [parent, candidate, training])
    assert not candidate.get('all_development_retention_passed')
    assert candidate['hard_state_comparison'] == dict(gained=[], lost=[], all_safe_success=False)
    assert training['all_reference_stop_labels_retained'] and training['new_ppo_steps'] == 1665
    parent_rows = {r['case_id']: r for g in parent['groups'].values() for r in g['rows']}
    new_hard = {r['case_id']: r for r in candidate['groups']['hard3']['rows']}
    old, new = parent_rows['legal_wait_parent_t90'], new_hard['legal_wait_parent_t90']
    assert old['safe_success'] and new['safe_success']
    assert new['post_release_seconds'] < old['post_release_seconds']*.8
    assert new['travel_m'] <= old['travel_m']*1.02
    for group in candidate['groups'].values():
        assert all(not any(r[k] for k in ['collisions', 'false_arrival', 'physical_guards']) for r in group['rows'])
    baseline = ROOT / parent['selected_model'] / 'portable'
    frozen = read(frozen_path)
    assert frozen['frozen_before_ppo'] and digest(cases_path) == frozen['cases_sha256']
    assert str(cases_path.relative_to(ROOT)) not in read(training_plan)['inputs']
    cases = read(cases_path)
    assert len(cases) == len({c['id'] for c in cases}) == 8
    for name in ['hardware.json', 'localizer.pt', 'router.pt', 'decision_feedback.json',
            'release_observation.json', 'docking_contract.json', 'localization_contract.json']:
        assert digest(baseline / name) == digest(MODEL / name), name
    files = [Path(__file__), task_path, parent_path, candidate_path, training_path,
        training_plan, cases_path, frozen_path, HERE / 'BATCH_POLICY_EFFECT_AUDIT.json']
    for bundle in [baseline, MODEL]:
        manifest = read(bundle / 'manifest.json')
        assert not manifest['deployment_approved']
        files += [bundle / 'manifest.json'] + [bundle / name for name in manifest['files']]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    recoveries = [*tasks['actual95_actual90'], tasks['delayed_wait']]
    files += [ROOT / s['trace'] for s in recoveries]
    scenes = {c['scene'] for c in tasks['normal18'] + tasks['sensor5'] + tasks['heading2'] + cases}
    scenes |= {s['case']['scene'] for s in recoveries}
    for scene in scenes:
        files += [p for p in (ROOT / 'training' / scene).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, exact_dependency=DEPENDENCY,
        expected_invocation=INVOCATION, maximum_physical_episodes=44,
        scope='Separate prospective retention study triggered by observed96s recovery speedup',
        original_failed_expansion_gate_unchanged=True,
        late97_6s_and106s_recovery_requirements_not_waived=True,
        trigger_comparison={name: {k: row[k] for k in ['safe_success', 'release_sensor_timestamp_s',
            'post_release_seconds', 'travel_m']} for name, row in [('parent', old), ('candidate', new)]},
        development_groups=dict(normal18=18, sensor5=5, source_recovery3=3, heading2=2),
        frozen_seed_pairs=8, frozen_cases_sha256=digest(cases_path),
        full_fixed_retention_required_before_frozen_evaluation=True,
        unchanged_models_actions_625_observation_hardware_masks_4cm_and_docking=True,
        no_optimizer_updates=True, no_parallel_physical_job=True,
        same_known_map_not_unseen_map_or_real_car=True,
        no_automatic_model_promotion=True, goal_achieved=False, deployment_approved=False))
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', DEPENDENCY,
            '--property=MainPID,ActiveState,InvocationID,ExecMainStatus'],
            capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if state['InvocationID'] not in ('', INVOCATION):
            raise RuntimeError('Pinned predecessor was restarted or replaced')
        if state['MainPID'] == '0' and state['ActiveState'] in ('inactive', 'failed'):
            break
        write(output / 'STATUS.json', dict(status='waiting_for_exact_memory_diagnostic',
            dependency=state, new_physical_episodes=0, new_ppo_steps=0))
        time.sleep(5.)
    assert state['ActiveState'] == 'inactive' and state['ExecMainStatus'] == '0'
    diagnostic_path = BASE / 'earlier_memory_evidence_20261017/RESULTS.json'
    assert read(diagnostic_path)['completed']
    hashes[str(diagnostic_path.relative_to(ROOT))] = digest(diagnostic_path)

    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.release_recovery import evaluate_recovery
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    from neural_navigation.goal_pose_runtime import check_files

    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    for bundle in [baseline, MODEL]:
        check_files(bundle, read(bundle / 'manifest.json'))
    write(output / 'DEPENDENCIES.json', dict(inputs=hashes, predecessor_terminal=state))
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    physical = output / 'physical'
    physical.mkdir()
    report = dict(completed=False, groups={}, frozen_rows=[], gates={}, new_physical_episodes=0,
        new_optimizer_updates=0, new_ppo_steps=0, candidate_training_steps=1665,
        original_hard_recovery_result=dict(safe_successes=1, total=3, timeouts=2),
        full_goal_still_requires_late_recovery_or_verified_infeasibility_and_prevention=True,
        goal_achieved=False, deployment_approved=False)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        report.update(completed=True, reason=reason, inputs_verified=len(hashes))
        save()
        write(output / 'STATUS.json', dict(status='complete', reason=reason,
            completed_episodes=report['new_physical_episodes']))

    def run(spec, group, recovery=False, bundle=MODEL, variant='candidate'):
        write(output / 'STATUS.json', dict(status='physical_retention_validation',
            group=group, case_id=spec['id'], variant=variant,
            completed_episodes=report['new_physical_episodes']))
        row, trace = (evaluate_recovery if recovery else evaluate)(spec, bundle, budget)
        path = physical / (group + '_' + spec['id'] + '_' + variant + '.npz')
        np.savez_compressed(path, **trace)
        row.update(variant=variant, trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
            bundle=str(bundle.relative_to(ROOT)), no_optimizer_updates=True)
        if group == 'frozen8':
            report['frozen_rows'].append(row)
        else:
            report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        print(json.dumps(dict(group=group, **row)), flush=True)
        return row

    def retained(row, reference):
        time_key = 'post_release_seconds' if row.get('recovery_test') else 'seconds'
        return bool(row['protocol_valid'] and row['safe_success']
            and not any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards'])
            and all(row[k] <= reference[k]*1.2 + 1e-8 for k in [time_key, 'travel_m']))

    fixed_recoveries = [*tasks['reference']['actual95_actual90'], *tasks['reference']['delayed_wait']]
    fixed = {group: {r['case_id']: r for r in rows} for group, rows in [
        ('normal18', tasks['reference']['normal18']), ('sensor5', tasks['reference']['sensor5']),
        ('source_recovery3', fixed_recoveries), ('heading2', tasks['reference']['heading2'])]}
    save()
    for group, specifications, recovery in [('normal18', tasks['normal18'], False),
            ('sensor5', tasks['sensor5'], False), ('source_recovery3', recoveries, True),
            ('heading2', tasks['heading2'], False)]:
        for spec in specifications:
            row = run(spec, group, recovery)
            okay = retained(row, fixed[group][row['case_id']]) and retained(row, parent_rows[row['case_id']])
            report['gates'][row['case_id']] = dict(passed=okay)
            if not okay:
                finish('Fixed retention failed; frozen expansion stopped; no model promotion')
                return
    normal = report['groups']['normal18']['rows']
    report['normal18_against_fixed'] = compare(tasks['reference']['normal18'], normal)
    report['normal18_against_parent'] = compare([parent_rows[r['case_id']] for r in normal], normal)
    report['all_fixed_retention_passed'] = bool(report['normal18_against_fixed']['larger_training_gate_passed']
        and report['normal18_against_parent']['larger_training_gate_passed'])
    if not report['all_fixed_retention_passed']:
        finish('Aggregate efficiency retention failed; frozen expansion stopped')
        return
    for case in cases:
        for variant, bundle in [('parent', baseline), ('candidate', MODEL)]:
            row = run(case, 'frozen8', bundle=bundle, variant=variant)
            if variant == 'candidate' and any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']):
                finish('Candidate safety failure in pre-frozen pair; evaluation stopped')
                return
    before = [r for r in report['frozen_rows'] if r['variant'] == 'parent']
    after = [r for r in report['frozen_rows'] if r['variant'] == 'candidate']
    assert len(before) == len(after) == 8
    report['frozen_paired_comparison'] = compare(before, after)
    report['all_frozen_retention_passed'] = report['frozen_paired_comparison']['larger_training_gate_passed']
    finish('Separate retention study complete; original late-recovery goal remains unresolved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
