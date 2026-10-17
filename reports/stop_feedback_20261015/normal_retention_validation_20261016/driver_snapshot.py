"""Bounded closed-loop validation after full source fit, with fail-fast gates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.long_risk_actor import digest
from experiments.paired_navigation_gate import compare
from experiments.release_recovery import evaluate_recovery
from neural_navigation.experiment_budget import ExperimentBudget
from neural_navigation.goal_pose_runtime import check_files


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    fits, output = args.fits.resolve(), args.output.resolve()
    paired_fit = json.loads((fits / 'RESULTS.json').read_text())
    assert paired_fit['completed']
    selected = paired_fit['selected_for_bounded_physical_validation']
    assert selected is not None, 'No source-fit-qualified candidate; no physics'
    model = ROOT / selected
    fitted = json.loads((model / 'RESULTS.json').read_text())
    assert fitted['completed'] and fitted['exported_for_physical_validation']
    assert fitted['correct_source_labels'] == fitted['total_source_labels'] == 3728
    assert fitted['original3252_all_retained'] and fitted['safe_stop_labels_all_retained']
    assert digest(model / 'checkpoint.pt') == fitted['checkpoint_sha256']
    bundle = model / 'portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert not manifest['deployment_approved']
    check_files(bundle, manifest)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=2800, directory=output)
    tasks_path = BASE / 'context_feedback_longfit_protocol_20261016/tasks.json'
    tasks = json.loads(tasks_path.read_text())
    late_path = BASE / 'parent_wait_protocol_20261016/specifications.json'
    late = json.loads(late_path.read_text())
    fixed_ids = ['recovery_holdout_03', 'motion_profile_002']
    fixed = [next(c for c in tasks['normal18'] if c['id'] == name) for name in fixed_ids]
    others = [c for c in tasks['normal18'] if c['id'] not in fixed_ids]
    source = [*tasks['actual95_actual90'], tasks['delayed_wait']]
    hard = [tasks['actual106'], *late]
    assert len(others) == 16 and len(source) == len(hard) == 3
    source_reference = [*tasks['reference']['actual95_actual90'], *tasks['reference']['delayed_wait']]
    assert {s['id'] for s in source} == {r['case_id'] for r in source_reference}
    for group in ['heading2', 'sensor5', 'normal18']:
        assert {s['id'] for s in tasks[group]} == {r['case_id'] for r in tasks['reference'][group]}
    files = [Path(__file__), fits / 'RESULTS.json', fits / 'PLAN.json',
        model / 'RESULTS.json', model / 'checkpoint.pt', bundle / 'manifest.json', tasks_path, late_path]
    files += [bundle / name for name in manifest['files']]
    files += [ROOT / spec['trace'] for spec in [*source, *hard]]
    for folder in ['neural_navigation', 'navrl', 'experiments']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    cases = [*fixed, *others, *tasks['heading2'], *tasks['sensor5'], *[s['case'] for s in source + hard]]
    for case in cases:
        files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
        selected_model=selected, source_fit_all3728_passed=True,
        maximum_new_physical_episodes=31,
        ordered_groups=['normal_repaired2', 'source_recovery3', 'heading2', 'sensor5', 'normal_other16', 'hard3'],
        stop_on_safety_success_or_efficiency_regression_before_hard_group=True,
        hard3_timeouts_are_unresolved_and_do_not_count_as_success=True,
        normal18_requires_both_normal_groups_complete=True,
        all_tasks_are_development=True, no_unseen_generalization_claim=True,
        reused_teacher_sources_not_independent_success=True,
        no_runtime_teacher_or_forced_motion=True, no_physics_after_fit_failure=True,
        original_actions_body4cm_margin_sensors_masks_and_deadlines=True,
        no_optimizer_updates=True, new_ppo_steps=0, deployment_approved=False))
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    physical = output / 'physical'; physical.mkdir()
    report = dict(completed=False, groups={}, gates={}, new_physical_episodes=0,
        selected_model=selected, new_ppo_steps=0, new_optimizer_updates=0,
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

    def run(spec, group, recovery=False):
        write(output / 'STATUS.json', dict(status='bounded_physical_validation', group=group,
            case_id=spec['id'], completed_episodes=report['new_physical_episodes']))
        print(json.dumps(dict(starting=spec['id'], group=group)), flush=True)
        row, trace = (evaluate_recovery if recovery else evaluate)(spec, bundle, budget)
        path = physical / (group + '_' + spec['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_path=str(path.relative_to(ROOT)), trace_sha256=digest(path),
            variant=selected, pure_learned_model=True,
            training_regression_or_development=True)
        report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        return row

    def check(row, reference):
        return bool(row['safe_success'] and row['protocol_valid'] and not row['physical_guards']
            and all(row[key] <= reference[key] * 1.2 + 1e-8 for key in ['seconds', 'travel_m']))

    save()
    for case in fixed:
        row = run(case, 'normal_repaired2')
        reference = next(r for r in tasks['reference']['normal18'] if r['case_id'] == row['case_id'])
        passed = check(row, reference)
        report['gates'][case['id']] = dict(passed=passed)
        if not passed:
            finish('Known normal mission regression; no wider physics expansion')
            return
    for spec in source:
        row = run(spec, 'source_recovery3', True)
        reference = next(r for r in source_reference if r['case_id'] == row['case_id'])
        if not check(row, reference):
            finish('Original source recovery regression; no wider expansion')
            return
    for group in ['heading2', 'sensor5']:
        for case in tasks[group]:
            row = run(case, group)
            reference = next(r for r in tasks['reference'][group] if r['case_id'] == row['case_id'])
            if not check(row, reference):
                finish(group + ' safety or efficiency regression; no wider expansion')
                return
    for case in others:
        row = run(case, 'normal_other16')
        reference = next(r for r in tasks['reference']['normal18'] if r['case_id'] == row['case_id'])
        if not check(row, reference):
            finish('Normal mission safety or efficiency regression; no wider expansion')
            return
    normal = report['groups']['normal_repaired2']['rows'] + report['groups']['normal_other16']['rows']
    assert len(normal) == 18
    report['normal18_pair'] = compare(tasks['reference']['normal18'], normal)
    report['normal18_completed'] = True
    if not report['normal18_pair']['all_regression_checks_passed']:
        finish('Complete normal18 aggregate efficiency regression; no hard-state expansion')
        return
    for spec in hard:
        row = run(spec, 'hard3', True)
        if not row['protocol_valid'] or any(row[k] for k in ['collisions', 'false_arrival', 'physical_guards']):
            finish('Hard-state safety regression; stopped')
            return
    report['hard_states_all_solved'] = all(r['safe_success'] for r in report['groups']['hard3']['rows'])
    finish('Declared bounded validation finished; hard-state timeouts remain unresolved if present')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['fits', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
