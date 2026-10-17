"""Full serial development regressions after the nine sensor-release checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments.evaluate_docking_navigation import evaluate
from experiments.experiment_provenance import record_provenance
from experiments.long_risk_actor import digest
from experiments.paired_navigation_gate import compare
from experiments.release_recovery import evaluate_recovery
from neural_navigation.experiment_budget import ExperimentBudget


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/stop_feedback_20261015'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    model = args.model.resolve()
    pilot = args.pilot.resolve()
    prior = json.loads((pilot / 'RESULTS.json').read_text())
    assert prior['completed'] and len(prior['rows']) == 9
    assert prior['gates'] and all(g['passed'] for g in prior['gates'].values())
    process = subprocess.check_output(['systemctl', '--user', 'show',
        args.predecessor_unit, '-p', 'MainPID', '-p', 'ActiveState'], text=True)
    process = dict(line.split('=',1) for line in process.splitlines())
    assert process['MainPID'] == '0' and process['ActiveState'] == 'inactive'
    admission = json.loads((pilot / 'MODEL_ADMISSION.json').read_text())
    own = next(r for r in admission['models'] if r['model'] == str(model.relative_to(ROOT)))
    assert own['checkpoint_sha256'] == digest(model / 'checkpoint.pt')
    prior_inputs = json.loads((pilot / 'PLAN.json').read_text())['inputs']
    for name, expected in prior_inputs.items():
        assert digest(ROOT / name) == expected, name
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
    full, fresh, prefixes, w11, headings = [json.loads(p.read_text())
        for p in [full_path, fresh_path, prefix_path, w11_path, heading_path]]
    direct, direct_first, retained, wait_parent = [json.loads(p.read_text())
        for p in [direct_path, direct_first_path, retained_path, wait_parent_path]]
    assert all(r['completed'] for r in [direct, direct_first, retained, wait_parent])
    assert (len(full), len(fresh), len(prefixes), len(headings)) == (18,12,2,2)
    reused_heading = [r for r in prior['rows'] if r['variant'] == 'candidate' and r['group'] == 'heading']
    assert {r['case_id'] for r in reused_heading} == {c['id'] for c in headings}
    assert all(r['safe_success'] for r in reused_heading)
    actual = [next(s for s in w11 if s['id'] == 'w11_training_stall_t' + suffix)
        for suffix in ['95','90']]
    actual106 = next(s for s in w11 if s['id'] == 'w11_training_stall_t106')
    delayed = dict(actual[0], id='w11_actual95_wait2_4', release_s=97.4, trace=str(wait_trace_path))
    files = [Path(__file__), pilot / 'RESULTS.json', pilot / 'PLAN.json', pilot / 'MODEL_ADMISSION.json',
        full_path, fresh_path, prefix_path, w11_path, heading_path, direct_path, direct_first_path,
        retained_path, wait_parent_path, wait_trace_path, model / 'checkpoint.pt', model / 'RESULTS.json',
        bundle / 'manifest.json']
    files += [bundle / name for name in manifest['files']]
    files += [ROOT / s['trace'] for s in [*prefixes,*w11]]
    files += [ROOT / r['trace_path'] for r in reused_heading]
    for folder in ['neural_navigation','navrl','experiments']:
        files += list((ROOT / 'training' / folder).rglob('*.py'))
    files += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    for case in [*full,*fresh,*headings,*[s['case'] for s in [*prefixes,*w11]]]:
        files += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in set(files)}
    write(output / 'PLAN.json', dict(inputs=hashes, maximum_new_physical_episodes=36,
        ordered_groups=['actual95_actual90','normal18','extra12','prefixes2','added_wait2_4','actual106_diagnostic'],
        reused_heading_cases=[r['case_id'] for r in reused_heading], reused_results_are_not_new_episodes=True,
        heading_model_and_source_inputs_pinned=True, complete_extra12_requires_twelve_new_cases=True,
        stop_after_recovery_or_normal_safety_efficiency_failure=True,
        prefix_commands_initialize_only_and_no_teacher_after_release=True,
        original_actions_masks_body_margin_blind_memory_and_deadlines=True,
        new_ppo_steps=0, new_optimizer_updates=0, deployment_approved=False))
    record_provenance(output, bundle, [Path(__file__)])
    (output / 'driver_snapshot.py').write_text(Path(__file__).read_text())
    physical = output / 'physical';physical.mkdir()
    report = dict(completed=False, groups={'heading2':dict(rows=reused_heading, reused=True)},
        new_physical_episodes=0, new_ppo_steps=0, new_optimizer_updates=0,
        actual106_not_assumed_solved=True, goal_achieved=False, deployment_approved=False)
    report['groups']['heading2']['paired'] = compare(direct['groups']['heading2']['rows'], reused_heading)

    def save():
        budget.tick()
        report.update(at=datetime.now(timezone.utc).isoformat(), resource=budget.statistics())
        write(output / 'RESULTS.json', report)

    def finish(reason):
        for name, expected in hashes.items():
            assert digest(ROOT / name) == expected, name
        assert report['new_physical_episodes'] <= 36
        report.update(completed=True, reason=reason)
        save()
        write(output / 'STATUS.json', dict(status='complete', reason=reason,
            new_physical_episodes=report['new_physical_episodes']))

    def run(case, group, recovery=False):
        write(output / 'STATUS.json', dict(status='physical_regression', group=group, case_id=case['id'],
            finished_new_episodes=report['new_physical_episodes']))
        print(json.dumps(dict(status='starting', group=group, case_id=case['id'])), flush=True)
        row, trace = (evaluate_recovery if recovery else evaluate)(case, bundle, budget)
        path = physical / (group + '_' + case['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(variant=model.name, trace_path=str(path.relative_to(ROOT)),
            trace_sha256=digest(path), model_only_after_initialization=True)
        report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['new_physical_episodes'] += 1
        save()
        print(json.dumps(row), flush=True)
        return row

    save()
    for spec in actual:
        run(spec, 'actual95_actual90', True)
    group = report['groups']['actual95_actual90']
    reference = [next(r for r in direct_first['rows'] if r['case_id'] == s['id']) for s in actual]
    group['paired'] = compare(reference, group['rows'])
    if not all(r['safe_success'] for r in group['rows']) or not group['paired']['all_regression_checks_passed']:
        finish('Retained 90/95 recovery gate failed; wider expansion stopped')
        return
    for case in full:
        run(case, 'normal18')
    group = report['groups']['normal18']
    group['paired'] = compare(direct['full18_rows'], group['rows'])
    if not all(r['safe_success'] for r in group['rows']) or not group['paired']['all_regression_checks_passed']:
        finish('Normal18 safety or efficiency gate failed; wider expansion stopped')
        return
    for case in fresh:
        run(case, 'extra12')
    extra = report['groups']['extra12']
    extra['followup_pair'] = compare(direct['groups']['fresh12']['rows'], extra['rows'])
    extra['retained_2k_pair'] = compare(retained['rows'], extra['rows'])
    for spec in prefixes:
        run(spec, 'prefixes2', True)
    report['groups']['prefixes2']['paired'] = compare(direct['groups']['original_prefixes2']['rows'],
        report['groups']['prefixes2']['rows'])
    ready = (all(r['safe_success'] for r in extra['rows'])
        and all(extra[k]['all_regression_checks_passed'] for k in ['followup_pair','retained_2k_pair'])
        and all(report['groups'][k]['paired']['all_regression_checks_passed'] for k in ['prefixes2','heading2']))
    report['added_wait_admitted'] = ready
    save()
    if not ready:
        finish('Extra/prefix/heading gate failed; delayed recovery not expanded')
        return
    row = run(delayed, 'added_wait2_4', True)
    old_wait = next(r for r in wait_parent['rows'] if r['variant'] == 'new_data')
    report['groups']['added_wait2_4']['paired'] = compare([old_wait], [row])
    if not row['safe_success'] or not report['groups']['added_wait2_4']['paired']['all_regression_checks_passed']:
        finish('Delayed recovery safety or efficiency failed; actual106 not expanded')
        return
    run(actual106, 'actual106_diagnostic', True)
    report['all_broad_development_cases_safe'] = all(r['safe_success']
        for group in report['groups'].values() for r in group['rows'])
    finish('Declared broad checks finished; assess every failed or unverified requirement separately')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--predecessor-unit', default='rl-sensor-release-validation-20261016.service')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write(args.output / 'FAILURE.json', dict(error=repr(error), automatically_restarted=False))
        raise
