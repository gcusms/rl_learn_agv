"""Complete the new residual candidate's frozen development regressions.

Reuse five completed normal tasks and one extra task; evaluate only the rest. Further
groups are evaluated only if the full eighteen retain their safety/efficiency.
No optimizer, action override, guard change or deployment is performed.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/stop_feedback_20261015'
MODEL = ROOT / 'models/stop_feedback_20261015/recovery_followup_bc400_s231016'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Do not restart or redefine the regression plan')
    import numpy as np
    import torch
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.evaluate_feedback_recovery import evaluate_recovery
    from experiments.experiment_provenance import record_provenance
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    known_path = BASE / 'recovery_followup_validation_20261016/RESULTS.json'
    known = json.loads(known_path.read_text())
    assert known['completed'] and known['bounded_gates_passed']
    import subprocess
    terminal = subprocess.check_output(['systemctl', '--user', 'show', 'rl-recovery-followup-validation-20261016.service', '-p', 'MainPID', '-p', 'ActiveState', '-p', 'Result'], text=True)
    assert 'MainPID=0' in terminal and 'ActiveState=inactive' in terminal and 'Result=success' in terminal
    direct_path = BASE / 'actual95_broad_regression_20261016/RESULTS.json'
    direct = json.loads(direct_path.read_text()); assert direct['completed']
    old_path = BASE / 'early_transfer_regression_20261016/RESULTS.json'
    old = json.loads(old_path.read_text())
    assert old['completed'] and len(old['candidate_rows']) == 18
    parent_groups_path = BASE / 'actual_stop_validation_20261016/RESULTS_main.json'
    parent_groups = json.loads(parent_groups_path.read_text())
    assert parent_groups['completed'] and parent_groups['fixed35_groups_completed']
    all_cases_path = BASE / 'w11_recovery_bc800_navigation_20261016/cases.json'
    all_cases = json.loads(all_cases_path.read_text())
    done = [r for r in known['rows'] if r['group'] in ('normal4', 'motion15')]
    fresh_done = [r for r in known['rows'] if r['group'] == 'regressed_fresh33']
    assert len(fresh_done) == 1
    done_ids = {r['case_id'] for r in done}
    pending = [c for c in all_cases if c['id'] not in done_ids]
    assert len(pending) == 13 and len(done_ids) == 5
    fresh_path = BASE / 'proactive_augmented_protocol_20261016/fresh_cases.json'
    prefixes_path = BASE / 'w11_recovery_bc800_prefixes_20261016/specifications.json'
    headings_path = BASE / 'heading_probe_cases.json'
    fresh = json.loads(fresh_path.read_text())
    prefixes = json.loads(prefixes_path.read_text())
    headings = json.loads(headings_path.read_text())
    w11_path = BASE / 'w11_recovery_bc800_w11_20261016/specifications.json'
    w11 = []  # The new model already passed the exact90/95 prefixes; never count them twice.
    assert (len(fresh), len(prefixes), len(headings), len(w11)) == (12, 2, 2, 0)
    write('pending_cases.json', pending)
    write('finished_rows.json', done)
    bundle = MODEL / 'bc_portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    paths = [Path(__file__), known_path, old_path, direct_path, parent_groups_path, all_cases_path,
             fresh_path, prefixes_path, headings_path, w11_path,
             OUT / 'pending_cases.json', OUT / 'finished_rows.json',
             MODEL / 'policy_bc.zip', MODEL / 'RESULTS.json', bundle / 'manifest.json']
    paths += [bundle / name for name in manifest['files']]
    for folder in ['experiments', 'neural_navigation', 'navrl']:
        paths += list((ROOT / 'training' / folder).rglob('*.py'))
    paths += [ROOT / 'training/neural_navigation/static_tracking_kernel.cpp']
    for case in [*all_cases, *fresh, *headings, *(p['case'] for p in [*prefixes, *w11])]:
        paths += [p for p in (ROOT / 'training' / case['scene']).iterdir() if p.is_file()]
    for row in [*done, *fresh_done]:
        trace = ROOT / row['trace_path']
        assert digest(trace) == row['trace_sha256']
        paths.append(trace)
    for prefix in [*prefixes, *w11]:
        paths.append(ROOT / prefix['trace'])
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in set(paths)}
    plan = dict(at=stamp(), inputs=inputs, checkpoint_sha256=digest(MODEL / 'policy_bc.zip'),
        reused_finished_full_tasks=sorted(done_ids), new_full_tasks=13,
        subsequent_groups=dict(additional_conditions=11, original_prefixes=2, heading_only=2, w11_t90=0),
        subsequent_groups_require_full18_safety_and_efficiency=True,
        maximum_new_physical_cases=28, reused_extra_task_ids=[r['case_id'] for r in fresh_done],
        fixed_predecessor_state=terminal, actual106_not_claimed_fixed=True,
        unchanged_body_lidar_actions_masks_guard_and_timeout=True,
        new_ppo_steps=0, new_optimizer_updates=0,
        resource=dict(soft_cpu_cores=.35, hard_cpu_percent=50, memory_high_mib=3072, memory_max_mib=4096),
        no_parallel_physical_main=True, deployment_approved=False, goal_achieved=False)
    write('PLAN.json', plan)
    def verify():
        for name, expected in inputs.items():
            if digest(ROOT / name) != expected:
                raise ValueError('A fixed regression input changed: ' + name)
    verify()
    physical = OUT / 'physical'
    physical.mkdir()
    record_provenance(physical, bundle, [Path(__file__)])
    budget = ExperimentBudget(.35, memory_mib=3072, directory=physical)
    report = dict(completed=False, at=stamp(), full18_rows=list(done), groups={'fresh12':dict(rows=list(fresh_done))},
                  new_ppo_steps=0, new_optimizer_updates=0, deployment_approved=False, goal_achieved=False)
    def run(case, group, recovery=False):
        verify()
        name = case['id']
        write('STATUS.json', dict(at=stamp(), status='physical_regression', group=group, case_id=name,
            finished_new_cases=sum(len(g.get('rows', [])) for g in report['groups'].values())-len(fresh_done)))
        row, trace = (evaluate_recovery if recovery else evaluate)(case, bundle, budget)
        path = physical / (group + '_' + name + '.npz')
        np.savez_compressed(path, **trace)
        row.update(variant='recovery_followup_bc400', trace_sha256=digest(path),
                   trace_path=str(path.relative_to(ROOT)), model_only_after_initialization=True)
        report['groups'].setdefault(group, dict(rows=[]))['rows'].append(row)
        report['resource'] = budget.statistics()
        write('RESULTS.json', report)
        print(json.dumps(row), flush=True)
        return row
    for case in pending:
        report['full18_rows'].append(run(case, 'remaining_full13'))
    assert len(report['full18_rows']) == len({r['case_id'] for r in report['full18_rows']}) == 18
    report['full18_pair'] = compare(old['candidate_rows'], report['full18_rows'])
    report['full18_direct_parent_pair'] = compare(direct['full18_rows'], report['full18_rows'])
    report['full18_completed'] = True
    report['subsequent_groups_admitted'] = (report['full18_pair']['all_regression_checks_passed']
        and report['full18_direct_parent_pair']['all_regression_checks_passed'])
    write('RESULTS.json', report)
    if report['subsequent_groups_admitted']:
        for case in fresh:
            if case['id'] not in {r['case_id'] for r in fresh_done}:
                run(case, 'fresh12')
        assert len(report['groups']['fresh12']['rows']) == 12
        report['groups']['fresh12']['direct_parent_pair'] = compare(direct['groups']['fresh12']['rows'], report['groups']['fresh12']['rows'])
        report['groups']['fresh12']['paired'] = compare(parent_groups['groups']['fresh']['rows'], report['groups']['fresh12']['rows'])
        for prefix in prefixes:
            run(prefix, 'original_prefixes2', recovery=True)
        report['groups']['original_prefixes2']['paired'] = compare(parent_groups['groups']['prefixes']['rows'], report['groups']['original_prefixes2']['rows'])
        for case in headings:
            run(case, 'heading2')
        for prefix in w11:
            run(prefix, 'w11_t90', recovery=True)
    verify()
    report.update(completed=True, at=stamp(), resource=budget.statistics(),
        broader_regression_completed=report['subsequent_groups_admitted'],
        scope='Reused development cases only; five normal tasks and one extra case reused, not counted as new. No actual106 repair or deployment claim.')
    write('RESULTS.json', report)
    write('STATUS.json', dict(at=stamp(), status='complete', new_ppo_steps=0, goal_achieved=False))


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('FAILURE.json', dict(at=stamp(), error=repr(error), automatically_restarted=False,
                                  goal_achieved=False, deployment_approved=False))
        raise
