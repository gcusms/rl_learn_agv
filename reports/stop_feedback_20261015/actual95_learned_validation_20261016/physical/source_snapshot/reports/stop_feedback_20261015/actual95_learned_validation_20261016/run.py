"""Test the fitted model alone after the actual 95/106 s histories."""
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
MODEL = ROOT / 'models/stop_feedback_20261015/actual95_transfer_bc400_s231016'
DEPENDENCY = 'rl-actual95-transfer-fit-20261016.service'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def state():
    data = subprocess.check_output(['systemctl', '--user', 'show', DEPENDENCY,
        '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID', '-p', 'InvocationID'], text=True)
    return dict(line.split('=', 1) for line in data.splitlines())


def verify(plan):
    for name, expected in plan['inputs'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('A fixed validation input changed: ' + name)


def main():
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Do not restart an existing validation plan')
    dependency = state()
    prefixes = []
    original = json.loads((BASE / 'actual_stop_validation_20261016/actual_prefix.json').read_text())
    for release in [95, 106]:
        prefix = dict(original, id='w11_training_stall_t' + str(release), release_s=float(release))
        prefixes.append(prefix)
    write('prefixes.json', prefixes)
    cases_path = BASE / 'w11_recovery_bc800_navigation_20261016/cases.json'
    all_cases = json.loads(cases_path.read_text())
    selected = ['train_failure_w11_e7', 'motion_profile_002', 'recovery_holdout_00', 'recovery_holdout_01']
    cases = [next(c for c in all_cases if c['id'] == name) for name in selected]
    write('cases.json', cases)
    paths = [Path(__file__), OUT / 'prefixes.json', OUT / 'cases.json', cases_path,
        BASE / 'actual_stop_validation_20261016/RESULTS_main.json',
        BASE / 'early_transfer_regression_20261016/RESULTS.json',
        BASE / 'actual95_transfer_20261016/run.py',
        ROOT / 'training/experiments/evaluate_feedback_recovery.py',
        ROOT / 'training/experiments/evaluate_docking_navigation.py']
    plan = dict(at=stamp(), dependency_unit=DEPENDENCY, dependency_invocation=dependency['InvocationID'],
        dependency_at_prepare=dependency, inputs={str(p.relative_to(ROOT)): digest(p) for p in paths},
        candidate=str(MODEL.relative_to(ROOT)), recovery_cases=2, normal_regression_cases=selected,
        normal_regression_runs_only_after_one_complete_actual_recovery=True,
        no_expert_after_physical_prefix_release=True, no_new_ppo=True,
        preserves_original_physical_footprint_guard_lidar_actions_and_timeout=True,
        resource=dict(soft_cpu_cores=.35, hard_cpu_percent=50, memory_high_mib=3072, memory_max_mib=4096),
        larger_training_admitted=False, deployment_approved=False, goal_achieved=False)
    write('PLAN.json', plan)
    while True:
        current = state()
        if current['InvocationID'] not in ('', plan['dependency_invocation']):
            raise ValueError('The supervised fitting job was restarted')
        if current['ActiveState'] in ('inactive', 'failed'):
            if current['ActiveState'] != 'inactive' or current['Result'] != 'success' or current['MainPID'] != '0':
                raise RuntimeError('The supervised fitting job did not complete successfully')
            break
        write('STATUS.json', dict(status='waiting_for_fitted_model', at=stamp(),
            physical_simulation_started=False, dependency=current))
        time.sleep(5)
    verify(plan)
    fit = json.loads((MODEL / 'RESULTS.json').read_text())
    if not fit['completed'] or not fit['admitted_for_bounded_physical_diagnosis_only']:
        write('RESULTS.json', dict(completed=True, skipped=True, at=stamp(),
            reason='The fitted candidate did not preserve necessary stopping labels.', goal_achieved=False))
        return
    import numpy as np
    import torch
    from experiments.evaluate_feedback_recovery import evaluate_recovery
    from experiments.evaluate_docking_navigation import evaluate
    from experiments.experiment_provenance import record_provenance
    from experiments.paired_navigation_gate import compare
    from neural_navigation.experiment_budget import ExperimentBudget
    torch.set_num_threads(1)
    bundle = MODEL / 'bc_portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    candidate_pins = {str((MODEL / name).relative_to(ROOT)): digest(MODEL / name)
        for name in ['policy_bc.zip', 'RESULTS.json', 'config.json']}
    for name in ['manifest.json', *manifest['files']]:
        p = bundle / name
        candidate_pins[str(p.relative_to(ROOT))] = digest(p)
    plan['inputs'].update(candidate_pins)
    verify(plan)
    write('CANDIDATE_ADMISSION.json', dict(at=stamp(), candidate_pins=candidate_pins,
        dependency_terminal=current, no_automatic_deployment_or_ppo=True))
    output = OUT / 'physical'
    output.mkdir()
    record_provenance(output, bundle, [Path(__file__)])
    budget = ExperimentBudget(.35, memory_mib=3072, directory=output)
    results = dict(completed=False, recovered_rows=[], normal_rows=[],
        new_ppo_steps=0, deployment_approved=False, goal_achieved=False)
    for prefix in prefixes:
        verify(plan)
        write('STATUS.json', dict(status='actual_recovery_validation', at=stamp(), case=prefix['id']))
        row, trace = evaluate_recovery(prefix, bundle, budget)
        path = output / (prefix['id'] + '.npz')
        np.savez_compressed(path, **trace)
        row.update(trace_sha256=digest(path), variant='actual95_transfer_bc400', no_expert_after_release=True)
        results['recovered_rows'].append(row)
        (output / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    recovered = any(r['safe_success'] for r in results['recovered_rows'])
    if recovered:
        for case in cases:
            verify(plan)
            write('STATUS.json', dict(status='normal_regression', at=stamp(), case=case['id']))
            row, trace = evaluate(case, bundle, budget)
            path = output / (case['id'] + '.npz')
            np.savez_compressed(path, **trace)
            row.update(trace_sha256=digest(path), variant='actual95_transfer_bc400')
            results['normal_rows'].append(row)
            (output / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    results.update(completed=True, at=stamp(), normal_regression_skipped_for_no_actual_recovery=not recovered,
        resource=budget.statistics(), scope='Reused development tasks; fixed sensor/action prefix is initialization, not PPO or model success.')
    reference = json.loads((BASE / 'actual_stop_validation_20261016/RESULTS_main.json').read_text())
    old_prefixes = [r for r in reference['groups']['w11']['rows'] if r['case_id'] in [p['id'] for p in prefixes]]
    results['recovery_pair'] = compare(old_prefixes, results['recovered_rows'])
    if results['normal_rows']:
        old_normal = json.loads((BASE / 'early_transfer_regression_20261016/RESULTS.json').read_text())
        original_rows = old_normal['candidate_rows']
        results['normal_pair'] = compare([r for r in original_rows if r['case_id'] in selected], results['normal_rows'])
    (output / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
    write('RESULTS.json', results)
    write('STATUS.json', dict(status='complete', at=stamp(), actual_recovery_gained=recovered,
        goal_achieved=False, deployment_approved=False))


if __name__ == '__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as error:
        write('STATUS.json', dict(status='failed', at=stamp(), error=repr(error), automatically_restarted=False))
        raise
