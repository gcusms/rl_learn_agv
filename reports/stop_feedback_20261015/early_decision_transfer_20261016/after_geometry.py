"""Run three declared candidate checks after the existing auxiliary pipeline.

This data-only BC factor stays separate from the live PPO-retention comparison
and from the changed-guard rotation diagnostic. Waiting for those experiments
does not require occupying another physical simulation slot.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def write(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def state(unit):
    return dict(line.split('=', 1) for line in subprocess.check_output([
        'systemctl', '--user', 'show', unit, '-p', 'Id', '-p', 'MainPID',
        '-p', 'ActiveState', '-p', 'Result', '-p', 'InvocationID'], text=True).splitlines())


def verify_inputs(plan):
    for name, expected in plan['inputs'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen candidate validation input changed: ' + name)


def wait_dependencies(plan):
    last = None
    while True:
        states = {}
        finished = True
        for unit, invocation in plan['dependencies'].items():
            current = state(unit)
            if current['InvocationID'] not in ('', invocation):
                raise ValueError('Dependency was restarted: ' + unit)
            if current['ActiveState'] in ('inactive', 'failed'):
                if (current['ActiveState'] != 'inactive' or current['Result'] != 'success'
                        or current['MainPID'] != '0'):
                    raise RuntimeError('Dependency did not finish successfully: ' + unit)
            else:
                finished = False
            states[unit] = current
        if finished:
            return states
        signature = json.dumps(states, sort_keys=True)
        if signature != last:
            write('QUICK_STATUS.json', dict(status='waiting', at=stamp(), dependency_states=states,
                physical_simulation_started=False))
            last = signature
        time.sleep(5)


def main():
    os.chdir(ROOT)
    plan = json.loads((OUT / 'QUICK_PLAN.json').read_text())
    verify_inputs(plan)
    states = wait_dependencies(plan)
    verify_inputs(plan)
    model = Path(plan['candidate_model'])
    fitted = json.loads((model / 'RESULTS.json').read_text())
    status = json.loads((model / 'status.json').read_text())
    if (not fitted['completed'] or status['status'] != 'offline_bc_complete'
            or fitted['additional_ppo_steps'] or fitted['environment_steps']):
        raise ValueError('Complete offline-only initialization is required')
    geometry = json.loads(Path(plan['geometry_results']).read_text())
    if not geometry['completed']:
        raise ValueError('Complete preceding geometry diagnostic required')
    if (fitted['correct_examples'] != 2355 or fitted['correct_original_examples'] != 2089
            or fitted['correct_original_wait_examples'] != 37):
        write('QUICK_ADMISSION.json', dict(admitted=False, at=stamp(),
            reason='Verified original and new supervised labels are not all retained',
            fitted_result=fitted, dependency_states=states, physical_simulation_started=False))
        write('QUICK_STATUS.json', dict(status='rejected_before_physical_test', at=stamp(),
            automatically_restarted=False, goal_achieved=False))
        return
    finalization = model / 'METADATA_FINALIZATION.json'
    if not finalization.exists():
        subprocess.run([sys.executable, str(OUT / 'finalize_metadata.py')], check=True)
    finalized = json.loads(finalization.read_text())
    if not finalized['completed'] or not finalized['weights_optimizer_data_and_rng_unchanged']:
        raise ValueError('Reference-BC metadata was not finalized without changing artifacts')
    candidate = model / 'bc_portable'
    baseline = Path(plan['baseline_bundle'])
    for name in ('hardware.json', 'localizer.pt', 'router.pt', 'decision_feedback.json'):
        if (candidate / name).read_bytes() != (baseline / name).read_bytes():
            raise ValueError('Candidate changed an unselected runtime component: ' + name)
    source = json.loads(Path(plan['baseline_results']).read_text())
    if not source['completed']:
        raise ValueError('Frozen BC navigation reference is incomplete')
    ids = set(plan['roles'])
    old = [row for row in source['rows'] if row['case_id'] in ids]
    if len(old) != 3:
        raise ValueError('Exactly three frozen reference outcomes required')
    write('QUICK_ADMISSION.json', dict(admitted=True, at=stamp(), dependency_states=states,
        source_bc_results_reused_not_reexecuted=True, main_ppo_pair_may_continue_independently=True,
        soft_cpu_cores=.15, guard_matches_original_reference=True, data_only_factor=True,
        timed_intervention_used=False, current_ppo_comparison_changed=False,
        no_additional_auxiliary_simulation=True, roles=plan['roles']))
    write('QUICK_STATUS.json', dict(status='running', at=stamp(), stage='three_complete_physical_tasks'))
    command = [sys.executable, '-m', 'experiments.evaluate_docking_navigation',
        '--candidate', str(candidate), '--candidate-name', 'early_transfer_bc800',
        '--cases', str(OUT / 'quick_cases.json'), '--output', plan['output'], '--cpu-cores', '.15']
    print(json.dumps(dict(starting=command)), flush=True)
    subprocess.run(command, check=True)
    report = json.loads((Path(plan['output']) / 'RESULTS.json').read_text())
    if not report['completed'] or len(report['rows']) != 3:
        raise ValueError('All three predeclared physical outcomes required')
    from experiments.paired_navigation_gate import compare
    teacher = json.loads(Path(plan['teacher_results']).read_text())
    training_ids = {key for key, value in plan['roles'].items() if value == 'new_training_rehearsal'}
    teacher_rows = [row for row in teacher['rows'] if row['case_id'] in training_ids]
    training_rows = [row for row in report['rows'] if row['case_id'] in training_ids]
    result = dict(completed=True, at=stamp(), candidate_rows=report['rows'],
        bc_to_data_transfer=compare(old, report['rows']),
        successful_ppo_teacher_to_training_rehearsal=compare(teacher_rows, training_rows),
        roles=plan['roles'], same_original_guard=True, timed_intervention_used=False,
        training_cases_not_reported_as_unseen=True, main_retention_pair_results_not_mixed=True,
        original_fourteen_successes_fully_verified=False, extra_eight_cases_not_yet_executed=True,
        no_automatic_larger_evaluation_or_training=True, new_ppo_steps=0,
        scope='One known-success regression and two newly trained rehearsals; not complete acceptance.',
        goal_achieved=False, deployment_approved=False)
    write('QUICK_RESULTS.json', result)
    write('QUICK_STATUS.json', dict(status='complete', at=stamp(), automatically_promoted=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write('QUICK_STATUS.json', dict(status='failed', at=stamp(), error=repr(error),
            automatically_restarted=False, goal_achieved=False))
        raise
