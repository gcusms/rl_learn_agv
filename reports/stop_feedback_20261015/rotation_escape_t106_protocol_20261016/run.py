"""Reuse the auxiliary slot for one declared guard-geometry experiment.

The policy is frozen. Enabling the existing rotation escape changes the mask,
so these results are not scores under the original protection configuration.
The optional fixed-window probe is a diagnostic, never a deployment teacher.
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


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def state(unit):
    return dict(line.split('=', 1) for line in subprocess.check_output([
        'systemctl', '--user', 'show', unit, '-p', 'Id', '-p', 'MainPID',
        '-p', 'ActiveState', '-p', 'Result', '-p', 'InvocationID'], text=True).splitlines())


def verify_inputs(plan):
    for name, expected in plan['inputs'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen diagnostic input changed: ' + name)


def run(module, *arguments):
    command = [sys.executable, '-m', module, *map(str, arguments)]
    print(json.dumps(dict(starting=command)), flush=True)
    subprocess.run(command, check=True)


def complete_result(directory, diagnostic=False):
    report = json.loads((Path(directory) / 'RESULTS.json').read_text())
    if not report['completed'] or len(report['rows']) != 1:
        raise ValueError('One complete physical continuation is required')
    if diagnostic and not report['not_a_trained_policy']:
        raise ValueError('The fixed-window probe must be labelled diagnostic')
    row = report['rows'][0]
    if not row['prefix_sensor_and_physical_parity_verified']:
        raise ValueError('Physical prefix was not reproduced')
    return row


def safe_to_continue(row):
    return not (row['collisions'] or row['physical_guards'] or row['false_arrival'])


def main():
    os.chdir(ROOT)
    plan = json.loads((OUT / 'PLAN.json').read_text())
    verify_inputs(plan)
    while True:
        dependency = state(plan['dependency_unit'])
        if dependency['InvocationID'] not in ('', plan['dependency_invocation']):
            raise RuntimeError('Auxiliary dependency was restarted')
        if dependency['ActiveState'] in ('inactive', 'failed'):
            if (dependency['ActiveState'] != 'inactive' or dependency['Result'] != 'success'
                    or dependency['MainPID'] != '0'):
                raise RuntimeError('Auxiliary dependency did not finish successfully')
            break
        write('STATUS.json', dict(status='waiting', at=stamp(), dependency_state=dependency,
            physical_simulation_started=False))
        time.sleep(5)
    verify_inputs(plan)
    fresh = json.loads(Path(plan['dependency_completed_report']).read_text())
    paired = json.loads(Path(plan['dependency_paired_report']).read_text())
    if not fresh['completed'] or len(fresh['rows']) != 24 or not paired['completed']:
        raise ValueError('All 24 paired rows and their final paired report must finish first')
    write('ADMISSION.json', dict(admitted=True, at=stamp(), dependency_state=dependency,
        completed_dependency_rows=24, soft_cpu_cores=.15, original_protection_changed=True,
        original_deployment_replaced=False, new_ppo_steps=0))
    write('STATUS.json', dict(status='running', stage='frozen_policy_changed_guard', at=stamp()))
    run('experiments.evaluate_feedback_recovery', '--bundle', plan['diagnostic_bundle'],
        '--specifications', OUT / 'model_specifications.json', '--output', plan['model_output'],
        '--label', 'frozen_bc_rotation_escape', '--cpu-cores', '.15')
    model = complete_result(plan['model_output'])
    diagnostic = None
    skip = None
    if model['safe_success']:
        skip = 'Frozen model already completed the declared continuation safely'
    elif not safe_to_continue(model):
        skip = 'Unsafe complete model continuation; no further probe under this plan'
    else:
        verify_inputs(plan)
        write('STATUS.json', dict(status='running', stage='legal_rotation_window_diagnostic', at=stamp()))
        run('experiments.probe_recovery_prefix', '--bundle', plan['diagnostic_bundle'],
            '--prefix', OUT / 'prefix.json', '--specifications', OUT / 'probe_specifications.json',
            '--output', plan['probe_output'], '--cpu-cores', '.15')
        diagnostic = complete_result(plan['probe_output'], diagnostic=True)
    write('RESULTS.json', dict(completed=True, at=stamp(), frozen_model=model,
        timed_probe=diagnostic, timed_probe_skipped_reason=skip,
        guard_changed=True, current_ppo_configuration_changed=False,
        successful_timed_probe_is_not_a_learned_success=True, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False))
    write('STATUS.json', dict(status='complete', at=stamp(), automatically_promoted=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write('STATUS.json', dict(status='failed', at=stamp(), error=repr(error),
            automatically_restarted=False))
        raise
