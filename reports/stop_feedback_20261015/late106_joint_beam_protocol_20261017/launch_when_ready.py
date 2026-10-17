"""A short standard-library gate, called periodically by a transient timer.

No polling service stays live while the serialized neural probe waits for
quiescence. Only an actual failed current106 gate can launch offline physics.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE / 'late106_joint_beam_20261017'
TIMER = 'rl-late106-joint-beam-20261017.timer'
MODEL = 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'
DEPENDENCIES = [
    ('rl-full-navigation-joint-linear-20261017.service', '4410ede38b4a4369a6c1dfe388ab40fd'),
    ('rl-full-navigation-joint-linear-mechanism-20261017.service', '12f8bee1621e4efda6da7f4a2bcbc002'),
    ('rl-full-navigation-joint-linear-recorded-audits-20261017.service', '4da29fb265534d2a8589a637f6f7a723'),
]


def read(path):
    return json.loads(path.read_text())


def write(name, **value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        model_or_physics_not_loaded=True, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)
    path = HERE / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def stop_timer():
    subprocess.run(['systemctl', '--user', 'stop', TIMER], check=True)


def main():
    assert not {'numpy', 'torch', 'mujoco', 'rclpy'} & sys.modules.keys()
    if OUT.exists():
        stop_timer()
        write('TRIGGER_TERMINAL.json', stage='not_restarted',
            reason='Diagnostic already started; preserve terminal or interrupted evidence',
            result_exists=(OUT / 'RESULTS.json').exists(),
            failure_exists=(OUT / 'FAILURE.json').exists())
        return
    states = []
    for unit, invocation in DEPENDENCIES:
        result = subprocess.run(['systemctl', '--user', 'show', unit,
            '-p', 'MainPID', '-p', 'ActiveState', '-p', 'InvocationID',
            '-p', 'ExecMainStatus'], capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state.get('InvocationID') in ('', invocation), 'Dependency restarted'
        states.append(dict(unit=unit, **state))
        if state['MainPID'] != '0' or state['ActiveState'] not in ('inactive', 'failed'):
            write('TRIGGER_STATUS.json', stage='waiting_for_exact_dependency',
                dependency=states[-1], source_process_confirmed_live=state['MainPID'] != '0')
            return
        if state['ActiveState'] == 'failed' or state['ExecMainStatus'] != '0':
            stop_timer()
            write('TRIGGER_TERMINAL.json', stage='not_executed',
                reason='Dependency failed; no automatic restart or physics', dependencies=states)
            return
    validation = read(BASE / 'full_navigation_joint_linear_validation_20261017/RESULTS.json')
    assert validation['completed'] and validation['model'] == MODEL
    failures = [row for row in validation['groups'].get('hard3', {}).get('rows', [])
        if row['case_id'] == 'w11_training_stall_t106' and not row['safe_success']]
    if not failures:
        stop_timer()
        write('TRIGGER_TERMINAL.json', stage='not_executed',
            reason='No actual failed current106 gate; other failure or successful gate requires separate next action',
            validation_reason=validation.get('reason'), dependencies=states)
        return
    assert len(failures) == 1 and validation.get('all_fixed28_passed')
    failure = failures[0]
    assert failure['protocol_valid'] and not failure['collisions'] and not failure['false_arrival']
    folder = BASE / 'full_navigation_joint_linear_20261017'
    fixed, neural, fit = [read(folder / name) for name in
        ('FIXED28_AUDIT.json', 'NEURAL_MECHANISM.json', 'FIT_RESULTS.json')]
    assert all(value['completed'] and value['model'] == MODEL for value in (fixed, neural, fit))
    assert fixed['safe_cases'] == 28 and fit['fit_passed']
    checkpoint = ROOT / MODEL / 'checkpoint.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == fit['checkpoint_sha256']
    write('TRIGGER_STATUS.json', stage='admitted_offline_teacher_search',
        dependencies=states, actual_failed_case=failure['case_id'],
        checkpoint_sha256=fit['checkpoint_sha256'], maximum_physical_steps=4300,
        teacher_search_not_model_success=True, no_automatic_training=True)
    # Stop further timer activations before replacing this process. The
    # diagnostic's own entry point repeats exclusivity and immutable-input checks.
    stop_timer()
    os.execv(sys.executable, [sys.executable, '-u', str(HERE / 'run.py')])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        stop_timer()
        write('TRIGGER_FAILURE.json', error=repr(error), no_auto_restart=True)
        raise
