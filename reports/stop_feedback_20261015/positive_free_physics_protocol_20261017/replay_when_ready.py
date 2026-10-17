"""Stdlib-only conditional admission of one independent safe-teacher replay."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE / 'positive_free_replay_admission_20261017'
SOURCE = 'rl-positive-free-physics-20261017.service'
EXPECTED_PID = '2256859'
EXPECTED_INVOCATION = 'c82fb27feb3a4256afa74ef44c61e528'
OWN = 'rl-positive-free-replay-admission-20261017.service'
TARGET = 'rl-positive-free-fresh-replay-20261017.service'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID',
        '-p', 'ActiveState', '-p', 'InvocationID', '-p', 'ExecMainStatus'],
        capture_output=True, text=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def write(name, value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        model_or_physics_loaded=False, new_ppo_steps=0, new_training_updates=0,
        goal_achieved=False, deployment_approved=False)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    assert not OUT.exists()
    original = state(SOURCE)
    assert original['InvocationID'] in ('', EXPECTED_INVOCATION), original
    if original['MainPID'] != '0':
        assert original['MainPID'] == EXPECTED_PID and original['ActiveState'] == 'active'
        assert original['InvocationID'] == EXPECTED_INVOCATION
        original['start_ticks'] = (Path('/proc') / EXPECTED_PID / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    else:
        assert original['ActiveState'] == 'inactive' and original['ExecMainStatus'] == '0'
    paths = [Path(__file__), HERE / 'replay_worker.py', HERE / 'run.py',
        BASE / 'positive_free_shadow_v2_20261017/evidence_memory.py']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    OUT.mkdir()
    write('PLAN.json', dict(inputs=hashes, observed_source=original,
        exact_source_pid=EXPECTED_PID, exact_source_invocation=EXPECTED_INVOCATION,
        launch_one_fresh_replay_only_after_complete_safe_teacher=True,
        no_automatic_training_admission=True))
    while True:
        current = state(SOURCE)
        assert current['InvocationID'] in ('', EXPECTED_INVOCATION), current
        assert current['MainPID'] in ('0', EXPECTED_PID), current
        if current['MainPID'] == '0' and current['ActiveState'] in ('inactive', 'failed'):
            assert current['ActiveState'] == 'inactive' and current['ExecMainStatus'] == '0', current
            break
        write('STATUS.json', dict(stage='waiting_for_exact_physical_diagnostic', dependency=current))
        time.sleep(5.)
    parent = BASE / 'positive_free_physics_20261017'
    results = read(parent / 'RESULTS.json')
    assert results['completed'] and results['changed_memory_not_adopted']
    for name, sha in hashes.items():
        assert digest(ROOT / name) == sha, name
    if not results['safe_teacher_candidate_found']:
        write('RESULTS.json', dict(completed=True, replay_launched=False,
            reason='No complete safe teacher; no replay, no training labels, no infeasibility claim',
            source_result_sha256=digest(parent / 'RESULTS.json')))
        return
    row = next(value for value in results['rows'] if value['safe_success'])
    assert row['collisions'] == 0 and row['physical_guards'] == 0
    assert row['seconds'] <= 140. + 1e-8 and row['final_xy_error_m'] < .2
    assert row['minimum_dynamic_gap_m'] >= .04 - 1e-4
    assert digest(ROOT / row['trace_path']) == row['trace_sha256']
    plan = read(parent / 'PLAN.json')
    for name, sha in plan['inputs'].items():
        assert digest(ROOT / name) == sha, name
    inputs = dict(plan['inputs'])
    inputs.update(hashes)
    for path in (parent / 'RESULTS.json', parent / 'PLAN.json',
                 parent / 'FROM_INIT_ADMISSION.json', ROOT / row['trace_path']):
        inputs[str(path.relative_to(ROOT))] = digest(path)
    job = dict(directory='reports/stop_feedback_20261015/positive_free_fresh_replay_20261017',
        expected_trace=row['trace_path'], expected_sha256=row['trace_sha256'],
        case=plan['case'], bundle=plan['frozen_source_model'] + '/portable', inputs=inputs)
    write('REPLAY_JOB.json', job)
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running', '--no-legend', '--no-pager', 'rl-*'],
        capture_output=True, text=True, check=True)
    assert all(line.split()[0] == OWN for line in running.stdout.splitlines() if line.strip())
    command = ['systemd-run', '--user', '--unit=' + TARGET.removesuffix('.service'),
        '--working-directory=' + str(ROOT), '--property=MemoryMax=3G', '--property=MemoryHigh=2800M',
        '--property=TasksMax=32', '--property=Nice=19', '--property=KillMode=control-group',
        '--property=Restart=no', '--setenv=PYTHONPATH=training:.', '--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=MUJOCO_GL=egl', '/home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python',
        '-u', str(HERE / 'replay_worker.py'), str(OUT / 'REPLAY_JOB.json')]
    subprocess.run(command, capture_output=True, text=True, check=True)
    launched = state(TARGET)
    assert launched['MainPID'] != '0' and launched['ActiveState'] == 'active'
    write('RESULTS.json', dict(completed=True, replay_launched=True,
        target_live_state=launched, no_automatic_training_admission=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write('FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
