"""Stdlib-only exact-process continuation, never a second physical worker.

After the admitted search actually ends, independently replay a complete safe
winner in a fresh, separately limited process. Never restart either worker or
admit failed/partial trajectories as positive training data.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'late106_compact_beam_20261017'
DEPENDENCY = 'rl-late106-compact-beam-20261017.service'
EXPECTED_INVOCATION = 'aaf1528f6aaf4017a374919569122bd8'
REPLAY_UNIT = 'rl-late106-compact-fresh-replay-20261017.service'
OWN_UNIT = 'rl-late106-compact-replay-waiter-20261017.service'
PYTHON = '/home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '-p', 'MainPID', '-p', 'ActiveState', '-p', 'InvocationID', '-p', 'ExecMainStatus'],
        capture_output=True, text=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def write(name, value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        goal_achieved=False, deployment_approved=False, new_ppo_steps=0, new_training_updates=0)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    assert OUT.exists() and not (OUT / 'INDEPENDENT_REPLAY_RESULTS.json').exists()
    initial = state(DEPENDENCY)
    assert initial['MainPID'] != '0' and initial['ActiveState'] == 'active'
    assert initial['InvocationID'] == EXPECTED_INVOCATION
    observed_live = dict(initial)
    stat = Path('/proc') / initial['MainPID'] / 'stat'
    observed_live['start_ticks'] = stat.read_text().rsplit(')', 1)[1].split()[19]
    while True:
        current = state(DEPENDENCY)
        assert current['InvocationID'] in ('', EXPECTED_INVOCATION), current
        if current['MainPID'] == '0' and current['ActiveState'] in ('inactive', 'failed'):
            assert current['ActiveState'] == 'inactive' and current['ExecMainStatus'] == '0', current
            break
        write('INDEPENDENT_REPLAY_STATUS.json', dict(stage='waiting_for_exact_live_search',
            original_observed_live=observed_live, dependency_state=current,
            model_or_physics_loaded=False))
        time.sleep(5.)
    result = read(OUT / 'RESULTS.json')
    search = read(OUT / 'SEARCH_RESULTS.json')
    assert result['completed'] and search['completed']
    if not result['safe_teacher_candidate_found']:
        assert not search['safe_teacher_found'] and not (OUT / 'REPLAY_JOB_READY.json').exists()
        write('INDEPENDENT_REPLAY_RESULTS.json', dict(completed=True,
            replay_launched=False, reason='No complete safe teacher; no data or training admission',
            search_terminal_state=current, no_infeasibility_or_learned_success_claim=True))
        return
    ready = read(OUT / 'REPLAY_JOB_READY.json')
    job_path = ROOT / ready['job']
    job = read(job_path)
    assert job['mode'] == 'replay'
    assert digest(ROOT / job['expected_trace']) == job['expected_sha256']
    for path, sha in job['inputs'].items():
        assert digest(ROOT / path) == sha, path
    assert not (ROOT / job['directory']).exists()
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running', '--no-legend', '--no-pager', 'rl-*'], capture_output=True, text=True, check=True)
    assert all(line.split()[0] == OWN_UNIT for line in running.stdout.splitlines() if line.strip())
    worker = HERE.parent / 'late106_creep_protocol_20261017/worker.py'
    command = ['systemd-run', '--user', '--unit=' + REPLAY_UNIT.removesuffix('.service'),
        '--working-directory=' + str(ROOT), '--property=MemoryMax=3G', '--property=MemoryHigh=2800M',
        '--property=TasksMax=32', '--property=Nice=19', '--property=KillMode=control-group',
        '--property=Restart=no', '--setenv=PYTHONPATH=training:.', '--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=MUJOCO_GL=egl', PYTHON, '-u', str(worker), str(job_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    launched = state(REPLAY_UNIT)
    assert launched['MainPID'] != '0' and launched['ActiveState'] == 'active', launched
    invocation = launched['InvocationID']
    while True:
        current = state(REPLAY_UNIT)
        assert current['InvocationID'] in ('', invocation), current
        if current['MainPID'] == '0' and current['ActiveState'] in ('inactive', 'failed'):
            assert current['ActiveState'] == 'inactive' and current['ExecMainStatus'] == '0', current
            break
        write('INDEPENDENT_REPLAY_STATUS.json', dict(stage='fresh_process_replay',
            launch_state=launched, current_state=current,
            search_heavy_process_already_terminal=True, model_or_physics_loaded=False))
        time.sleep(5.)
    replay_path = ROOT / job['directory'] / 'RESULTS.json'
    replay = read(replay_path)
    assert replay['completed'] and replay['safe_teacher_success']
    assert replay['complete_fresh_process_teacher_replay_exact']
    assert replay['collisions'] == replay['physical_guards'] == 0
    assert replay['seconds'] <= 140. + 1e-8 and replay['final_xy_error_m'] < .2
    total = search['new_physical_steps'] + replay['new_physical_steps']
    assert total <= 4300
    write('INDEPENDENT_REPLAY_RESULTS.json', dict(completed=True,
        replay_launched=True, complete_safe_teacher_fresh_replay_passed=True,
        replay_result=str(replay_path.relative_to(ROOT)), replay_result_sha256=digest(replay_path),
        original_observed_search=observed_live, replay_launch_state=launched,
        replay_terminal_state=current, total_new_physical_steps=total,
        teacher_only_not_learned_success=True, no_automatic_training_admission=True))
    write('INDEPENDENT_REPLAY_STATUS.json', dict(stage='complete', complete_safe_teacher_fresh_replay_passed=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write('INDEPENDENT_REPLAY_FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
