"""Stdlib-only admission after the exact live positive-free shadow finishes."""
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
OUT = BASE / 'positive_free_physics_admission_20261017'
SOURCE = 'rl-positive-free-shadow-20261017.service'
INVOCATION = '6b1f6ec8843f46e593ba3436fc942fd7'
OWN = 'rl-positive-free-physics-admission-20261017.service'
TARGET = 'rl-positive-free-physics-20261017.service'


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
    assert original['MainPID'] != '0' and original['ActiveState'] == 'active'
    assert original['InvocationID'] == INVOCATION
    original['start_ticks'] = (Path('/proc') / original['MainPID'] / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    paths = [Path(__file__), HERE / 'run.py', BASE / 'positive_free_shadow_20261017/evidence_memory.py',
        BASE / 'positive_free_shadow_20261017/audit.py',
        BASE / 'late106_compact_beam_protocol_20261017/compact_state.py',
        BASE / 'late106_all_action_protocol_20261017/legal_goal_teacher.py']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    OUT.mkdir()
    write('PLAN.json', dict(inputs=hashes, observed_live=original,
        launch_only_after_successful_shadow_with_hard106_geometry_benefit=True,
        full_from_init_and_retention_still_required=True, no_automatic_training=True))
    while True:
        current = state(SOURCE)
        assert current['InvocationID'] in ('', INVOCATION), current
        assert current['MainPID'] in ('0', original['MainPID']), current
        if current['MainPID'] == '0' and current['ActiveState'] in ('inactive', 'failed'):
            assert current['ActiveState'] == 'inactive' and current['ExecMainStatus'] == '0', current
            break
        write('STATUS.json', dict(stage='waiting_for_exact_live_shadow', dependency=current,
            original_observed_live=original, neural_or_physics_worker_not_launched=True))
        time.sleep(5.)
    result_path = BASE / 'positive_free_shadow_20261017/RESULTS.json'
    result = read(result_path)
    assert result['completed'] and result['all_recorded_retained_point_sets_are_original_subsets']
    for path, sha in hashes.items():assert digest(ROOT / path) == sha, path
    if not any(row['trace'] in ('current106', 'creep04_failed')
            and row['sampled_extra_geometry_allowances'] > 0 for row in result['summaries']):
        write('RESULTS.json', dict(completed=True, physics_launched=False,
            reason='No demonstrated hard106geometry benefit; no automatic same experiment retry',
            shadow_result_sha256=digest(result_path)))
        return
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running', '--no-legend', '--no-pager', 'rl-*'], capture_output=True, text=True, check=True)
    assert all(line.split()[0] == OWN for line in running.stdout.splitlines() if line.strip())
    command = ['systemd-run', '--user', '--unit=' + TARGET.removesuffix('.service'),
        '--working-directory=' + str(ROOT), '--property=MemoryMax=3G', '--property=MemoryHigh=2800M',
        '--property=TasksMax=32', '--property=Nice=19', '--property=KillMode=control-group',
        '--property=Restart=no', '--setenv=PYTHONPATH=training:.', '--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=MUJOCO_GL=egl', '/home/ubuntu/rl_learn/rl_learn_agv/.venv/bin/python',
        '-u', str(HERE / 'run.py')]
    subprocess.run(command, capture_output=True, text=True, check=True)
    launched = state(TARGET)
    assert launched['MainPID'] != '0' and launched['ActiveState'] == 'active', launched
    write('RESULTS.json', dict(completed=True, physics_launched=True,
        target_live_state=launched, shadow_result_sha256=digest(result_path),
        no_model_success_or_training_admission=True))
    write('STATUS.json', dict(stage='launched_from_init_physical_diagnostic', target=launched))


if __name__ == '__main__':
    try:main()
    except Exception as error:
        if OUT.exists():write('FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
