"""Check saved original-gate traces while the exact validator progresses.

The coordinator waits using only the standard library. Its children load saved
arrays, never models or physics, and retain the original safety/efficiency gates.
Failure ends this audit without restarting or changing the physical validator.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent / 'full_navigation_joint_linear_validation_20261017'
MODEL = 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'
UNIT = 'rl-full-navigation-joint-linear-20261017.service'
INVOCATION = '4410ede38b4a4369a6c1dfe388ab40fd'
MAIN_PID = '2224412'
MAIN_START_TICKS = '50175760'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        new_training_updates=0, new_ppo_steps=0, new_physical_episodes=0,
        model_or_physics_not_loaded=True, goal_achieved=False,
        deployment_approved=False)
    path = HERE / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def dependency_state():
    result = subprocess.run(['systemctl', '--user', 'show', UNIT,
        '-p', 'MainPID', '-p', 'ActiveState', '-p', 'InvocationID',
        '-p', 'ExecMainStatus'], capture_output=True, text=True, check=True)
    state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    assert state.get('InvocationID') in ('', INVOCATION), 'Validator was restarted'
    if state['MainPID'] != '0':
        assert state['MainPID'] == MAIN_PID and state['InvocationID'] == INVOCATION
        stat = Path('/proc') / MAIN_PID / 'stat'
        assert stat.read_text().rsplit(')', 1)[1].split()[19] == MAIN_START_TICKS
        return state, False
    assert state['ActiveState'] in ('inactive', 'failed'), state
    return state, True


def main():
    assert not (HERE / 'RECORDED_AUDITS.json').exists()
    stages = [
        ('normal18', 'audit_normal18.py', 'NORMAL18_AUDIT.json', {'normal18': 18}),
        ('fixed28', 'audit_fixed28.py', 'FIXED28_AUDIT.json',
            {'normal18': 18, 'sensor5': 5, 'source_recovery3': 3, 'heading2': 2}),
    ]
    pinned = {str(HERE / script): digest(HERE / script) for _, script, _, _ in stages}
    pinned[str(HERE / 'FIT_RESULTS.json')] = digest(HERE / 'FIT_RESULTS.json')
    fit = read(HERE / 'FIT_RESULTS.json')
    assert fit['model'] == MODEL and fit['fit_passed']
    pinned[str(ROOT / MODEL / 'checkpoint.pt')] = fit['checkpoint_sha256']
    outputs = []
    for stage, script, output, expected in stages:
        assert not (HERE / output).exists(), 'Do not rerun a completed audit'
        while True:
            state, terminal = dependency_state()
            report = read(VALIDATION / 'RESULTS.json')
            assert report['model'] == MODEL
            counts = {key: len(group['rows']) for key, group in report['groups'].items()}
            ready = all(counts.get(key, 0) == count for key, count in expected.items())
            if ready:
                break
            if terminal:
                write('RECORDED_AUDITS.json', dict(completed=False,
                    reason='Exact validator ended before all original groups were recorded',
                    completed_audits=outputs, pending_stage=stage,
                    groups=counts, dependency=state, no_auto_restart=True))
                return
            assert not {'torch', 'numpy', 'mujoco', 'rclpy'} & sys.modules.keys()
            write('RECORDED_AUDITS_STATUS.json', dict(stage='waiting_for_saved_' + stage,
                groups=counts, dependency=state, completed_audits=outputs))
            time.sleep(5.)
        for path, expected_digest in pinned.items():
            assert digest(Path(path)) == expected_digest, path
        write('RECORDED_AUDITS_STATUS.json', dict(stage='checking_' + stage,
            groups=counts, dependency=state, completed_audits=outputs,
            child_soft_cpu_cores=.1, child_rss_budget_mib=550))
        with (HERE / (stage.upper() + '_AUDIT.log')).open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(HERE / script)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            write('RECORDED_AUDITS_STATUS.json', dict(stage='checking_' + stage,
                child_pid=child.pid, dependency=state, completed_audits=outputs))
            code = child.wait()
        assert code == 0, (stage, code)
        result = read(HERE / output)
        assert result['completed'] and result['model'] == MODEL
        assert result['model_or_physics_not_loaded']
        outputs.append(dict(stage=stage, output=output, sha256=digest(HERE / output)))
    write('RECORDED_AUDITS.json', dict(completed=True, completed_audits=outputs,
        original_fixed28_independently_checked=True,
        hard97_6_106_other_known_and_conditional_fresh_not_waived=True,
        input_hashes={str(Path(path).relative_to(ROOT)): value for path, value in pinned.items()}))
    write('RECORDED_AUDITS_STATUS.json', dict(stage='complete', completed_audits=outputs))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write('RECORDED_AUDITS_FAILURE.json', dict(error=repr(error), no_auto_restart=True))
        raise
