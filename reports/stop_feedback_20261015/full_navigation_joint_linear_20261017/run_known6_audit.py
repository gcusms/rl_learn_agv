"""Wait for six actual known missions, then check saved arrays only."""
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
UNIT = 'rl-full-navigation-joint-linear-20261017.service'
INVOCATION = '4410ede38b4a4369a6c1dfe388ab40fd'
MODEL = 'models/stop_feedback_20261015/full_navigation_joint_linear_s231017'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(**value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        new_physical_episodes=0, new_ppo_steps=0, new_training_updates=0,
        model_or_physics_not_loaded=True, goal_achieved=False, deployment_approved=False)
    path = HERE / 'KNOWN6_AUDITOR_STATUS.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    assert not (HERE / 'KNOWN6_PHYSICAL_AUDIT.json').exists()
    script = HERE / 'audit_known6.py'
    pinned_script = digest(script)
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', UNIT,
            '-p', 'MainPID', '-p', 'ActiveState', '-p', 'InvocationID',
            '-p', 'ExecMainStatus'], capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state['InvocationID'] in ('', INVOCATION), 'Exact validator restarted'
        report = read(VALIDATION / 'RESULTS.json')
        assert report['model'] == MODEL
        count = len(report['groups']['observed6']['rows'])
        if count == 6:
            break
        assert count < 6
        if state['MainPID'] == '0':
            assert state['ActiveState'] in ('inactive', 'failed'), state
            status(stage='not_executed', reason='Exact validator ended before six known missions',
                completed_known_cases=count, dependency=state)
            return
        assert state['MainPID'] == '2224412' and state['InvocationID'] == INVOCATION
        stat = Path('/proc/2224412/stat').read_text().rsplit(')', 1)[1].split()
        assert stat[19] == '50175760'
        assert not {'numpy', 'torch', 'mujoco', 'rclpy'} & sys.modules.keys()
        status(stage='waiting_for_actual_known6', completed_known_cases=count, dependency=state)
        time.sleep(5.)
    assert digest(script) == pinned_script
    with (HERE / 'KNOWN6_AUDIT.log').open('x') as stream:
        child = subprocess.Popen([sys.executable, '-u', str(script)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        status(stage='checking_saved_known6', child_pid=child.pid, dependency=state)
        code = child.wait()
    assert code == 0, ('Recorded known6 audit failed', code)
    audit = read(HERE / 'KNOWN6_PHYSICAL_AUDIT.json')
    assert audit['completed'] and audit['model'] == MODEL and audit['safe_cases'] == 6
    status(stage='complete', safe_cases=6,
        audit_sha256=digest(HERE / 'KNOWN6_PHYSICAL_AUDIT.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status(stage='failed', error=repr(error), no_auto_restart=True)
        raise
