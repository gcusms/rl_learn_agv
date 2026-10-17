"""Run the admitted retention correction and all original gates serially."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OWN_UNIT = 'rl-full-navigation-joint-linear-20261017.service'


def write(value):
    value.update(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        goal_achieved=False, deployment_approved=False)
    path = HERE / 'CYCLE_STATUS.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    for unit, invocation in [
        ('rl-full-navigation-margin-resume-cycle-20261017.service', 'ef95cccf487943edb1a993b9732a2520'),
        ('rl-full-navigation-joint-retention-20261017.service', 'b1503901b57f4ca0a815f869dfe94c54'),
        ('rl-full-navigation-mechanism-post-20261017.service', 'a4e3d1b2f3db4520ac7289f9bc3b5938')]:
        result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p',
            'ActiveState', '-p', 'InvocationID', '-p', 'ExecMainStatus'], capture_output=True, text=True, check=True)
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        assert state['InvocationID'] in ('', invocation), state
        assert state['MainPID'] == '0' and state['ActiveState'] == 'inactive' and state['ExecMainStatus'] == '0', state
    running = subprocess.run(['systemctl', '--user', 'list-units', '--type=service', '--state=running',
        '--no-legend', '--no-pager', 'rl-*'], capture_output=True, text=True, check=True)
    assert all(line.split()[0] == OWN_UNIT for line in running.stdout.splitlines() if line.strip()), running.stdout
    source = json.loads((BASE / 'full_navigation_margin_resume_validation_20261017/RESULTS.json').read_text())
    neural = json.loads((BASE / 'full_navigation_residual_20261017/NEURAL_MECHANISM.json').read_text())
    assert source['completed'] and source['all_fixed28_passed'] and neural['completed']
    assert any(r['case_id'] == 'pre_ppo_holdout_dynamic_02' and r['timeout']
        for r in source['groups']['observed6']['rows'])
    for stage in ('fit_linear', 'validate_joint'):
        if stage == 'validate_joint':
            fit = json.loads((HERE / 'FIT_RESULTS.json').read_text())
            if not all(fit[key] for key in ('completed', 'fit_passed', 'export_parity_passed', 'strict_runtime_admission_passed')):
                write(dict(status='complete', reason='Fit or export rejected; no physical expansion'))
                return
        child = subprocess.Popen([sys.executable, '-u', str(HERE / (stage + '.py'))], cwd=ROOT)
        stat = Path('/proc') / str(child.pid) / 'stat'
        tick = stat.read_text().rsplit(')', 1)[1].split()[19] if stat.exists() else None
        write(dict(status='running', stage=stage, child_pid=child.pid, child_start_ticks=tick,
            one_heavy_child_at_a_time=True))
        result = child.wait()
        if result:
            raise RuntimeError(f'{stage} child failed with {result}; no automatic restart')
    write(dict(status='complete', reason='Declared cycle finished; inspect full goal requirements'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(dict(status='failed', error=repr(error), no_auto_restart=True))
        raise
