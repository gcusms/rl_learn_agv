"""One bounded paired PPO experiment, sequential in the existing main slot."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')
PROTOCOL = BASE/'w11_success_augmented_protocol_20261016'


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def main():
    os.chdir(ROOT)
    plan = json.loads((OUT/'PLAN.json').read_text())
    def verify_inputs():
        for path, expected in plan['inputs'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Paired experiment input changed: '+path)
    verify_inputs()
    for unit in ['rl-proactive-augmented-ppo-pipeline-20261016.service',
                 'rl-late-recovery-probes-after-main-20261016.service',
                 'rl-w11-success-bc-validation-20261016.service']:
        state = dict(line.split('=', 1) for line in subprocess.check_output(
            ['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'Result', '-p', 'MainPID'],
            text=True).splitlines())
        if state != dict(ActiveState='inactive', Result='success', MainPID='0'):
            raise RuntimeError('Prior main/admission job must finish successfully: '+unit)
    write(OUT/'STATUS.json', dict(status='admitted', started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        existing_main_slot_reused=True, new_auxiliary_slot=False, automatically_promoted=False))
    def run(label, module, *arguments):
        write(OUT/'STATUS.json', dict(status='running', stage=label, module=module,
            at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), automatically_promoted=False))
        command = [sys.executable, '-m', module, *map(str, arguments)]
        print(json.dumps(dict(stage=label, command=command)), flush=True)
        subprocess.run(command, check=True)
    for variant, destination in plan['models'].items():
        verify_inputs()
        model = Path(destination)
        run(variant+'_training', 'experiments.train_prefix_recovery',
            '--bundle', 'models/stop_feedback_20261015/baseline/portable',
            '--checkpoint', 'models/quality_recovery_20261014/policy_2000.zip',
            '--training-cases', BASE/'proactive_protocol_20261016/ppo_training_cases.json',
            '--prefixes', PROTOCOL/'ppo_training_prefixes.json',
            '--demonstrations', PROTOCOL/'demonstrations.json',
            '--reference-bc', plan['reference_bc'], '--bc-updates', plan['bc_updates'],
            '--proactive-recovery', '--steps', plan['steps_per_model'], '--seed', plan['seed'],
            '--cpu-cores', '.35', '--demo-retention-coefficient', plan['retention_coefficients'][variant],
            '--demo-retention-batch-size', plan['retention_batch_size'], '--output', model)
        run(variant+'_audit', 'experiments.audit_prefix_recovery_rollouts',
            '--model-run', model, '--output', BASE/('w11_'+variant+'2000_audit_20261016'))
        run(variant+'_learning', 'experiments.analyze_recovery_learning',
            '--model-run', model, '--output', BASE/('w11_'+variant+'2000_learning_20261016'),
            '--cpu-cores', '.05')
    results = {}
    for variant, destination in plan['models'].items():
        verify_inputs()
        model = Path(destination)
        stem = 'w11_'+variant+'2000_'
        for suffix, module, options, expected in [
            ('navigation', 'experiments.evaluate_docking_navigation',
             ['--candidate', model/'portable', '--candidate-name', variant+'2000', '--cases', PROTOCOL/'gate_cases.json'], 18),
            ('prefixes', 'experiments.evaluate_feedback_recovery',
             ['--bundle', model/'portable', '--label', variant+'2000', '--specifications', BASE/'frozen_recovery_cases.json'], 2),
            ('w11', 'experiments.evaluate_feedback_recovery',
             ['--bundle', model/'portable', '--label', variant+'2000', '--specifications', BASE/'proactive_protocol_20261016/w11_prefixes.json'], 3),
            ('fresh', 'experiments.evaluate_docking_navigation',
             ['--candidate', model/'portable', '--candidate-name', variant+'2000', '--cases', PROTOCOL/'fresh_cases.json'], 12)]:
            output = BASE/(stem+suffix+'_20261016')
            run(variant+'_'+suffix, module, *options, '--output', output, '--cpu-cores', '.35')
            report = json.loads((output/'RESULTS.json').read_text())
            if not report['completed'] or len(report['rows']) != expected:
                raise ValueError('All predeclared outcomes required: '+str(output))
            results[variant+'_'+suffix] = dict(report=str(output/'RESULTS.json'),
                cases=expected, safe_successes=sum(r['safe_success'] for r in report['rows']),
                collisions=sum(r['collisions'] for r in report['rows']),
                timeouts=sum(r['timeout'] for r in report['rows']))
    write(OUT/'RESULTS.json', dict(completed=True, results=results,
        independently_trained_steps_per_model=2000, total_steps_across_two_models=4000,
        neither_model_is_an_8k_resume=True, automatically_promoted=False,
        goal_achieved=False, deployment_approved=False))
    write(OUT/'STATUS.json', dict(status='complete', at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        automatically_promoted=False, goal_achieved=False, deployment_approved=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(OUT/'STATUS.json', dict(status='failed', error=repr(error),
            automatically_promoted=False, goal_achieved=False, deployment_approved=False))
        raise
