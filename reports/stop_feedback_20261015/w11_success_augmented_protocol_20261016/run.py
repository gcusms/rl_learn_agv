"""Independent demonstration fit and learned closed-loop tests, never promotion."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')
MODEL = Path('models/stop_feedback_20261015/w11_recovery_bc800_s231015')


def run(module, *arguments):
    command = [sys.executable, '-m', module, *map(str, arguments)]
    print(json.dumps(dict(starting=command)), flush=True)
    subprocess.run(command, check=True)


def main():
    os.chdir(ROOT)
    subprocess.run([sys.executable, str(PROTOCOL/'prepare.py')], check=True)
    run('experiments.train_prefix_recovery',
        '--bundle', 'models/stop_feedback_20261015/baseline/portable',
        '--checkpoint', 'models/quality_recovery_20261014/policy_2000.zip',
        '--training-cases', BASE/'proactive_protocol_20261016/ppo_training_cases.json',
        '--prefixes', PROTOCOL/'ppo_training_prefixes.json',
        '--demonstrations', PROTOCOL/'demonstrations.json',
        '--output', MODEL, '--bc-updates', '800', '--proactive-recovery',
        '--initialization-only', '--steps', '1000', '--seed', '231015', '--cpu-cores', '.15')
    run('experiments.inspect_recovery_initialization', '--model', MODEL,
        '--demonstrations', PROTOCOL/'demonstrations.json', '--retention-frames', '2030',
        '--output', BASE/'w11_recovery_bc800_fit_20261016', '--cpu-cores', '.15')
    run('experiments.evaluate_feedback_recovery', '--bundle', MODEL/'bc_portable',
        '--specifications', PROTOCOL/'w11_t90_evaluation.json',
        '--output', BASE/'w11_recovery_bc800_early_20261016',
        '--label', 'w11_augmented_bc800', '--cpu-cores', '.15')
    run('experiments.evaluate_docking_navigation', '--candidate', MODEL/'bc_portable',
        '--cases', PROTOCOL/'gate_cases.json',
        '--output', BASE/'w11_recovery_bc800_navigation_20261016',
        '--candidate-name', 'w11_augmented_bc800', '--cpu-cores', '.15')
    run('experiments.evaluate_feedback_recovery', '--bundle', MODEL/'bc_portable',
        '--specifications', BASE/'frozen_recovery_cases.json',
        '--output', BASE/'w11_recovery_bc800_prefixes_20261016',
        '--label', 'w11_augmented_bc800', '--cpu-cores', '.15')
    run('experiments.evaluate_feedback_recovery', '--bundle', MODEL/'bc_portable',
        '--specifications', BASE/'proactive_protocol_20261016/w11_prefixes.json',
        '--output', BASE/'w11_recovery_bc800_w11_20261016',
        '--label', 'w11_augmented_bc800', '--cpu-cores', '.15')
    print(json.dumps(dict(completed=True, demonstration_updates_only=True,
        additional_ppo_steps=0, automatically_promoted=False,
        goal_achieved=False, deployment_approved=False)), flush=True)


if __name__ == '__main__':
    main()
