"""Run complete BC missions after the exact local initialization invocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')
MODEL = Path('models/stop_feedback_20261015/proactive_bc_s231015_prepared')


def wait_for(invocation,unit='rl-proactive-bc-prepared-20261016.service'):
    while True:
        state = dict(line.split('=', 1) for line in subprocess.check_output([
            'systemctl', '--user', 'show', unit,
            '-p', 'InvocationID', '-p', 'ActiveState', '-p', 'Result'], text=True).splitlines())
        if state['InvocationID'] and state['InvocationID'] != invocation:
            raise RuntimeError('BC dependency invocation changed')
        if state['ActiveState'] in ('inactive', 'failed'):
            if state['ActiveState'] != 'inactive' or state['Result'] != 'success':
                raise RuntimeError('BC initialization failed; do not restart it automatically')
            return
        time.sleep(5)


def run(module, *arguments):
    command = [sys.executable, '-m', module, *map(str, arguments)]
    print(json.dumps(dict(starting=command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['navigation', 'holdout'], required=True)
    parser.add_argument('--initialization-invocation', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    wait_for(args.initialization_invocation)
    status = json.loads((MODEL/'status.json').read_text())
    fitting = json.loads((MODEL/'INITIALIZATION.json').read_text())
    final = fitting['history'][-1]
    if (status['status'] != 'initialization_verified' or status['additional_ppo_steps'] != 0
            or final['correct_choices'] != final['training_examples']):
        raise ValueError('Verified physics initialization and exact BC label fit required')
    if args.phase == 'navigation':
        target = BASE/'proactive_bc_navigation_20261016'
        run('experiments.evaluate_docking_navigation', '--candidate', MODEL/'bc_portable',
            '--cases', PROTOCOL/'evaluation_cases.json', '--output', target,
            '--candidate-name', 'proactive_bc', '--cpu-cores', '.35')
        from experiments.paired_navigation_gate import compare
        old = json.loads((BASE/'prefix_recovery_threeway_20261016/RESULTS.json').read_text())
        new = json.loads((target/'RESULTS.json').read_text())
        if not new['completed'] or len(new['rows']) != 14:
            raise ValueError('All original complete development tasks required')
        gates = {name:compare([r for r in old['rows'] if r['variant']==name], new['rows'])
            for name in ['frozen_reference','bc_only','recovery_ppo']}
        (target/'SUMMARY.json').write_text(json.dumps(dict(completed=True,paired_gates=gates,
            training_examples_reused=True,unseen_map_generalization=False,
            additional_ppo_steps=0,deployment_approved=False),indent=2)+'\n')
        run('experiments.evaluate_feedback_recovery','--bundle',MODEL/'bc_portable',
            '--specifications',BASE/'frozen_recovery_cases.json',
            '--output',BASE/'proactive_bc_prefix_recovery_20261016',
            '--label','proactive_bc','--cpu-cores','.35')
    else:
        run('experiments.evaluate_docking_navigation','--candidate',MODEL/'bc_portable',
            '--baseline', 'models/stop_feedback_20261015/ppo_4000_s221015/initial_portable',
            '--cases',BASE/'recovery_holdout_reference_20261016/cases.json',
            '--output',BASE/'proactive_bc_full_reset_holdout_20261016',
            '--candidate-name','proactive_bc','--baseline-name','frozen_reference',
            '--cpu-cores','.15')
    print(json.dumps(dict(phase=args.phase,completed=True)),flush=True)


if __name__ == '__main__':
    main()
