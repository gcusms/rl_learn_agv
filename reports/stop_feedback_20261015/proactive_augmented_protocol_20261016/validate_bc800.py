"""Run complete regression after the exact initialization audit invocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=Path(__file__).resolve().parent
BASE=Path('reports/stop_feedback_20261015')
MODEL=Path('models/stop_feedback_20261015/proactive_augmented_bc_800_s231015')


def run(module,*arguments):
    command=[sys.executable,'-m',module,*map(str,arguments)]
    print(json.dumps(dict(starting=command)),flush=True)
    subprocess.run(command,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-invocation',required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    unit='rl-proactive-augmented-bc800-fitcheck-20261016.service'
    while True:
        state=dict(line.split('=',1) for line in subprocess.check_output([
            'systemctl','--user','show',unit,'-p','InvocationID','-p','ActiveState','-p','Result'],text=True).splitlines())
        if state['InvocationID'] and state['InvocationID']!=args.fit_invocation:
            raise RuntimeError('Fit audit invocation changed')
        if state['ActiveState'] in ('inactive','failed'):
            if state['ActiveState']!='inactive' or state.get('Result')!='success':
                raise RuntimeError('Initialization audit failed; no automatic restart')
            break
        time.sleep(5)
    fit=json.loads((BASE/'proactive_augmented_fit800_20261016/RESULTS.json').read_text())
    if (not fit['completed'] or fit['correct']!=fit['total'] or fit['retention_errors']
            or not fit['portable_actions_identical_on_all_training_states']):
        raise ValueError('Exact retained-action fit and portable parity required before regression')
    target=BASE/'proactive_augmented_bc800_gate_20261016'
    run('experiments.evaluate_docking_navigation','--candidate',MODEL/'bc_portable',
        '--cases',PROTOCOL/'gate_cases.json','--output',target,
        '--candidate-name','augmented_bc800','--cpu-cores','.35')
    new=json.loads((target/'RESULTS.json').read_text())
    if not new['completed'] or len(new['rows'])!=18:
        raise ValueError('All original fourteen plus all four different conditions required')
    from experiments.paired_navigation_gate import compare
    original=json.loads((BASE/'prefix_recovery_threeway_20261016/RESULTS.json').read_text())
    previous=json.loads((BASE/'proactive_bc_navigation_20261016/RESULTS.json').read_text())
    conditions=json.loads((BASE/'proactive_bc_full_reset_holdout_20261016/RESULTS.json').read_text())
    reference=[r for r in original['rows'] if r['variant']=='frozen_reference']+[
        r for r in conditions['rows'] if r['variant']=='frozen_reference']
    prior=previous['rows']+[r for r in conditions['rows'] if r['variant']=='proactive_bc']
    summary=dict(completed=True,additional_ppo_steps=0,
        fitting_confidence_target_met=fit['confidence_target_met'],
        paired_gates=dict(frozen_reference=compare(reference,new['rows']),
            preceding_proactive_bc=compare(prior,new['rows'])),
        all_eighteen_safe=all(r['safe_success'] for r in new['rows']),
        development_tasks_reused=True,unseen_map_generalization=False,
        automatically_launches_ppo=False,deployment_approved=False)
    (target/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    run('experiments.evaluate_feedback_recovery','--bundle',MODEL/'bc_portable',
        '--specifications',BASE/'frozen_recovery_cases.json',
        '--output',BASE/'proactive_augmented_bc800_prefixes_20261016',
        '--label','augmented_bc800','--cpu-cores','.35')
    print(json.dumps(dict(phase='bc_complete_regression',completed=True,
        automatically_started_ppo=False)),flush=True)


if __name__=='__main__':
    main()
