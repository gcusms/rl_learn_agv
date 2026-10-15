"""Admit an experimental PPO run only after complete preservation checks.

Hard different-condition failures are learning targets, not deployment passes.
The old fourteen successes and the two safe continuations must be retained.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=Path(__file__).resolve().parent
BASE=Path('reports/stop_feedback_20261015')
BC=Path('models/stop_feedback_20261015/proactive_augmented_bc_800_s231015')
MODEL=Path('models/stop_feedback_20261015/proactive_augmented_ppo_8000_s231015')


def wait_for(unit,invocation):
    while True:
        state=dict(line.split('=',1) for line in subprocess.check_output([
            'systemctl','--user','show',unit,'-p','InvocationID','-p','ActiveState','-p','Result'],text=True).splitlines())
        if state['InvocationID'] and state['InvocationID']!=invocation:
            raise RuntimeError('Dependency invocation changed: '+unit)
        if state['ActiveState'] in ('inactive','failed'):
            if state['ActiveState']!='inactive' or state.get('Result')!='success':
                raise RuntimeError('Dependency failed; no automatic restart: '+unit)
            return
        time.sleep(5)


def run(module,*arguments):
    command=[sys.executable,'-m',module,*map(str,arguments)]
    print(json.dumps(dict(starting=command)),flush=True)
    subprocess.run(command,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-invocation',required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    wait_for('rl-proactive-augmented-bc800-validation-20261016.service',args.validation_invocation)
    gate=json.loads((BASE/'proactive_augmented_bc800_gate_20261016/RESULTS.json').read_text())
    prefixes=json.loads((BASE/'proactive_augmented_bc800_prefixes_20261016/RESULTS.json').read_text())
    originals=json.loads((BASE/'proactive_protocol_20261016/evaluation_cases.json').read_text())
    identities={c['id'] for c in originals}
    normal=[r for r in gate['rows'] if r['case_id'] in identities]
    if (not gate['completed'] or len(gate['rows'])!=18 or len(normal)!=14
            or not prefixes['completed'] or len(prefixes['rows'])!=2):
        raise ValueError('All complete preservation and different-condition results required')
    admitted=all(r['safe_success'] for r in normal+prefixes['rows'])
    admission=dict(completed=True,at=datetime.now().astimezone().isoformat(),admitted=admitted,
        normal_safe=sum(r['safe_success'] for r in normal),normal_total=14,
        prefix_safe=sum(r['safe_success'] for r in prefixes['rows']),prefix_total=2,
        different_condition_failures=[r['case_id'] for r in gate['rows'] if r['case_id'] not in identities and not r['safe_success']],
        hard_condition_results_not_counted_as_passes=True,
        training_is_experimental=True,reference_initialization=str(BC/'policy_bc.zip'),
        original_deployment_replaced=False,goal_achieved=False,deployment_approved=False)
    (PROTOCOL/'PPO_ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n')
    if not admitted:
        print(json.dumps(dict(training_started=False,reason='normal_navigation_or_safe_recovery_regressed')),flush=True)
        return
    run('experiments.train_prefix_recovery',
        '--bundle','models/stop_feedback_20261015/baseline/portable',
        '--checkpoint','models/quality_recovery_20261014/policy_2000.zip',
        '--training-cases',BASE/'proactive_protocol_20261016/ppo_training_cases.json',
        '--prefixes',PROTOCOL/'ppo_training_prefixes.json',
        '--demonstrations',PROTOCOL/'demonstrations.json',
        '--reference-bc',BC/'policy_bc.zip','--bc-updates','800',
        '--output',MODEL,'--proactive-recovery','--steps','8000','--seed','231015','--cpu-cores','.35')
    status=json.loads((MODEL/'status.json').read_text())
    if status['status']!='complete' or status['additional_ppo_steps']!=8000:
        raise ValueError('Actual 8000-step PPO completion required')
    run('experiments.audit_prefix_recovery_rollouts','--model-run',MODEL,
        '--output',BASE/'proactive_augmented_ppo8000_audit_20261016')
    run('experiments.analyze_recovery_learning','--model-run',MODEL,
        '--output',BASE/'proactive_augmented_ppo8000_learning_20261016','--cpu-cores','.35')
    target=BASE/'proactive_augmented_ppo8000_navigation_20261016'
    run('experiments.evaluate_docking_navigation','--candidate',MODEL/'portable',
        '--cases',PROTOCOL/'gate_cases.json','--output',target,
        '--candidate-name','augmented_ppo8000','--cpu-cores','.35')
    from experiments.paired_navigation_gate import compare
    final=json.loads((target/'RESULTS.json').read_text())
    (target/'SUMMARY.json').write_text(json.dumps(dict(completed=True,
        own_bc_versus_ppo=compare(gate['rows'],final['rows']),
        bc_results_reused_not_reexecuted=True,bc_checkpoint_parity_checked_before_training=True,
        additional_ppo_steps=8000,goal_achieved=False,deployment_approved=False),indent=2)+'\n')
    run('experiments.evaluate_feedback_recovery','--bundle',MODEL/'portable',
        '--specifications',BASE/'frozen_recovery_cases.json',
        '--output',BASE/'proactive_augmented_ppo8000_prefixes_20261016',
        '--label','augmented_ppo8000','--cpu-cores','.35')
    run('experiments.evaluate_feedback_recovery','--bundle',MODEL/'portable',
        '--specifications',BASE/'proactive_protocol_20261016/w11_prefixes.json',
        '--output',BASE/'proactive_augmented_ppo8000_w11_20261016',
        '--label','augmented_ppo8000','--cpu-cores','.35')
    run('experiments.evaluate_docking_navigation','--candidate',MODEL/'portable',
        '--bc',BC/'bc_portable','--cases',PROTOCOL/'fresh_cases.json',
        '--output',BASE/'proactive_augmented_ppo8000_fresh_20261016',
        '--candidate-name','augmented_ppo8000','--cpu-cores','.35')
    print(json.dumps(dict(phase='experimental_ppo_and_declared_validation',completed=True,
        automatically_promoted=False,goal_achieved=False)),flush=True)


if __name__=='__main__':
    main()
