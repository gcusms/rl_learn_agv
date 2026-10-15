"""Reuse the one auxiliary slot for two explicitly separated diagnoses."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(name,data):
    path = OUT/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def verify(plan):
    for name,expected in plan['inputs'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Diagnostic input changed: '+name)


def wait_dependency(plan):
    while True:
        current = dict(line.split('=',1) for line in subprocess.check_output([
            'systemctl','--user','show',plan['dependency_unit'],
            '-p','Id','-p','ActiveState','-p','MainPID','-p','Result','-p','InvocationID'],
            text=True).splitlines())
        if current['InvocationID'] not in ('',plan['dependency_invocation']):
            raise ValueError('The preceding auxiliary job was restarted')
        if current['ActiveState'] in ('failed','inactive'):
            if (current['ActiveState']!='inactive' or current['Result']!='success'
                    or current['MainPID']!='0'):
                raise RuntimeError('The exact preceding auxiliary job failed')
            return current
        write('STATUS.json',dict(status='waiting',at=stamp(),dependency=current,
            physical_simulation_started=False))
        time.sleep(5)


def main():
    os.chdir(ROOT)
    plan = json.loads((OUT/'PLAN.json').read_text())
    verify(plan)
    dependency = wait_dependency(plan)
    verify(plan)
    quick = ROOT/'reports/stop_feedback_20261015/early_decision_transfer_20261016'
    quick_status = json.loads((quick/'QUICK_STATUS.json').read_text())
    admission = json.loads((quick/'QUICK_ADMISSION.json').read_text())
    if (quick_status['status']!='rejected_before_physical_test'
            or admission['admitted'] or admission['physical_simulation_started']):
        raise ValueError('Original quick admission must remain rejected without simulation')
    write('ADMISSION.json',dict(admitted_diagnostic_only=True,at=stamp(),
        original_quick_admission_still_rejected=True,dependency=dependency,
        no_third_physical_simulation=True,soft_cpu_cores=.15,
        recorded_collision_replay_is_not_model_performance=True,
        candidate_full_navigation_is_model_only=True,
        source_training_rehearsals_not_unseen_tests=True,
        original14_and_directional_acceptance_still_required=True,
        no_automatic_larger_evaluation_training_or_promotion=True))

    def run(stage,arguments):
        verify(plan)
        write('STATUS.json',dict(status='running',stage=stage,at=stamp(),
            no_automatic_promotion=True))
        command=[sys.executable,*map(str,arguments)]
        print(json.dumps(dict(stage=stage,command=command)),flush=True)
        subprocess.run(command,check=True)

    run('exact_recorded_collision_replay',[OUT/'replay_collision.py'])
    collision=json.loads((OUT/'collision_replay/RESULTS.json').read_text())
    if not collision['completed'] or not collision['all25sampled_states_masks_commands_and_rewards_reproduced']:
        raise ValueError('Exact collision replay did not finish')
    candidate_output=ROOT/'reports/stop_feedback_20261015/early_transfer_diagnostic_navigation_20261016'
    reference_output=ROOT/'reports/stop_feedback_20261015/early_transfer_diagnostic_reference_20261016'
    run('candidate_six_complete_tasks',['-m','experiments.evaluate_docking_navigation',
        '--candidate','models/stop_feedback_20261015/early_transfer_bc800_s231015/bc_portable',
        '--candidate-name','early_transfer_diagnostic',
        '--cases',OUT/'candidate_cases.json','--output',candidate_output,'--cpu-cores','.15'])
    run('reference_three_additional_tasks',['-m','experiments.evaluate_docking_navigation',
        '--candidate','models/stop_feedback_20261015/w11_recovery_bc800_s231015/bc_portable',
        '--candidate-name','w11_bc800_reference',
        '--cases',OUT/'reference_additional_cases.json','--output',reference_output,'--cpu-cores','.15'])
    candidate=json.loads((candidate_output/'RESULTS.json').read_text())
    reference=json.loads((reference_output/'RESULTS.json').read_text())
    reused=json.loads((OUT/'reused_reference_rows.json').read_text())
    if not candidate['completed'] or len(candidate['rows'])!=6:
        raise ValueError('All six complete candidate outcomes are required')
    if not reference['completed'] or len(reference['rows'])!=3:
        raise ValueError('All three additional complete reference outcomes are required')
    from experiments.paired_navigation_gate import compare
    baseline_rows=reused['rows']+reference['rows']
    result=dict(completed=True,at=stamp(),scope='Separate diagnostic of six known training/rehearsal tasks; original quick gate remains failed.',
        collision_replay=dict(report=str((OUT/'collision_replay/RESULTS.json').relative_to(ROOT)),
            exact_recorded_states=25,model_evaluation=False,collisions=collision['collisions']),
        candidate_rows=candidate['rows'],reference_rows=baseline_rows,
        paired_navigation=compare(baseline_rows,candidate['rows']),
        original_admission_failed=True,original_fourteen_and_directional_suite_not_fully_verified=True,
        additional_eight_conditions_not_run=True,
        no_automatic_expansion_training_or_promotion=True,new_ppo_steps=0,
        goal_achieved=False,deployment_approved=False)
    write('RESULTS.json',result)
    write('STATUS.json',dict(status='complete',at=stamp(),goal_achieved=False,deployment_approved=False))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write('STATUS.json',dict(status='failed',at=stamp(),error=repr(error),
            automatically_restarted=False,goal_achieved=False,deployment_approved=False))
        raise
