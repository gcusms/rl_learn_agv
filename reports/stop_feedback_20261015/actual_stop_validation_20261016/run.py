"""Bounded unchanged-guard validation and an explicit actual-failure probe.

main: finish the existing 35-task evidence for the diagnostic learned model.
probe: test action9 in the genuine106s failure; timed actions are not a model.
The roles use one main and one auxiliary physical slot, with separate outputs.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = Path('reports/stop_feedback_20261015')


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    path = OUT/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def state(unit):
    return dict(line.split('=',1) for line in subprocess.check_output([
        'systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID',
        '-p','Result','-p','InvocationID'],text=True).splitlines())


def verify(plan):
    for name,expected in plan['inputs'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Frozen validation input changed: '+name)


def prepare():
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Do not replace a prepared validation')
    dependency = state('rl-early-transfer-regression-20261016.service')
    if dependency['ActiveState']!='inactive' or dependency['MainPID']!='0' or dependency['Result']!='success':
        raise ValueError('The precise preceding auxiliary run must finish first')
    full_path=ROOT/BASE/'early_transfer_regression_20261016/RESULTS.json'
    full=json.loads(full_path.read_text())
    if (not full['completed'] or len(full['candidate_rows'])!=18
            or not full['all18']['all_regression_checks_passed']
            or not full['original14']['all_regression_checks_passed']):
        raise ValueError('Complete original18 and original14 outcome regressions are required')
    specs=json.loads((ROOT/BASE/'proactive_protocol_20261016/w11_prefixes.json').read_text())
    prefix=next(s for s in specs if s['id']=='w11_training_stall_t106')
    write('actual_prefix.json',prefix)
    windows=[dict(id='actual_t106_action9_'+str(n)+'frames',choice=9,
        start_s=106.,end_s=106.+n*.2) for n in [4,12,24]]
    write('probe_windows.json',windows)
    groups=[dict(name='w11',kind='recovery',count=3,
        specifications=str(BASE/'proactive_protocol_20261016/w11_prefixes.json'),
        reference=str(BASE/'w11_recovery_bc800_w11_20261016/RESULTS.json'),
        output=str(BASE/'early_transfer_verified_w11_20261016')),
        dict(name='prefixes',kind='recovery',count=2,
        specifications=str(BASE/'frozen_recovery_cases.json'),
        reference=str(BASE/'w11_recovery_bc800_prefixes_20261016/RESULTS.json'),
        output=str(BASE/'early_transfer_verified_prefixes_20261016')),
        dict(name='fresh',kind='navigation',count=12,
        specifications=str(BASE/'w11_success_augmented_protocol_20261016/fresh_cases.json'),
        reference=str(BASE/'w11_recovery_bc800_fresh_20261016/RESULTS.json'),
        output=str(BASE/'early_transfer_verified_fresh_20261016'))]
    old_path=ROOT/BASE/'early_transfer_regression_20261016/PLAN.json'
    inputs=dict(json.loads(old_path.read_text())['inputs'])
    additional=[Path(__file__),OUT/'actual_prefix.json',OUT/'probe_windows.json',old_path,full_path,
        ROOT/BASE/'supported_motion_gap_20261016/scout.py',
        ROOT/BASE/'supported_motion_gap_20261016/RESULTS.json',ROOT/prefix['trace'],
        ROOT/'training/experiments/probe_recovery_prefix.py',
        ROOT/'training/experiments/probe_feedback_alternatives.py']
    for group in groups:
        additional += [ROOT/group['specifications'],ROOT/group['reference']]
    for path in additional:
        inputs[str(path.relative_to(ROOT))]=digest(path)
    bundle='models/stop_feedback_20261015/early_transfer_bc800_s231015/bc_portable'
    for path in (ROOT/bundle).rglob('*'):
        if path.is_file():inputs[str(path.relative_to(ROOT))]=digest(path)
    plan=dict(created_at=stamp(),inputs=inputs,bundle=bundle,groups=groups,
        preceding_auxiliary_terminal_state=dependency,original18_complete_and_passed=True,
        original_offline_all_label_gate_still_failed=True,closed_loop_outcome_gate_is_separate=True,
        model_main_soft_cpu_cores=.35,model_main_hard_cpu_percent=50,
        physical_probe_soft_cpu_cores=.15,physical_probe_hard_cpu_percent=20,
        no_third_physical_simulation=True,probe_is_explicit_teacher_diagnosis_not_model_score=True,
        unchanged_footprint_actions_masks_lidar_dynamics_and_timeout=True,
        probe_output=str(BASE/'actual_t106_action9_probe_20261016'),
        new_ppo_steps=0,no_automatic_training_or_promotion=True,
        goal_achieved=False,deployment_approved=False)
    verify(plan);write('PLAN.json',plan)
    print(json.dumps(dict(prepared=True,groups=3,model_only_cases=17,actual_failure_probes=3,pinned_inputs=len(inputs))))


def run(role):
    plan=json.loads((OUT/'PLAN.json').read_text());verify(plan)
    final=OUT/('RESULTS_'+role+'.json')
    if final.exists():raise FileExistsError('Do not restart a completed role')
    def execute(stage,arguments):
        verify(plan)
        write('STATUS_'+role+'.json',dict(status='running',stage=stage,at=stamp()))
        print(json.dumps(dict(stage=stage,command=arguments)),flush=True)
        subprocess.run([sys.executable,*map(str,arguments)],check=True)
    if role=='main':
        results={}
        from experiments.paired_navigation_gate import compare
        for group in plan['groups']:
            output=ROOT/group['output']
            if output.exists():raise FileExistsError('Never restart a prior group: '+str(output))
            module='experiments.evaluate_feedback_recovery' if group['kind']=='recovery' else 'experiments.evaluate_docking_navigation'
            arguments=['-m',module,'--bundle' if group['kind']=='recovery' else '--candidate',plan['bundle'],
                '--label' if group['kind']=='recovery' else '--candidate-name','early_transfer_verified',
                '--specifications' if group['kind']=='recovery' else '--cases',group['specifications'],
                '--output',group['output'],'--cpu-cores','.35']
            execute(group['name'],arguments)
            report=json.loads((output/'RESULTS.json').read_text())
            if not report['completed'] or len(report['rows'])!=group['count']:
                raise ValueError('Incomplete fixed validation group')
            reference=json.loads((ROOT/group['reference']).read_text())
            reference_rows=[r for r in reference['rows'] if r['variant']=='w11_augmented_bc800']
            if len(reference_rows)!=group['count']:raise ValueError('Reference group or variant differs')
            results[group['name']]=dict(rows=report['rows'],paired=compare(reference_rows,report['rows']))
        write(final.name,dict(completed=True,at=stamp(),groups=results,
            original18_report=str(BASE/'early_transfer_regression_20261016/RESULTS.json'),
            fixed35_groups_completed=True,rows_are_correlated_not_independent_generalization=True,
            original_offline_label_gate_unchanged=True,new_ppo_steps=0,goal_achieved=False,deployment_approved=False))
    else:
        execute('three_original_guard_actual_failure_windows',['-m','experiments.probe_recovery_prefix',
            '--bundle',plan['bundle'],'--prefix',OUT/'actual_prefix.json',
            '--specifications',OUT/'probe_windows.json','--output',plan['probe_output'],'--cpu-cores','.15'])
        report=json.loads((ROOT/plan['probe_output']/'RESULTS.json').read_text())
        if not report['completed'] or len(report['rows'])!=3 or not report['not_a_trained_policy']:
            raise ValueError('All three actual-failure interventions must complete')
        rows=[]
        for row in report['rows']:
            entries=row['intervention']
            rows.append({**{k:row[k] for k in ['case_id','safe_success','collisions','physical_guards',
                'timeout','seconds','travel_m','final_xy_error_m','trace_sha256','specification']},
                'requested_frames':len(entries),'accepted_frames':sum(e['requested_choice_legal'] for e in entries)})
        write(final.name,dict(completed=True,at=stamp(),rows=rows,original_guard_unchanged=True,
            no_labels_admitted=True,not_a_trained_policy=True,new_ppo_steps=0,
            goal_achieved=False,deployment_approved=False))
    write('STATUS_'+role+'.json',dict(status='complete',at=stamp(),goal_achieved=False,deployment_approved=False))


if __name__=='__main__':
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','main','probe'])
    args=parser.parse_args()
    try:
        prepare() if args.action=='prepare' else run(args.action)
    except Exception as error:
        write('STATUS_'+args.action+'.json',dict(status='failed',at=stamp(),error=repr(error),automatically_restarted=False))
        raise
