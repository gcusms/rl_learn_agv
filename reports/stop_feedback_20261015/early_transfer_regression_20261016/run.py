"""Complete the remaining original cases after the existing auxiliary job."""
import argparse
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
BASE = Path('reports/stop_feedback_20261015')
DEPENDENCY = 'rl-reference-wait-t32-20261016.service'
INVOCATION = '5fff3c9bd509498f98fa49fe15800c46'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    path = OUT/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def state():
    return dict(line.split('=',1) for line in subprocess.check_output([
        'systemctl','--user','show',DEPENDENCY,'-p','Id','-p','ActiveState',
        '-p','MainPID','-p','Result','-p','InvocationID'],text=True).splitlines())


def verify(plan):
    for name, expected in plan['inputs'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Frozen regression input changed: '+name)


def prepare():
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Do not replace a prepared experiment')
    current = state()
    if current['ActiveState']=='active':
        if current['InvocationID']!=INVOCATION or current['MainPID']=='0':
            raise ValueError('Unexpected running auxiliary invocation')
    elif current['ActiveState']!='inactive' or current['Result']!='success' or current['MainPID']!='0':
        raise ValueError('Previous auxiliary job did not complete normally')
    auxiliary = json.loads((ROOT/BASE/'post_aux_diagnostics_fixed_20261016/RESULTS.json').read_text())
    if (not auxiliary['completed'] or len(auxiliary['candidate_rows'])!=6
            or not auxiliary['paired_navigation']['all_regression_checks_passed']
            or not all(row['safe_success'] for row in auxiliary['candidate_rows'])):
        raise ValueError('Completed six-case evidence required before extending diagnostics')
    cases = json.loads((ROOT/BASE/'w11_success_augmented_protocol_20261016/gate_cases.json').read_text())
    seen_cases = {c['id']:c for c in json.loads((ROOT/BASE/'early_transfer_diagnostic_navigation_20261016/cases.json').read_text())}
    all_ids = {c['id'] for c in cases}
    reused = [r for r in auxiliary['candidate_rows'] if r['case_id'] in all_ids]
    reused_ids = {r['case_id'] for r in reused}
    if len(cases)!=18 or len(reused)!=3:
        raise ValueError('Exactly three already completed original cases expected')
    for case in cases:
        if case['id'] in reused_ids and seen_cases[case['id']]!=case:
            raise ValueError('Reused physical-case specification differs')
    remaining = [c for c in cases if c['id'] not in reused_ids]
    write('remaining_cases.json',remaining)
    write('reused_candidate_rows.json',dict(rows=reused,already_counted_not_reexecuted=True))
    inputs = {}
    for folder in ['post_aux_diagnostics_fixed_20261016','reference_wait_t32_protocol_20261016']:
        source = ROOT/BASE/folder/'PLAN.json'
        old = json.loads(source.read_text())
        for name,expected in old['inputs'].items():
            if name in inputs and inputs[name]!=expected:
                raise ValueError('Original protocols disagree on an input')
            inputs[name]=expected
        inputs[str(source.relative_to(ROOT))]=digest(source)
    extra = [Path(__file__),OUT/'remaining_cases.json',OUT/'reused_candidate_rows.json',
        ROOT/BASE/'post_aux_diagnostics_fixed_20261016/RESULTS.json',
        ROOT/BASE/'early_transfer_diagnostic_navigation_20261016/RESULTS.json',
        ROOT/BASE/'early_transfer_diagnostic_navigation_20261016/cases.json',
        ROOT/BASE/'w11_recovery_bc800_navigation_20261016/RESULTS.json',
        ROOT/BASE/'w11_recovery_bc800_navigation_20261016/cases.json',
        ROOT/BASE/'w11_success_augmented_protocol_20261016/gate_cases.json']
    for row in reused:
        extra.append(ROOT/BASE/'early_transfer_diagnostic_navigation_20261016'/(
            row['case_id']+'_'+row['variant']+'.npz'))
    for path in extra:
        inputs[str(path.relative_to(ROOT))]=digest(path)
    plan = dict(created_at=stamp(),dependency_unit=DEPENDENCY,dependency_invocation=INVOCATION,
        source_dependency_at_prepare=current,inputs=inputs,
        bundle='models/stop_feedback_20261015/early_transfer_bc800_s231015/bc_portable',
        output=str(BASE/'early_transfer_remaining15_navigation_20261016'),
        candidate_name='early_transfer_diagnostic',remaining_case_count=15,reused_case_count=3,
        main_physical_job_unchanged=True,new_auxiliary_physical_slot=False,
        soft_cpu_cores=.15,hard_cpu_percent=20,memory_high_mib=2560,memory_max_mib=3072,
        original_offline_admission_still_failed=True,
        scope='New bounded regression diagnosis following six successful known cases; not a revision of the rejected quick gate.',
        no_automatic_training_expansion_or_promotion=True,
        new_ppo_steps=0,goal_achieved=False,deployment_approved=False)
    verify(plan)
    write('PLAN.json',plan)
    print(json.dumps(dict(prepared=True,remaining_cases=15,reused_cases=3,pinned_inputs=len(inputs))))


def run():
    if (OUT/'RESULTS.json').exists() or (ROOT/BASE/'early_transfer_remaining15_navigation_20261016').exists():
        raise FileExistsError('Do not restart an admitted regression run')
    plan = json.loads((OUT/'PLAN.json').read_text())
    verify(plan)
    while True:
        current = state()
        if current['InvocationID'] not in ('',plan['dependency_invocation']):
            raise ValueError('The preceding auxiliary job was restarted')
        if current['ActiveState'] in ('failed','inactive'):
            if current['ActiveState']!='inactive' or current['Result']!='success' or current['MainPID']!='0':
                raise ValueError('Preceding auxiliary did not exit successfully')
            break
        write('STATUS.json',dict(status='waiting',at=stamp(),dependency=current,physical_simulation_started=False))
        time.sleep(5)
    previous = json.loads((ROOT/BASE/'reference_wait_t32_protocol_20261016/RESULTS.json').read_text())
    if not previous['completed']:
        raise ValueError('Full preceding diagnostic outcome required, including failures')
    verify(plan)
    write('ADMISSION.json',dict(admitted_for_regression_diagnosis_only=True,at=stamp(),
        previous_auxiliary_complete=True,third_physical_simulation=False,
        original_offline_gate_unchanged=True,model_promotion=False))
    write('STATUS.json',dict(status='running',at=stamp(),stage='remaining15_model_only_tasks'))
    subprocess.run([sys.executable,'-m','experiments.evaluate_docking_navigation',
        '--candidate',plan['bundle'],'--candidate-name',plan['candidate_name'],
        '--cases',str(OUT/'remaining_cases.json'),'--output',plan['output'],'--cpu-cores','.15'],check=True)
    report = json.loads((ROOT/plan['output']/'RESULTS.json').read_text())
    reused = json.loads((OUT/'reused_candidate_rows.json').read_text())['rows']
    reference = json.loads((ROOT/BASE/'w11_recovery_bc800_navigation_20261016/RESULTS.json').read_text())
    if not report['completed'] or len(report['rows'])!=15:
        raise ValueError('All 15 predeclared remaining results are required')
    rows = reused+report['rows']
    original = {r['case_id'] for r in reference['rows'] if not r['case_id'].startswith('recovery_holdout_')}
    from experiments.paired_navigation_gate import compare
    write('RESULTS.json',dict(completed=True,at=stamp(),candidate_rows=rows,
        all18=compare(reference['rows'],rows),
        original14=compare([r for r in reference['rows'] if r['case_id'] in original],
                           [r for r in rows if r['case_id'] in original]),
        reused_completed_cases=3,new_model_only_cases=15,new_ppo_steps=0,
        original_offline_gate_remains_failed=True,full35_not_yet_verified=True,
        no_automatic_expansion_training_or_promotion=True,goal_achieved=False,deployment_approved=False))
    write('STATUS.json',dict(status='complete',at=stamp(),goal_achieved=False,deployment_approved=False))


if __name__=='__main__':
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    try:
        prepare() if args.action=='prepare' else run()
    except Exception as error:
        write('STATUS.json',dict(status='failed',at=stamp(),error=repr(error),automatically_restarted=False))
        raise
