"""Repair a pre-simulation Path logging error without editing live main inputs."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(name,value):
    path=OUT/name;temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    os.chdir(ROOT)
    plan=json.loads((OUT/'PLAN.json').read_text())
    def verify():
        for name,expected in plan['inputs'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('A fixed original input changed: '+name)
    verify()
    if (OUT/'RESULTS_probe_fixed.json').exists() or (ROOT/plan['probe_output']).exists():
        raise FileExistsError('Do not restart a physical probe or overwrite results')
    failed=json.loads((OUT/'STATUS_probe.json').read_text())
    if failed['status']!='failed' or 'PosixPath is not JSON serializable' not in failed['error']:
        raise ValueError('This adapter only fixes the recorded pre-simulation logger failure')
    original_state=dict(line.split('=',1) for line in subprocess.check_output([
        'systemctl','--user','show','rl-actual-stop-validation-probe-20261016.service',
        '-p','ActiveState','-p','MainPID','-p','Result','-p','ExecMainStatus'],text=True).splitlines())
    if (original_state['ActiveState']!='failed' or original_state['MainPID']!='0'
            or original_state['Result']!='exit-code' or original_state['ExecMainStatus']!='1'):
        raise ValueError('Original launcher must be authoritatively terminal before replacement')
    write('PROBE_FIXED_ADMISSION.json',dict(at=stamp(),original_terminal_state=original_state,
        previous_physical_output_absent=True,previous_failed_status=failed,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256((OUT/'PLAN.json').read_bytes()).hexdigest(),
        original182inputs_verified=True,main_driver_and_plan_unchanged=True,
        physical_auxiliary_slot_reused=True,new_ppo_steps=0))
    write('STATUS_probe_fixed.json',dict(status='running',at=stamp()))
    arguments=list(map(str,['-m','experiments.probe_recovery_prefix','--bundle',plan['bundle'],
        '--prefix',OUT/'actual_prefix.json','--specifications',OUT/'probe_windows.json',
        '--output',plan['probe_output'],'--cpu-cores','.15']))
    print(json.dumps(dict(command=arguments)),flush=True)
    subprocess.run([sys.executable,*arguments],check=True)
    verify()
    report=json.loads((ROOT/plan['probe_output']/'RESULTS.json').read_text())
    if not report['completed'] or len(report['rows'])!=3 or not report['not_a_trained_policy']:
        raise ValueError('Three completed explicit intervention outcomes are required')
    rows=[]
    for row in report['rows']:
        entries=row['intervention']
        rows.append({**{k:row[k] for k in ['case_id','safe_success','collisions','physical_guards',
            'timeout','seconds','travel_m','final_xy_error_m','trace_sha256','specification']},
            'requested_frames':len(entries),'accepted_frames':sum(e['requested_choice_legal'] for e in entries)})
    write('RESULTS_probe_fixed.json',dict(completed=True,at=stamp(),rows=rows,
        original_guard_unchanged=True,no_labels_admitted=True,not_a_trained_policy=True,
        new_ppo_steps=0,goal_achieved=False,deployment_approved=False))
    write('STATUS_probe_fixed.json',dict(status='complete',at=stamp(),goal_achieved=False))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write('STATUS_probe_fixed.json',dict(status='failed',at=stamp(),error=repr(error),automatically_restarted=False))
        raise
